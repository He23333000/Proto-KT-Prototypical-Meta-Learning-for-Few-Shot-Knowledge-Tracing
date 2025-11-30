"""
Meta-learning training for Proto-KT and MAML baselines.

Implements bi-level optimization with inner/outer loops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import higher  # For differentiable optimization
from tqdm import tqdm
import numpy as np
from pathlib import Path


class MetaLearner:
    """
    Meta-learner for Proto-KT and MAML-SAKT.
    
    Implements Algorithm 1 from the paper with bi-level optimization.
    """
    
    def __init__(
        self,
        meta_model,
        device='cuda',
        inner_lr=0.001,
        meta_lr=0.0001,
        inner_steps=1,
        use_first_order=False,  # Set True for memory efficiency
        gradient_checkpointing=False
    ):
        """
        Args:
            meta_model: ProtoKT or MAML model
            device: 'cuda' or 'cpu'
            inner_lr: Learning rate for inner loop adaptation
            meta_lr: Learning rate for outer loop meta-update
            inner_steps: Number of gradient steps in inner loop
            use_first_order: Use first-order approximation (FOMAML)
            gradient_checkpointing: Enable gradient checkpointing (saves memory)
        """
        self.meta_model = meta_model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.use_first_order = use_first_order
        self.gradient_checkpointing = gradient_checkpointing
        
        # Meta-optimizer (for outer loop)
        self.meta_optimizer = Adam(self.meta_model.parameters(), lr=meta_lr)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def inner_loop_adapt(self, task_model, support_data, create_graph=True):
        """
        Perform inner loop adaptation on support set.
        
        Args:
            task_model: SAKT model to adapt
            support_data: Dict with 'question_ids', 'responses'
            create_graph: Whether to create computational graph for meta-gradient
            
        Returns:
            adapted_model: Model after inner loop updates
            inner_loss: Final loss on support set
        """
        question_ids = support_data['question_ids'].to(self.device)
        responses = support_data['responses'].to(self.device)
        
        # Create differentiable optimizer using 'higher'
        with higher.innerloop_ctx(
            task_model,
            SGD(task_model.parameters(), lr=self.inner_lr),
            copy_initial_weights=False,
            track_higher_grads=create_graph and not self.use_first_order
        ) as (fmodel, diffopt):
            
            for step in range(self.inner_steps):
                # Forward pass
                # Predict each position using previous context
                loss = 0
                batch_size, seq_len = question_ids.shape
                
                for t in range(1, seq_len):
                    # Use interactions 0:t to predict response at position t
                    past_q = question_ids[:, :t]
                    next_q = question_ids[:, t:t+1]
                    
                    # Build past responses - should have same length as past_q
                    # Response at index i corresponds to question at index i
                    # We use zeros for positions where we don't have response yet
                    past_r = torch.zeros((batch_size, t), dtype=torch.long, device=self.device)
                    if t > 1:
                        # Fill in actual responses (excluding the current position)
                        past_r[:, :t-1] = responses[:, :t-1]
                    # past_r[:, t-1] stays 0 (we don't use current response to predict itself)
                    
                    # Predict
                    pred = fmodel(past_q, past_r, next_q)
                    target = responses[:, t].float()
                    
                    # Ensure shapes match
                    pred = pred.view_as(target)
                    loss = loss + F.binary_cross_entropy(pred, target)
                
                loss = loss / (seq_len - 1)
                
                # Inner loop update
                diffopt.step(loss)
            
            inner_loss = loss.item()
            
            return fmodel, inner_loss
    
    def compute_query_loss(self, task_model, query_data):
        """
        Compute loss on query set.
        
        Args:
            task_model: Adapted SAKT model
            query_data: Dict with 'question_ids', 'responses', 'mask', 'lengths'
            
        Returns:
            loss: Scalar loss
            metrics: Dict with additional metrics (accuracy, etc.)
        """
        question_ids = query_data['question_ids'].to(self.device)
        responses = query_data['responses'].to(self.device)
        mask = query_data['mask'].to(self.device)
        
        batch_size, seq_len = question_ids.shape
        
        total_loss = 0
        correct = 0
        total = 0
        
        for t in range(1, seq_len):
            # Mask check: only compute loss on valid positions
            valid_mask = mask[:, t]
            if valid_mask.sum() == 0:
                continue
            
            # Use previous interactions to predict current
            past_q = question_ids[:, :t]
            past_r_indices = torch.arange(t-1, device=self.device)
            past_r = torch.zeros((batch_size, t), dtype=torch.long, device=self.device)
            
            if t > 1:
                past_r[:, 1:] = responses[:, :t-1]
            
            next_q = question_ids[:, t:t+1]
            
            # Predict
            pred = task_model(past_q, past_r, next_q)
            target = responses[:, t].float()
            
            # Ensure pred is properly shaped
            pred = pred.view(-1)  # Flatten to 1D
            
            # Apply mask
            pred_masked = pred[valid_mask.bool()]
            target_masked = target[valid_mask.bool()]
            
            if len(pred_masked) > 0:
                loss = F.binary_cross_entropy(pred_masked, target_masked)
                total_loss = total_loss + loss
                
                # Compute accuracy
                pred_binary = (pred_masked > 0.5).long()
                correct += (pred_binary == target_masked.long()).sum().item()
                total += len(pred_masked)
        
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, {'accuracy': accuracy, 'total': total}
    
    def meta_train_step(self, batch):
        """
        Single meta-training step on a batch of tasks.
        
        Args:
            batch: Dict with 'support' and 'query' data
            
        Returns:
            meta_loss: Scalar meta-loss
            metrics: Dict with training metrics
        """
        support_data = batch['support']
        query_data = batch['query']
        batch_size = support_data['question_ids'].size(0)
        
        # Generate initial parameters for each task
        proto_output = self.meta_model(
            support_data['question_ids'].to(self.device),
            support_data['responses'].to(self.device)
        )
        
        theta_inits = proto_output['theta_init']  # (batch_size, total_params)
        
        meta_loss = 0
        inner_losses = []
        accuracies = []
        
        # Process each task in the batch
        for i in range(batch_size):
            # Create task-specific model with generated initialization
            task_model = self.meta_model.create_task_model(theta_inits[i])
            task_model = task_model.to(self.device)
            
            # Extract single task data
            task_support = {
                'question_ids': support_data['question_ids'][i:i+1],
                'responses': support_data['responses'][i:i+1]
            }
            
            task_query = {
                'question_ids': query_data['question_ids'][i:i+1],
                'responses': query_data['responses'][i:i+1],
                'mask': query_data['mask'][i:i+1],
                'lengths': query_data['lengths'][i:i+1]
            }
            
            # Inner loop: adapt on support set
            adapted_model, inner_loss = self.inner_loop_adapt(
                task_model,
                task_support,
                create_graph=True
            )
            
            # Outer loop: evaluate on query set
            query_loss, query_metrics = self.compute_query_loss(adapted_model, task_query)
            
            meta_loss = meta_loss + query_loss
            inner_losses.append(inner_loss)
            accuracies.append(query_metrics['accuracy'])
        
        # Average meta-loss over batch
        meta_loss = meta_loss / batch_size
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        metrics = {
            'meta_loss': meta_loss.item(),
            'inner_loss': np.mean(inner_losses),
            'query_accuracy': np.mean(accuracies)
        }
        
        return meta_loss.item(), metrics
    
    def meta_validate(self, val_loader, num_batches=None):
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation dataloader
            num_batches: Number of batches to evaluate (None = all)
            
        Returns:
            metrics: Dict with validation metrics
        """
        self.meta_model.eval()
        
        total_loss = 0
        total_acc = 0
        num_tasks = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if num_batches and batch_idx >= num_batches:
                    break
                
                support_data = batch['support']
                query_data = batch['query']
                batch_size = support_data['question_ids'].size(0)
                
                # Generate initial parameters
                proto_output = self.meta_model(
                    support_data['question_ids'].to(self.device),
                    support_data['responses'].to(self.device)
                )
                
                theta_inits = proto_output['theta_init']
                
                for i in range(batch_size):
                    # Create and adapt task model
                    task_model = self.meta_model.create_task_model(theta_inits[i])
                    task_model = task_model.to(self.device)
                    task_model.eval()
                    
                    task_support = {
                        'question_ids': support_data['question_ids'][i:i+1],
                        'responses': support_data['responses'][i:i+1]
                    }
                    
                    task_query = {
                        'question_ids': query_data['question_ids'][i:i+1],
                        'responses': query_data['responses'][i:i+1],
                        'mask': query_data['mask'][i:i+1],
                        'lengths': query_data['lengths'][i:i+1]
                    }
                    
                    # Adapt (without gradients for validation)
                    with torch.enable_grad():
                        adapted_model, _ = self.inner_loop_adapt(
                            task_model,
                            task_support,
                            create_graph=False
                        )
                    
                    # Evaluate
                    query_loss, query_metrics = self.compute_query_loss(adapted_model, task_query)
                    
                    total_loss += query_loss.item()
                    total_acc += query_metrics['accuracy']
                    num_tasks += 1
        
        self.meta_model.train()
        
        return {
            'val_loss': total_loss / max(num_tasks, 1),
            'val_accuracy': total_acc / max(num_tasks, 1),
            'num_tasks': num_tasks
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs=100,
        early_stopping_patience=10,
        save_dir='checkpoints',
        log_interval=10
    ):
        """
        Full meta-training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of meta-training epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save checkpoints
            log_interval: Log metrics every N batches
            
        Returns:
            training_history: Dict with training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"Starting meta-training for {num_epochs} epochs...")
        print(f"Inner LR: {self.inner_lr}, Meta LR: {self.meta_optimizer.param_groups[0]['lr']}")
        print(f"Inner steps: {self.inner_steps}, First-order: {self.use_first_order}")
        
        for epoch in range(num_epochs):
            self.meta_model.train()
            epoch_losses = []
            epoch_accs = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Meta-training step
                meta_loss, metrics = self.meta_train_step(batch)
                
                epoch_losses.append(metrics['meta_loss'])
                epoch_accs.append(metrics['query_accuracy'])
                
                if batch_idx % log_interval == 0:
                    pbar.set_postfix({
                        'meta_loss': f"{metrics['meta_loss']:.4f}",
                        'inner_loss': f"{metrics['inner_loss']:.4f}",
                        'query_acc': f"{metrics['query_accuracy']:.4f}"
                    })
            
            # Epoch metrics
            train_loss = np.mean(epoch_losses)
            train_acc = np.mean(epoch_accs)
            
            # Validation
            val_metrics = self.meta_validate(val_loader, num_batches=50)
            val_loss = val_metrics['val_loss']
            val_acc = val_metrics['val_accuracy']
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.meta_model.state_dict(),
                    'optimizer_state': self.meta_optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                
                torch.save(checkpoint, save_dir / 'best_model.pt')
                print(f"  Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Early stopping counter: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        print("\nMeta-training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Load best model
        best_checkpoint = torch.load(save_dir / 'best_model.pt')
        self.meta_model.load_state_dict(best_checkpoint['model_state'])
        
        return history


if __name__ == "__main__":
    # Test meta-learner (placeholder - needs actual data)
    print("Meta-learner module loaded successfully")
    print("Run from training scripts for actual meta-training")

