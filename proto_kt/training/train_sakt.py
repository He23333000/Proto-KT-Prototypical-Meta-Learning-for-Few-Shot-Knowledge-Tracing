"""
Training script for baseline SAKT model (pre-train + fine-tune baseline).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import SAKT
from data import create_meta_dataloaders


def train_sakt_epoch(model, train_loader, optimizer, device, criterion):
    """Train SAKT for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Concatenate support and query for standard training
        support_q = batch['support']['question_ids'].to(device)
        support_r = batch['support']['responses'].to(device)
        query_q = batch['query']['question_ids'].to(device)
        query_r = batch['query']['responses'].to(device)
        query_mask = batch['query']['mask'].to(device)
        
        batch_size = support_q.size(0)
        
        batch_loss = 0
        batch_correct = 0
        batch_total = 0
        
        for i in range(batch_size):
            # Concatenate support and query for this student
            full_q = torch.cat([support_q[i:i+1], query_q[i:i+1]], dim=1)
            full_r = torch.cat([support_r[i:i+1], query_r[i:i+1]], dim=1)
            
            # Create mask for full sequence
            support_mask = torch.ones_like(support_q[i:i+1])
            full_mask = torch.cat([support_mask, query_mask[i:i+1]], dim=1)
            
            seq_len = full_q.size(1)
            
            # Predict each position
            for t in range(1, seq_len):
                if full_mask[0, t] == 0:
                    continue
                
                past_q = full_q[:, :t]
                past_r = torch.zeros((1, t), dtype=torch.long, device=device)
                if t > 1:
                    past_r[:, 1:] = full_r[:, :t-1]
                
                next_q = full_q[:, t:t+1]
                
                pred = model(past_q, past_r, next_q)
                target = full_r[:, t].float()
                
                # Ensure prediction tensor matches target shape (avoid scalar squeeze)
                pred = pred.view_as(target)
                loss = criterion(pred, target)
                batch_loss = batch_loss + loss
                
                pred_binary = (pred.squeeze() > 0.5).long()
                batch_correct += (pred_binary == target.long()).sum().item()
                batch_total += 1
        
        if batch_total > 0:
            avg_batch_loss = batch_loss / batch_total
            
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += avg_batch_loss.item()
            correct += batch_correct
            total += batch_total
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def evaluate_sakt(model, val_loader, device, criterion):
    """Evaluate SAKT on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            support_q = batch['support']['question_ids'].to(device)
            support_r = batch['support']['responses'].to(device)
            query_q = batch['query']['question_ids'].to(device)
            query_r = batch['query']['responses'].to(device)
            query_mask = batch['query']['mask'].to(device)
            
            batch_size = support_q.size(0)
            
            for i in range(batch_size):
                full_q = torch.cat([support_q[i:i+1], query_q[i:i+1]], dim=1)
                full_r = torch.cat([support_r[i:i+1], query_r[i:i+1]], dim=1)
                support_mask = torch.ones_like(support_q[i:i+1])
                full_mask = torch.cat([support_mask, query_mask[i:i+1]], dim=1)
                
                seq_len = full_q.size(1)
                
                for t in range(1, seq_len):
                    if full_mask[0, t] == 0:
                        continue
                    
                    past_q = full_q[:, :t]
                    past_r = torch.zeros((1, t), dtype=torch.long, device=device)
                    if t > 1:
                        past_r[:, 1:] = full_r[:, :t-1]
                    
                    next_q = full_q[:, t:t+1]
                    
                    pred = model(past_q, past_r, next_q)
                    target = full_r[:, t].float()
                    
                    pred = pred.view_as(target)
                    loss = criterion(pred, target)
                    total_loss += loss.item()
                    
                    pred_binary = (pred.squeeze() > 0.5).long()
                    correct += (pred_binary == target.long()).sum().item()
                    total += 1
    
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print(f"\nLoading data from {args.data_path}...")
    train_loader, val_loader, test_loader, num_questions = create_meta_dataloaders(
        args.data_path,
        support_size=config['data']['support_size'],
        meta_batch_size=config['training']['meta_batch_size'],
        num_workers=0,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        subset_seed=args.subset_seed
    )
    
    # Create SAKT model
    print(f"\nInitializing SAKT model...")
    model = SAKT(
        num_questions=num_questions,
        embed_dim=config['model']['sakt']['embedding_dim'],
        num_heads=config['model']['sakt']['num_heads'],
        num_layers=config['model']['sakt']['num_layers'],
        dropout=config['model']['sakt']['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer and criterion
    optimizer = Adam(model.parameters(), lr=config['training']['meta_lr'])
    criterion = nn.BCELoss()
    
    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience = 0
    
    print(f"\n{'='*60}")
    print("Starting SAKT training...")
    print(f"{'='*60}\n")
    
    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_sakt_epoch(
            model, train_loader, optimizer, device, criterion
        )
        
        # Validate
        val_loss, val_acc = evaluate_sakt(
            model, val_loader, device, criterion
        )
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            
            torch.save(checkpoint, save_dir / 'best_model.pt')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience += 1
            print(f"  Early stopping counter: {patience}/{config['training']['early_stopping_patience']}")
        
        # Early stopping
        if patience >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nTraining completed! Model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline SAKT")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to processed dataset (.pkl)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='checkpoints/sakt',
                        help='Directory to save checkpoints')
    parser.add_argument('--train_fraction', type=float, default=1.0,
                        help='Fraction of train students to use (e.g., 0.1 for 10%)')
    parser.add_argument('--val_fraction', type=float, default=1.0,
                        help='Fraction of val students to use')
    parser.add_argument('--test_fraction', type=float, default=1.0,
                        help='Fraction of test students to use for evaluation')
    parser.add_argument('--subset_seed', type=int, default=42,
                        help='Random seed for subset sampling')
    
    args = parser.parse_args()
    
    main(args)

