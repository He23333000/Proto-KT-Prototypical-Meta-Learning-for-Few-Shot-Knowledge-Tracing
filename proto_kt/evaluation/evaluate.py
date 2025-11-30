"""
Evaluation framework for few-shot knowledge tracing.

Evaluates meta-learned models on new students with sequential adaptation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle
import higher

from .metrics import compute_all_metrics


class FewShotEvaluator:
    """
    Evaluates meta-learned models in the few-shot setting.
    
    For each test student:
    1. Use first K interactions as support set
    2. Adapt model on support set
    3. Evaluate on remaining interactions sequentially
    4. Track performance at each interaction step
    """
    
    def __init__(
        self,
        meta_model,
        device='cuda',
        inner_lr=0.001,
        inner_steps=1,
        support_size=5,
        max_eval_len=51
    ):
        self.meta_model = meta_model
        self.device = device
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.support_size = support_size
        self.max_eval_len = max_eval_len
        
        self.meta_model.eval()
    
    def adapt_and_evaluate_student(self, student_data):
        """
        Adapt to a single student and evaluate sequentially.
        
        Args:
            student_data: Dict with 'question_ids', 'responses', 'length', 'user_id'
            
        Returns:
            results: Dict with predictions and targets at each step
        """
        question_ids = student_data['question_ids'].to(self.device)
        responses = student_data['responses'].to(self.device)
        length = min(student_data['length'].item(), self.max_eval_len)
        
        # Split into support and query
        support_q = question_ids[:, :self.support_size]
        support_r = responses[:, :self.support_size]
        
        # Generate initial parameters
        with torch.no_grad():
            proto_output = self.meta_model(support_q, support_r)
            theta_init = proto_output['theta_init'][0]  # Single student
        
        # Create task-specific model
        task_model = self.meta_model.create_task_model(theta_init)
        task_model = task_model.to(self.device)
        
        # Inner loop adaptation on support
        task_model.train()
        with higher.innerloop_ctx(
            task_model,
            SGD(task_model.parameters(), lr=self.inner_lr),
            copy_initial_weights=False,
            track_higher_grads=False
        ) as (fmodel, diffopt):
            
            for step in range(self.inner_steps):
                loss = 0
                for t in range(1, self.support_size):
                    past_q = support_q[:, :t]
                    past_r = torch.zeros((1, t), dtype=torch.long, device=self.device)
                    if t > 1:
                        past_r[:, 1:] = support_r[:, :t-1]
                    
                    next_q = support_q[:, t:t+1]
                    pred = fmodel(past_q, past_r, next_q)
                    target = support_r[:, t].float()
                    
                    loss = loss + F.binary_cross_entropy(pred.squeeze(), target.squeeze())
                
                if self.support_size > 1:
                    loss = loss / (self.support_size - 1)
                    diffopt.step(loss)
            
            # Now evaluate on query set (interactions after support)
            fmodel.eval()
            
            predictions = []
            targets = []
            interaction_indices = []
            
            with torch.no_grad():
                for t in range(self.support_size + 1, length):
                    # Use all previous interactions to predict current
                    past_q = question_ids[:, :t]
                    past_r = torch.zeros((1, t), dtype=torch.long, device=self.device)
                    past_r[:, 1:] = responses[:, :t-1]
                    
                    next_q = question_ids[:, t:t+1]
                    
                    pred = fmodel(past_q, past_r, next_q)
                    target = responses[:, t]
                    
                    predictions.append(pred.squeeze().cpu().item())
                    targets.append(target.cpu().item())
                    interaction_indices.append(t)
        
        results = {
            'predictions': predictions,
            'targets': targets,
            'interaction_indices': interaction_indices,
            'user_id': student_data['user_id'],
            'attention_weights': proto_output.get('attention_weights', None)
        }
        
        return results
    
    def evaluate_test_set(self, test_loader, save_path=None):
        """
        Evaluate on entire test set.
        
        Args:
            test_loader: DataLoader for test students
            save_path: Optional path to save detailed results
            
        Returns:
            aggregated_results: Dict with aggregated metrics
        """
        all_results = []
        
        print(f"Evaluating on {len(test_loader)} test students...")
        
        for student_data in tqdm(test_loader):
            results = self.adapt_and_evaluate_student(student_data)
            all_results.append(results)
        
        # Aggregate results
        aggregated = self.aggregate_results(all_results)
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'individual_results': all_results,
                    'aggregated_results': aggregated
                }, f)
            
            print(f"Results saved to {save_path}")
        
        return aggregated
    
    def aggregate_results(self, all_results):
        """
        Aggregate results across all students.
        
        Computes:
        - Overall metrics (AUC, accuracy, etc.)
        - Metrics by interaction window (1-10, 1-20, 1-50)
        - Learning curves (performance vs. number of interactions)
        """
        # Collect all predictions and targets
        all_preds = []
        all_targets = []
        
        for result in all_results:
            all_preds.extend(result['predictions'])
            all_targets.extend(result['targets'])
        
        # Overall metrics
        overall_metrics = compute_all_metrics(all_preds, all_targets)
        
        # Metrics by interaction window
        window_metrics = {}
        
        for window in [10, 20, 50]:
            window_preds = []
            window_targets = []
            
            for result in all_results:
                # Take predictions up to interaction window
                for i, idx in enumerate(result['interaction_indices']):
                    if idx <= self.support_size + window:
                        window_preds.append(result['predictions'][i])
                        window_targets.append(result['targets'][i])
            
            if len(window_preds) > 0:
                window_metrics[f'interactions_1-{window}'] = compute_all_metrics(
                    window_preds, window_targets
                )
        
        # Learning curve: AUC at each interaction position
        max_interactions = max(
            max(result['interaction_indices']) if result['interaction_indices'] else 0
            for result in all_results
        )
        
        learning_curve = {'interaction': [], 'auc': [], 'accuracy': [], 'count': []}
        
        for t in range(self.support_size + 1, min(max_interactions + 1, self.max_eval_len)):
            preds_at_t = []
            targets_at_t = []
            
            for result in all_results:
                for i, idx in enumerate(result['interaction_indices']):
                    if idx <= t:
                        preds_at_t.append(result['predictions'][i])
                        targets_at_t.append(result['targets'][i])
            
            if len(preds_at_t) > 0:
                metrics_at_t = compute_all_metrics(preds_at_t, targets_at_t)
                
                learning_curve['interaction'].append(t)
                learning_curve['auc'].append(metrics_at_t['auc'])
                learning_curve['accuracy'].append(metrics_at_t['accuracy'])
                learning_curve['count'].append(len(preds_at_t))
        
        # Per-student metrics for statistical analysis
        student_metrics = []
        
        for result in all_results:
            if len(result['predictions']) > 0:
                student_met = compute_all_metrics(
                    result['predictions'],
                    result['targets']
                )
                student_met['user_id'] = result['user_id']
                student_met['num_predictions'] = len(result['predictions'])
                student_metrics.append(student_met)
        
        return {
            'overall_metrics': overall_metrics,
            'window_metrics': window_metrics,
            'learning_curve': learning_curve,
            'student_metrics': student_metrics,
            'num_students': len(all_results)
        }
    
    def print_results(self, aggregated_results):
        """Print aggregated results in a readable format."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nNumber of test students: {aggregated_results['num_students']}")
        
        print("\nOverall Metrics:")
        for metric, value in aggregated_results['overall_metrics'].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        
        print("\nMetrics by Interaction Window:")
        for window, metrics in aggregated_results['window_metrics'].items():
            print(f"\n  {window}:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"    {metric}: {value:.4f}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print("Use from experiment scripts for actual evaluation")

