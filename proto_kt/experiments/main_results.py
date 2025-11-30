"""
Main results experiment: Compare Proto-KT against baselines.

Generates:
- Learning curves (AUC vs. interactions)
- Table 1: Performance at different interaction windows
"""
import torch
import yaml
import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import ProtoKT, MAML_SAKT, SAKT
from data import create_meta_dataloaders
from evaluation import FewShotEvaluator, bootstrap_confidence_interval
from training import MetaLearner


def load_model(model_type, checkpoint_path, config, num_questions):
    """Load a trained model from checkpoint."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'proto_kt':
        model = ProtoKT(
            num_questions=num_questions,
            num_prototypes=config.get('num_prototypes', 8),
            embed_dim=config['model']['sakt']['embedding_dim'],
            context_dim=config['model']['proto_kt']['context_dim'],
            num_heads=config['model']['sakt']['num_heads'],
            num_layers=config['model']['sakt']['num_layers'],
            dropout=config['model']['sakt']['dropout'],
            use_alignment=config['model']['proto_kt']['use_parameter_alignment']
        )
    elif model_type == 'maml':
        model = MAML_SAKT(
            num_questions=num_questions,
            embed_dim=config['model']['sakt']['embedding_dim'],
            num_heads=config['model']['sakt']['num_heads'],
            num_layers=config['model']['sakt']['num_layers'],
            dropout=config['model']['sakt']['dropout']
        )
    elif model_type == 'sakt':
        model = SAKT(
            num_questions=num_questions,
            embed_dim=config['model']['sakt']['embedding_dim'],
            num_heads=config['model']['sakt']['num_heads'],
            num_layers=config['model']['sakt']['num_layers'],
            dropout=config['model']['sakt']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'sakt':
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} from {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model


def evaluate_sakt_baseline(model, test_loader, config, save_path):
    """Evaluate SAKT baseline (no meta-learning adaptation)."""
    from evaluation.metrics import compute_all_metrics
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    all_results = []
    support_size = config['data']['support_size']
    
    with torch.no_grad():
        for student_data in tqdm(test_loader, desc="Evaluating on test students"):
            question_ids = student_data['question_ids'][0].to(device)
            responses = student_data['responses'][0].to(device)
            
            seq_len = question_ids.shape[0]
            
            predictions = []
            targets = []
            interaction_indices = []
            
            # Evaluate sequentially, using all past interactions
            for t in range(support_size, min(seq_len, 51)):
                past_q = question_ids[:t].unsqueeze(0)
                past_r = responses[:t].unsqueeze(0)
                next_q = question_ids[t:t+1].unsqueeze(0)
                
                pred = model(past_q, past_r, next_q)
                target = responses[t].item()
                
                predictions.append(pred.item())
                targets.append(target)
                interaction_indices.append(t + 1)
            
            if len(predictions) > 0:
                all_results.append({
                    'predictions': np.array(predictions),
                    'targets': np.array(targets),
                    'interaction_indices': np.array(interaction_indices)
                })
    
    # Aggregate results
    aggregated = {}
    
    # Overall metrics
    all_preds = np.concatenate([r['predictions'] for r in all_results])
    all_targets = np.concatenate([r['targets'] for r in all_results])
    aggregated['overall'] = compute_all_metrics(all_preds, all_targets)
    
    # By interaction window
    for window_name, (start, end) in [('1-10', (1, 10)), ('1-20', (1, 20)), ('1-50', (1, 50))]:
        window_preds = []
        window_targets = []
        
        for result in all_results:
            mask = (result['interaction_indices'] >= start) & (result['interaction_indices'] <= end)
            if mask.any():
                window_preds.append(result['predictions'][mask])
                window_targets.append(result['targets'][mask])
        
        if len(window_preds) > 0:
            window_preds = np.concatenate(window_preds)
            window_targets = np.concatenate(window_targets)
            aggregated[f'interactions_{window_name}'] = compute_all_metrics(window_preds, window_targets)
    
    # Learning curve (AUC at each interaction step)
    from sklearn.metrics import roc_auc_score
    max_interaction = max([r['interaction_indices'].max() for r in all_results])
    learning_curve = {'interaction': [], 'auc': []}
    
    for t in range(support_size + 1, int(max_interaction) + 1):
        preds_at_t = []
        targets_at_t = []
        
        for result in all_results:
            mask = result['interaction_indices'] <= t
            if mask.any():
                preds_at_t.append(result['predictions'][mask])
                targets_at_t.append(result['targets'][mask])
        
        if len(preds_at_t) > 0:
            preds_at_t = np.concatenate(preds_at_t)
            targets_at_t = np.concatenate(targets_at_t)
            
            # Only compute AUC if we have both classes
            if len(np.unique(targets_at_t)) > 1:
                auc = roc_auc_score(targets_at_t, preds_at_t)
            else:
                auc = None
            
            learning_curve['interaction'].append(t)
            learning_curve['auc'].append(auc)
    
    aggregated['learning_curve'] = learning_curve
    aggregated['num_students'] = len(all_results)
    aggregated['by_student'] = all_results
    
    # Save results
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(aggregated, f)
        print(f"Results saved to {save_path}")
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nNumber of test students: {aggregated['num_students']}")
    print("\nOverall Metrics:")
    for k, v in aggregated['overall'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nMetrics by Interaction Window:")
    for window in ['interactions_1-10', 'interactions_1-20', 'interactions_1-50']:
        if window in aggregated:
            print(f"\n  {window}:")
            for k, v in aggregated[window].items():
                print(f"    {k}: {v:.4f}")
    print("\n" + "="*60)
    
    return aggregated


def evaluate_model(model, test_loader, config, save_path):
    """Evaluate a model on test set."""
    evaluator = FewShotEvaluator(
        meta_model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        inner_lr=config['training']['inner_lr'],
        inner_steps=config['training']['inner_steps'],
        support_size=config['data']['support_size'],
        max_eval_len=51
    )
    
    results = evaluator.evaluate_test_set(test_loader, save_path=save_path)
    evaluator.print_results(results)
    
    return results


def plot_learning_curves(results_dict, save_path):
    """
    Plot learning curves for all methods.
    
    Args:
        results_dict: Dict mapping method names to their results
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    colors = {
        'Proto-KT (k=8)': '#2E86AB',
        'MAML-SAKT': '#A23B72',
        'SAKT Baseline': '#F18F01'
    }
    
    for method_name, results in results_dict.items():
        curve = results['learning_curve']
        interactions = curve['interaction']
        aucs = [auc if auc is not None else 0 for auc in curve['auc']]
        
        color = colors.get(method_name, None)
        plt.plot(interactions, aucs, marker='o', label=method_name, 
                linewidth=2, markersize=4, color=color, alpha=0.8)
    
    plt.xlabel('Number of Interactions', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('Few-Shot Learning Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {save_path}")
    
    plt.close()


def create_results_table(results_dict, save_path):
    """
    Create Table 1: Performance at different interaction windows.
    
    Args:
        results_dict: Dict mapping method names to their results
        save_path: Path to save table
    """
    # Extract metrics for each method and window
    table_data = []
    
    for method_name, results in results_dict.items():
        row = {'Method': method_name}
        
        # Handle different result formats
        # Meta-learning models (Proto-KT, MAML) have 'overall_metrics' and 'window_metrics'
        # SAKT baseline has 'overall' and 'interactions_X-Y' directly
        if 'overall_metrics' in results:
            # Meta-learning format
            overall = results['overall_metrics']
            row['Overall AUC'] = f"{overall['auc']:.4f}" if overall['auc'] else "N/A"
            row['Overall Acc'] = f"{overall['accuracy']:.4f}"
            
            # Windows
            for window in [10, 20, 50]:
                window_key = f'interactions_1-{window}'
                if window_key in results['window_metrics']:
                    metrics = results['window_metrics'][window_key]
                    row[f'AUC@{window}'] = f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A"
                    row[f'Acc@{window}'] = f"{metrics['accuracy']:.4f}"
                else:
                    row[f'AUC@{window}'] = "N/A"
                    row[f'Acc@{window}'] = "N/A"
        else:
            # SAKT baseline format
            overall = results['overall']
            row['Overall AUC'] = f"{overall['auc']:.4f}" if overall['auc'] else "N/A"
            row['Overall Acc'] = f"{overall['accuracy']:.4f}"
            
            # Windows
            for window in [10, 20, 50]:
                window_key = f'interactions_1-{window}'
                if window_key in results:
                    metrics = results[window_key]
                    row[f'AUC@{window}'] = f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A"
                    row[f'Acc@{window}'] = f"{metrics['accuracy']:.4f}"
                else:
                    row[f'AUC@{window}'] = "N/A"
                    row[f'Acc@{window}'] = "N/A"
        
        table_data.append(row)
    
    # Save as LaTeX table
    with open(save_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Few-shot performance comparison across methods and interaction windows.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{l|cc|cc|cc|cc}\n")
        f.write("\\toprule\n")
        f.write("Method & \\multicolumn{2}{c|}{1-10 Int.} & \\multicolumn{2}{c|}{1-20 Int.} & "
                "\\multicolumn{2}{c|}{1-50 Int.} & \\multicolumn{2}{c}{Overall} \\\\\n")
        f.write(" & AUC & Acc & AUC & Acc & AUC & Acc & AUC & Acc \\\\\n")
        f.write("\\midrule\n")
        
        for row in table_data:
            f.write(f"{row['Method']} & {row['AUC@10']} & {row['Acc@10']} & "
                   f"{row['AUC@20']} & {row['Acc@20']} & "
                   f"{row['AUC@50']} & {row['Acc@50']} & "
                   f"{row['Overall AUC']} & {row['Overall Acc']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved results table to {save_path}")
    
    # Also save as CSV for easy viewing
    csv_path = Path(str(save_path).replace('.tex', '.csv'))
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
        writer.writeheader()
        writer.writerows(table_data)
    
    print(f"Saved CSV table to {csv_path}")


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    _, _, test_loader, num_questions = create_meta_dataloaders(
        args.data_path,
        support_size=config['data']['support_size'],
        meta_batch_size=1,  # Evaluate one at a time
        num_workers=0
    )
    
    print(f"\nEvaluating on {len(test_loader)} test students")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    results_dict = {}
    
    # Proto-KT
    if args.proto_kt_checkpoint:
        print("\n" + "="*60)
        print("Evaluating Proto-KT...")
        print("="*60)
        
        proto_config = config.copy()
        proto_config['num_prototypes'] = args.num_prototypes
        
        proto_model = load_model('proto_kt', args.proto_kt_checkpoint, proto_config, num_questions)
        proto_results = evaluate_model(
            proto_model,
            test_loader,
            config,
            save_path=output_dir / 'proto_kt_results.pkl'
        )
        results_dict[f'Proto-KT (k={args.num_prototypes})'] = proto_results
    
    # MAML-SAKT
    if args.maml_checkpoint:
        print("\n" + "="*60)
        print("Evaluating MAML-SAKT...")
        print("="*60)
        
        maml_model = load_model('maml', args.maml_checkpoint, config, num_questions)
        maml_results = evaluate_model(
            maml_model,
            test_loader,
            config,
            save_path=output_dir / 'maml_results.pkl'
        )
        results_dict['MAML-SAKT'] = maml_results
    
    # SAKT Baseline (no meta-learning)
    if args.sakt_checkpoint:
        print("\n" + "="*60)
        print("Evaluating SAKT Baseline...")
        print("="*60)
        
        sakt_model = load_model('sakt', args.sakt_checkpoint, config, num_questions)
        sakt_results = evaluate_sakt_baseline(
            sakt_model,
            test_loader,
            config,
            save_path=output_dir / 'sakt_results.pkl'
        )
        results_dict['SAKT Baseline'] = sakt_results
    
    # Generate plots and tables
    if len(results_dict) > 0:
        print("\n" + "="*60)
        print("Generating visualizations...")
        print("="*60)
        
        plot_learning_curves(results_dict, output_dir / 'learning_curves.png')
        create_results_table(results_dict, output_dir / 'table_1_main_results.tex')
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main results experiment")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--sakt_checkpoint', type=str, default=None)
    parser.add_argument('--proto_kt_checkpoint', type=str, default=None)
    parser.add_argument('--maml_checkpoint', type=str, default=None)
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='results/main')
    
    args = parser.parse_args()
    
    main(args)

