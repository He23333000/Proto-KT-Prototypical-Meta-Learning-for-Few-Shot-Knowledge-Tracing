"""
Ablation study: Effect of number of prototypes (k).

Tests k ∈ {1, 2, 4, 8, 16} to determine optimal number of prototypes.
Generates Table 2.
"""
import torch
import yaml
import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import ProtoKT
from data import create_meta_dataloaders
from evaluation import FewShotEvaluator


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    _, _, test_loader, num_questions = create_meta_dataloaders(
        args.data_path,
        support_size=config['data']['support_size'],
        meta_batch_size=1,
        num_workers=0
    )
    
    print(f"\nAblation study: Testing k ∈ {args.k_values}")
    print(f"Evaluating on {len(test_loader)} test students")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    ablation_results = {}
    
    # Evaluate for each k
    for k in args.k_values:
        print("\n" + "="*60)
        print(f"Evaluating Proto-KT with k={k} prototypes")
        print("="*60)
        
        # Find checkpoint
        checkpoint_path = Path(args.checkpoint_dir) / f'proto_kt_k{k}' / 'best_model.pt'
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}, skipping...")
            continue
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = ProtoKT(
            num_questions=num_questions,
            num_prototypes=k,
            embed_dim=config['model']['sakt']['embedding_dim'],
            context_dim=config['model']['proto_kt']['context_dim'],
            num_heads=config['model']['sakt']['num_heads'],
            num_layers=config['model']['sakt']['num_layers'],
            dropout=config['model']['sakt']['dropout'],
            use_alignment=config['model']['proto_kt']['use_parameter_alignment']
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        
        # Evaluate
        evaluator = FewShotEvaluator(
            meta_model=model,
            device=device,
            inner_lr=config['training']['inner_lr'],
            inner_steps=config['training']['inner_steps'],
            support_size=config['data']['support_size'],
            max_eval_len=51
        )
        
        results = evaluator.evaluate_test_set(
            test_loader,
            save_path=output_dir / f'proto_kt_k{k}_results.pkl'
        )
        
        evaluator.print_results(results)
        ablation_results[k] = results
    
    # Generate ablation table
    if len(ablation_results) > 0:
        print("\n" + "="*60)
        print("Generating ablation table...")
        print("="*60)
        
        create_ablation_table(ablation_results, output_dir / 'table_2_ablation.tex')
        plot_ablation_results(ablation_results, output_dir / 'ablation_plot.png')
    
    print(f"\nAblation study completed! Results saved to {output_dir}")


def create_ablation_table(results_dict, save_path):
    """
    Create Table 2: Ablation study on number of prototypes.
    """
    table_data = []
    
    for k in sorted(results_dict.keys()):
        results = results_dict[k]
        row = {'k': k}
        
        # Get AUC at different windows
        for window in [10, 20, 50]:
            window_key = f'interactions_1-{window}'
            if window_key in results['window_metrics']:
                auc = results['window_metrics'][window_key]['auc']
                acc = results['window_metrics'][window_key]['accuracy']
                row[f'AUC@{window}'] = f"{auc:.4f}" if auc else "N/A"
                row[f'Acc@{window}'] = f"{acc:.4f}"
            else:
                row[f'AUC@{window}'] = "N/A"
                row[f'Acc@{window}'] = "N/A"
        
        # Overall
        overall = results['overall_metrics']
        row['Overall AUC'] = f"{overall['auc']:.4f}" if overall['auc'] else "N/A"
        row['Overall Acc'] = f"{overall['accuracy']:.4f}"
        
        table_data.append(row)
    
    # Save as LaTeX
    with open(save_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation study: Effect of number of prototypes $k$ on few-shot performance.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{c|cc|cc|cc|cc}\n")
        f.write("\\toprule\n")
        f.write("$k$ & \\multicolumn{2}{c|}{1-10 Int.} & \\multicolumn{2}{c|}{1-20 Int.} & "
                "\\multicolumn{2}{c|}{1-50 Int.} & \\multicolumn{2}{c}{Overall} \\\\\n")
        f.write(" & AUC & Acc & AUC & Acc & AUC & Acc & AUC & Acc \\\\\n")
        f.write("\\midrule\n")
        
        for row in table_data:
            f.write(f"{row['k']} & {row['AUC@10']} & {row['Acc@10']} & "
                   f"{row['AUC@20']} & {row['Acc@20']} & "
                   f"{row['AUC@50']} & {row['Acc@50']} & "
                   f"{row['Overall AUC']} & {row['Overall Acc']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved ablation table to {save_path}")
    
    # CSV version
    import csv
    csv_path = save_path.replace('.tex', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
        writer.writeheader()
        writer.writerows(table_data)
    
    print(f"Saved CSV table to {csv_path}")


def plot_ablation_results(results_dict, save_path):
    """
    Plot AUC vs. number of prototypes for different interaction windows.
    """
    windows = [10, 20, 50]
    k_values = sorted(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for window in windows:
        aucs = []
        for k in k_values:
            window_key = f'interactions_1-{window}'
            if window_key in results_dict[k]['window_metrics']:
                auc = results_dict[k]['window_metrics'][window_key]['auc']
                aucs.append(auc if auc else 0)
            else:
                aucs.append(0)
        
        ax.plot(k_values, aucs, marker='o', label=f'1-{window} interactions',
               linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Prototypes (k)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Ablation Study: AUC vs. Number of Prototypes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ablation plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study on k")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints for different k values')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                       help='Values of k to test')
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    
    args = parser.parse_args()
    
    main(args)

