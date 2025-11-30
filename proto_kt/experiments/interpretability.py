"""
Interpretability analysis: Visualize and characterize learned prototypes.

Generates:
- Figure 1: UMAP visualization of student contexts colored by prototype assignment
- Figure 2: Attention weight distributions
- Table 3: Prototype cluster characteristics
"""
import torch
import yaml
import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import ProtoKT
from data import create_meta_dataloaders
from tqdm import tqdm


def extract_contexts_and_prototypes(model, test_loader, device):
    """
    Extract context vectors and prototype assignments for all test students.
    """
    model.eval()
    
    contexts = []
    attention_weights = []
    student_stats = []
    
    print("Extracting contexts and prototype assignments...")
    
    with torch.no_grad():
        for student_data in tqdm(test_loader):
            question_ids = student_data['question_ids'].to(device)
            responses = student_data['responses'].to(device)
            
            # Use support set (first K interactions)
            support_size = 5
            support_q = question_ids[:, :support_size]
            support_r = responses[:, :support_size]
            
            # Forward through Proto-KT
            output = model(support_q, support_r)
            
            context = output['context'][0].cpu().numpy()
            attn = output['attention_weights'][0].cpu().numpy()
            
            contexts.append(context)
            attention_weights.append(attn)
            
            # Compute student statistics
            support_acc = support_r[0].float().mean().cpu().item()
            
            # Get full sequence stats (if available)
            full_r = responses[0, :min(student_data['length'].item(), 51)]
            final_acc = full_r.float().mean().cpu().item()
            
            student_stats.append({
                'user_id': student_data['user_id'],
                'support_accuracy': support_acc,
                'final_accuracy': final_acc,
                'assigned_prototype': int(np.argmax(attn))
            })
    
    contexts = np.array(contexts)
    attention_weights = np.array(attention_weights)
    
    return contexts, attention_weights, student_stats


def plot_umap_visualization(contexts, student_stats, save_path):
    """
    Create UMAP visualization of student contexts colored by prototype assignment.
    """
    from umap import UMAP
    
    print("Computing UMAP projection...")
    
    # Reduce to 2D
    reducer = UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(contexts)
    
    # Get prototype assignments
    prototype_assignments = [s['assigned_prototype'] for s in student_stats]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=prototype_assignments,
        cmap='tab10',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('Student Context Embeddings Colored by Prototype Assignment', 
                fontsize=14, fontweight='bold')
    
    # Color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Assigned Prototype', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP visualization to {save_path}")
    plt.close()


def plot_attention_distributions(attention_weights, save_path):
    """
    Plot attention weight distributions for each prototype.
    """
    num_prototypes = attention_weights.shape[1]
    
    fig, axes = plt.subplots(2, (num_prototypes + 1) // 2, figsize=(14, 6))
    axes = axes.flatten()
    
    for k in range(num_prototypes):
        ax = axes[k]
        ax.hist(attention_weights[:, k], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Attention Weight', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Prototype {k+1}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for k in range(num_prototypes, len(axes)):
        axes[k].axis('off')
    
    plt.suptitle('Attention Weight Distributions Across Prototypes', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention distributions to {save_path}")
    plt.close()


def analyze_prototype_characteristics(student_stats, save_path):
    """
    Analyze characteristics of students assigned to each prototype.
    Creates Table 3.
    """
    from collections import defaultdict
    
    # Group by prototype
    prototype_groups = defaultdict(list)
    for stats in student_stats:
        prototype_groups[stats['assigned_prototype']].append(stats)
    
    # Compute statistics for each prototype
    table_data = []
    
    for k in sorted(prototype_groups.keys()):
        group = prototype_groups[k]
        
        support_accs = [s['support_accuracy'] for s in group]
        final_accs = [s['final_accuracy'] for s in group]
        
        # Learning gain
        learning_gains = [f - s for s, f in zip(support_accs, final_accs)]
        
        row = {
            'Prototype': k + 1,
            'Count': len(group),
            'Avg Support Acc': f"{np.mean(support_accs):.3f}",
            'Std Support Acc': f"{np.std(support_accs):.3f}",
            'Avg Final Acc': f"{np.mean(final_accs):.3f}",
            'Std Final Acc': f"{np.std(final_accs):.3f}",
            'Avg Learning Gain': f"{np.mean(learning_gains):.3f}",
            'Characterization': characterize_prototype(
                np.mean(support_accs),
                np.mean(final_accs),
                np.mean(learning_gains)
            )
        }
        
        table_data.append(row)
    
    # Save as LaTeX
    with open(save_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Learned prototype characteristics. Each prototype captures distinct student behaviors.}\n")
        f.write("\\label{tab:prototype_characteristics}\n")
        f.write("\\begin{tabular}{c|c|cc|cc|c|l}\n")
        f.write("\\toprule\n")
        f.write("Prototype & Count & \\multicolumn{2}{c|}{Support Acc} & \\multicolumn{2}{c|}{Final Acc} & Learning & Characterization \\\\\n")
        f.write(" & & Mean & Std & Mean & Std & Gain & \\\\\n")
        f.write("\\midrule\n")
        
        for row in table_data:
            f.write(f"{row['Prototype']} & {row['Count']} & "
                   f"{row['Avg Support Acc']} & {row['Std Support Acc']} & "
                   f"{row['Avg Final Acc']} & {row['Std Final Acc']} & "
                   f"{row['Avg Learning Gain']} & {row['Characterization']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved prototype characteristics table to {save_path}")
    
    # CSV version
    import csv
    csv_path = save_path.replace('.tex', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
        writer.writeheader()
        writer.writerows(table_data)
    
    print(f"Saved CSV table to {csv_path}")


def characterize_prototype(support_acc, final_acc, learning_gain):
    """
    Provide a semantic characterization of a prototype based on metrics.
    """
    if support_acc > 0.7:
        if learning_gain > 0.05:
            return "High-performing improvers"
        else:
            return "Consistent high-performers"
    elif support_acc < 0.4:
        if learning_gain > 0.1:
            return "Struggling improvers"
        else:
            return "Persistent strugglers"
    else:
        if learning_gain > 0.05:
            return "Moderate learners"
        else:
            return "Average performers"


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    _, _, test_loader, num_questions = create_meta_dataloaders(
        args.data_path,
        support_size=config['data']['support_size'],
        meta_batch_size=1,
        num_workers=0
    )
    
    # Load Proto-KT model
    print(f"\nLoading Proto-KT model with k={args.num_prototypes}...")
    
    model = ProtoKT(
        num_questions=num_questions,
        num_prototypes=args.num_prototypes,
        embed_dim=config['model']['sakt']['embedding_dim'],
        context_dim=config['model']['proto_kt']['context_dim'],
        num_heads=config['model']['sakt']['num_heads'],
        num_layers=config['model']['sakt']['num_layers'],
        dropout=config['model']['sakt']['dropout'],
        use_alignment=config['model']['proto_kt']['use_parameter_alignment']
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Loaded model from {args.checkpoint_path}")
    
    # Extract contexts and prototype assignments
    contexts, attention_weights, student_stats = extract_contexts_and_prototypes(
        model, test_loader, device
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations and tables
    print("\n" + "="*60)
    print("Generating interpretability analyses...")
    print("="*60)
    
    plot_umap_visualization(contexts, student_stats, output_dir / 'figure_1_umap.png')
    plot_attention_distributions(attention_weights, output_dir / 'figure_2_attention_dist.png')
    analyze_prototype_characteristics(student_stats, output_dir / 'table_3_prototypes.tex')
    
    # Save raw data
    with open(output_dir / 'interpretability_data.pkl', 'wb') as f:
        pickle.dump({
            'contexts': contexts,
            'attention_weights': attention_weights,
            'student_stats': student_stats
        }, f)
    
    print(f"\nInterpretability analysis completed! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretability analysis")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='results/interpretability')
    
    args = parser.parse_args()
    
    main(args)

