"""
Training script for MAML-SAKT baseline.
"""
import torch
import yaml
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.maml import MAML_SAKT
from data import create_meta_dataloaders
from training import MetaLearner


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
    
    # Create MAML-SAKT model
    print(f"\nInitializing MAML-SAKT model...")
    print(f"  Embedding dim: {config['model']['sakt']['embedding_dim']}")
    
    model = MAML_SAKT(
        num_questions=num_questions,
        embed_dim=config['model']['sakt']['embedding_dim'],
        num_heads=config['model']['sakt']['num_heads'],
        num_layers=config['model']['sakt']['num_layers'],
        dropout=config['model']['sakt']['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create meta-learner (same training procedure as Proto-KT)
    meta_learner = MetaLearner(
        meta_model=model,
        device=device,
        inner_lr=config['training']['inner_lr'],
        meta_lr=config['training']['meta_lr'],
        inner_steps=config['training']['inner_steps'],
        use_first_order=config['training']['use_first_order'],
        gradient_checkpointing=config['training']['gradient_checkpointing']
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting MAML meta-training...")
    print(f"{'='*60}\n")
    
    history = meta_learner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        save_dir=args.save_dir
    )
    
    print(f"\nTraining completed! Models saved to {args.save_dir}")
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAML-SAKT")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to processed dataset (.pkl)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='checkpoints/maml',
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

