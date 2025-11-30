"""
Complete NeurIPS-Level Experimental Pipeline for Proto-KT

Runs all experiments required by main.tex:
1. Train all baseline models (SAKT, MAML, Proto-KT)
2. Run ablation studies (Proto-KT with k=1,2,4,8,16)
3. Evaluate and generate all tables/figures
4. Statistical significance testing

Usage:
    python run_complete_experiments.py --quick  # 5% subset for testing
    python run_complete_experiments.py --full   # 100% for final results
"""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime


class ExperimentPipeline:
    """Top-tier experimental pipeline for NeurIPS submission."""
    
    def __init__(self, data_fraction=0.05, config_path='configs/config.yaml'):
        """
        Args:
            data_fraction: Fraction of data to use (0.05=5%, 1.0=100%)
            config_path: Path to config file
        """
        self.data_fraction = data_fraction
        self.config_path = config_path
        self.base_dir = Path(__file__).parent
        self.data_path = self.base_dir / 'data' / 'processed' / 'assistments2009_processed.pkl'
        
        # Results directory
        fraction_str = f"{int(data_fraction*100)}pct" if data_fraction < 1.0 else "full"
        self.results_dir = self.base_dir / 'results' / f'neurips_{fraction_str}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoints directory
        self.ckpt_dir = self.base_dir / 'checkpoints' / fraction_str
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment log
        self.log_file = self.results_dir / 'experiment_log.json'
        self.log = self._load_log()
        
    def _load_log(self):
        """Load or create experiment log."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {
            'started': str(datetime.now()),
            'data_fraction': self.data_fraction,
            'experiments': {}
        }
    
    def _save_log(self):
        """Save experiment log."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log, f, indent=2)
    
    def _run_command(self, cmd, name):
        """Run command and log results."""
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}\n")
        
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=False, text=True)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.log['experiments'][name] = {
            'status': 'success' if result.returncode == 0 else 'failed',
            'duration_seconds': duration,
            'completed': str(end_time)
        }
        self._save_log()
        
        if result.returncode != 0:
            print(f"\n❌ FAILED: {name}")
            return False
        else:
            print(f"\n✅ SUCCESS: {name} (took {duration:.1f}s)")
            return True
    
    def train_sakt(self):
        """Train SAKT baseline."""
        cmd = [
            sys.executable, 'training/train_sakt.py',
            '--data_path', str(self.data_path),
            '--config', self.config_path,
            '--save_dir', str(self.ckpt_dir / 'sakt'),
            '--train_fraction', str(self.data_fraction),
            '--val_fraction', str(min(self.data_fraction * 2, 0.2))
        ]
        return self._run_command(cmd, 'SAKT_baseline')
    
    def train_maml(self):
        """Train MAML-SAKT baseline."""
        cmd = [
            sys.executable, 'training/train_maml.py',
            '--data_path', str(self.data_path),
            '--config', self.config_path,
            '--save_dir', str(self.ckpt_dir / 'maml'),
            '--train_fraction', str(self.data_fraction),
            '--val_fraction', str(min(self.data_fraction * 2, 0.2))
        ]
        return self._run_command(cmd, 'MAML_SAKT')
    
    def train_proto_kt(self, k=8):
        """Train Proto-KT with k prototypes."""
        cmd = [
            sys.executable, 'training/train_proto_kt.py',
            '--data_path', str(self.data_path),
            '--config', self.config_path,
            '--num_prototypes', str(k),
            '--save_dir', str(self.ckpt_dir / f'proto_kt_k{k}'),
            '--train_fraction', str(self.data_fraction),
            '--val_fraction', str(min(self.data_fraction * 2, 0.2))
        ]
        return self._run_command(cmd, f'Proto_KT_k{k}')
    
    def run_main_results(self):
        """Generate main results (Table 1, Figure 1)."""
        cmd = [
            sys.executable, 'experiments/main_results.py',
            '--data_path', str(self.data_path),
            '--config', self.config_path,
            '--sakt_checkpoint', str(self.ckpt_dir / 'sakt' / 'best_model.pt'),
            '--maml_checkpoint', str(self.ckpt_dir / 'maml' / 'best_model.pt'),
            '--proto_kt_checkpoint', str(self.ckpt_dir / 'proto_kt_k8' / 'best_model.pt'),
            '--num_prototypes', '8',
            '--output_dir', str(self.results_dir / 'main')
        ]
        return self._run_command(cmd, 'Main_Results_Experiment')
    
    def run_ablation(self):
        """Generate ablation study (Table 2)."""
        cmd = [
            sys.executable, 'experiments/ablation.py',
            '--data_path', str(self.data_path),
            '--config', self.config_path,
            '--checkpoint_dir', str(self.ckpt_dir),
            '--output_dir', str(self.results_dir / 'ablation')
        ]
        return self._run_command(cmd, 'Ablation_Study')
    
    def run_interpretability(self):
        """Generate interpretability analysis (Figure 2, Table 3)."""
        cmd = [
            sys.executable, 'experiments/interpretability.py',
            '--data_path', str(self.data_path),
            '--config', self.config_path,
            '--checkpoint', str(self.ckpt_dir / 'proto_kt_k8' / 'best_model.pt'),
            '--num_prototypes', '8',
            '--output_dir', str(self.results_dir / 'interpretability')
        ]
        return self._run_command(cmd, 'Interpretability_Analysis')
    
    def run_all(self, skip_training=False):
        """Run complete experimental pipeline."""
        print(f"\n{'#'*70}")
        print(f"#  NeurIPS-LEVEL EXPERIMENTAL PIPELINE")
        print(f"#  Data Fraction: {self.data_fraction*100:.1f}%")
        print(f"#  Results Dir: {self.results_dir}")
        print(f"{'#'*70}\n")
        
        # Phase 1: Training
        if not skip_training:
            print("\n" + "="*70)
            print("PHASE 1: MODEL TRAINING")
            print("="*70)
            
            # # Train baselines
            # if not self.train_sakt():
            #     print("\n⚠️ SAKT training failed, stopping pipeline")
            #     return False
            
            # if not self.train_maml():
            #     print("\n⚠️ MAML training failed, stopping pipeline")
            #     return False
            
            # Train Proto-KT variants for ablation
            for k in [1, 2, 4, 8, 16]:
                print(f"\n--- Training Proto-KT with k={k} prototypes ---")
                if not self.train_proto_kt(k):
                    print(f"\n⚠️ Proto-KT k={k} training failed")
                    if k == 8:  # k=8 is critical
                        return False
        else:
            print("\n⏭️  SKIPPING TRAINING (using existing checkpoints)")
        
        # Phase 2: Evaluation
        print("\n" + "="*70)
        print("PHASE 2: EVALUATION & RESULT GENERATION")
        print("="*70)
        
        # Main results
        if not self.run_main_results():
            print("\n⚠️ Main results generation failed")
            return False
        
        # Ablation study
        if not self.run_ablation():
            print("\n⚠️ Ablation study failed (continuing anyway)")
        
        # Interpretability
        if not self.run_interpretability():
            print("\n⚠️ Interpretability analysis failed (continuing anyway)")
        
        # Summary
        print("\n" + "#"*70)
        print("# EXPERIMENT PIPELINE COMPLETE")
        print("#"*70)
        print(f"\nResults saved to: {self.results_dir}")
        print(f"Experiment log: {self.log_file}")
        print("\nGenerated files:")
        for file in sorted(self.results_dir.rglob('*')):
            if file.is_file():
                print(f"  - {file.relative_to(self.results_dir)}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete experimental pipeline for Proto-KT NeurIPS submission"
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with 5%% data (for validation)')
    parser.add_argument('--subset', action='store_true',
                       help='Run with 10%% data (faster iteration)')
    parser.add_argument('--full', action='store_true',
                       help='Run with 100%% data (final results)')
    parser.add_argument('--data_fraction', type=float, default=None,
                       help='Custom data fraction (overrides --quick/--subset/--full)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only run evaluation (use existing checkpoints)')
    
    args = parser.parse_args()
    
    # Determine data fraction
    if args.data_fraction is not None:
        fraction = args.data_fraction
    elif args.quick:
        fraction = 0.05  # 5% for quick testing
    elif args.subset:
        fraction = 0.10  # 10% for faster iteration
    elif args.full:
        fraction = 1.0   # 100% for final results
    else:
        fraction = 0.05  # Default to quick mode
        print("⚠️ No mode specified, defaulting to --quick (5% data)")
        print("   Use --subset for 10%, --full for 100%, or --data_fraction X.X")
    
    # Run pipeline
    pipeline = ExperimentPipeline(data_fraction=fraction)
    success = pipeline.run_all(skip_training=args.skip_training)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

