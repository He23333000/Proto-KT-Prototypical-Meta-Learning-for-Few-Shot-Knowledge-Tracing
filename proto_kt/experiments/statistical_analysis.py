"""
Statistical significance testing and confidence interval computation.

Performs:
- Bootstrap confidence intervals for all metrics
- Paired t-tests between methods
- Bonferroni correction for multiple comparisons
"""
import torch
import yaml
import argparse
import pickle
from pathlib import Path
import numpy as np
from scipy import stats
import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import bootstrap_confidence_interval, paired_t_test, bonferroni_correction


def load_results(result_path):
    """Load evaluation results from pickle file."""
    with open(result_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_student_aucs(results):
    """Extract per-student AUC scores."""
    if 'individual_results' in results:
        # Results from FewShotEvaluator
        aucs = []
        for student_result in results['individual_results']:
            if len(student_result['predictions']) > 0:
                from evaluation.metrics import compute_auc
                auc = compute_auc(
                    student_result['predictions'],
                    student_result['targets']
                )
                if auc is not None:
                    aucs.append(auc)
        return np.array(aucs)
    elif 'student_metrics' in results:
        # Aggregated results
        aucs = [s['auc'] for s in results['student_metrics'] if s['auc'] is not None]
        return np.array(aucs)
    else:
        raise ValueError("Cannot extract AUCs from results")


def compute_confidence_intervals(results_dict, confidence=0.95, n_bootstrap=1000):
    """
    Compute bootstrap confidence intervals for all methods.
    """
    print("\n" + "="*60)
    print("Computing Bootstrap Confidence Intervals")
    print("="*60)
    
    ci_results = {}
    
    for method_name, results in results_dict.items():
        print(f"\n{method_name}:")
        
        aucs = extract_student_aucs(results)
        
        if len(aucs) == 0:
            print("  No valid AUC scores")
            continue
        
        lower, mean, upper = bootstrap_confidence_interval(
            aucs, confidence=confidence, n_bootstrap=n_bootstrap
        )
        
        print(f"  AUC: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
        print(f"  Std: {np.std(aucs):.4f}")
        print(f"  N: {len(aucs)} students")
        
        ci_results[method_name] = {
            'auc_mean': mean,
            'auc_lower': lower,
            'auc_upper': upper,
            'auc_std': np.std(aucs),
            'n_students': len(aucs)
        }
    
    return ci_results


def perform_pairwise_tests(results_dict, alpha=0.05):
    """
    Perform pairwise significance tests between all methods.
    """
    print("\n" + "="*60)
    print("Pairwise Significance Testing")
    print("="*60)
    
    method_names = list(results_dict.keys())
    test_results = []
    p_values = []
    
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            method_1 = method_names[i]
            method_2 = method_names[j]
            
            aucs_1 = extract_student_aucs(results_dict[method_1])
            aucs_2 = extract_student_aucs(results_dict[method_2])
            
            # Ensure same students (take intersection)
            min_len = min(len(aucs_1), len(aucs_2))
            aucs_1 = aucs_1[:min_len]
            aucs_2 = aucs_2[:min_len]
            
            # Paired t-test
            t_stat, p_value = paired_t_test(aucs_1, aucs_2)
            
            mean_diff = np.mean(aucs_1) - np.mean(aucs_2)
            
            test_results.append({
                'method_1': method_1,
                'method_2': method_2,
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_diff': mean_diff,
                'significant': p_value < alpha
            })
            
            p_values.append(p_value)
            
            print(f"\n{method_1} vs. {method_2}:")
            print(f"  Mean difference: {mean_diff:.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {p_value < alpha}")
    
    # Bonferroni correction
    if len(p_values) > 1:
        corrected_alpha, significant = bonferroni_correction(p_values, alpha)
        
        print("\n" + "-"*60)
        print(f"Bonferroni Correction (α={alpha}):")
        print(f"  Corrected α: {corrected_alpha:.4f}")
        print(f"  Significant after correction:")
        
        for i, result in enumerate(test_results):
            result['significant_bonferroni'] = significant[i]
            if significant[i]:
                print(f"    {result['method_1']} vs. {result['method_2']}: "
                     f"p={result['p_value']:.4f} *")
    
    return test_results


def create_statistical_table(ci_results, test_results, save_path):
    """
    Create LaTeX table with confidence intervals and significance tests.
    """
    # CI table
    with open(save_path, 'w') as f:
        f.write("% Confidence Intervals\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Mean AUC with 95\\% bootstrap confidence intervals.}\n")
        f.write("\\label{tab:confidence_intervals}\n")
        f.write("\\begin{tabular}{l|c|c|c}\n")
        f.write("\\toprule\n")
        f.write("Method & Mean AUC & 95\\% CI & N \\\\\n")
        f.write("\\midrule\n")
        
        for method, ci in ci_results.items():
            f.write(f"{method} & {ci['auc_mean']:.4f} & "
                   f"[{ci['auc_lower']:.4f}, {ci['auc_upper']:.4f}] & "
                   f"{ci['n_students']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Significance table
        f.write("% Pairwise Significance Tests\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Pairwise significance tests (paired t-test). "
               "* indicates significance after Bonferroni correction.}\n")
        f.write("\\label{tab:significance_tests}\n")
        f.write("\\begin{tabular}{ll|c|c|c}\n")
        f.write("\\toprule\n")
        f.write("Method 1 & Method 2 & Mean Diff & t-statistic & p-value \\\\\n")
        f.write("\\midrule\n")
        
        for result in test_results:
            sig_marker = "*" if result.get('significant_bonferroni', False) else ""
            f.write(f"{result['method_1']} & {result['method_2']} & "
                   f"{result['mean_diff']:.4f} & {result['t_statistic']:.4f} & "
                   f"{result['p_value']:.4f}{sig_marker} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\nSaved statistical tables to {save_path}")


def main(args):
    print("="*60)
    print("Statistical Significance Analysis")
    print("="*60)
    
    # Load results
    results_dict = {}
    
    if args.proto_kt_results:
        print(f"\nLoading Proto-KT results from {args.proto_kt_results}")
        results_dict['Proto-KT'] = load_results(args.proto_kt_results)
    
    if args.maml_results:
        print(f"Loading MAML results from {args.maml_results}")
        results_dict['MAML-SAKT'] = load_results(args.maml_results)
    
    if args.sakt_results:
        print(f"Loading SAKT results from {args.sakt_results}")
        results_dict['SAKT'] = load_results(args.sakt_results)
    
    if len(results_dict) < 2:
        print("\nError: Need at least 2 result files for comparison")
        return
    
    # Compute confidence intervals
    ci_results = compute_confidence_intervals(
        results_dict,
        confidence=args.confidence,
        n_bootstrap=args.n_bootstrap
    )
    
    # Perform pairwise tests
    test_results = perform_pairwise_tests(
        results_dict,
        alpha=args.alpha
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tables
    create_statistical_table(
        ci_results,
        test_results,
        output_dir / 'statistical_tables.tex'
    )
    
    # Save raw results
    with open(output_dir / 'statistical_analysis.pkl', 'wb') as f:
        pickle.dump({
            'confidence_intervals': ci_results,
            'significance_tests': test_results
        }, f)
    
    print(f"\nStatistical analysis completed! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical significance analysis")
    parser.add_argument('--proto_kt_results', type=str, default=None)
    parser.add_argument('--maml_results', type=str, default=None)
    parser.add_argument('--sakt_results', type=str, default=None)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str, default='results/statistical')
    
    args = parser.parse_args()
    
    main(args)

