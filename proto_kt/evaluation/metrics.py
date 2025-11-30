"""
Evaluation metrics for knowledge tracing.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import torch


def compute_auc(predictions, targets):
    """
    Compute AUC score.
    
    Args:
        predictions: Array of predicted probabilities
        targets: Array of true labels (0 or 1)
        
    Returns:
        auc: AUC score, or None if not computable
    """
    if len(np.unique(targets)) < 2:
        # Need both classes for AUC
        return None
    
    try:
        auc = roc_auc_score(targets, predictions)
        return auc
    except:
        return None


def compute_accuracy(predictions, targets, threshold=0.5):
    """
    Compute accuracy.
    
    Args:
        predictions: Array of predicted probabilities
        targets: Array of true labels
        threshold: Classification threshold
        
    Returns:
        accuracy: Accuracy score
    """
    pred_binary = (predictions > threshold).astype(int)
    return accuracy_score(targets, pred_binary)


def compute_bce_loss(predictions, targets):
    """
    Compute binary cross-entropy loss.
    
    Args:
        predictions: Array of predicted probabilities
        targets: Array of true labels
        
    Returns:
        bce: Binary cross-entropy loss
    """
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    
    # Handle edge case: if all labels are the same, sklearn's log_loss fails
    # In this case, compute BCE manually
    unique_labels = np.unique(targets)
    if len(unique_labels) == 1:
        # Manual BCE computation: -[y*log(p) + (1-y)*log(1-p)]
        bce = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return np.mean(bce)
    
    return log_loss(targets, predictions)


def compute_ece(predictions, targets, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Measures how well-calibrated the predicted probabilities are.
    
    Args:
        predictions: Array of predicted probabilities
        targets: Array of true labels
        n_bins: Number of bins for calibration
        
    Returns:
        ece: Expected calibration error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_samples = len(predictions)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(targets[in_bin])
            avg_confidence_in_bin = np.mean(predictions[in_bin])
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_all_metrics(predictions, targets):
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Array or list of predicted probabilities
        targets: Array or list of true labels
        
    Returns:
        dict of metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = {
        'auc': compute_auc(predictions, targets),
        'accuracy': compute_accuracy(predictions, targets),
        'bce_loss': compute_bce_loss(predictions, targets),
        'ece': compute_ece(predictions, targets)
    }
    
    return metrics


def bootstrap_confidence_interval(values, confidence=0.95, n_bootstrap=1000):
    """
    Compute bootstrap confidence interval.
    
    Args:
        values: Array of values
        confidence: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        (lower, mean, upper): Confidence interval bounds and mean
    """
    values = np.array(values)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    mean = np.mean(values)
    
    return lower, mean, upper


def paired_t_test(values1, values2):
    """
    Perform paired t-test.
    
    Args:
        values1, values2: Arrays of paired observations
        
    Returns:
        (t_statistic, p_value): Test results
    """
    from scipy import stats
    
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    return t_stat, p_value


def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        corrected_alpha: Corrected significance threshold
        significant: Boolean array indicating significant results
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    significant = np.array(p_values) < corrected_alpha
    
    return corrected_alpha, significant


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    predictions = np.random.rand(100)
    targets = (np.random.rand(100) > 0.5).astype(int)
    
    metrics = compute_all_metrics(predictions, targets)
    
    print("Test metrics:")
    for name, value in metrics.items():
        if value is not None:
            print(f"  {name}: {value:.4f}")
    
    # Test bootstrap CI
    values = np.random.normal(0.75, 0.05, 50)
    lower, mean, upper = bootstrap_confidence_interval(values)
    print(f"\nBootstrap CI: [{lower:.4f}, {mean:.4f}, {upper:.4f}]")

