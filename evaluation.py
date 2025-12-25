"""
Comprehensive evaluation: Calibration, OOD analysis, Bootstrap CIs, failure analysis
Time budget: 30-45 minutes
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os
from collections import defaultdict

def compute_calibration_metrics(confidences: np.ndarray, 
                                correctness: np.ndarray,
                                n_bins: int = 10) -> Dict:
    """
    Compute calibration metrics (ECE, MCE) and calibration curve data
    
    Args:
        confidences: Array of confidence scores (0-1)
        correctness: Binary array indicating if prediction was correct
        n_bins: Number of bins for calibration
    
    Returns:
        Dict with ECE, MCE, and calibration curve data
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correctness[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            mce = max(mce, np.abs(accuracy_in_bin - avg_confidence_in_bin))
            
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    
    return {
        'ece': ece,
        'mce': mce,
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts,
        'n_bins': n_bins
    }

def plot_calibration_curve(calibration_data: Dict, save_path: str):
    """Plot calibration curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bin_confs = calibration_data['bin_confidences']
    bin_accs = calibration_data['bin_accuracies']
    bin_counts = calibration_data['bin_counts']
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(bin_confs, bin_accs, 'o-', label='Model calibration')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Calibration Curve\nECE: {calibration_data["ece"]:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bin counts
    ax2.bar(range(len(bin_counts)), bin_counts)
    ax2.set_xlabel('Bin')
    ax2.set_ylabel('Count')
    ax2.set_title('Samples per Confidence Bin')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved calibration plot to {save_path}")

def bootstrap_confidence_interval(data: np.ndarray, 
                                  statistic_func=np.mean,
                                  n_bootstrap: int = 1000,
                                  confidence_level: float = 0.95) -> Tuple:
    """
    Compute bootstrap confidence interval for a statistic
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    # Use scipy's bootstrap
    result = bootstrap(
        (data,),
        statistic_func,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        method='percentile'
    )
    
    mean_value = statistic_func(data)
    ci_lower = result.confidence_interval.low
    ci_upper = result.confidence_interval.high
    
    return mean_value, ci_lower, ci_upper

def compute_cider_score(generated_captions: List[str],
                       reference_captions: List[str]) -> float:
    """
    Simplified CIDEr-like score based on n-gram overlap
    (Full CIDEr requires pycocoevalcap, this is a lightweight approximation)
    """
    from collections import Counter
    
    def get_ngrams(text: str, n: int = 4):
        """Extract n-grams from text"""
        words = text.lower().split()
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    scores = []
    
    for gen, ref in zip(generated_captions, reference_captions):
        # Compute 1-4 gram overlaps
        gram_scores = []
        for n in range(1, 5):
            gen_grams = Counter(get_ngrams(gen, n))
            ref_grams = Counter(get_ngrams(ref, n))
            
            if len(gen_grams) == 0 or len(ref_grams) == 0:
                gram_scores.append(0)
                continue
            
            # Compute precision and recall
            overlap = sum((gen_grams & ref_grams).values())
            precision = overlap / sum(gen_grams.values()) if sum(gen_grams.values()) > 0 else 0
            recall = overlap / sum(ref_grams.values()) if sum(ref_grams.values()) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            gram_scores.append(f1)
        
        # Average across n-grams
        scores.append(np.mean(gram_scores))
    
    return np.mean(scores) * 100  # Scale to 0-100

class FailureAnalyzer:
    """Analyze and categorize failure cases"""
    
    def __init__(self, similarity_matrix, captions, generated_captions=None):
        """
        Args:
            similarity_matrix: (n_images, n_captions) similarity scores
            captions: Ground truth captions
            generated_captions: Optional generated captions for captioning model
        """
        self.similarity = similarity_matrix
        self.captions = captions
        self.generated = generated_captions
        
        # Compute predictions
        self.predictions = torch.argmax(similarity_matrix, dim=1).cpu().numpy()
        self.ground_truth = np.arange(len(captions))
        self.correct = (self.predictions == self.ground_truth)
        
        self.failure_indices = np.where(~self.correct)[0]
    
    def categorize_failures(self) -> Dict:
        """
        Categorize failures into types:
        - Low confidence: Model uncertain (low max similarity)
        - High confusion: Multiple similar candidates
        - Semantic mismatch: Retrieved caption semantically different
        """
        categories = defaultdict(list)
        
        for idx in self.failure_indices:
            correct_idx = self.ground_truth[idx]
            pred_idx = self.predictions[idx]
            
            similarities = self.similarity[idx].cpu().numpy()
            max_sim = similarities.max()
            correct_sim = similarities[correct_idx]
            
            # Categorize
            if max_sim < 0.2:  # Low confidence threshold
                categories['low_confidence'].append({
                    'index': int(idx),
                    'predicted_caption': self.captions[pred_idx],
                    'correct_caption': self.captions[correct_idx],
                    'max_similarity': float(max_sim),
                    'correct_similarity': float(correct_sim)
                })
            elif (similarities > max_sim - 0.05).sum() > 3:  # Multiple close candidates
                categories['high_confusion'].append({
                    'index': int(idx),
                    'predicted_caption': self.captions[pred_idx],
                    'correct_caption': self.captions[correct_idx],
                    'max_similarity': float(max_sim),
                    'correct_similarity': float(correct_sim),
                    'num_close_candidates': int((similarities > max_sim - 0.05).sum())
                })
            else:  # Semantic mismatch
                categories['semantic_mismatch'].append({
                    'index': int(idx),
                    'predicted_caption': self.captions[pred_idx],
                    'correct_caption': self.captions[correct_idx],
                    'max_similarity': float(max_sim),
                    'correct_similarity': float(correct_sim)
                })
        
        # Add statistics
        summary = {
            'total_failures': len(self.failure_indices),
            'total_samples': len(self.ground_truth),
            'failure_rate': len(self.failure_indices) / len(self.ground_truth),
            'categories': {
                'low_confidence': len(categories['low_confidence']),
                'high_confusion': len(categories['high_confusion']),
                'semantic_mismatch': len(categories['semantic_mismatch'])
            }
        }
        
        return {
            'summary': summary,
            'failures': dict(categories)
        }
    
    def get_representative_failures(self, n_per_category: int = 5) -> Dict:
        """Get representative examples from each failure category"""
        analysis = self.categorize_failures()
        
        representatives = {}
        for category, failures in analysis['failures'].items():
            # Sample n examples (or all if less than n)
            n_samples = min(n_per_category, len(failures))
            if n_samples > 0:
                sampled = np.random.choice(len(failures), n_samples, replace=False)
                representatives[category] = [failures[i] for i in sampled]
        
        return {
            'summary': analysis['summary'],
            'representative_failures': representatives
        }

def evaluate_with_calibration(model_outputs: Dict, 
                              config,
                              split_name: str = "test") -> Dict:
    """
    Complete evaluation with calibration and bootstrap CIs
    
    Args:
        model_outputs: Dict containing similarity_matrix, captions, etc.
        config: Configuration object
        split_name: Name of the split being evaluated
    
    Returns:
        Dict with all evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluation with Calibration - {split_name}")
    print(f"{'='*60}")
    
    similarity = model_outputs['similarity']
    captions = model_outputs['captions']
    is_ood = model_outputs.get('is_ood', None)
    
    # Compute retrieval metrics
    num_samples = similarity.size(0)
    ranks = []
    confidences = []
    correctness = []
    
    for i in range(num_samples):
        scores = similarity[i]
        max_conf = scores.max().item()
        rank = (scores > scores[i]).sum().item() + 1
        
        ranks.append(rank)
        confidences.append(max_conf)
        correctness.append(1 if rank == 1 else 0)
    
    ranks = np.array(ranks)
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    # Compute metrics with bootstrap CIs
    print("\nComputing bootstrap confidence intervals...")
    
    recall1_data = (ranks == 1).astype(float) * 100
    recall5_data = (ranks <= 5).astype(float) * 100
    recall10_data = (ranks <= 10).astype(float) * 100
    
    r1_mean, r1_lower, r1_upper = bootstrap_confidence_interval(recall1_data, n_bootstrap=1000)
    r5_mean, r5_lower, r5_upper = bootstrap_confidence_interval(recall5_data, n_bootstrap=1000)
    r10_mean, r10_lower, r10_upper = bootstrap_confidence_interval(recall10_data, n_bootstrap=1000)
    
    metrics = {
        'recall@1': {
            'mean': float(r1_mean),
            'ci_lower': float(r1_lower),
            'ci_upper': float(r1_upper),
            'ci_95': f"[{r1_lower:.2f}, {r1_upper:.2f}]"
        },
        'recall@5': {
            'mean': float(r5_mean),
            'ci_lower': float(r5_lower),
            'ci_upper': float(r5_upper),
            'ci_95': f"[{r5_lower:.2f}, {r5_upper:.2f}]"
        },
        'recall@10': {
            'mean': float(r10_mean),
            'ci_lower': float(r10_lower),
            'ci_upper': float(r10_upper),
            'ci_95': f"[{r10_lower:.2f}, {r10_upper:.2f}]"
        }
    }
    
    # Separate in-dist and OOD if available
    if is_ood is not None:
        is_ood_array = np.array(is_ood)
        in_dist_mask = ~is_ood_array
        
        if in_dist_mask.sum() > 0:
            in_dist_r1 = (ranks[in_dist_mask] == 1).astype(float) * 100
            id_mean, id_lower, id_upper = bootstrap_confidence_interval(in_dist_r1)
            metrics['in_dist_recall@1'] = {
                'mean': float(id_mean),
                'ci_95': f"[{id_lower:.2f}, {id_upper:.2f}]"
            }
        
        if is_ood_array.sum() > 0:
            ood_r1 = (ranks[is_ood_array] == 1).astype(float) * 100
            ood_mean, ood_lower, ood_upper = bootstrap_confidence_interval(ood_r1)
            metrics['ood_recall@1'] = {
                'mean': float(ood_mean),
                'ci_95': f"[{ood_lower:.2f}, {ood_upper:.2f}]"
            }
    
    # Compute calibration metrics
    print("\nComputing calibration metrics...")
    calibration = compute_calibration_metrics(confidences, correctness, n_bins=10)
    
    # Plot calibration curve
    cal_plot_path = os.path.join(
        config.experiment.output_dir,
        f'calibration_{split_name}.png'
    )
    plot_calibration_curve(calibration, cal_plot_path)
    
    # Failure analysis
    print("\nAnalyzing failures...")
    analyzer = FailureAnalyzer(similarity, captions)
    failure_analysis = analyzer.get_representative_failures(n_per_category=5)
    
    # Print results
    print(f"\nResults:")
    print(f"  Recall@1:  {metrics['recall@1']['mean']:.2f}% {metrics['recall@1']['ci_95']}")
    print(f"  Recall@5:  {metrics['recall@5']['mean']:.2f}% {metrics['recall@5']['ci_95']}")
    print(f"  Recall@10: {metrics['recall@10']['mean']:.2f}% {metrics['recall@10']['ci_95']}")
    print(f"\n  Calibration ECE: {calibration['ece']:.4f}")
    print(f"  Calibration MCE: {calibration['mce']:.4f}")
    
    print(f"\n  Failures: {failure_analysis['summary']['total_failures']}/{failure_analysis['summary']['total_samples']}")
    print(f"  Failure categories:")
    for cat, count in failure_analysis['summary']['categories'].items():
        print(f"    {cat}: {count}")
    
    return {
        'metrics': metrics,
        'calibration': calibration,
        'failure_analysis': failure_analysis
    }

def aggregate_results_across_seeds(results_list: List[Dict]) -> Dict:
    """
    Aggregate results across multiple seeds
    
    Args:
        results_list: List of result dicts from different seeds
    
    Returns:
        Aggregated statistics (mean ± std, 95% CI)
    """
    print(f"\n{'='*60}")
    print(f"Aggregating Results Across {len(results_list)} Seeds")
    print(f"{'='*60}")
    
    # Extract metrics from each seed
    recall1_values = [r['metrics']['recall@1']['mean'] for r in results_list]
    recall5_values = [r['metrics']['recall@5']['mean'] for r in results_list]
    recall10_values = [r['metrics']['recall@10']['mean'] for r in results_list]
    
    recall1_array = np.array(recall1_values)
    recall5_array = np.array(recall5_values)
    recall10_array = np.array(recall10_values)
    
    # Compute statistics
    aggregated = {
        'recall@1': {
            'mean': float(recall1_array.mean()),
            'std': float(recall1_array.std()),
            'ci_95': f"[{np.percentile(recall1_array, 2.5):.2f}, {np.percentile(recall1_array, 97.5):.2f}]",
            'values': recall1_values
        },
        'recall@5': {
            'mean': float(recall5_array.mean()),
            'std': float(recall5_array.std()),
            'ci_95': f"[{np.percentile(recall5_array, 2.5):.2f}, {np.percentile(recall5_array, 97.5):.2f}]",
            'values': recall5_values
        },
        'recall@10': {
            'mean': float(recall10_array.mean()),
            'std': float(recall10_array.std()),
            'ci_95': f"[{np.percentile(recall10_array, 2.5):.2f}, {np.percentile(recall10_array, 97.5):.2f}]",
            'values': recall10_values
        },
        'num_seeds': len(results_list)
    }
    
    print(f"\nAggregated Results:")
    print(f"  Recall@1:  {aggregated['recall@1']['mean']:.2f} ± {aggregated['recall@1']['std']:.2f}%")
    print(f"             95% CI: {aggregated['recall@1']['ci_95']}")
    print(f"  Recall@5:  {aggregated['recall@5']['mean']:.2f} ± {aggregated['recall@5']['std']:.2f}%")
    print(f"             95% CI: {aggregated['recall@5']['ci_95']}")
    print(f"  Recall@10: {aggregated['recall@10']['mean']:.2f} ± {aggregated['recall@10']['std']:.2f}%")
    print(f"             95% CI: {aggregated['recall@10']['ci_95']}")
    
    return aggregated
