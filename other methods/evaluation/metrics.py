"""
Evaluation Metrics for ECG Arrhythmia Classification

This module provides comprehensive evaluation metrics for ECG arrhythmia
classification including accuracy, F1-score, AUROC, AUPRC, and confusion matrix.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, cohen_kappa_score, matthews_corrcoef
)
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None,
                     class_names: Optional[List[str]] = None) -> Dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC/AUPRC)
        class_names: Names of classes (optional)
        
    Returns:
        Dictionary with all metrics
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Additional metrics
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC and AUPRC (if probabilities provided)
    auroc = None
    auprc = None
    if y_prob is not None:
        try:
            # Multi-class ROC AUC
            if len(class_names) == 2:
                auroc = roc_auc_score(y_true, y_prob[:, 1])
                auprc = average_precision_score(y_true, y_prob[:, 1])
            else:
                auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                auprc = average_precision_score(y_true, y_prob, average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC/AUPRC: {str(e)}")
    
    # Create per-class dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1_score': f1_per_class[i]
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'kappa': kappa,
        'matthews_corrcoef': mcc,
        'auroc': auroc,
        'auprc': auprc,
        'per_class': per_class_metrics,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, target_names=class_names)
    }

def calculate_cross_subject_metrics(results: Dict) -> Dict:
    """
    Calculate metrics across multiple subjects.
    
    Args:
        results: Dictionary with results per subject
        
    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = []
    
    for subject, subject_results in results.items():
        metrics = calculate_metrics(
            subject_results['y_true'],
            subject_results['y_pred'],
            subject_results.get('y_prob')
        )
        all_metrics.append(metrics)
    
    # Aggregate metrics
    aggregated = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'kappa', 'matthews_corrcoef']:
        values = [m[metric] for m in all_metrics if m[metric] is not None]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
    
    # Aggregate confusion matrices
    all_cms = [m['confusion_matrix'] for m in all_metrics]
    aggregated['confusion_matrix_sum'] = np.sum(all_cms, axis=0)
    aggregated['confusion_matrix_mean'] = np.mean(all_cms, axis=0)
    
    return aggregated

def calculate_temporal_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             timestamps: np.ndarray) -> Dict:
    """
    Calculate temporal performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        timestamps: Timestamps for each prediction
        
    Returns:
        Dictionary with temporal metrics
    """
    # Sort by timestamp
    sorted_indices = np.argsort(timestamps)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    # Calculate sliding window metrics
    window_size = min(100, len(y_true) // 10)  # 10% of data or 100 samples
    temporal_metrics = []
    
    for i in range(0, len(y_true) - window_size, window_size // 2):
        window_true = y_true_sorted[i:i + window_size]
        window_pred = y_pred_sorted[i:i + window_size]
        
        if len(np.unique(window_true)) > 1:  # Only if multiple classes in window
            window_metrics = calculate_metrics(window_true, window_pred)
            temporal_metrics.append({
                'start_idx': i,
                'end_idx': i + window_size,
                'accuracy': window_metrics['accuracy'],
                'f1_score': window_metrics['f1_score']
            })
    
    return {
        'temporal_metrics': temporal_metrics,
        'mean_temporal_accuracy': np.mean([m['accuracy'] for m in temporal_metrics]),
        'std_temporal_accuracy': np.std([m['accuracy'] for m in temporal_metrics])
    }

def calculate_lead_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  lead_attributions: np.ndarray) -> Dict:
    """
    Calculate lead-specific performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        lead_attributions: Lead attribution scores
        
    Returns:
        Dictionary with lead-specific metrics
    """
    num_leads = lead_attributions.shape[1]
    lead_metrics = {}
    
    for lead in range(num_leads):
        # Calculate performance for samples where this lead is most important
        lead_importance = lead_attributions[:, lead]
        importance_threshold = np.percentile(lead_importance, 75)  # Top 25%
        
        important_mask = lead_importance >= importance_threshold
        if np.sum(important_mask) > 10:  # Need sufficient samples
            important_true = y_true[important_mask]
            important_pred = y_pred[important_mask]
            
            metrics = calculate_metrics(important_true, important_pred)
            lead_metrics[f'lead_{lead}'] = {
                'num_samples': np.sum(important_mask),
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'mean_importance': np.mean(lead_importance[important_mask])
            }
    
    return lead_metrics

def calculate_interpretability_metrics(attributions: np.ndarray,
                                     ablation_results: Dict) -> Dict:
    """
    Calculate interpretability-specific metrics.
    
    Args:
        attributions: Feature attribution scores
        ablation_results: Results from ablation studies
        
    Returns:
        Dictionary with interpretability metrics
    """
    # Faithfulness: correlation between attributions and ablation results
    faithfulness_scores = []
    
    if 'temporal' in ablation_results:
        temporal_ablation = ablation_results['temporal']
        for result in temporal_ablation['ablation_results']:
            start, end = result['start'], result['end']
            segment_attribution = np.mean(np.abs(attributions[:, :, start:end]), axis=(1, 2))
            confidence_drop = result['confidence_drop']
            
            if len(segment_attribution) > 1:
                correlation = np.corrcoef(segment_attribution, confidence_drop)[0, 1]
                if not np.isnan(correlation):
                    faithfulness_scores.append(correlation)
    
    # Sparsity: percentage of features needed for prediction
    attribution_magnitudes = np.abs(attributions)
    total_importance = np.sum(attribution_magnitudes, axis=(1, 2))
    
    # Find features needed for 80% of total importance
    sparsity_scores = []
    for i in range(len(attribution_magnitudes)):
        sorted_importance = np.sort(attribution_magnitudes[i].flatten())[::-1]
        cumulative_importance = np.cumsum(sorted_importance)
        threshold_idx = np.where(cumulative_importance >= 0.8 * total_importance[i])[0]
        if len(threshold_idx) > 0:
            sparsity = threshold_idx[0] / len(sorted_importance)
            sparsity_scores.append(sparsity)
    
    return {
        'faithfulness_mean': np.mean(faithfulness_scores) if faithfulness_scores else None,
        'faithfulness_std': np.std(faithfulness_scores) if faithfulness_scores else None,
        'sparsity_mean': np.mean(sparsity_scores) if sparsity_scores else None,
        'sparsity_std': np.std(sparsity_scores) if sparsity_scores else None
    }

def generate_metrics_report(metrics: Dict, save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive metrics report.
    
    Args:
        metrics: Dictionary with all metrics
        save_path: Path to save report (optional)
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("ECG Arrhythmia Classification Metrics Report")
    report.append("=" * 50)
    report.append("")
    
    # Overall metrics
    report.append("Overall Performance:")
    report.append("-" * 20)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'kappa', 'matthews_corrcoef']:
        if metric in metrics and metrics[metric] is not None:
            report.append(f"{metric.replace('_', ' ').title()}: {metrics[metric]:.4f}")
    
    if metrics.get('auroc') is not None:
        report.append(f"AUROC: {metrics['auroc']:.4f}")
    if metrics.get('auprc') is not None:
        report.append(f"AUPRC: {metrics['auprc']:.4f}")
    
    report.append("")
    
    # Per-class metrics
    if 'per_class' in metrics:
        report.append("Per-Class Performance:")
        report.append("-" * 20)
        for class_name, class_metrics in metrics['per_class'].items():
            report.append(f"\n{class_name}:")
            for metric, value in class_metrics.items():
                report.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    report.append("")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        report.append("Confusion Matrix:")
        report.append("-" * 20)
        cm = metrics['confusion_matrix']
        report.append(str(cm))
    
    report.append("")
    
    # Classification report
    if 'classification_report' in metrics:
        report.append("Detailed Classification Report:")
        report.append("-" * 30)
        report.append(metrics['classification_report'])
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text

def main():
    """Example usage of the metrics module."""
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 1000)
    y_pred = np.random.randint(0, 5, 1000)
    y_prob = np.random.rand(1000, 5)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Generate report
    report = generate_metrics_report(metrics)
    print(report)


if __name__ == "__main__":
    main()

