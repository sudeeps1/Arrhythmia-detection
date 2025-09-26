"""
Visualization Module for ECG Arrhythmia Analysis

This module provides functions for creating comprehensive visualizations
of training results, model performance, and interpretability analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_history(history: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Training and validation loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training and validation accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss vs Accuracy
    axes[1, 1].scatter(history['train_loss'], history['train_acc'], 
                      alpha=0.6, label='Training', s=50)
    axes[1, 1].scatter(history['val_loss'], history['val_acc'], 
                      alpha=0.6, label='Validation', s=50)
    axes[1, 1].set_title('Loss vs Accuracy')
    axes[1, 1].set_xlabel('Loss')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray, 
                         class_names: List[str],
                         save_path: Optional[str] = None,
                         normalize: bool = True) -> None:
    """
    Plot confusion matrix with heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Names of classes
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Replace NaN with 0
    else:
        cm = confusion_matrix
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_class_distribution(labels: np.ndarray, 
                          class_names: List[str],
                          save_path: Optional[str] = None) -> None:
    """
    Plot class distribution in the dataset.
    
    Args:
        labels: Array of labels
        class_names: Names of classes
        save_path: Path to save the plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(unique)), counts, color=sns.color_palette("husl", len(unique)))
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(len(unique)), class_names, rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                str(count), ha='center', va='bottom')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    plt.title('Class Distribution (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_attribution_heatmap(attributions: np.ndarray,
                           lead_names: List[str] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot attribution heatmap showing feature importance over time and leads.
    
    Args:
        attributions: Attribution tensor (batch, leads, time)
        lead_names: Names of ECG leads
        save_path: Path to save the plot
    """
    if lead_names is None:
        lead_names = [f'Lead {i+1}' for i in range(attributions.shape[1])]
    
    # Average across batch dimension
    avg_attributions = np.mean(np.abs(attributions), axis=0)
    
    plt.figure(figsize=(15, 8))
    
    # Create heatmap
    sns.heatmap(avg_attributions, 
                xticklabels=50,  # Show every 50th time point
                yticklabels=lead_names,
                cmap='Reds',
                cbar_kws={'label': 'Attribution Magnitude'})
    
    plt.title('Feature Attribution Heatmap')
    plt.xlabel('Time (samples)')
    plt.ylabel('ECG Leads')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_lead_importance(lead_importance: np.ndarray,
                        lead_names: List[str] = None,
                        save_path: Optional[str] = None) -> None:
    """
    Plot lead importance comparison.
    
    Args:
        lead_importance: Lead importance scores
        lead_names: Names of ECG leads
        save_path: Path to save the plot
    """
    if lead_names is None:
        lead_names = [f'Lead {i+1}' for i in range(len(lead_importance))]
    
    plt.figure(figsize=(10, 6))
    
    # Bar plot
    bars = plt.bar(range(len(lead_importance)), lead_importance, 
                   color=sns.color_palette("husl", len(lead_importance)))
    
    plt.title('Lead Importance Comparison')
    plt.xlabel('ECG Leads')
    plt.ylabel('Importance Score')
    plt.xticks(range(len(lead_importance)), lead_names)
    
    # Add value labels on bars
    for bar, importance in zip(bars, lead_importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(lead_importance),
                f'{importance:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_ablation_results(ablation_results: Dict,
                         save_path: Optional[str] = None) -> None:
    """
    Plot ablation study results.
    
    Args:
        ablation_results: Results from ablation studies
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Temporal ablation
    if 'temporal' in ablation_results:
        temporal_results = ablation_results['temporal']['ablation_results']
        positions = [result['start'] for result in temporal_results]
        confidence_drops = [np.mean(result['confidence_drop']) for result in temporal_results]
        
        axes[0].bar(range(len(positions)), confidence_drops, alpha=0.7)
        axes[0].set_title('Temporal Ablation Results')
        axes[0].set_xlabel('Segment Position')
        axes[0].set_ylabel('Mean Confidence Drop')
        axes[0].grid(True, alpha=0.3)
    
    # Lead ablation
    if 'lead' in ablation_results:
        lead_results = ablation_results['lead']['ablation_results']
        leads = [result['lead'] for result in lead_results]
        confidence_drops = [np.mean(result['confidence_drop']) for result in lead_results]
        
        axes[1].bar(leads, confidence_drops, alpha=0.7)
        axes[1].set_title('Lead Ablation Results')
        axes[1].set_xlabel('Lead')
        axes[1].set_ylabel('Mean Confidence Drop')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_ecg_with_attributions(signal: np.ndarray,
                              attributions: np.ndarray,
                              lead_names: List[str] = None,
                              save_path: Optional[str] = None) -> None:
    """
    Plot ECG signal with attribution overlay.
    
    Args:
        signal: ECG signal data (leads, time)
        attributions: Attribution scores (leads, time)
        lead_names: Names of ECG leads
        save_path: Path to save the plot
    """
    if lead_names is None:
        lead_names = [f'Lead {i+1}' for i in range(signal.shape[0])]
    
    num_leads = signal.shape[0]
    fig, axes = plt.subplots(num_leads, 1, figsize=(15, 3*num_leads))
    
    if num_leads == 1:
        axes = [axes]
    
    for i in range(num_leads):
        # Plot ECG signal
        axes[i].plot(signal[i], 'b-', alpha=0.7, linewidth=1, label='ECG Signal')
        
        # Plot attribution overlay
        attribution_norm = np.abs(attributions[i]) / np.max(np.abs(attributions[i]))
        axes[i].fill_between(range(len(signal[i])), signal[i], alpha=0.3, 
                           color='red', where=attribution_norm > 0.5)
        
        axes[i].set_title(f'{lead_names[i]} - ECG Signal with Attribution')
        axes[i].set_xlabel('Time (samples)')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_performance_comparison(metrics_dict: Dict,
                              save_path: Optional[str] = None) -> None:
    """
    Plot performance comparison across different models or configurations.
    
    Args:
        metrics_dict: Dictionary with metrics for different models
        save_path: Path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        model_names = list(metrics_dict.keys())
        metric_values = [metrics_dict[model].get(metric, 0) for model in model_names]
        
        bars = axes[i].bar(model_names, metric_values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_interactive_ecg_plot(signal: np.ndarray,
                               attributions: np.ndarray,
                               lead_names: List[str] = None) -> go.Figure:
    """
    Create interactive ECG plot with attributions using Plotly.
    
    Args:
        signal: ECG signal data (leads, time)
        attributions: Attribution scores (leads, time)
        lead_names: Names of ECG leads
        
    Returns:
        Plotly figure object
    """
    if lead_names is None:
        lead_names = [f'Lead {i+1}' for i in range(signal.shape[0])]
    
    fig = make_subplots(
        rows=signal.shape[0], cols=1,
        subplot_titles=lead_names,
        vertical_spacing=0.05
    )
    
    for i in range(signal.shape[0]):
        # Add ECG signal
        fig.add_trace(
            go.Scatter(
                x=list(range(len(signal[i]))),
                y=signal[i],
                mode='lines',
                name=f'{lead_names[i]} - Signal',
                line=dict(color='blue', width=1),
                showlegend=(i == 0)
            ),
            row=i+1, col=1
        )
        
        # Add attribution overlay
        attribution_norm = np.abs(attributions[i]) / np.max(np.abs(attributions[i]))
        fig.add_trace(
            go.Scatter(
                x=list(range(len(signal[i]))),
                y=signal[i],
                mode='markers',
                name=f'{lead_names[i]} - Attribution',
                marker=dict(
                    color=attribution_norm,
                    colorscale='Reds',
                    size=3,
                    showscale=(i == 0),
                    colorbar=dict(title="Attribution")
                ),
                showlegend=False
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title="Interactive ECG Signal with Attributions",
        height=300 * signal.shape[0],
        showlegend=True
    )
    
    return fig

def plot_cross_subject_performance(subject_metrics: Dict,
                                 save_path: Optional[str] = None) -> None:
    """
    Plot performance across different subjects.
    
    Args:
        subject_metrics: Dictionary with metrics per subject
        save_path: Path to save the plot
    """
    subjects = list(subject_metrics.keys())
    metrics = ['accuracy', 'f1_score', 'auroc']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [subject_metrics[subject].get(metric, 0) for subject in subjects]
        
        bars = axes[i].bar(range(len(subjects)), values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()} by Subject')
        axes[i].set_xlabel('Subject')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_xticks(range(len(subjects)))
        axes[i].set_xticklabels(subjects, rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Example usage of the visualization module."""
    # Generate sample data
    np.random.seed(42)
    
    # Sample training history
    history = {
        'train_loss': np.random.exponential(0.5, 20),
        'val_loss': np.random.exponential(0.6, 20),
        'train_acc': 1 - np.random.exponential(0.3, 20),
        'val_acc': 1 - np.random.exponential(0.4, 20),
        'learning_rate': np.logspace(-2, -4, 20)
    }
    
    # Sample confusion matrix
    cm = np.random.randint(0, 100, (5, 5))
    np.fill_diagonal(cm, np.random.randint(50, 100, 5))
    
    # Sample attributions
    attributions = np.random.randn(2, 1000)
    
    # Create plots
    plot_training_history(history)
    plot_confusion_matrix(cm, ['Normal', 'SV', 'V', 'Fusion', 'Paced'])
    plot_lead_importance(np.random.rand(2), ['Lead I', 'Lead II'])
    plot_attribution_heatmap(attributions.reshape(1, 2, 1000))


if __name__ == "__main__":
    main()

