"""
Advanced ECG Interpretability Module

This module provides advanced interpretability techniques for ECG arrhythmia detection,
including SHAP for time-series data, multi-method attribution, and clinical visualization.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import shap
from captum.attr import IntegratedGradients, Occlusion, Saliency
from captum.attr import visualization as viz

logger = logging.getLogger(__name__)

class AdvancedECGInterpreter:
    """
    Advanced ECG interpreter with multiple attribution methods and clinical insights.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize the advanced interpreter.
        
        Args:
            model: Trained ECG model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.occlusion = Occlusion(self.model)
        self.saliency = Saliency(self.model)
        
    def compute_shap_attributions(self, 
                                data: torch.Tensor, 
                                background_data: torch.Tensor = None,
                                n_background: int = 100) -> np.ndarray:
        """
        Compute SHAP attributions for time-series ECG data.
        
        Args:
            data: Input ECG data (batch, leads, timesteps)
            background_data: Background data for SHAP
            n_background: Number of background samples
            
        Returns:
            SHAP attributions
        """
        logger.info("Computing SHAP attributions...")
        
        # Prepare background data
        if background_data is None:
            # Use random samples as background
            indices = np.random.choice(len(data), min(n_background, len(data)), replace=False)
            background_data = data[indices]
        
        # Create SHAP explainer
        explainer = shap.DeepExplainer(self.model, background_data)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(data)
        
        # For multi-class, take the mean across classes
        if isinstance(shap_values, list):
            shap_values = np.mean(shap_values, axis=0)
        
        return shap_values
    
    def compute_multi_method_attributions(self, 
                                        data: torch.Tensor,
                                        target_class: int = None) -> Dict[str, np.ndarray]:
        """
        Compute attributions using multiple methods.
        
        Args:
            data: Input ECG data
            target_class: Target class for attribution
            
        Returns:
            Dictionary of attributions from different methods
        """
        logger.info("Computing multi-method attributions...")
        
        attributions = {}
        
        # Integrated Gradients
        if target_class is not None:
            attributions['integrated_gradients'] = self.integrated_gradients.attribute(
                data, target=target_class
            ).cpu().numpy()
        else:
            attributions['integrated_gradients'] = self.integrated_gradients.attribute(
                data
            ).cpu().numpy()
        
        # Occlusion
        attributions['occlusion'] = self.occlusion.attribute(
            data, target=target_class, sliding_window_shapes=(1, 50)
        ).cpu().numpy()
        
        # Saliency
        attributions['saliency'] = self.saliency.attribute(
            data, target=target_class
        ).cpu().numpy()
        
        return attributions
    
    def extract_beat_level_importance(self, 
                                    attributions: np.ndarray,
                                    beat_annotations: List[Dict] = None,
                                    window_size: int = 1080) -> Dict:
        """
        Map attributions back to individual QRS complexes for beat-level importance.
        
        Args:
            attributions: Attribution values
            beat_annotations: Beat annotations with timing information
            window_size: Size of the analysis window
            
        Returns:
            Beat-level importance scores
        """
        logger.info("Extracting beat-level importance...")
        
        beat_importance = {}
        
        if beat_annotations is None:
            # Simple approach: find peaks in attribution
            for lead in range(attributions.shape[1]):
                lead_attributions = attributions[0, lead, :]  # First sample
                
                # Find peaks in attribution (potential QRS complexes)
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(np.abs(lead_attributions), height=0.1)
                
                beat_importance[f'lead_{lead}'] = {
                    'peak_positions': peaks.tolist(),
                    'peak_importance': lead_attributions[peaks].tolist(),
                    'mean_importance': float(np.mean(np.abs(lead_attributions)))
                }
        else:
            # Use actual beat annotations
            for annotation in beat_annotations:
                beat_time = annotation.get('time', 0)
                beat_sample = int(beat_time * 360)  # Assuming 360 Hz
                
                if 0 <= beat_sample < window_size:
                    importance_scores = []
                    for lead in range(attributions.shape[1]):
                        # Get importance around the beat
                        start_idx = max(0, beat_sample - 50)
                        end_idx = min(window_size, beat_sample + 50)
                        importance_scores.append(
                            float(np.mean(np.abs(attributions[0, lead, start_idx:end_idx])))
                        )
                    
                    beat_importance[f'beat_{beat_time:.2f}'] = {
                        'time': beat_time,
                        'sample': beat_sample,
                        'lead_importance': importance_scores,
                        'total_importance': float(np.sum(importance_scores))
                    }
        
        return beat_importance
    
    def extract_clinical_rules(self, 
                             attributions: np.ndarray,
                             data: torch.Tensor,
                             labels: np.ndarray,
                             feature_names: List[str] = None) -> Dict:
        """
        Extract clinical rules from attribution patterns.
        
        Args:
            attributions: Attribution values
            data: Original ECG data
            labels: True labels
            feature_names: Names of features/leads
            
        Returns:
            Clinical rules and patterns
        """
        logger.info("Extracting clinical rules...")
        
        if feature_names is None:
            feature_names = ['MLII', 'V1']
        
        # Extract features from attributions
        features = []
        for i in range(len(data)):
            sample_features = []
            
            # Lead-specific features
            for lead in range(attributions.shape[1]):
                lead_attr = attributions[i, lead, :]
                
                # Statistical features
                sample_features.extend([
                    np.mean(np.abs(lead_attr)),  # Mean importance
                    np.std(lead_attr),           # Variability
                    np.max(np.abs(lead_attr)),   # Peak importance
                    np.sum(np.abs(lead_attr)),   # Total importance
                    np.percentile(np.abs(lead_attr), 90),  # 90th percentile
                ])
            
            features.append(sample_features)
        
        features = np.array(features)
        
        # Create feature names
        feature_names_extended = []
        for lead_name in feature_names:
            feature_names_extended.extend([
                f'{lead_name}_mean_importance',
                f'{lead_name}_std_importance',
                f'{lead_name}_max_importance',
                f'{lead_name}_total_importance',
                f'{lead_name}_90th_percentile'
            ])
        
        # Train interpretable models
        rules = {}
        
        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(features, labels)
        dt_accuracy = accuracy_score(labels, dt.predict(features))
        
        rules['decision_tree'] = {
            'accuracy': dt_accuracy,
            'feature_importance': dict(zip(feature_names_extended, dt.feature_importances_)),
            'n_leaves': dt.get_n_leaves(),
            'max_depth': dt.get_depth()
        }
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(features, labels)
        rf_accuracy = accuracy_score(labels, rf.predict(features))
        
        rules['random_forest'] = {
            'accuracy': rf_accuracy,
            'feature_importance': dict(zip(feature_names_extended, rf.feature_importances_)),
            'n_estimators': 100
        }
        
        # Extract top features
        top_features = sorted(
            zip(feature_names_extended, rf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:10]
        
        rules['top_features'] = top_features
        
        return rules
    
    def create_clinical_visualizations(self, 
                                     data: torch.Tensor,
                                     attributions: Dict[str, np.ndarray],
                                     labels: np.ndarray,
                                     save_dir: str = 'results/interpretability') -> None:
        """
        Create comprehensive clinical visualizations.
        
        Args:
            data: ECG data
            attributions: Attribution results
            labels: True labels
            save_dir: Directory to save visualizations
        """
        logger.info("Creating clinical visualizations...")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Attribution heatmaps
        self._plot_attribution_heatmaps(data, attributions, labels, save_path)
        
        # 2. Lead comparison
        self._plot_lead_comparison(data, attributions, save_path)
        
        # 3. Temporal importance
        self._plot_temporal_importance(data, attributions, save_path)
        
        # 4. Method comparison
        self._plot_method_comparison(attributions, save_path)
        
        logger.info(f"Visualizations saved to {save_path}")
    
    def _plot_attribution_heatmaps(self, data, attributions, labels, save_path):
        """Plot attribution heatmaps."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (method, attr) in enumerate(attributions.items()):
            if i >= 4:  # Limit to 4 methods
                break
                
            ax = axes[i // 2, i % 2]
            
            # Plot heatmap for first sample
            im = ax.imshow(attr[0], aspect='auto', cmap='RdBu_r', 
                          vmin=-np.max(np.abs(attr[0])), vmax=np.max(np.abs(attr[0])))
            
            ax.set_title(f'{method.replace("_", " ").title()}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Leads')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['MLII', 'V1'])
            
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(save_path / 'attribution_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lead_comparison(self, data, attributions, save_path):
        """Plot lead comparison."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original signals
        ax1 = axes[0]
        time_steps = np.arange(data.shape[2])
        ax1.plot(time_steps, data[0, 0, :].cpu().numpy(), label='MLII', alpha=0.7)
        ax1.plot(time_steps, data[0, 1, :].cpu().numpy(), label='V1', alpha=0.7)
        ax1.set_title('Original ECG Signals')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot attributions
        ax2 = axes[1]
        for method, attr in attributions.items():
            if method == 'integrated_gradients':  # Use one method for clarity
                ax2.plot(time_steps, attr[0, 0, :], label=f'MLII ({method})', alpha=0.7)
                ax2.plot(time_steps, attr[0, 1, :], label=f'V1 ({method})', alpha=0.7)
                break
        
        ax2.set_title('Attribution Values')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Attribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'lead_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_importance(self, data, attributions, save_path):
        """Plot temporal importance."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for i, lead_name in enumerate(['MLII', 'V1']):
            ax = axes[i]
            
            # Plot temporal importance for each method
            for method, attr in attributions.items():
                lead_attr = attr[0, i, :]
                time_steps = np.arange(len(lead_attr))
                ax.plot(time_steps, np.abs(lead_attr), label=method.replace('_', ' ').title(), alpha=0.7)
            
            ax.set_title(f'Temporal Importance - {lead_name}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Absolute Attribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'temporal_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_method_comparison(self, attributions, save_path):
        """Plot method comparison."""
        methods = list(attributions.keys())
        n_methods = len(methods)
        
        # Calculate summary statistics
        summary_stats = {}
        for method, attr in attributions.items():
            summary_stats[method] = {
                'mean_importance': np.mean(np.abs(attr)),
                'std_importance': np.std(attr),
                'max_importance': np.max(np.abs(attr)),
                'total_importance': np.sum(np.abs(attr))
            }
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        metrics = ['mean_importance', 'std_importance', 'max_importance', 'total_importance']
        metric_names = ['Mean Importance', 'Std Importance', 'Max Importance', 'Total Importance']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 2, i % 2]
            
            values = [summary_stats[method][metric] for method in methods]
            ax.bar(methods, values, alpha=0.7)
            ax.set_title(name)
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_clinical_report(self, 
                               attributions: Dict[str, np.ndarray],
                               rules: Dict,
                               beat_importance: Dict,
                               save_dir: str = 'results/interpretability') -> None:
        """
        Generate a comprehensive clinical report.
        
        Args:
            attributions: Attribution results
            rules: Extracted rules
            beat_importance: Beat-level importance
            save_dir: Directory to save report
        """
        logger.info("Generating clinical report...")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'summary': {
                'total_attribution_methods': len(attributions),
                'extracted_rules': len(rules),
                'beat_analysis': len(beat_importance)
            },
            'attribution_summary': {},
            'clinical_rules': rules,
            'beat_importance': beat_importance,
            'recommendations': []
        }
        
        # Attribution summary
        for method, attr in attributions.items():
            report['attribution_summary'][method] = {
                'mean_importance': float(np.mean(np.abs(attr))),
                'max_importance': float(np.max(np.abs(attr))),
                'std_importance': float(np.std(attr)),
                'lead_importance': {
                    'MLII': float(np.mean(np.abs(attr[:, 0, :]))),
                    'V1': float(np.mean(np.abs(attr[:, 1, :])))
                }
            }
        
        # Clinical recommendations
        if 'top_features' in rules:
            top_feature = rules['top_features'][0]
            report['recommendations'].append(
                f"Most important feature: {top_feature[0]} (importance: {top_feature[1]:.3f})"
            )
        
        # Lead-specific insights
        for method, attr in attributions.items():
            mlii_importance = np.mean(np.abs(attr[:, 0, :]))
            v1_importance = np.mean(np.abs(attr[:, 1, :]))
            
            if mlii_importance > v1_importance:
                report['recommendations'].append(
                    f"{method}: MLII lead shows higher importance than V1 lead"
                )
            else:
                report['recommendations'].append(
                    f"{method}: V1 lead shows higher importance than MLII lead"
                )
        
        # Save report
        with open(save_path / 'clinical_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate text report
        with open(save_path / 'clinical_report.txt', 'w') as f:
            f.write("ECG Arrhythmia Detection - Clinical Interpretability Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Methods analyzed: {report['summary']['total_attribution_methods']}\n")
            f.write(f"Rules extracted: {report['summary']['extracted_rules']}\n")
            f.write(f"Beats analyzed: {report['summary']['beat_analysis']}\n\n")
            
            f.write("CLINICAL RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
            f.write("\n")
            
            f.write("DETAILED ANALYSIS\n")
            f.write("-" * 20 + "\n")
            for method, summary in report['attribution_summary'].items():
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  Mean importance: {summary['mean_importance']:.4f}\n")
                f.write(f"  MLII importance: {summary['lead_importance']['MLII']:.4f}\n")
                f.write(f"  V1 importance: {summary['lead_importance']['V1']:.4f}\n")
        
        logger.info(f"Clinical report saved to {save_path}")

