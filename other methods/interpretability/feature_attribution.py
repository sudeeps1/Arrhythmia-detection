"""
Feature Attribution for ECG Arrhythmia Analysis

This module provides methods for computing temporal and lead-specific feature
importance using various attribution techniques including Integrated Gradients,
SHAP, and ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from captum.attr import IntegratedGradients, Occlusion, LayerIntegratedGradients
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logger = logging.getLogger(__name__)

class ECGFeatureAttribution:
    """
    Feature attribution analysis for ECG arrhythmia classification.
    
    Implements various attribution methods to identify critical temporal
    and lead-specific patterns in ECG signals.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize feature attribution analyzer.
        
        Args:
            model: Trained ECG classifier model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Create a wrapper function that returns logits for Captum
        def model_wrapper(x):
            outputs = self.model(x)
            return outputs['logits']
        
        self.model_wrapper = model_wrapper
        
        # Initialize attribution methods with the wrapper
        self.integrated_gradients = IntegratedGradients(self.model_wrapper)
        self.occlusion = Occlusion(self.model_wrapper)
        
    def compute_integrated_gradients(self, 
                                   x: torch.Tensor,
                                   target_class: Optional[int] = None,
                                   baseline: Optional[torch.Tensor] = None,
                                   steps: int = 50) -> torch.Tensor:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            x: Input tensor (batch, num_leads, sequence_length)
            target_class: Target class for attribution (None for predicted class)
            baseline: Baseline tensor (None for zero baseline)
            steps: Number of steps for integration
            
        Returns:
            Attribution tensor with same shape as input
        """
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        if target_class is None:
            with torch.no_grad():
                logits = self.model_wrapper(x)
                target_class = torch.argmax(logits, dim=1)
        
        attributions = self.integrated_gradients.attribute(
            inputs=x,
            target=target_class,
            baselines=baseline,
            n_steps=steps
        )
        
        return attributions
    
    def compute_occlusion_attribution(self,
                                    x: torch.Tensor,
                                    target_class: Optional[int] = None,
                                    window_size: int = 50,
                                    stride: int = 25) -> torch.Tensor:
        """
        Compute occlusion-based attribution.
        
        Args:
            x: Input tensor
            target_class: Target class for attribution
            window_size: Size of occlusion window
            stride: Stride for occlusion windows
            
        Returns:
            Attribution tensor
        """
        if target_class is None:
            with torch.no_grad():
                logits = self.model_wrapper(x)
                target_class = torch.argmax(logits, dim=1)
        
        attributions = self.occlusion.attribute(
            inputs=x,
            target=target_class,
            sliding_window_shapes=(1, window_size),
            strides=(1, stride)
        )
        
        return attributions
    
    def compute_temporal_importance(self,
                                  x: torch.Tensor,
                                  method: str = 'integrated_gradients',
                                  **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute temporal importance scores for ECG segments.
        
        Args:
            x: Input tensor
            method: Attribution method ('integrated_gradients', 'occlusion')
            **kwargs: Additional arguments for attribution method
            
        Returns:
            Dictionary with temporal importance scores
        """
        if method == 'integrated_gradients':
            attributions = self.compute_integrated_gradients(x, **kwargs)
        elif method == 'occlusion':
            attributions = self.compute_occlusion_attribution(x, **kwargs)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Compute temporal importance by averaging across leads
        temporal_importance = torch.mean(torch.abs(attributions), dim=1)  # (batch, seq_len)
        
        # Compute lead-specific temporal importance
        lead_temporal_importance = torch.abs(attributions)  # (batch, num_leads, seq_len)
        
        return {
            'attributions': attributions,
            'temporal_importance': temporal_importance,
            'lead_temporal_importance': lead_temporal_importance
        }
    
    def compute_lead_importance(self,
                               x: torch.Tensor,
                               method: str = 'integrated_gradients',
                               **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute lead-specific importance scores.
        
        Args:
            x: Input tensor
            method: Attribution method
            **kwargs: Additional arguments for attribution method
            
        Returns:
            Dictionary with lead importance scores
        """
        attributions = self.compute_temporal_importance(x, method, **kwargs)
        
        # Compute lead importance by averaging across time
        lead_importance = torch.mean(attributions['lead_temporal_importance'], dim=2)  # (batch, num_leads)
        
        # Compute lead importance by summing absolute attributions
        lead_importance_sum = torch.sum(torch.abs(attributions['attributions']), dim=2)  # (batch, num_leads)
        
        return {
            'lead_importance': lead_importance,
            'lead_importance_sum': lead_importance_sum,
            'attributions': attributions['attributions']
        }
    
    def compute_critical_segments(self,
                                 x: torch.Tensor,
                                 threshold_percentile: float = 90.0,
                                 min_segment_length: int = 50,
                                 method: str = 'integrated_gradients',
                                 **kwargs) -> Dict[str, List]:
        """
        Identify critical temporal segments in ECG signals.
        
        Args:
            x: Input tensor
            threshold_percentile: Percentile threshold for critical segments
            min_segment_length: Minimum length of critical segments
            method: Attribution method
            **kwargs: Additional arguments for attribution method
            
        Returns:
            Dictionary with critical segment information
        """
        temporal_importance = self.compute_temporal_importance(x, method, **kwargs)
        
        critical_segments = []
        lead_critical_segments = []
        
        batch_size = x.size(0)
        num_leads = x.size(1)
        
        for b in range(batch_size):
            # Overall temporal importance
            importance = temporal_importance['temporal_importance'][b].cpu().numpy()
            threshold = np.percentile(importance, threshold_percentile)
            
            # Find critical segments
            critical_mask = importance > threshold
            segments = self._find_continuous_segments(critical_mask, min_segment_length)
            critical_segments.append(segments)
            
            # Lead-specific critical segments
            lead_segments = []
            for l in range(num_leads):
                lead_importance = temporal_importance['lead_temporal_importance'][b, l].cpu().numpy()
                lead_threshold = np.percentile(lead_importance, threshold_percentile)
                lead_critical_mask = lead_importance > lead_threshold
                lead_segments.append(self._find_continuous_segments(lead_critical_mask, min_segment_length))
            lead_critical_segments.append(lead_segments)
        
        return {
            'critical_segments': critical_segments,
            'lead_critical_segments': lead_critical_segments,
            'temporal_importance': temporal_importance
        }
    
    def _find_continuous_segments(self, mask: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
        """
        Find continuous segments in a boolean mask.
        
        Args:
            mask: Boolean mask
            min_length: Minimum segment length
            
        Returns:
            List of (start, end) indices for segments
        """
        segments = []
        start = None
        
        for i, is_critical in enumerate(mask):
            if is_critical and start is None:
                start = i
            elif not is_critical and start is not None:
                if i - start >= min_length:
                    segments.append((start, i))
                start = None
        
        # Handle case where segment extends to end
        if start is not None and len(mask) - start >= min_length:
            segments.append((start, len(mask)))
        
        return segments
    
    def ablation_study(self,
                      x: torch.Tensor,
                      ablation_type: str = 'temporal',
                      ablation_size: int = 100,
                      stride: int = 50,
                      target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Perform ablation study to validate feature importance.
        
        Args:
            x: Input tensor
            ablation_type: Type of ablation ('temporal', 'lead', 'segment')
            ablation_size: Size of ablation window
            stride: Stride for ablation windows
            target_class: Target class for evaluation
            
        Returns:
            Dictionary with ablation results
        """
        if target_class is None:
            with torch.no_grad():
                logits = self.model_wrapper(x)
                target_class = torch.argmax(logits, dim=1)
        
        original_outputs = self.model(x)
        original_probs = F.softmax(original_outputs['logits'], dim=1)
        original_conf = original_probs[torch.arange(len(target_class)), target_class]
        
        ablation_results = []
        
        if ablation_type == 'temporal':
            # Temporal ablation
            for start in range(0, x.size(2) - ablation_size, stride):
                end = start + ablation_size
                
                # Create ablated input
                x_ablated = x.clone()
                x_ablated[:, :, start:end] = 0.0  # Zero out segment
                
                # Get model prediction
                with torch.no_grad():
                    ablated_outputs = self.model(x_ablated)
                    ablated_probs = F.softmax(ablated_outputs['logits'], dim=1)
                    ablated_conf = ablated_probs[torch.arange(len(target_class)), target_class]
                
                # Compute confidence drop
                confidence_drop = original_conf - ablated_conf
                ablation_results.append({
                    'start': start,
                    'end': end,
                    'confidence_drop': confidence_drop.detach().cpu().numpy()
                })
        
        elif ablation_type == 'lead':
            # Lead ablation
            for lead in range(x.size(1)):
                x_ablated = x.clone()
                x_ablated[:, lead, :] = 0.0  # Zero out lead
                
                with torch.no_grad():
                    ablated_outputs = self.model(x_ablated)
                    ablated_probs = F.softmax(ablated_outputs['logits'], dim=1)
                    ablated_conf = ablated_probs[torch.arange(len(target_class)), target_class]
                
                confidence_drop = original_conf - ablated_conf
                ablation_results.append({
                    'lead': lead,
                    'confidence_drop': confidence_drop.detach().cpu().numpy()
                })
        
        return {
            'ablation_results': ablation_results,
            'original_confidence': original_conf.detach().cpu().numpy(),
            'ablation_type': ablation_type
        }
    
    def compute_attribution_correlation(self,
                                      x: torch.Tensor,
                                      ablation_results: Dict,
                                      attribution_method: str = 'integrated_gradients') -> Dict:
        """
        Compute correlation between attribution scores and ablation results.
        
        Args:
            x: Input tensor
            ablation_results: Results from ablation study
            attribution_method: Attribution method to use
            
        Returns:
            Dictionary with correlation analysis
        """
        # Get attributions
        attributions = self.compute_temporal_importance(x, attribution_method)
        
        correlations = {}
        
        if ablation_results['ablation_type'] == 'temporal':
            # Correlate temporal attributions with temporal ablation
            temporal_importance = attributions['temporal_importance'].cpu().numpy()
            
            for result in ablation_results['ablation_results']:
                start, end = result['start'], result['end']
                segment_importance = np.mean(temporal_importance[:, start:end], axis=1)
                confidence_drop = result['confidence_drop']
                
                correlation = np.corrcoef(segment_importance, confidence_drop)[0, 1]
                correlations[f'segment_{start}_{end}'] = correlation
        
        elif ablation_results['ablation_type'] == 'lead':
            # Correlate lead attributions with lead ablation
            lead_importance = self.compute_lead_importance(x, attribution_method)
            lead_importance_scores = lead_importance['lead_importance'].cpu().numpy()
            
            for result in ablation_results['ablation_results']:
                lead = result['lead']
                confidence_drop = result['confidence_drop']
                importance_score = lead_importance_scores[:, lead]
                
                correlation = np.corrcoef(importance_score, confidence_drop)[0, 1]
                correlations[f'lead_{lead}'] = correlation
        
        return {
            'correlations': correlations,
            'mean_correlation': np.mean(list(correlations.values())),
            'attribution_method': attribution_method
        }
    
    def visualize_attributions(self,
                              x: torch.Tensor,
                              attributions: torch.Tensor,
                              lead_names: List[str] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Visualize attribution results.
        
        Args:
            x: Input tensor
            attributions: Attribution tensor
            lead_names: Names of ECG leads
            save_path: Path to save visualization
        """
        if lead_names is None:
            lead_names = [f'Lead {i+1}' for i in range(x.size(1))]
        
        fig, axes = plt.subplots(x.size(1), 2, figsize=(15, 5 * x.size(1)))
        if x.size(1) == 1:
            axes = axes.reshape(1, -1)
        
        for lead in range(x.size(1)):
            # Original signal
            axes[lead, 0].plot(x[0, lead, :].cpu().numpy(), label='Original Signal', alpha=0.7)
            axes[lead, 0].set_title(f'{lead_names[lead]} - Original Signal')
            axes[lead, 0].set_xlabel('Time (samples)')
            axes[lead, 0].set_ylabel('Amplitude')
            axes[lead, 0].legend()
            
            # Attribution overlay
            signal = x[0, lead, :].cpu().numpy()
            attribution = attributions[0, lead, :].cpu().numpy()
            
            # Normalize attribution for visualization
            attribution_norm = np.abs(attribution) / np.max(np.abs(attribution))
            
            axes[lead, 1].plot(signal, label='Original Signal', alpha=0.7)
            axes[lead, 1].fill_between(range(len(signal)), signal, alpha=0.3, 
                                     color='red', where=attribution_norm > 0.5)
            axes[lead, 1].set_title(f'{lead_names[lead]} - Attribution Overlay')
            axes[lead, 1].set_xlabel('Time (samples)')
            axes[lead, 1].set_ylabel('Amplitude')
            axes[lead, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_attribution_report(self,
                                  x: torch.Tensor,
                                  target_class: Optional[int] = None,
                                  save_dir: str = 'results/attribution') -> Dict:
        """
        Generate comprehensive attribution report.
        
        Args:
            x: Input tensor
            target_class: Target class for attribution
            save_dir: Directory to save results
            
        Returns:
            Dictionary with attribution report
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Compute attributions using multiple methods
        methods = ['integrated_gradients', 'occlusion']
        attribution_results = {}
        
        for method in methods:
            logger.info(f"Computing {method} attributions...")
            attribution_results[method] = self.compute_temporal_importance(
                x, method=method, target_class=target_class
            )
        
        # Compute critical segments
        logger.info("Identifying critical segments...")
        critical_segments = self.compute_critical_segments(
            x, method='integrated_gradients', target_class=target_class
        )
        
        # Perform ablation studies
        logger.info("Performing ablation studies...")
        ablation_results = {
            'temporal': self.ablation_study(x, 'temporal', target_class=target_class),
            'lead': self.ablation_study(x, 'lead', target_class=target_class)
        }
        
        # Compute correlations
        logger.info("Computing attribution correlations...")
        correlations = {}
        for method in methods:
            correlations[method] = self.compute_attribution_correlation(
                x, ablation_results['temporal'], method
            )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        for method in methods:
            self.visualize_attributions(
                x, attribution_results[method]['attributions'],
                save_path=f"{save_dir}/attributions_{method}.png"
            )
        
        # Create summary report
        report = {
            'attribution_results': attribution_results,
            'critical_segments': critical_segments,
            'ablation_results': ablation_results,
            'correlations': correlations,
            'summary': {
                'mean_temporal_importance': {
                    method: torch.mean(result['temporal_importance']).item()
                    for method, result in attribution_results.items()
                },
                'mean_correlation': {
                    method: corr['mean_correlation']
                    for method, corr in correlations.items()
                },
                'num_critical_segments': len(critical_segments['critical_segments'][0])
            }
        }
        
        # Save report
        import json
        with open(f"{save_dir}/attribution_report.json", 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_report = self._convert_tensors_to_lists(report)
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Attribution report saved to {save_dir}")
        return report
    
    def _convert_tensors_to_lists(self, obj):
        """Convert tensors to lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_tensors_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_lists(item) for item in obj]
        else:
            return obj


def main():
    """Example usage of the ECGFeatureAttribution class."""
    # This would be used with a trained model
    print("ECG Feature Attribution module initialized")
    print("Use with a trained ECG classifier model for attribution analysis")


if __name__ == "__main__":
    main()
