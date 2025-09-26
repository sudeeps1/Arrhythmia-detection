"""
ECG Data Augmentation Module

This module provides data augmentation techniques specifically designed for ECG signals,
including time-series aware oversampling, window-level augmentations, and synthetic
beat generation for improving class balance in arrhythmia detection.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class ECGAugmenter:
    """
    ECG-specific data augmentation for arrhythmia detection.
    
    Implements time-series aware oversampling, window-level augmentations,
    and synthetic beat generation to improve class balance.
    """
    
    def __init__(self, fs: int = 360, random_seed: int = 42):
        """
        Initialize ECG augmenter.
        
        Args:
            fs: Sampling frequency in Hz
            random_seed: Random seed for reproducibility
        """
        self.fs = fs
        self.random_seed = random_seed
        np.random.seed(random_seed)
        

    
    def window_augmentation(self, 
                           segment: np.ndarray, 
                           augmentation_type: str = 'random') -> np.ndarray:
        """
        Apply window-level augmentations to ECG segments.
        
        Args:
            segment: ECG segment (n_leads, n_timesteps)
            augmentation_type: Type of augmentation ('random', 'noise', 'warp', 'scale')
            
        Returns:
            Augmented segment
        """
        if augmentation_type == 'random':
            # Randomly choose augmentation
            aug_types = ['noise', 'warp', 'scale', 'none']
            augmentation_type = np.random.choice(aug_types)
        
        if augmentation_type == 'none':
            return segment.copy()
        
        augmented = segment.copy()
        
        if augmentation_type == 'noise':
            # Add Gaussian noise
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, segment.shape)
            augmented += noise
            
        elif augmentation_type == 'warp':
            # Time warping (stretch/compress)
            warp_factor = np.random.uniform(0.9, 1.1)
            n_timesteps = segment.shape[1]
            new_length = int(n_timesteps * warp_factor)
            
            # Resample the signal
            for lead in range(segment.shape[0]):
                original_time = np.linspace(0, 1, n_timesteps)
                new_time = np.linspace(0, 1, new_length)
                
                # Interpolate to new time points
                interpolator = interp1d(original_time, segment[lead], 
                                      kind='linear', bounds_error=False, 
                                      fill_value='extrapolate')
                warped_signal = interpolator(new_time)
                
                # Resample back to original length
                if len(warped_signal) != n_timesteps:
                    warped_signal = signal.resample(warped_signal, n_timesteps)
                
                augmented[lead] = warped_signal
                
        elif augmentation_type == 'scale':
            # Amplitude scaling
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented *= scale_factor
        
        return augmented
    
    def generate_synthetic_vbeats(self, 
                                 normal_segments: np.ndarray,
                                 vbeat_template: np.ndarray,
                                 insertion_prob: float = 0.3) -> np.ndarray:
        """
        Generate synthetic ventricular ectopic beats by inserting VEB templates
        into normal sequences.
        
        Args:
            normal_segments: Normal ECG segments
            vbeat_template: Template VEB pattern
            insertion_prob: Probability of inserting VEB into each segment
            
        Returns:
            Segments with synthetic VEBs
        """
        synthetic_segments = []
        
        for segment in normal_segments:
            if np.random.random() < insertion_prob:
                # Insert VEB template at random position
                segment_len = segment.shape[1]
                vbeat_len = vbeat_template.shape[1]
                
                # Ensure VEB fits within segment
                if vbeat_len <= segment_len:
                    # Random insertion point
                    insert_start = np.random.randint(0, segment_len - vbeat_len + 1)
                    
                    # Create synthetic segment
                    synthetic_segment = segment.copy()
                    
                    # Blend VEB template with normal signal
                    blend_factor = np.random.uniform(0.6, 0.9)
                    synthetic_segment[:, insert_start:insert_start + vbeat_len] = (
                        (1 - blend_factor) * segment[:, insert_start:insert_start + vbeat_len] +
                        blend_factor * vbeat_template
                    )
                    
                    synthetic_segments.append(synthetic_segment)
                else:
                    synthetic_segments.append(segment)
            else:
                synthetic_segments.append(segment)
        
        return np.array(synthetic_segments)
    
    def extract_vbeat_template(self, 
                             segments: np.ndarray, 
                             labels: np.ndarray,
                             vbeat_class: int = 1) -> np.ndarray:
        """
        Extract a representative VEB template from VEB segments.
        
        Args:
            segments: ECG segments
            labels: Segment labels
            vbeat_class: Class label for VEBs
            
        Returns:
            VEB template
        """
        vbeat_indices = np.where(labels == vbeat_class)[0]
        
        if len(vbeat_indices) == 0:
            logger.warning("No VEB segments found for template extraction")
            return None
        
        # Use the first VEB segment as template
        # In a more sophisticated approach, you could cluster VEBs and use centroids
        template = segments[vbeat_indices[0]]
        
        return template
    
    def augment_dataset(self, 
                       segments: np.ndarray, 
                       labels: np.ndarray,
                       augmentation_config: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply comprehensive data augmentation to the dataset.
        
        Args:
            segments: ECG segments
            labels: Segment labels
            augmentation_config: Configuration for augmentation
            
        Returns:
            Tuple of (augmented_segments, augmented_labels)
        """
        if augmentation_config is None:
            augmentation_config = {
                'window_augment': True,
                'synthetic_vbeats': True,
                'augmentation_prob': 0.5
            }
        
        logger.info("Starting data augmentation...")
        
        # Step 1: Window-level augmentations for minority classes
        if augmentation_config.get('window_augment', False):
            logger.info("Applying window-level augmentations...")
            augmented_segments = []
            augmented_labels = []
            
            for i, (segment, label) in enumerate(zip(segments, labels)):
                augmented_segments.append(segment)
                augmented_labels.append(label)
                
                # Apply augmentation to minority classes with some probability
                class_counts = Counter(labels)
                majority_class = max(class_counts, key=class_counts.get)
                
                if (label != majority_class and 
                    np.random.random() < augmentation_config.get('augmentation_prob', 0.5)):
                    
                    aug_segment = self.window_augmentation(segment, 'random')
                    augmented_segments.append(aug_segment)
                    augmented_labels.append(label)
            
            segments = np.array(augmented_segments)
            labels = np.array(augmented_labels)
        
        # Step 2: Synthetic VEB generation
        if augmentation_config.get('synthetic_vbeats', False):
            logger.info("Generating synthetic VEBs...")
            vbeat_template = self.extract_vbeat_template(segments, labels)
            
            if vbeat_template is not None:
                # Find normal segments
                normal_indices = np.where(labels == 0)[0]  # Assuming 0 is normal class
                if len(normal_indices) > 0:
                    normal_segments = segments[normal_indices]
                    
                    # Generate synthetic VEBs
                    synthetic_segments = self.generate_synthetic_vbeats(
                        normal_segments, vbeat_template
                    )
                    
                    # Add synthetic segments with VEB labels
                    synthetic_labels = np.full(len(synthetic_segments), 1)  # VEB class
                    
                    segments = np.concatenate([segments, synthetic_segments], axis=0)
                    labels = np.concatenate([labels, synthetic_labels], axis=0)
        
        logger.info(f"Augmentation complete. Final dataset: {len(segments)} samples")
        logger.info(f"Class distribution: {Counter(labels)}")
        
        return segments, labels
    


