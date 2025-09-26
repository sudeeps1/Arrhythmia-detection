"""
ECG Signal Processing Module

This module provides functionality for preprocessing ECG signals including
filtering, segmentation, normalization, and feature extraction.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, resample
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ECGSignalProcessor:
    """
    ECG signal processing class for preprocessing MIT-BIH data.
    
    Implements bandpass filtering, normalization, segmentation, and
    feature extraction for ECG arrhythmia analysis.
    """
    
    def __init__(self, fs: int = 360, low_freq: float = 0.5, high_freq: float = 40.0):
        """
        Initialize the ECG signal processor.
        
        Args:
            fs: Sampling frequency in Hz
            low_freq: Low frequency cutoff for bandpass filter
            high_freq: High frequency cutoff for bandpass filter
        """
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        
        # Design bandpass filter
        self.b, self.a = self._design_bandpass_filter()
        
    def _design_bandpass_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design a Butterworth bandpass filter.
        
        Returns:
            Filter coefficients (b, a)
        """
        nyquist = self.fs / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        # Ensure frequencies are within valid range
        low = max(0.001, min(low, 0.99))
        high = max(0.001, min(high, 0.99))
        
        b, a = butter(4, [low, high], btype='band')
        return b, a
    
    def apply_bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to remove baseline wander and high-frequency noise.
        
        Args:
            signal_data: Input signal data (samples, channels)
            
        Returns:
            Filtered signal data
        """
        filtered_data = np.zeros_like(signal_data)
        
        for i in range(signal_data.shape[1]):
            # Apply zero-phase filtering to avoid phase distortion
            filtered_data[:, i] = filtfilt(self.b, self.a, signal_data[:, i])
        
        return filtered_data
    
    def normalize_signal(self, signal_data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize signal data.
        
        Args:
            signal_data: Input signal data
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized signal data
        """
        normalized_data = np.zeros_like(signal_data)
        
        for i in range(signal_data.shape[1]):
            signal = signal_data[:, i]
            
            if method == 'zscore':
                # Z-score normalization
                mean_val = np.mean(signal)
                std_val = np.std(signal)
                if std_val > 0:
                    normalized_data[:, i] = (signal - mean_val) / std_val
                else:
                    normalized_data[:, i] = signal - mean_val
                    
            elif method == 'minmax':
                # Min-max normalization to [0, 1]
                min_val = np.min(signal)
                max_val = np.max(signal)
                if max_val > min_val:
                    normalized_data[:, i] = (signal - min_val) / (max_val - min_val)
                else:
                    normalized_data[:, i] = signal - min_val
                    
            elif method == 'robust':
                # Robust normalization using median and MAD
                median_val = np.median(signal)
                mad = np.median(np.abs(signal - median_val))
                if mad > 0:
                    normalized_data[:, i] = (signal - median_val) / mad
                else:
                    normalized_data[:, i] = signal - median_val
                    
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized_data
    
    def segment_signal(self, signal_data: np.ndarray, annotations: pd.DataFrame,
                      window_size: float = 3.0, overlap: float = 0.5,
                      min_beats: int = 3) -> Dict:
        """
        Segment ECG signal into windows with beat annotations.
        
        Args:
            signal_data: ECG signal data (samples, channels)
            annotations: Beat annotations DataFrame
            window_size: Window size in seconds
            overlap: Overlap ratio (0-1)
            min_beats: Minimum number of beats required in a window
            
        Returns:
            Dictionary containing segmented data and labels
        """
        window_samples = int(window_size * self.fs)
        step_samples = int(window_samples * (1 - overlap))
        
        segments = []
        segment_labels = []
        segment_metadata = []
        
        # Convert annotation times to sample indices
        annotation_samples = (annotations['time'] * self.fs).astype(int)
        
        for start_idx in range(0, len(signal_data) - window_samples, step_samples):
            end_idx = start_idx + window_samples
            
            # Extract segment
            segment = signal_data[start_idx:end_idx, :]
            
            # Find beats in this window
            window_beats = annotations[
                (annotation_samples >= start_idx) & 
                (annotation_samples < end_idx)
            ]
            
            if len(window_beats) >= min_beats:
                # Determine label based on majority beat type
                beat_counts = window_beats['code'].value_counts()
                majority_beat = beat_counts.index[0]
                
                # Map beat types to arrhythmia categories
                arrhythmia_label = self._map_beat_to_arrhythmia(majority_beat)
                
                segments.append(segment)
                segment_labels.append(arrhythmia_label)
                
                segment_metadata.append({
                    'start_time': start_idx / self.fs,
                    'end_time': end_idx / self.fs,
                    'num_beats': len(window_beats),
                    'beat_types': beat_counts.to_dict(),
                    'majority_beat': majority_beat
                })
        
        return {
            'segments': np.array(segments),
            'labels': np.array(segment_labels),
            'metadata': segment_metadata
        }
    
    def _map_beat_to_arrhythmia(self, beat_code: str) -> str:
        """
        Map beat annotation codes to arrhythmia categories.
        
        Args:
            beat_code: Beat annotation code
            
        Returns:
            Arrhythmia category
        """
        # Normal beats
        if beat_code in ['N', 'L', 'R', 'B']:
            return 'normal'
        
        # Supraventricular arrhythmias
        elif beat_code in ['A', 'a', 'J', 'S', 'e', 'j', 'n']:
            return 'supraventricular'
        
        # Ventricular arrhythmias
        elif beat_code in ['V', 'r', 'E']:
            return 'ventricular'
        
        # Fusion beats
        elif beat_code in ['F', 'f']:
            return 'fusion'
        
        # Paced beats
        elif beat_code in ['/']:
            return 'paced'
        
        # Unknown/unclassifiable
        else:
            return 'unknown'
    
    def extract_rr_intervals(self, annotations: pd.DataFrame) -> np.ndarray:
        """
        Extract R-R intervals from beat annotations.
        
        Args:
            annotations: Beat annotations DataFrame
            
        Returns:
            Array of R-R intervals in seconds
        """
        # Filter for R-peaks (normal beats and ventricular beats)
        r_peaks = annotations[annotations['code'].isin(['N', 'L', 'R', 'B', 'V', 'r'])]
        
        if len(r_peaks) < 2:
            return np.array([])
        
        # Calculate R-R intervals
        rr_intervals = np.diff(r_peaks['time'].values)
        
        return rr_intervals
    
    def extract_qrs_features(self, signal_data: np.ndarray, 
                           annotations: pd.DataFrame) -> pd.DataFrame:
        """
        Extract QRS complex features from ECG signal.
        
        Args:
            signal_data: ECG signal data
            annotations: Beat annotations DataFrame
            
        Returns:
            DataFrame with QRS features
        """
        features = []
        
        for _, beat in annotations.iterrows():
            sample_idx = int(beat['sample'])
            
            # Extract window around R-peak
            window_size = int(0.2 * self.fs)  # 200ms window
            start_idx = max(0, sample_idx - window_size // 2)
            end_idx = min(len(signal_data), sample_idx + window_size // 2)
            
            if end_idx - start_idx < window_size // 4:
                continue
            
            # Extract QRS segment
            qrs_segment = signal_data[start_idx:end_idx, 0]  # Use first lead
            
            # Calculate features
            feature_dict = {
                'beat_time': beat['time'],
                'beat_code': beat['code'],
                'qrs_amplitude': np.max(qrs_segment) - np.min(qrs_segment),
                'qrs_duration': self._estimate_qrs_duration(qrs_segment),
                'qrs_area': np.trapz(np.abs(qrs_segment)),
                'qrs_slope': np.max(np.diff(qrs_segment)),
                'qrs_energy': np.sum(qrs_segment ** 2)
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _estimate_qrs_duration(self, qrs_segment: np.ndarray) -> float:
        """
        Estimate QRS duration using envelope detection.
        
        Args:
            qrs_segment: QRS complex segment
            
        Returns:
            Estimated QRS duration in seconds
        """
        # Calculate signal envelope
        analytic_signal = signal.hilbert(qrs_segment)
        envelope = np.abs(analytic_signal)
        
        # Find threshold for QRS detection
        threshold = 0.5 * np.max(envelope)
        
        # Find QRS boundaries
        qrs_mask = envelope > threshold
        
        if np.sum(qrs_mask) == 0:
            return 0.08  # Default QRS duration
        
        # Find first and last points above threshold
        qrs_indices = np.where(qrs_mask)[0]
        qrs_start = qrs_indices[0]
        qrs_end = qrs_indices[-1]
        
        qrs_duration = (qrs_end - qrs_start) / self.fs
        
        return qrs_duration
    
    def resample_signal(self, signal_data: np.ndarray, 
                       target_fs: int = 250) -> np.ndarray:
        """
        Resample signal to target frequency.
        
        Args:
            signal_data: Input signal data
            target_fs: Target sampling frequency
            
        Returns:
            Resampled signal data
        """
        if target_fs == self.fs:
            return signal_data
        
        # Calculate resampling ratio
        ratio = target_fs / self.fs
        
        resampled_data = np.zeros((int(len(signal_data) * ratio), signal_data.shape[1]))
        
        for i in range(signal_data.shape[1]):
            resampled_data[:, i] = resample(signal_data[:, i], 
                                          int(len(signal_data) * ratio))
        
        return resampled_data
    
    def detect_artifacts(self, signal_data: np.ndarray, 
                        threshold: float = 3.0) -> np.ndarray:
        """
        Detect artifacts in ECG signal using statistical methods.
        
        Args:
            signal_data: ECG signal data
            threshold: Threshold for artifact detection (in standard deviations)
            
        Returns:
            Boolean array indicating artifact samples
        """
        artifacts = np.zeros(len(signal_data), dtype=bool)
        
        for i in range(signal_data.shape[1]):
            signal = signal_data[:, i]
            
            # Calculate moving statistics
            window_size = int(5 * self.fs)  # 5-second window
            if len(signal) < window_size:
                continue
            
            # Calculate moving mean and std
            moving_mean = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
            moving_std = np.sqrt(np.convolve((signal - moving_mean)**2, 
                                           np.ones(window_size)/window_size, mode='same'))
            
            # Detect outliers
            z_scores = np.abs(signal - moving_mean) / (moving_std + 1e-8)
            artifacts |= z_scores > threshold
        
        return artifacts
    
    def preprocess_record(self, signal_data: np.ndarray, 
                         annotations: pd.DataFrame,
                         window_size: float = 3.0,
                         overlap: float = 0.5,
                         normalize_method: str = 'zscore',
                         target_fs: Optional[int] = None) -> Dict:
        """
        Complete preprocessing pipeline for a single record.
        
        Args:
            signal_data: Raw ECG signal data
            annotations: Beat annotations
            window_size: Window size for segmentation
            overlap: Overlap ratio for segmentation
            normalize_method: Normalization method
            target_fs: Target sampling frequency (None to keep original)
            
        Returns:
            Dictionary with preprocessed data
        """
        logger.info("Starting ECG preprocessing pipeline...")
        
        # Step 1: Apply bandpass filter
        logger.info("Applying bandpass filter...")
        filtered_data = self.apply_bandpass_filter(signal_data)
        
        # Step 2: Detect and handle artifacts
        logger.info("Detecting artifacts...")
        artifacts = self.detect_artifacts(filtered_data)
        # For now, we'll keep all data but flag artifacts
        # In a production system, you might want to interpolate or remove artifacts
        
        # Step 3: Normalize signal
        logger.info(f"Normalizing signal using {normalize_method} method...")
        normalized_data = self.normalize_signal(filtered_data, method=normalize_method)
        
        # Step 4: Resample if needed
        if target_fs is not None and target_fs != self.fs:
            logger.info(f"Resampling from {self.fs} Hz to {target_fs} Hz...")
            resampled_data = self.resample_signal(normalized_data, target_fs)
            # Update annotations for new sampling rate
            annotations_resampled = annotations.copy()
            annotations_resampled['sample'] = (annotations['sample'] * target_fs / self.fs).astype(int)
            annotations_resampled['time'] = annotations_resampled['sample'] / target_fs
        else:
            resampled_data = normalized_data
            annotations_resampled = annotations
        
        # Step 5: Segment signal
        logger.info("Segmenting signal into windows...")
        segmentation_result = self.segment_signal(
            resampled_data, annotations_resampled, 
            window_size=window_size, overlap=overlap
        )
        
        # Step 6: Extract additional features
        logger.info("Extracting QRS features...")
        qrs_features = self.extract_qrs_features(resampled_data, annotations_resampled)
        rr_intervals = self.extract_rr_intervals(annotations_resampled)
        
        logger.info("Preprocessing completed successfully!")
        
        return {
            'segments': segmentation_result['segments'],
            'labels': segmentation_result['labels'],
            'metadata': segmentation_result['metadata'],
            'qrs_features': qrs_features,
            'rr_intervals': rr_intervals,
            'artifacts': artifacts,
            'preprocessing_info': {
                'original_fs': self.fs,
                'target_fs': target_fs or self.fs,
                'filter_freqs': [self.low_freq, self.high_freq],
                'normalize_method': normalize_method,
                'window_size': window_size,
                'overlap': overlap
            }
        }


def main():
    """Example usage of the ECGSignalProcessor."""
    # This would be used in conjunction with the data loader
    processor = ECGSignalProcessor(fs=360)
    
    print("ECG Signal Processor initialized with:")
    print(f"  Sampling frequency: {processor.fs} Hz")
    print(f"  Bandpass filter: {processor.low_freq}-{processor.high_freq} Hz")


if __name__ == "__main__":
    main()

