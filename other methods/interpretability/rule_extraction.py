"""
Neuro-Symbolic Rule Extraction for ECG Arrhythmia Analysis

This module implements methods for extracting human-readable decision rules
from trained ECG classifiers, linking ECG patterns to arrhythmia predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class ECGRuleExtractor:
    """
    Neuro-symbolic rule extractor for ECG arrhythmia classification.
    
    Extracts human-readable decision rules from trained models by analyzing
    feature importance and decision boundaries.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize rule extractor.
        
        Args:
            model: Trained ECG classifier model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Feature names for interpretable rules (matching the features extracted in pipeline)
        self.feature_names = [
            'mean_amplitude', 'std_amplitude', 'max_amplitude',
            'min_amplitude', 'signal_energy', 'zero_crossings'
        ]
        
    def extract_ecg_features(self, 
                           signal_data: np.ndarray,
                           annotations: pd.DataFrame,
                           fs: int = 360) -> pd.DataFrame:
        """
        Extract interpretable ECG features for rule generation.
        
        Args:
            signal_data: ECG signal data (samples, leads)
            annotations: Beat annotations DataFrame
            fs: Sampling frequency
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        # Group annotations by time windows (e.g., 10-second windows)
        window_size = 10  # seconds
        window_samples = window_size * fs
        
        for start_idx in range(0, len(signal_data) - window_samples, window_samples):
            end_idx = start_idx + window_samples
            
            # Get annotations in this window
            window_annotations = annotations[
                (annotations['sample'] >= start_idx) & 
                (annotations['sample'] < end_idx)
            ]
            
            if len(window_annotations) < 3:  # Need at least 3 beats
                continue
            
            # Extract features for this window
            window_features = self._extract_window_features(
                signal_data[start_idx:end_idx, :],
                window_annotations,
                fs
            )
            
            if window_features is not None:
                features.append(window_features)
        
        return pd.DataFrame(features)
    
    def _extract_window_features(self,
                                window_signal: np.ndarray,
                                window_annotations: pd.DataFrame,
                                fs: int) -> Optional[Dict]:
        """
        Extract features from a time window.
        
        Args:
            window_signal: Signal data for the window
            window_annotations: Annotations in the window
            fs: Sampling frequency
            
        Returns:
            Dictionary of features or None if insufficient data
        """
        try:
            # R-R intervals
            rr_intervals = np.diff(window_annotations['time'].values)
            
            if len(rr_intervals) < 2:
                return None
            
            # QRS features
            qrs_features = self._extract_qrs_features(window_signal, window_annotations, fs)
            
            # Heart rate features
            heart_rates = 60.0 / rr_intervals  # Convert to BPM
            
            # P-wave and T-wave features (simplified)
            p_wave_amp = self._estimate_p_wave_amplitude(window_signal, window_annotations, fs)
            t_wave_amp = self._estimate_t_wave_amplitude(window_signal, window_annotations, fs)
            
            # ST segment features
            st_deviation = self._estimate_st_deviation(window_signal, window_annotations, fs)
            
            # QT interval features
            qt_intervals = self._estimate_qt_intervals(window_signal, window_annotations, fs)
            qtc_intervals = self._correct_qt_intervals(qt_intervals, rr_intervals)
            
            features = {
                'RR_interval_mean': np.mean(rr_intervals),
                'RR_interval_std': np.std(rr_intervals),
                'RR_interval_cv': np.std(rr_intervals) / np.mean(rr_intervals),
                'QRS_duration_mean': np.mean(qrs_features['duration']),
                'QRS_duration_std': np.std(qrs_features['duration']),
                'QRS_amplitude_mean': np.mean(qrs_features['amplitude']),
                'QRS_amplitude_std': np.std(qrs_features['amplitude']),
                'QRS_area_mean': np.mean(qrs_features['area']),
                'QRS_slope_mean': np.mean(qrs_features['slope']),
                'QRS_energy_mean': np.mean(qrs_features['energy']),
                'Heart_rate_mean': np.mean(heart_rates),
                'Heart_rate_std': np.std(heart_rates),
                'P_wave_amplitude_mean': np.mean(p_wave_amp),
                'T_wave_amplitude_mean': np.mean(t_wave_amp),
                'ST_segment_deviation_mean': np.mean(st_deviation),
                'QT_interval_mean': np.mean(qt_intervals),
                'QT_interval_std': np.std(qt_intervals),
                'QTc_interval_mean': np.mean(qtc_intervals),
                'QTc_interval_std': np.std(qtc_intervals)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting window features: {str(e)}")
            return None
    
    def _extract_qrs_features(self,
                             signal: np.ndarray,
                             annotations: pd.DataFrame,
                             fs: int) -> Dict[str, np.ndarray]:
        """Extract QRS complex features."""
        features = {'duration': [], 'amplitude': [], 'area': [], 'slope': [], 'energy': []}
        
        for _, beat in annotations.iterrows():
            sample_idx = int(beat['sample'])
            
            # Extract window around R-peak
            window_size = int(0.2 * fs)  # 200ms window
            start_idx = max(0, sample_idx - window_size // 2)
            end_idx = min(len(signal), sample_idx + window_size // 2)
            
            if end_idx - start_idx < window_size // 4:
                continue
            
            # Extract QRS segment
            qrs_segment = signal[start_idx:end_idx, 0]  # Use first lead
            
            # Calculate features
            features['amplitude'].append(np.max(qrs_segment) - np.min(qrs_segment))
            features['duration'].append(self._estimate_qrs_duration(qrs_segment, fs))
            features['area'].append(np.trapz(np.abs(qrs_segment)))
            features['slope'].append(np.max(np.diff(qrs_segment)))
            features['energy'].append(np.sum(qrs_segment ** 2))
        
        return {k: np.array(v) for k, v in features.items()}
    
    def _estimate_qrs_duration(self, qrs_segment: np.ndarray, fs: int) -> float:
        """Estimate QRS duration using envelope detection."""
        from scipy.signal import hilbert
        
        # Calculate signal envelope
        analytic_signal = hilbert(qrs_segment)
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
        
        qrs_duration = (qrs_end - qrs_start) / fs
        
        return qrs_duration
    
    def _estimate_p_wave_amplitude(self,
                                  signal: np.ndarray,
                                  annotations: pd.DataFrame,
                                  fs: int) -> np.ndarray:
        """Estimate P-wave amplitude (simplified)."""
        amplitudes = []
        
        for _, beat in annotations.iterrows():
            sample_idx = int(beat['sample'])
            
            # Look for P-wave before R-peak
            p_window_start = max(0, sample_idx - int(0.2 * fs))
            p_window_end = sample_idx
            
            if p_window_end - p_window_start < int(0.1 * fs):
                amplitudes.append(0.0)
                continue
            
            p_segment = signal[p_window_start:p_window_end, 0]
            amplitudes.append(np.max(p_segment) - np.min(p_segment))
        
        return np.array(amplitudes)
    
    def _estimate_t_wave_amplitude(self,
                                  signal: np.ndarray,
                                  annotations: pd.DataFrame,
                                  fs: int) -> np.ndarray:
        """Estimate T-wave amplitude (simplified)."""
        amplitudes = []
        
        for _, beat in annotations.iterrows():
            sample_idx = int(beat['sample'])
            
            # Look for T-wave after R-peak
            t_window_start = sample_idx
            t_window_end = min(len(signal), sample_idx + int(0.4 * fs))
            
            if t_window_end - t_window_start < int(0.1 * fs):
                amplitudes.append(0.0)
                continue
            
            t_segment = signal[t_window_start:t_window_end, 0]
            amplitudes.append(np.max(t_segment) - np.min(t_segment))
        
        return np.array(amplitudes)
    
    def _estimate_st_deviation(self,
                              signal: np.ndarray,
                              annotations: pd.DataFrame,
                              fs: int) -> np.ndarray:
        """Estimate ST segment deviation."""
        deviations = []
        
        for _, beat in annotations.iterrows():
            sample_idx = int(beat['sample'])
            
            # ST segment window (typically 80ms after R-peak)
            st_window_start = sample_idx + int(0.08 * fs)
            st_window_end = min(len(signal), st_window_start + int(0.04 * fs))
            
            if st_window_end - st_window_start < int(0.02 * fs):
                deviations.append(0.0)
                continue
            
            st_segment = signal[st_window_start:st_window_end, 0]
            
            # Calculate deviation from baseline (simplified)
            baseline = np.mean(signal[max(0, sample_idx - int(0.1 * fs)):sample_idx, 0])
            st_level = np.mean(st_segment)
            deviation = st_level - baseline
            
            deviations.append(deviation)
        
        return np.array(deviations)
    
    def _estimate_qt_intervals(self,
                              signal: np.ndarray,
                              annotations: pd.DataFrame,
                              fs: int) -> np.ndarray:
        """Estimate QT intervals."""
        qt_intervals = []
        
        for i, (_, beat) in enumerate(annotations.iterrows()):
            sample_idx = int(beat['sample'])
            
            # Find next R-peak for QT measurement
            if i + 1 >= len(annotations):
                qt_intervals.append(0.4)  # Default QT interval
                continue
            
            next_sample_idx = int(annotations.iloc[i + 1]['sample'])
            
            # QT interval window
            qt_window_start = sample_idx
            qt_window_end = min(len(signal), sample_idx + int(0.6 * fs))
            
            if qt_window_end - qt_window_start < int(0.2 * fs):
                qt_intervals.append(0.4)
                continue
            
            # Simplified QT estimation (from R-peak to T-wave end)
            qt_segment = signal[qt_window_start:qt_window_end, 0]
            
            # Find T-wave end using signal derivative
            derivative = np.diff(qt_segment)
            t_end_idx = np.argmin(derivative[int(0.2 * fs):]) + int(0.2 * fs)
            
            qt_interval = t_end_idx / fs
            qt_intervals.append(qt_interval)
        
        return np.array(qt_intervals)
    
    def _correct_qt_intervals(self, qt_intervals: np.ndarray, rr_intervals: np.ndarray) -> np.ndarray:
        """Apply Bazett's formula for QT correction."""
        # Bazett's formula: QTc = QT / sqrt(RR)
        qtc_intervals = []
        
        for i, qt in enumerate(qt_intervals):
            if i < len(rr_intervals):
                rr = rr_intervals[i]
                if rr > 0:
                    qtc = qt / np.sqrt(rr)
                else:
                    qtc = qt
            else:
                qtc = qt
            
            qtc_intervals.append(qtc)
        
        return np.array(qtc_intervals)
    
    def train_rule_classifier(self,
                             features: pd.DataFrame,
                             labels: np.ndarray,
                             classifier_type: str = 'decision_tree',
                             **kwargs) -> Dict:
        """
        Train a rule-based classifier on extracted features.
        
        Args:
            features: Extracted ECG features
            labels: Arrhythmia labels
            classifier_type: Type of classifier ('decision_tree', 'random_forest', 'logistic')
            **kwargs: Additional arguments for classifier
            
        Returns:
            Dictionary with trained classifier and performance metrics
        """
        # Prepare data
        X = features.values
        y = labels
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        if classifier_type == 'decision_tree':
            classifier = DecisionTreeClassifier(
                max_depth=5, min_samples_split=10, min_samples_leaf=5,
                random_state=42, **kwargs
            )
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, random_state=42, **kwargs
            )
        elif classifier_type == 'logistic':
            classifier = LogisticRegression(
                max_iter=1000, random_state=42, **kwargs
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Train model
        classifier.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate rules
        rules = self._extract_rules(classifier, classifier_type, scaler)
        
        return {
            'classifier_type': classifier_type,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'rules': rules,
            'feature_importance': self._get_feature_importance(classifier, classifier_type)
        }
    
    def _extract_rules(self,
                      classifier,
                      classifier_type: str,
                      scaler: StandardScaler) -> List[str]:
        """
        Extract human-readable rules from trained classifier.
        
        Args:
            classifier: Trained classifier
            classifier_type: Type of classifier
            scaler: Feature scaler
            
        Returns:
            List of human-readable rules
        """
        rules = []
        
        if classifier_type == 'decision_tree':
            # Extract decision tree rules
            tree_rules = export_text(classifier, feature_names=self.feature_names)
            
            # Parse and format rules
            for line in tree_rules.split('\n'):
                if 'class:' in line:
                    # Extract class prediction
                    class_pred = line.split('class: ')[1].strip()
                    rules.append(f"-> Predict: {class_pred}")
                elif '|' in line and ('<=' in line or '>' in line):
                    # Extract condition
                    condition = line.strip()
                    # Convert scaled values back to original scale
                    condition = self._convert_scaled_condition(condition, scaler)
                    rules.append(condition)
        
        elif classifier_type == 'random_forest':
            # Extract rules from random forest
            feature_importance = classifier.feature_importances_
            top_features = np.argsort(feature_importance)[-5:]  # Top 5 features
            
            rules.append("Random Forest Feature Importance:")
            for i, feature_idx in enumerate(reversed(top_features)):
                importance = feature_importance[feature_idx]
                feature_name = self.feature_names[feature_idx]
                rules.append(f"{i+1}. {feature_name}: {importance:.3f}")
        
        elif classifier_type == 'logistic':
            # Extract logistic regression coefficients
            coefficients = classifier.coef_[0]
            intercept = classifier.intercept_[0]
            
            rules.append("Logistic Regression Coefficients:")
            for i, (coef, feature_name) in enumerate(zip(coefficients, self.feature_names)):
                if abs(coef) > 0.1:  # Only show significant coefficients
                    rules.append(f"{feature_name}: {coef:.3f}")
            
            rules.append(f"Intercept: {intercept:.3f}")
        
        return rules
    
    def _convert_scaled_condition(self, condition: str, scaler: StandardScaler) -> str:
        """
        Convert scaled condition back to original scale.
        
        Args:
            condition: Condition string from decision tree
            scaler: Feature scaler
            
        Returns:
            Converted condition string
        """
        # This is a simplified conversion - in practice, you'd need more sophisticated parsing
        return condition
    
    def _get_feature_importance(self, classifier, classifier_type: str) -> Dict[str, float]:
        """
        Get feature importance from classifier.
        
        Args:
            classifier: Trained classifier
            classifier_type: Type of classifier
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if classifier_type == 'decision_tree':
            importance = classifier.feature_importances_
        elif classifier_type == 'random_forest':
            importance = classifier.feature_importances_
        elif classifier_type == 'logistic':
            importance = np.abs(classifier.coef_[0])
        else:
            return {}
        
        return {feature_name: importance[i] for i, feature_name in enumerate(self.feature_names)}
    
    def generate_rule_report(self,
                           features: pd.DataFrame,
                           labels: np.ndarray,
                           save_dir: str = 'results/rules') -> Dict:
        """
        Generate comprehensive rule extraction report.
        
        Args:
            features: Extracted ECG features
            labels: Arrhythmia labels
            save_dir: Directory to save results
            
        Returns:
            Dictionary with rule extraction report
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Train multiple rule classifiers
        classifiers = ['decision_tree', 'random_forest', 'logistic']
        results = {}
        
        for classifier_type in classifiers:
            logger.info(f"Training {classifier_type} classifier...")
            results[classifier_type] = self.train_rule_classifier(
                features, labels, classifier_type
            )
        
        # Generate visualizations
        self._visualize_feature_importance(results, save_dir)
        self._visualize_rule_performance(results, save_dir)
        
        # Create summary report
        report = {
            'classifier_results': results,
            'summary': {
                'best_classifier': max(results.keys(), 
                                     key=lambda k: results[k]['accuracy']),
                'accuracies': {k: v['accuracy'] for k, v in results.items()},
                'num_rules': {k: len(v['rules']) for k, v in results.items()}
            }
        }
        
        # Save rules to text file
        with open(f"{save_dir}/extracted_rules.txt", 'w', encoding='utf-8') as f:
            f.write("ECG Arrhythmia Classification Rules\n")
            f.write("=" * 50 + "\n\n")
            
            for classifier_type, result in results.items():
                f.write(f"\n{classifier_type.upper()} CLASSIFIER\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {result['accuracy']:.3f}\n\n")
                
                for rule in result['rules']:
                    f.write(f"{rule}\n")
                f.write("\n")
        
        # Save report as JSON
        import json
        with open(f"{save_dir}/rule_report.json", 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_report = self._convert_numpy_to_lists(report)
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Rule extraction report saved to {save_dir}")
        return report
    
    def _visualize_feature_importance(self, results: Dict, save_dir: str):
        """Visualize feature importance across classifiers."""
        fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
        if len(results) == 1:
            axes = [axes]
        
        for i, (classifier_type, result) in enumerate(results.items()):
            importance = result['feature_importance']
            
            # Sort by importance
            sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, scores = zip(*sorted_items[:10])  # Top 10 features
            
            axes[i].barh(range(len(features)), scores)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_title(f'{classifier_type.replace("_", " ").title()}')
            axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_rule_performance(self, results: Dict, save_dir: str):
        """Visualize performance comparison across classifiers."""
        classifiers = list(results.keys())
        accuracies = [results[c]['accuracy'] for c in classifiers]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classifiers, accuracies)
        plt.title('Rule Classifier Performance Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _convert_numpy_to_lists(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        elif hasattr(obj, '__class__') and 'sklearn' in str(obj.__class__):
            # Handle sklearn objects by converting to string representation
            return str(obj.__class__.__name__)
        else:
            return obj


def main():
    """Example usage of the ECGRuleExtractor class."""
    print("ECG Rule Extractor module initialized")
    print("Use with extracted ECG features to generate human-readable rules")


if __name__ == "__main__":
    main()
