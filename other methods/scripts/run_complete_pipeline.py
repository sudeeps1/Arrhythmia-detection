#!/usr/bin/env python3
"""
Complete ECG Arrhythmia Analysis Pipeline

This script demonstrates the complete workflow for interpretable ECG arrhythmia
classification, including data preprocessing, model training, interpretability
analysis, and rule extraction.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import MITBIHDataLoader
from preprocessing.signal_processing import ECGSignalProcessor
from models.ecg_classifier import create_model, get_model_summary
from interpretability.feature_attribution import ECGFeatureAttribution
from interpretability.rule_extraction import ECGRuleExtractor
from evaluation.metrics import calculate_metrics, generate_metrics_report
from evaluation.visualization import plot_training_history, plot_confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteECGPipeline:
    """
    Complete ECG arrhythmia analysis pipeline.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the complete pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'data_path': 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                'save_dir': 'results',
                'model_type': 'lightweight',
                'num_leads': 2,
                'input_size': 1080,
                'num_classes': 5,
                'sampling_rate': 360,
                'window_size': 3.0,
                'overlap': 0.5,
                'batch_size': 32,
                'learning_rate': 0.001,
                'max_epochs': 5,  # Reduced for testing
                'patience': 5,
                'device': 'auto'
            }
        
        # Resolve device configuration
        device_config = self.config.get('device', 'cpu')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # Create output directory
        self.save_dir = Path(self.config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized with device: {self.device}")
        logger.info(f"Results will be saved to: {self.save_dir}")
    
    def run_data_preprocessing(self):
        """Run complete data preprocessing pipeline."""
        logger.info("=== Starting Data Preprocessing ===")
        
        # Initialize data loader and processor
        self.data_loader = MITBIHDataLoader(self.config['data_path'])
        self.signal_processor = ECGSignalProcessor(
            fs=self.config['sampling_rate'],
            low_freq=0.5,
            high_freq=40.0
        )
        
        logger.info(f"Found {len(self.data_loader.records)} records")
        
        # Process a subset of records for demo
        max_records = min(5, len(self.data_loader.records))  # Process first 5 records
        records_to_process = self.data_loader.records[:max_records]
        
        all_segments = []
        all_labels = []
        all_metadata = []
        
        for record_name in records_to_process:
            logger.info(f"Processing record {record_name}...")
            
            try:
                # Load record
                record_data = self.data_loader.load_record(record_name)
                if record_data is None:
                    continue
                
                # Preprocess record
                preprocessed = self.signal_processor.preprocess_record(
                    signal_data=record_data['signals'],
                    annotations=record_data['beat_annotations'],
                    window_size=self.config['window_size'],
                    overlap=self.config['overlap'],
                    normalize_method='zscore'
                )
                
                if preprocessed['segments'].size == 0:
                    continue
                
                # Add record information to metadata
                for i, metadata in enumerate(preprocessed['metadata']):
                    metadata['record_name'] = record_name
                    metadata['segment_idx'] = i
                
                all_segments.append(preprocessed['segments'])
                all_labels.append(preprocessed['labels'])
                all_metadata.extend(preprocessed['metadata'])
                
                logger.info(f"  Processed {len(preprocessed['segments'])} segments")
                
            except Exception as e:
                logger.error(f"Error processing record {record_name}: {str(e)}")
                continue
        
        # Combine all data
        if not all_segments:
            raise ValueError("No valid segments found in any records")
        
        self.segments = np.concatenate(all_segments, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        
        # Convert labels to numerical indices
        label_mapping = {
            'normal': 0, 'supraventricular': 1, 'ventricular': 2,
            'fusion': 3, 'paced': 4, 'unknown': 4
        }
        
        self.numerical_labels = np.array([label_mapping.get(label, 4) for label in self.labels])
        
        logger.info(f"Preprocessing completed: {len(self.segments)} segments, {len(np.unique(self.numerical_labels))} classes")
        logger.info(f"Class distribution: {np.bincount(self.numerical_labels)}")
        
        return True
    
    def run_model_training(self):
        """Run model training pipeline."""
        logger.info("=== Starting Model Training ===")
        
        # Create model
        self.model = create_model(
            model_type=self.config['model_type'],
            num_leads=self.config['num_leads'],
            input_size=self.config['input_size'],
            num_classes=self.config['num_classes']
        )
        
        self.model.to(self.device)
        logger.info(f"Model created: {get_model_summary(self.model)}")
        
        # Prepare data for training
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.segments, self.numerical_labels, 
            test_size=0.2, random_state=42, stratify=self.numerical_labels
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # Training setup
        import torch.nn as nn
        import torch.optim as optim
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Training loop
        self.model.train()
        train_losses = []
        test_accuracies = []
        
        for epoch in range(self.config['max_epochs']):
            # Training
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs['logits'], target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Evaluation
            self.model.eval()
            correct = 0
            total = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.model(data)
                    pred = outputs['logits'].argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            accuracy = correct / total
            train_losses.append(epoch_loss / len(train_loader))
            test_accuracies.append(accuracy)
            
            logger.info(f"Epoch {epoch+1}/{self.config['max_epochs']}: "
                       f"Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save training history
        self.training_history = {
            'train_loss': train_losses,
            'test_accuracy': test_accuracies
        }
        
        # Calculate final metrics
        self.final_metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions)
        )
        
        logger.info(f"Training completed! Final accuracy: {self.final_metrics['accuracy']:.4f}")
        
        return True
    
    def run_interpretability_analysis(self):
        """Run interpretability analysis."""
        logger.info("=== Starting Interpretability Analysis ===")
        
        # Initialize interpretability modules
        self.feature_attribution = ECGFeatureAttribution(self.model, device=str(self.device))
        self.rule_extractor = ECGRuleExtractor(self.model, device=str(self.device))
        
        # Get a sample of test data for analysis
        sample_data = torch.FloatTensor(self.segments[:10]).to(self.device)
        
        # Feature attribution analysis
        logger.info("Computing feature attributions...")
        attributions = self.feature_attribution.compute_temporal_importance(
            sample_data, method='integrated_gradients'
        )
        
        # Lead importance analysis
        lead_importance = self.feature_attribution.compute_lead_importance(
            sample_data, method='integrated_gradients'
        )
        
        # Critical segments analysis
        critical_segments = self.feature_attribution.compute_critical_segments(
            sample_data, threshold_percentile=90.0
        )
        
        # Ablation studies
        logger.info("Performing ablation studies...")
        ablation_results = {
            'temporal': self.feature_attribution.ablation_study(sample_data, 'temporal'),
            'lead': self.feature_attribution.ablation_study(sample_data, 'lead')
        }
        
        # Save interpretability results
        interpretability_results = {
            'attributions': attributions,
            'lead_importance': lead_importance,
            'critical_segments': critical_segments,
            'ablation_results': ablation_results
        }
        
        with open(self.save_dir / 'interpretability_results.json', 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = self._convert_tensors_to_lists(interpretability_results)
            json.dump(json_results, f, indent=2)
        
        logger.info("Interpretability analysis completed!")
        
        return True
    
    def run_rule_extraction(self):
        """Run neuro-symbolic rule extraction."""
        logger.info("=== Starting Rule Extraction ===")
        
        # Extract ECG features for rule generation
        logger.info("Extracting ECG features...")
        
        # Use a subset of data for rule extraction
        sample_indices = np.random.choice(len(self.segments), min(100, len(self.segments)), replace=False)
        sample_segments = self.segments[sample_indices]
        sample_labels = self.numerical_labels[sample_indices]
        
        # Extract features (simplified for demo)
        features = []
        for segment in sample_segments:
            # Extract basic features
            feature_dict = {
                'mean_amplitude': np.mean(segment),
                'std_amplitude': np.std(segment),
                'max_amplitude': np.max(segment),
                'min_amplitude': np.min(segment),
                'signal_energy': np.sum(segment ** 2),
                'zero_crossings': np.sum(np.diff(np.sign(segment)) != 0)
            }
            features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        
        # Generate rule report
        logger.info("Generating rule extraction report...")
        rule_report = self.rule_extractor.generate_rule_report(
            features_df, sample_labels, str(self.save_dir / 'rules')
        )
        
        logger.info("Rule extraction completed!")
        
        return True
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        logger.info("=== Generating Final Report ===")
        
        report = {
            'pipeline_summary': {
                'timestamp': datetime.now().isoformat(),
                'device_used': str(self.device),
                'config_used': self.config
            },
            'data_summary': {
                'total_segments': len(self.segments),
                'num_classes': len(np.unique(self.numerical_labels)),
                'class_distribution': np.bincount(self.numerical_labels).tolist()
            },
            'training_summary': {
                'final_accuracy': self.final_metrics['accuracy'],
                'final_f1_score': self.final_metrics['f1_score'],
                'training_epochs': len(self.training_history['train_loss']),
                'best_accuracy': max(self.training_history['test_accuracy'])
            },
            'model_summary': {
                'model_type': self.config['model_type'],
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        # Save final report
        with open(self.save_dir / 'final_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate metrics report
        metrics_report = generate_metrics_report(self.final_metrics)
        with open(self.save_dir / 'metrics_report.txt', 'w') as f:
            f.write(metrics_report)
        
        logger.info("Final report generated!")
        logger.info(f"Results saved to: {self.save_dir}")
        
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
    
    def run_complete_pipeline(self):
        """Run the complete pipeline."""
        logger.info("=== Starting Complete ECG Arrhythmia Analysis Pipeline ===")
        
        try:
            # Step 1: Data preprocessing
            self.run_data_preprocessing()
            
            # Step 2: Model training
            self.run_model_training()
            
            # Step 3: Interpretability analysis
            self.run_interpretability_analysis()
            
            # Step 4: Rule extraction
            self.run_rule_extraction()
            
            # Step 5: Generate final report
            final_report = self.generate_final_report()
            
            logger.info("=== Pipeline Completed Successfully! ===")
            logger.info(f"Final accuracy: {self.final_metrics['accuracy']:.4f}")
            logger.info(f"Results saved to: {self.save_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='Run complete ECG arrhythmia analysis pipeline')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, 
                       default='mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                       help='Path to MIT-BIH database')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = CompleteECGPipeline(args.config)
    
    # Update config with command line arguments
    pipeline.config['data_path'] = args.data_path
    pipeline.config['save_dir'] = args.save_dir
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved to: {pipeline.save_dir}")
        print("\n📊 Key Results:")
        print(f"   - Final Accuracy: {pipeline.final_metrics['accuracy']:.4f}")
        print(f"   - F1 Score: {pipeline.final_metrics['f1_score']:.4f}")
        print(f"   - Total Segments: {len(pipeline.segments)}")
        print(f"   - Classes: {len(np.unique(pipeline.numerical_labels))}")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")

if __name__ == "__main__":
    main()
