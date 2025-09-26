#!/usr/bin/env python3
"""
Improved ECG Arrhythmia Training Script

This script implements enhanced training strategies for ECG arrhythmia detection,
including data augmentation, improved training schedules, and cross-subject validation.
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
from datetime import datetime
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import MITBIHDataLoader
from preprocessing.signal_processing import ECGSignalProcessor
from preprocessing.data_augmentation import ECGAugmenter
from preprocessing.preprocessed_data_loader import load_preprocessed_data
from models.ecg_classifier import create_model, get_model_summary
from evaluation.metrics import calculate_metrics, generate_metrics_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ECGDataset(torch.utils.data.Dataset):
    """Custom dataset for ECG segments."""
    
    def __init__(self, segments, labels, subject_ids=None):
        self.segments = torch.FloatTensor(segments)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        if self.subject_ids is not None:
            return self.segments[idx], self.labels[idx], self.subject_ids[idx]
        return self.segments[idx], self.labels[idx]

class ImprovedECGTrainer:
    """
    Improved ECG trainer with enhanced training strategies.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the improved trainer.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default improved configuration
            self.config = {
                'data_path': 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                'save_dir': 'results/improved',
                'processed_data_dir': 'data/processed',
                'model_type': 'lightweight',
                'num_leads': 2,
                'input_size': 1080,
                'num_classes': 5,
                'sampling_rate': 360,
                'window_size': 3.0,
                'overlap': 0.5,
                'batch_size': 32,
                'learning_rate': 0.001,
                'max_epochs': 50,  # Increased epochs
                'patience': 15,    # Increased patience
                'device': 'auto',
                'augmentation': {
                    'window_augment': True,
                    'synthetic_vbeats': True,
                    'augmentation_prob': 0.5
                },
                'training': {
                    'scheduler': 'cosine',
                    'gradient_clip': 1.0,
                    'weight_decay': 1e-4,
                    'dropout_rate': 0.4
                },
                'validation': {
                    'cross_subject': True,
                    'loso_cv': True
                }
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
        
        logger.info(f"Improved trainer initialized with device: {self.device}")
        logger.info(f"Results will be saved to: {self.save_dir}")
    
    def prepare_data(self):
        """Prepare data with augmentation and cross-subject splits."""
        logger.info("=== Preparing Data with Augmentation ===")
        
        # Check if preprocessed data exists
        processed_data_dir = self.config.get('processed_data_dir', 'data/processed')
        logger.info(f"Checking for preprocessed data in: {processed_data_dir}")
        logger.info(f"Directory exists: {os.path.exists(processed_data_dir)}")
        
        if os.path.exists(processed_data_dir):
            try:
                logger.info(f"Loading preprocessed data from {processed_data_dir}")
                data_loader = load_preprocessed_data(processed_data_dir)
                if data_loader:
                    # Get the full dataset
                    all_segments, all_labels, all_subject_ids = data_loader.get_full_dataset()
                    
                    logger.info(f"Successfully loaded preprocessed data: {len(all_segments)} segments")
                    logger.info(f"Class distribution: {np.bincount(all_labels)}")
                    
                    self.all_segments = all_segments
                    self.all_labels = all_labels
                    self.all_subject_ids = all_subject_ids
                    
                    return all_segments, all_labels, all_subject_ids
                else:
                    logger.warning("Data loader returned None")
            except Exception as e:
                logger.warning(f"Failed to load preprocessed data: {e}")
                logger.info("Falling back to real-time processing...")
        else:
            logger.info(f"Preprocessed data directory not found: {processed_data_dir}")
            logger.info("Falling back to real-time processing...")
        
        # Load and preprocess data
        data_loader = MITBIHDataLoader(self.config['data_path'])
        signal_processor = ECGSignalProcessor(
            fs=self.config['sampling_rate'],
            low_freq=0.5,
            high_freq=40.0
        )
        
        # Process all records
        all_segments = []
        all_labels = []
        all_subject_ids = []
        
        for record in data_loader.records[:self.config.get('max_records', 47)]:
            try:
                record_data = data_loader.load_record(record)
                if record_data:
                    processed = signal_processor.preprocess_record(
                        signal_data=record_data['signals'],
                        annotations=record_data['beat_annotations'],
                        window_size=self.config['window_size'],
                        overlap=self.config['overlap']
                    )
                    
                    if processed and len(processed['segments']) > 0:
                        all_segments.extend(processed['segments'])
                        all_labels.extend(processed['labels'])
                        all_subject_ids.extend([record] * len(processed['segments']))
                        
            except Exception as e:
                logger.warning(f"Failed to process record {record}: {e}")
        
        all_segments = np.array(all_segments)
        all_labels = np.array(all_labels)
        all_subject_ids = np.array(all_subject_ids)
        
        # Convert string labels to numerical indices
        label_mapping = {
            'normal': 0, 'supraventricular': 1, 'ventricular': 2,
            'fusion': 3, 'paced': 4, 'unknown': 4
        }
        
        numerical_labels = np.array([label_mapping.get(label, 4) for label in all_labels])
        
        logger.info(f"Original dataset: {len(all_segments)} segments")
        logger.info(f"Class distribution: {np.bincount(numerical_labels)}")
        
        # Update labels to numerical values
        all_labels = numerical_labels
        
        # Apply data augmentation
        if self.config.get('augmentation'):
            augmenter = ECGAugmenter(fs=self.config['sampling_rate'])
            all_segments, all_labels = augmenter.augment_dataset(
                all_segments, all_labels, self.config['augmentation']
            )
        
        self.all_segments = all_segments
        self.all_labels = all_labels
        self.all_subject_ids = all_subject_ids
        
        return all_segments, all_labels, all_subject_ids
    
    def create_data_loaders(self, train_indices, val_indices, test_indices=None):
        """Create data loaders for training and validation."""
        
        # Split data
        train_segments = self.all_segments[train_indices]
        train_labels = self.all_labels[train_indices]
        train_subjects = self.all_subject_ids[train_indices]
        
        val_segments = self.all_segments[val_indices]
        val_labels = self.all_labels[val_indices]
        val_subjects = self.all_subject_ids[val_indices]
        
        # Calculate class weights for weighted sampling
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        
        # Create datasets
        train_dataset = ECGDataset(train_segments, train_labels, train_subjects)
        val_dataset = ECGDataset(val_segments, val_labels, val_subjects)
        
        # Create weighted sampler for training
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def create_model_and_optimizer(self):
        """Create model and optimizer with improved settings."""
        
        # Create model
        model = create_model(
            model_type=self.config['model_type'],
            num_leads=self.config['num_leads'],
            input_size=self.config['input_size'],
            num_classes=self.config['num_classes'],
            dropout_rate=self.config['training'].get('dropout_rate', 0.4)
        )
        
        model = model.to(self.device)
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 1e-4)
        )
        
        # Create scheduler
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config['max_epochs'],
                eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        else:
            scheduler = None
        
        # Create loss function with class weights
        class_counts = np.bincount(self.all_labels)
        class_weights = torch.FloatTensor(1.0 / class_counts).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        return model, optimizer, scheduler, criterion
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output['logits'], target)
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output['logits'].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, model, val_loader, criterion):
        """Validate the model."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output['logits'], target)
                
                total_loss += loss.item()
                pred = output['logits'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train_model(self, train_loader, val_loader):
        """Train the model with improved strategies."""
        logger.info("=== Starting Improved Training ===")
        
        model, optimizer, scheduler, criterion = self.create_model_and_optimizer()
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['max_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['max_epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Learning Rate: {current_lr:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config
                }, self.save_dir / 'best_model.pth')
                
                logger.info(f'New best model saved with val_loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logger.info(f'Early stopping after {patience_counter} epochs without improvement')
                    break
        
        # Load best model
        checkpoint = torch.load(self.save_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        final_val_loss, final_val_acc, final_preds, final_targets = self.validate(
            model, val_loader, criterion
        )
        
        # Calculate detailed metrics
        metrics = calculate_metrics(final_targets, final_preds)
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_metrics': metrics,
            'config': self.config
        }
        
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed. Best val_loss: {best_val_loss:.4f}")
        logger.info(f"Final metrics: {metrics}")
        
        return model, history
    
    def cross_subject_validation(self):
        """Perform cross-subject validation."""
        logger.info("=== Cross-Subject Validation ===")
        
        # Prepare data
        segments, labels, subject_ids = self.prepare_data()
        
        # Get unique subjects
        unique_subjects = np.unique(subject_ids)
        logger.info(f"Performing cross-subject validation on {len(unique_subjects)} subjects")
        
        # Initialize cross-validation
        logo = LeaveOneGroupOut()
        
        subject_results = []
        all_predictions = []
        all_targets = []
        
        for train_idx, test_idx in logo.split(segments, labels, subject_ids):
            test_subject = subject_ids[test_idx[0]]
            logger.info(f"Testing on subject: {test_subject}")
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(train_idx, test_idx)
            
            # Train model
            model, history = self.train_model(train_loader, val_loader)
            
            # Evaluate
            _, _, predictions, targets = self.validate(model, val_loader, None)
            
            # Calculate metrics
            metrics = calculate_metrics(targets, predictions)
            
            subject_results.append({
                'subject': test_subject,
                'metrics': metrics,
                'predictions': predictions,
                'targets': targets
            })
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            logger.info(f"Subject {test_subject} - Accuracy: {metrics['accuracy']:.4f}")
        
        # Overall results
        overall_metrics = calculate_metrics(all_targets, all_predictions)
        
        # Save results
        results = {
            'subject_results': subject_results,
            'overall_metrics': overall_metrics,
            'config': self.config
        }
        
        with open(self.save_dir / 'cross_subject_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Cross-subject validation completed.")
        logger.info(f"Overall accuracy: {overall_metrics['accuracy']:.4f}")
        
        return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Improved ECG Training')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--cross_subject', action='store_true', help='Run cross-subject validation')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ImprovedECGTrainer(args.config)
    
    if args.cross_subject:
        # Run cross-subject validation
        results = trainer.cross_subject_validation()
    else:
        # Run single training run
        segments, labels, subject_ids = trainer.prepare_data()
        
        # Simple train/val split
        n_samples = len(segments)
        train_size = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_loader, val_loader = trainer.create_data_loaders(train_indices, val_indices)
        model, history = trainer.train_model(train_loader, val_loader)

if __name__ == "__main__":
    main()

