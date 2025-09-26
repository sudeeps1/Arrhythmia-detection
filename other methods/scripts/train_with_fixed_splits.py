#!/usr/bin/env python3
"""
Training Script with Fixed Data Splits

This script trains the ECG arrhythmia classifier using the fixed data splits
that have proper class balance in the validation set.
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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocessed_data_loader import load_preprocessed_data
from models.ecg_classifier import create_model, get_model_summary
from evaluation.metrics import calculate_metrics, generate_metrics_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_fixed_splits.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FixedSplitsECGTrainer:
    """
    ECG trainer that uses fixed data splits with proper class balance.
    """
    
    def __init__(self, config_path: str = None, data_dir: str = 'data/processed', data_file: str = None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
            data_dir: Directory containing preprocessed data
            data_file: Specific data file to load (None for auto-detection)
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'save_dir': 'results/fixed_splits_training',
                'model_type': 'lightweight',
                'num_leads': 2,
                'input_size': 1080,
                'num_classes': 5,
                'batch_size': 32,
                'learning_rate': 0.001,
                'max_epochs': 50,
                'patience': 15,
                'device': 'auto',
                'training': {
                    'scheduler': 'cosine',
                    'gradient_clip': 1.0,
                    'weight_decay': 1e-4,
                    'dropout_rate': 0.4,
                    'use_weighted_sampling': True,
                    'sampling_strategy': 'inverse_frequency'
                }
            }
        
        # Resolve device configuration
        device_config = self.config.get('device', 'cpu')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # Load preprocessed data with fixed splits
        if data_file:
            self.data_loader = load_preprocessed_data(data_dir, data_file)
        else:
            # Try to find the fixed splits file
            fixed_file = os.path.join(data_dir, 'preprocessed_data_fixed_splits.pkl')
            if os.path.exists(fixed_file):
                self.data_loader = load_preprocessed_data(data_dir, 'preprocessed_data_fixed_splits.pkl')
                logger.info("Using fixed data splits file")
            else:
                self.data_loader = load_preprocessed_data(data_dir)
                logger.warning("Fixed splits file not found, using original data")
        
        # Create output directory
        self.save_dir = Path(self.config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Results will be saved to: {self.save_dir}")
        
        # Print data summary
        self.data_loader.print_summary()
    
    def _calculate_sample_weights(self, labels: np.ndarray, strategy: str = 'inverse_frequency') -> np.ndarray:
        """Calculate sample weights for weighted sampling."""
        if labels is None or len(labels) == 0:
            raise ValueError("Labels cannot be None or empty")
        
        if strategy not in ['inverse_frequency', 'sqrt_inverse', 'balanced']:
            raise ValueError(f"Unknown weighting strategy: {strategy}")
        
        # Convert string labels to integers if needed
        if labels.dtype.kind in ['U', 'S', 'O']:  # String types
            if not hasattr(self.data_loader, 'class_mapping') or self.data_loader.class_mapping is None:
                raise ValueError("Class mapping not available in data loader")
            numeric_labels = np.array([self.data_loader.class_mapping.get(str(label), 4) for label in labels])
        else:
            numeric_labels = labels
        
        # Count occurrences of each class
        unique_classes, class_counts = np.unique(numeric_labels, return_counts=True)
        
        if len(unique_classes) == 0:
            raise ValueError("No classes found in labels")
        
        if strategy == 'inverse_frequency':
            class_weights = 1.0 / (class_counts + 1e-8)
        elif strategy == 'sqrt_inverse':
            class_weights = 1.0 / np.sqrt(class_counts + 1e-8)
        elif strategy == 'balanced':
            class_weights = np.ones_like(class_counts, dtype=float)
        
        # Create sample weights
        sample_weights = class_weights[numeric_labels]
        
        # Validate output
        if np.any(sample_weights <= 0):
            raise ValueError("All sample weights must be positive")
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights)
        
        logger.debug(f"Strategy: {strategy}")
        logger.debug(f"Class counts: {class_counts}")
        logger.debug(f"Class weights: {class_weights}")
        logger.debug(f"Sample weights sum: {np.sum(sample_weights):.6f}")
        
        return sample_weights
    
    def create_model_and_optimizer(self):
        """Create model and optimizer."""
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
            weight_decay=float(self.config['training'].get('weight_decay', 1e-4))
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
        class_weights = self.data_loader.get_class_weights().to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        return model, optimizer, scheduler, criterion
    
    def create_data_loaders(self, batch_size: int):
        """Create data loaders with proper data shape handling."""
        try:
            datasets = self.data_loader.create_datasets('splits')
            loaders = {}
            
            for split_name, dataset in datasets.items():
                if split_name == 'train' and self.config['training'].get('use_weighted_sampling', True):
                    try:
                        # Use weighted sampling for training
                        strategy = self.config['training'].get('sampling_strategy', 'inverse_frequency')
                        
                        # Get labels for weight calculation
                        if hasattr(dataset, 'labels'):
                            labels = dataset.labels.numpy()
                        else:
                            # Extract labels from dataset
                            labels = []
                            for i in range(min(len(dataset), 1000)):
                                try:
                                    item = dataset[i]
                                    if len(item) >= 2:
                                        labels.append(item[1])
                                    else:
                                        logger.warning(f"Dataset item {i} has unexpected structure: {len(item)} elements")
                                except Exception as e:
                                    logger.warning(f"Error accessing dataset item {i}: {e}")
                            
                            if len(labels) == 0:
                                raise ValueError("Could not extract any labels from dataset")
                            
                            labels = np.array(labels)
                        
                        sample_weights = self._calculate_sample_weights(labels, strategy)
                        
                        # Create weighted sampler
                        sampler = WeightedRandomSampler(
                            weights=sample_weights,
                            num_samples=len(sample_weights),
                            replacement=True
                        )
                        
                        loaders[split_name] = DataLoader(
                            dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=self.config.get('num_workers', 4),
                            pin_memory=True if self.device.type == 'cuda' else False
                        )
                        
                        logger.info(f"Created {split_name} loader with weighted sampling (strategy: {strategy})")
                        
                    except Exception as sampling_error:
                        logger.warning(f"Weighted sampling failed for {split_name}: {sampling_error}")
                        logger.warning("Falling back to regular shuffling")
                        
                        loaders[split_name] = DataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=self.config.get('num_workers', 4),
                            pin_memory=True if self.device.type == 'cuda' else False
                        )
                        
                        logger.info(f"Created {split_name} loader with regular shuffling (fallback)")
                        
                else:
                    # Regular data loader for validation/test
                    shuffle = split_name == 'train'
                    loaders[split_name] = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=self.config.get('num_workers', 4),
                        pin_memory=True if self.device.type == 'cuda' else False
                    )
            
            return loaders
            
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            raise RuntimeError(f"Failed to create data loaders: {e}")
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if len(batch) == 3:  # segments, labels, subject_ids
                data, target, _ = batch
            else:  # segments, labels
                data, target = batch
            
            # Fix data shape: (batch, seq_len, num_leads) -> (batch, num_leads, seq_len)
            if data.dim() == 3 and data.size(1) == 1080 and data.size(2) == 2:
                data = data.transpose(1, 2)  # (batch, 2, 1080)
            
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
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
            
            # Calculate metrics
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
            for batch in val_loader:
                # Handle different batch formats
                if len(batch) == 3:  # segments, labels, subject_ids
                    data, target, _ = batch
                else:  # segments, labels
                    data, target = batch
                
                # Fix data shape: (batch, seq_len, num_leads) -> (batch, num_leads, seq_len)
                if data.dim() == 3 and data.size(1) == 1080 and data.size(2) == 2:
                    data = data.transpose(1, 2)  # (batch, 2, 1080)
                
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = model(data)
                loss = criterion(output['logits'], target)
                
                total_loss += loss.item()
                pred = output['logits'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        # Concatenate predictions
        all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train_model(self):
        """Train the model using fixed data splits."""
        logger.info("=== Starting Training with Fixed Data Splits ===")
        
        # Create data loaders
        data_loaders = self.create_data_loaders(batch_size=self.config['batch_size'])
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        logger.info(f"Training on {len(train_loader.dataset)} samples")
        logger.info(f"Validating on {len(val_loader.dataset)} samples")
        
        # Create model and optimizer
        model, optimizer, scheduler, criterion = self.create_model_and_optimizer()
        
        # Print model summary
        logger.info(get_model_summary(model))
        
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
        checkpoint = torch.load(self.save_dir / 'best_model.pth', map_location=self.device)
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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Training with Fixed Data Splits')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Preprocessed data directory')
    parser.add_argument('--data_file', type=str, default='preprocessed_data_fixed_splits.pkl', help='Specific data file to use')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        logger.info("Initializing trainer with fixed data splits...")
        trainer = FixedSplitsECGTrainer(args.config, args.data_dir, args.data_file)
        
        # Start training
        logger.info("Starting training...")
        model, history = trainer.train_model()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

