#!/usr/bin/env python3
"""
Training Script with Preprocessed Data

This script trains the ECG arrhythmia classifier using preprocessed data
that was saved by the comprehensive preprocessing script.
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
        logging.FileHandler('training_preprocessed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PreprocessedECGTrainer:
    """
    ECG trainer that uses preprocessed data for fast training.
    """
    
    def __init__(self, config_path: str = None, data_dir: str = 'data/processed'):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
            data_dir: Directory containing preprocessed data
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'save_dir': 'results/preprocessed_training',
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
                    'use_weighted_sampling': True,  # Enable weighted sampling
                    'sampling_strategy': 'sqrt_inverse',  # softened sampling weights
                    'class_weight_strategy': 'sqrt_inverse',  # soften loss weights
                    'class_weight_max_ratio': 5.0  # cap extreme ratios
                }
            }
        
        # Resolve device configuration
        device_config = self.config.get('device', 'cpu')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # Ensure numeric config values are properly typed
        if 'training' in self.config:
            if 'weight_decay' in self.config['training']:
                self.config['training']['weight_decay'] = float(self.config['training']['weight_decay'])
            if 'gradient_clip' in self.config['training']:
                self.config['training']['gradient_clip'] = float(self.config['training']['gradient_clip'])
            if 'dropout_rate' in self.config['training']:
                self.config['training']['dropout_rate'] = float(self.config['training']['dropout_rate'])
            
            # Add fallback for weighted sampling if it fails
            if 'use_weighted_sampling' in self.config['training'] and self.config['training']['use_weighted_sampling']:
                # Check if we should enable fallback mode
                if self.config['training'].get('enable_fallback', False):
                    logger.info("Fallback mode enabled - will use regular shuffling if weighted sampling fails")
        
        # Ensure other numeric config values are properly typed
        if 'learning_rate' in self.config:
            self.config['learning_rate'] = float(self.config['learning_rate'])
        if 'batch_size' in self.config:
            self.config['batch_size'] = int(self.config['batch_size'])
        if 'max_epochs' in self.config:
            self.config['max_epochs'] = int(self.config['max_epochs'])
        if 'patience' in self.config:
            self.config['patience'] = int(self.config['patience'])
        if 'num_workers' in self.config:
            self.config['num_workers'] = int(self.config['num_workers'])
        
        # Load preprocessed data
        self.data_loader = load_preprocessed_data(data_dir)
        
        # Create output directory
        self.save_dir = Path(self.config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Results will be saved to: {self.save_dir}")
        
        # Print data summary
        self.data_loader.print_summary()
    
    def _calculate_sample_weights(self, labels: np.ndarray, strategy: str = 'inverse_frequency') -> np.ndarray:
        """
        Calculate sample weights for weighted sampling.
        
        Args:
            labels: Array of labels
            strategy: Weighting strategy ('inverse_frequency', 'sqrt_inverse', 'balanced')
            
        Returns:
            Array of sample weights
        """
        # Validate inputs
        if labels is None or len(labels) == 0:
            raise ValueError("Labels cannot be None or empty")
        
        if strategy not in ['inverse_frequency', 'sqrt_inverse', 'balanced']:
            raise ValueError(f"Unknown weighting strategy: {strategy}. Must be one of: inverse_frequency, sqrt_inverse, balanced")
        
        # Convert string labels to integers if needed
        if labels.dtype.kind in ['U', 'S', 'O']:  # String types
            if not hasattr(self.data_loader, 'class_mapping') or self.data_loader.class_mapping is None:
                raise ValueError("Class mapping not available in data loader")
            numeric_labels = np.array([self.data_loader.class_mapping.get(str(label), 4) for label in labels])
        else:
            numeric_labels = labels
        
        # Count occurrences of each class
        unique_classes, class_counts = np.unique(numeric_labels, return_counts=True)
        
        # Ensure we have at least one class
        if len(unique_classes) == 0:
            raise ValueError("No classes found in labels")
        
        if strategy == 'inverse_frequency':
            # Inverse frequency weighting
            class_weights = 1.0 / (class_counts + 1e-8)
        elif strategy == 'sqrt_inverse':
            # Square root of inverse frequency (less aggressive)
            class_weights = 1.0 / np.sqrt(class_counts + 1e-8)
        elif strategy == 'balanced':
            # Balanced weighting (equal weight for all classes)
            class_weights = np.ones_like(class_counts, dtype=float)
        
        # Create sample weights directly from class weights
        # Each sample gets the weight corresponding to its class
        sample_weights = class_weights[numeric_labels]
        
        # Validate output
        if np.any(sample_weights <= 0):
            raise ValueError("All sample weights must be positive")
        
        # For WeightedRandomSampler, weights should sum to 1.0 for proper probability distribution
        # Normalize the sample weights to sum to 1.0
        sample_weights = sample_weights / np.sum(sample_weights)
        
        # Debug information
        logger.debug(f"Strategy: {strategy}")
        logger.debug(f"Class counts: {class_counts}")
        logger.debug(f"Class weights: {class_weights}")
        logger.debug(f"Sample weights sum: {np.sum(sample_weights):.6f}")
        logger.debug(f"Sample weights min: {np.min(sample_weights):.6f}, max: {np.max(sample_weights):.6f}")
        
        return sample_weights
    
    def _validate_training_config(self):
        """Validate the training configuration."""
        logger.info("Validating training configuration...")
        
        # Check required fields
        required_fields = ['batch_size', 'learning_rate', 'max_epochs', 'patience']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Check training-specific fields
        if 'training' in self.config:
            training_config = self.config['training']
            
            # Check weighted sampling configuration
            if training_config.get('use_weighted_sampling', False):
                if 'sampling_strategy' not in training_config:
                    logger.warning("use_weighted_sampling is True but sampling_strategy not specified. Using 'inverse_frequency'")
                    training_config['sampling_strategy'] = 'inverse_frequency'
                
                strategy = training_config['sampling_strategy']
                if strategy not in ['inverse_frequency', 'sqrt_inverse', 'balanced']:
                    raise ValueError(f"Invalid sampling_strategy: {strategy}. Must be one of: inverse_frequency, sqrt_inverse, balanced")
        
        # Validate numeric values
        if self.config['batch_size'] <= 0:
            raise ValueError(f"batch_size must be positive, got: {self.config['batch_size']}")
        if self.config['learning_rate'] <= 0:
            raise ValueError(f"learning_rate must be positive, got: {self.config['learning_rate']}")
        if self.config['max_epochs'] <= 0:
            raise ValueError(f"max_epochs must be positive, got: {self.config['max_epochs']}")
        if self.config['patience'] <= 0:
            raise ValueError(f"patience must be positive, got: {self.config['patience']}")
        
        logger.info("Configuration validation passed!")
    
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
        
        # Create loss function with softened class weights
        class_weights = self.data_loader.get_class_weights().to(self.device)

        # Apply configurable softening/capping of weights to reduce overfocus on rare classes
        cw_strategy = self.config['training'].get('class_weight_strategy', 'sqrt_inverse')
        if cw_strategy == 'sqrt_inverse':
            class_weights = torch.pow(class_weights, 0.5)
        elif cw_strategy == 'balanced':
            class_weights = torch.ones_like(class_weights)
        # Normalize to mean 1.0 for stability
        class_weights = class_weights / class_weights.mean().clamp(min=1e-8)
        # Cap maximum ratio between classes
        max_ratio = float(self.config['training'].get('class_weight_max_ratio', 5.0))
        class_weights = torch.clamp(class_weights, max=max_ratio)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        return model, optimizer, scheduler, criterion
    
    def create_data_loaders_with_sampling(self, batch_size: int, split_type: str = 'splits'):
        """
        Create data loaders with optional weighted sampling.
        
        Args:
            batch_size: Batch size for data loaders
            split_type: 'splits' for train/val/test or 'full' for full dataset
            
        Returns:
            Dictionary of data loaders
        """
        try:
            datasets = self.data_loader.create_datasets(split_type)
            loaders = {}
            
            for split_name, dataset in datasets.items():
                if split_name == 'train' and self.config['training'].get('use_weighted_sampling', True):
                    try:
                        # Use weighted sampling for training
                        strategy = self.config['training'].get('sampling_strategy', 'inverse_frequency')
                        
                        # Get labels for weight calculation
                        if hasattr(dataset, 'labels'):
                            labels = dataset.labels.numpy()
                            logger.debug(f"Using dataset.labels attribute, shape: {labels.shape}")
                        else:
                            # Extract labels from dataset
                            logger.debug("Extracting labels from dataset items")
                            labels = []
                            for i in range(min(len(dataset), 1000)):  # Sample first 1000 to avoid memory issues
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
                            logger.debug(f"Extracted labels shape: {labels.shape}")
                        
                        # Validate labels
                        if len(labels) == 0:
                            raise ValueError(f"No labels found in {split_name} dataset")
                        
                        # Debug label information
                        unique_labels, label_counts = np.unique(labels, return_counts=True)
                        logger.info(f"Labels for {split_name}: {dict(zip(unique_labels, label_counts))}")
                        
                        sample_weights = self._calculate_sample_weights(labels, strategy)
                        
                        # Validate sample weights
                        if np.any(np.isnan(sample_weights)) or np.any(np.isinf(sample_weights)):
                            raise ValueError(f"Invalid sample weights detected: NaN or Inf values")
                        
                        # Create weighted sampler
                        sampler = WeightedRandomSampler(
                            weights=sample_weights,
                            num_samples=len(sample_weights),
                            replacement=True
                        )
                        
                        loaders[split_name] = DataLoader(
                            dataset,
                            batch_size=batch_size,
                            sampler=sampler,  # Use sampler instead of shuffle
                            num_workers=self.config.get('num_workers', 4),
                            pin_memory=True if self.device.type == 'cuda' else False
                        )
                        
                        logger.info(f"Created {split_name} loader with weighted sampling (strategy: {strategy})")
                        
                    except Exception as sampling_error:
                        logger.warning(f"Weighted sampling failed for {split_name}: {sampling_error}")
                        logger.warning("Falling back to regular shuffling")
                        
                        # Fallback to regular data loader
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
                    shuffle = split_name == 'train'  # Only shuffle training if not using weighted sampling
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
        
        # Pre-allocate tensors for efficiency
        batch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Efficient batch unpacking
            if len(batch) == 3:  # segments, labels, subject_ids
                data, target, _ = batch
            else:  # segments, labels
                data, target = batch
            
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
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
            
            # Efficient metric calculation
            total_loss += loss.item()
            pred = output['logits'].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Logging optimization - reduce string formatting overhead
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
                # Efficient batch unpacking
                if len(batch) == 3:  # segments, labels, subject_ids
                    data, target, _ = batch
                else:  # segments, labels
                    data, target = batch
                
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = model(data)
                loss = criterion(output['logits'], target)
                
                total_loss += loss.item()
                pred = output['logits'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Efficient prediction collection
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        # Concatenate predictions efficiently
        all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train_model(self):
        """Train the model using preprocessed data."""
        logger.info("=== Starting Training with Preprocessed Data ===")
        
        # Validate configuration before starting
        self._validate_training_config()
        
        # Create data loaders with weighted sampling
        data_loaders = self.create_data_loaders_with_sampling(
            batch_size=self.config['batch_size'],
            split_type='splits'
        )
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        logger.info(f"Training on {len(train_loader.dataset)} samples")
        logger.info(f"Validating on {len(val_loader.dataset)} samples")
        
        # Create model and optimizer
        model, optimizer, scheduler, criterion = self.create_model_and_optimizer()
        
        # Print model summary
        logger.info(get_model_summary(model))
        
        # Training history - use lists for efficiency
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
    
    def cross_subject_validation(self):
        """Perform cross-subject validation using preprocessed data."""
        logger.info("=== Cross-Subject Validation with Preprocessed Data ===")
        
        # Get full dataset
        segments, labels, subject_ids = self.data_loader.get_full_dataset()
        
        # Get unique subjects
        unique_subjects = np.unique(subject_ids)
        logger.info(f"Performing cross-subject validation on {len(unique_subjects)} subjects")
        
        # Initialize cross-validation
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        
        subject_results = []
        all_predictions = []
        all_targets = []
        
        for train_idx, test_idx in logo.split(segments, labels, subject_ids):
            test_subject = subject_ids[test_idx[0]]
            logger.info(f"Testing on subject: {test_subject}")
            
            # Create data loaders for this split
            train_segments = segments[train_idx]
            train_labels = labels[train_idx]
            train_subjects = subject_ids[train_idx]
            
            test_segments = segments[test_idx]
            test_labels = labels[test_idx]
            test_subjects = subject_ids[test_idx]
            
            # Create datasets
            from preprocessing.preprocessed_data_loader import PreprocessedECGDataset
            train_dataset = PreprocessedECGDataset(train_segments, train_labels, train_subjects)
            test_dataset = PreprocessedECGDataset(test_segments, test_labels, test_subjects)
            
            # Create data loaders with weighted sampling for training
            if self.config['training'].get('use_weighted_sampling', True):
                strategy = self.config['training'].get('sampling_strategy', 'inverse_frequency')
                train_sample_weights = self._calculate_sample_weights(train_labels, strategy)
                train_sampler = WeightedRandomSampler(
                    weights=train_sample_weights,
                    num_samples=len(train_sample_weights),
                    replacement=True
                )
                train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                                        sampler=train_sampler, num_workers=self.config.get('num_workers', 4))
            else:
                train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                                        shuffle=True, num_workers=self.config.get('num_workers', 4))
            
            test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], 
                                   shuffle=False, num_workers=self.config.get('num_workers', 4))
            
            # Train model
            model, optimizer, scheduler, criterion = self.create_model_and_optimizer()
            
            # Quick training (fewer epochs for cross-validation)
            best_val_loss = float('inf')
            for epoch in range(10):  # Reduced epochs for cross-validation
                # Train
                train_loss, _ = self.train_epoch(model, train_loader, optimizer, criterion)
                
                # Validate
                val_loss, val_acc, val_preds, val_targets = self.validate(model, test_loader, criterion)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_preds = val_preds
                    best_targets = val_targets
            
            # Calculate metrics
            metrics = calculate_metrics(best_targets, best_preds)
            
            subject_results.append({
                'subject': test_subject,
                'metrics': metrics,
                'predictions': best_preds,
                'targets': best_targets
            })
            
            all_predictions.extend(best_preds)
            all_targets.extend(best_targets)
            
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
    parser = argparse.ArgumentParser(description='Training with Preprocessed Data')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Preprocessed data directory')
    parser.add_argument('--cross_subject', action='store_true', help='Run cross-subject validation')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = PreprocessedECGTrainer(args.config, args.data_dir)
        
        if args.cross_subject:
            # Run cross-subject validation
            logger.info("Running cross-subject validation...")
            results = trainer.cross_subject_validation()
            logger.info("Cross-subject validation completed successfully!")
        else:
            # Run single training run
            logger.info("Starting training...")
            model, history = trainer.train_model()
            logger.info("Training completed successfully!")
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure the data directory and configuration files exist.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your configuration file.")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        logger.error("Please check your data and model configuration.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error("Please check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

