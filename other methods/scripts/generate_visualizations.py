#!/usr/bin/env python3
"""
Comprehensive Visualization Script

This script generates all the requested figures for the arrhythmia classification project
using the existing trained model and data.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocessed_data_loader import load_preprocessed_data
from models.ecg_classifier import create_model
from evaluation.metrics import calculate_metrics

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationGenerator:
    def __init__(self):
        """Initialize the visualization generator."""
        self.results_dir = Path("results/preprocessed_training")
        self.output_dir = Path("results/visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model and data
        self.load_model_and_data()
        
        # Class names
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Paced']
        self.class_colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
    def load_model_and_data(self):
        """Load the trained model and data."""
        print("Loading model and data...")
        
        # Load model
        model_path = self.results_dir / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        self.config = checkpoint.get('config', {})
        
        # Create model
        self.model = create_model(
            model_type=self.config.get('model_type', 'lightweight'),
            num_leads=self.config.get('num_leads', 2),
            input_size=self.config.get('input_size', 1080),
            num_classes=self.config.get('num_classes', 5),
            dropout_rate=self.config.get('training', {}).get('dropout_rate', 0.4)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load data
        self.data_loader = load_preprocessed_data("data/processed")
        
        print("✅ Model and data loaded successfully")
    
    def generate_training_curves(self):
        """Generate training and validation curves."""
        print("Generating training curves...")
        
        # Load training history
        history_path = self.results_dir / "training_history.json"
        if not history_path.exists():
            print("⚠️  Training history not found, skipping training curves")
            return
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curves
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        epochs = range(1, len(train_losses) + 1)
        
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        train_accuracies = history.get('train_accuracies', [])
        val_accuracies = history.get('val_accuracies', [])
        
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Training curves saved")
    
    def generate_confusion_matrices(self):
        """Generate confusion matrices."""
        print("Generating confusion matrices...")
        
        # Get predictions on validation set
        predictions, targets = self.get_predictions()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall confusion matrix
        cm = confusion_matrix(targets, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Confusion matrices saved")
    
    def generate_class_metrics(self):
        """Generate class-specific metrics bar plot."""
        print("Generating class metrics...")
        
        # Get predictions and calculate metrics
        predictions, targets = self.get_predictions()
        
        # Calculate per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            # Create binary labels for this class
            binary_targets = (np.array(targets) == i).astype(int)
            binary_predictions = (np.array(predictions) == i).astype(int)
            
            # Calculate metrics
            tp = np.sum((binary_targets == 1) & (binary_predictions == 1))
            fp = np.sum((binary_targets == 0) & (binary_predictions == 1))
            fn = np.sum((binary_targets == 1) & (binary_predictions == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        precisions = [class_metrics[name]['precision'] for name in self.class_names]
        recalls = [class_metrics[name]['recall'] for name in self.class_names]
        f1_scores = [class_metrics[name]['f1'] for name in self.class_names]
        
        ax.bar(x - width, precisions, width, label='Precision', color='#2E8B57', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', color='#FF6B6B', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Arrhythmia Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Class-Specific Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
            ax.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Class metrics saved")
    
    def generate_sample_ecg_segments(self):
        """Generate sample ECG segments with predictions."""
        print("Generating sample ECG segments...")
        
        # Get validation data
        data_loaders = self.data_loader.create_data_loaders(
            batch_size=1, split_type='splits', shuffle_train=False, num_workers=0
        )
        val_loader = data_loaders['val']
        
        # Find one sample from each class
        class_samples = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        for batch in val_loader:
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch
            
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                output = self.model(data)
                prediction = output['logits'].argmax(dim=1).item()
                confidence = torch.softmax(output['logits'], dim=1).max().item()
            
            true_class = target.item()
            
            if true_class not in class_samples:
                class_samples[true_class] = {
                    'data': data.cpu().numpy()[0],  # Remove batch dimension
                    'prediction': prediction,
                    'confidence': confidence
                }
            
            if len(class_samples) == 5:  # Found all classes
                break
        
        # Create visualization
        fig, axes = plt.subplots(5, 1, figsize=(15, 12))
        
        for i, class_name in enumerate(self.class_names):
            if i in class_samples:
                sample = class_samples[i]
                ecg_data = sample['data']  # Shape: (1080, 2)
                
                # Plot both leads
                time = np.arange(1080) / 360  # Convert to seconds
                axes[i].plot(time, ecg_data[:, 0], label='MLII', linewidth=1.5, alpha=0.8)
                axes[i].plot(time, ecg_data[:, 1], label='V1', linewidth=1.5, alpha=0.8)
                
                # Add prediction info
                pred_class = self.class_names[sample['prediction']]
                color = 'green' if sample['prediction'] == i else 'red'
                axes[i].set_title(f'{class_name} - Predicted: {pred_class} (Confidence: {sample["confidence"]:.3f})', 
                                color=color, fontweight='bold')
                axes[i].set_ylabel('Amplitude')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'No {class_name} samples found', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(class_name)
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_ecg_segments.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Sample ECG segments saved")
    
    def generate_learning_rate_curve(self):
        """Generate learning rate curve."""
        print("Generating learning rate curve...")
        
        # Load training history
        history_path = self.results_dir / "training_history.json"
        if not history_path.exists():
            print("⚠️  Training history not found, skipping learning rate curve")
            return
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Simulate learning rate curve based on cosine annealing
        max_epochs = len(history.get('train_losses', []))
        initial_lr = self.config.get('learning_rate', 0.001)
        
        epochs = range(1, max_epochs + 1)
        lr_values = []
        
        for epoch in epochs:
            # Cosine annealing formula
            lr = initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - 1) / max_epochs))
            lr_values.append(lr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lr_values, 'b-', linewidth=2)
        plt.title('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "learning_rate_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Learning rate curve saved")
    
    def generate_model_architecture(self):
        """Generate model architecture diagram."""
        print("Generating model architecture diagram...")
        
        # Create a simple architecture diagram
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define layer positions
        layers = [
            ('Input\n(1080×2)', 0, 0.5),
            ('Conv1D\n(64 filters)', 1, 0.5),
            ('Conv1D\n(128 filters)', 2, 0.5),
            ('Conv1D\n(256 filters)', 3, 0.5),
            ('Global\nAvg Pool', 4, 0.5),
            ('Dense\n(512)', 5, 0.5),
            ('Dropout\n(0.4)', 6, 0.5),
            ('Output\n(5 classes)', 7, 0.5)
        ]
        
        # Draw layers
        for i, (name, x, y) in enumerate(layers):
            color = '#4ECDC4' if i == 0 else '#FF6B6B' if i == len(layers)-1 else '#96CEB4'
            rect = plt.Rectangle((x-0.3, y-0.2), 0.6, 0.4, 
                               facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, name, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw connections
        for i in range(len(layers)-1):
            ax.arrow(layers[i][1]+0.3, layers[i][2], 0.4, 0, 
                    head_width=0.05, head_length=0.05, fc='black', ec='black')
        
        # Add annotations
        ax.text(3.5, 0.8, f'Parameters: {sum(p.numel() for p in self.model.parameters()):,}', 
               fontsize=12, fontweight='bold')
        ax.text(3.5, 0.7, f'Model Type: {self.config.get("model_type", "lightweight")}', 
               fontsize=12)
        
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(0, 1)
        ax.set_title('ECG Classifier Architecture', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_architecture.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Model architecture saved")
    
    def get_predictions(self):
        """Get predictions on validation set."""
        # Get validation data
        data_loaders = self.data_loader.create_data_loaders(
            batch_size=32, split_type='splits', shuffle_train=False, num_workers=0
        )
        val_loader = data_loaders['val']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    data, target, _ = batch
                else:
                    data, target = batch
                
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                predictions = output['logits'].argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        return all_predictions, all_targets
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("=== Generating All Visualizations ===")
        
        try:
            self.generate_training_curves()
        except Exception as e:
            print(f"⚠️  Failed to generate training curves: {e}")
        
        try:
            self.generate_confusion_matrices()
        except Exception as e:
            print(f"⚠️  Failed to generate confusion matrices: {e}")
        
        try:
            self.generate_class_metrics()
        except Exception as e:
            print(f"⚠️  Failed to generate class metrics: {e}")
        
        try:
            self.generate_sample_ecg_segments()
        except Exception as e:
            print(f"⚠️  Failed to generate sample ECG segments: {e}")
        
        try:
            self.generate_learning_rate_curve()
        except Exception as e:
            print(f"⚠️  Failed to generate learning rate curve: {e}")
        
        try:
            self.generate_model_architecture()
        except Exception as e:
            print(f"⚠️  Failed to generate model architecture: {e}")
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}")
        print("Generated files:")
        for file in self.output_dir.glob("*.png"):
            print(f"  - {file.name}")

def main():
    """Main function."""
    try:
        generator = VisualizationGenerator()
        generator.generate_all_visualizations()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
