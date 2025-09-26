#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

This script evaluates the trained ECG arrhythmia classification model
and provides detailed performance metrics and analysis.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocessed_data_loader import load_preprocessed_data
from models.ecg_classifier import create_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveEvaluator:
    def __init__(self):
        """Initialize the comprehensive evaluator."""
        self.results_dir = Path("results/preprocessed_training")
        self.output_dir = Path("results/comprehensive_evaluation")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model and data
        self.load_model_and_data()
        
        # Class names and colors
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Paced']
        self.class_colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
    def load_model_and_data(self):
        """Load the trained model and data."""
        print("Loading model and data for comprehensive evaluation...")
        
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
    
    def get_predictions_and_probabilities(self):
        """Get predictions and probabilities on validation set."""
        print("Getting predictions on validation set...")
        
        # Get validation data
        data_loaders = self.data_loader.create_data_loaders(
            batch_size=32, split_type='splits', shuffle_train=False, num_workers=0
        )
        val_loader = data_loaders['val']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_subject_ids = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    data, target, subject_id = batch
                else:
                    data, target = batch
                    subject_id = None
                
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                probabilities = torch.softmax(output['logits'], dim=1)
                predictions = output['logits'].argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                if subject_id is not None:
                    if isinstance(subject_id, torch.Tensor):
                        all_subject_ids.extend(subject_id.cpu().numpy())
                    else:
                        all_subject_ids.extend(subject_id)
        
        all_probabilities = np.vstack(all_probabilities)
        return all_predictions, all_probabilities, all_targets, all_subject_ids
    
    def calculate_comprehensive_metrics(self, predictions, targets, probabilities):
        """Calculate comprehensive performance metrics."""
        print("Calculating comprehensive metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        
        # Per-class metrics
        precision = precision_score(targets, predictions, average=None, zero_division=0)
        recall = recall_score(targets, predictions, average=None, zero_division=0)
        f1 = f1_score(targets, predictions, average=None, zero_division=0)
        
        # Macro and weighted averages
        macro_precision = precision_score(targets, predictions, average='macro', zero_division=0)
        macro_recall = recall_score(targets, predictions, average='macro', zero_division=0)
        macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)
        
        weighted_precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        weighted_recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        weighted_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
        except:
            roc_auc = None
        
        # Per-class ROC AUC
        roc_auc_per_class = []
        for i in range(len(self.class_names)):
            try:
                auc = roc_auc_score((np.array(targets) == i).astype(int), probabilities[:, i])
                roc_auc_per_class.append(auc)
            except:
                roc_auc_per_class.append(None)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'roc_auc_per_class': roc_auc_per_class
        }
    
    def analyze_per_class_performance(self, predictions, targets, probabilities):
        """Analyze performance for each class."""
        print("Analyzing per-class performance...")
        
        class_analysis = {}
        
        for i, class_name in enumerate(self.class_names):
            # Binary classification for this class
            binary_targets = (np.array(targets) == i).astype(int)
            binary_predictions = (np.array(predictions) == i).astype(int)
            binary_probabilities = probabilities[:, i]
            
            # Calculate metrics
            tp = np.sum((binary_targets == 1) & (binary_predictions == 1))
            fp = np.sum((binary_targets == 0) & (binary_predictions == 1))
            tn = np.sum((binary_targets == 0) & (binary_predictions == 0))
            fn = np.sum((binary_targets == 1) & (binary_predictions == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Sample count
            total_samples = np.sum(binary_targets)
            correct_predictions = np.sum(binary_targets == binary_predictions)
            
            class_analysis[class_name] = {
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'accuracy': correct_predictions / total_samples if total_samples > 0 else 0,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
        
        return class_analysis
    
    def generate_comprehensive_report(self, metrics, class_analysis):
        """Generate a comprehensive evaluation report."""
        print("Generating comprehensive report...")
        
        report_file = self.output_dir / "comprehensive_evaluation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE ECG ARRHYTHMIA CLASSIFICATION EVALUATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OVERALL PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
            f.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n")
            f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted Precision: {metrics['weighted_precision']:.4f}\n")
            f.write(f"Weighted Recall: {metrics['weighted_recall']:.4f}\n")
            f.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n")
            if metrics['roc_auc'] is not None:
                f.write(f"ROC AUC (Macro): {metrics['roc_auc']:.4f}\n")
            f.write("\n")
            
            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-" * 25 + "\n")
            for i, class_name in enumerate(self.class_names):
                analysis = class_analysis[class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Total Samples: {analysis['total_samples']}\n")
                f.write(f"  Correct Predictions: {analysis['correct_predictions']}\n")
                f.write(f"  Accuracy: {analysis['accuracy']:.4f} ({analysis['accuracy']*100:.2f}%)\n")
                f.write(f"  Precision: {analysis['precision']:.4f}\n")
                f.write(f"  Recall: {analysis['recall']:.4f}\n")
                f.write(f"  F1-Score: {analysis['f1']:.4f}\n")
                f.write(f"  Specificity: {analysis['specificity']:.4f}\n")
                f.write(f"  True Positives: {analysis['tp']}\n")
                f.write(f"  False Positives: {analysis['fp']}\n")
                f.write(f"  True Negatives: {analysis['tn']}\n")
                f.write(f"  False Negatives: {analysis['fn']}\n")
                if metrics['roc_auc_per_class'][i] is not None:
                    f.write(f"  ROC AUC: {metrics['roc_auc_per_class'][i]:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print("✅ Comprehensive evaluation report saved")
    
    def plot_comprehensive_metrics(self, metrics, class_analysis):
        """Generate comprehensive visualization plots."""
        print("Generating comprehensive visualizations...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 3, 1)
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # 2. Per-class metrics comparison
        ax2 = plt.subplot(3, 3, 2)
        class_names = list(class_analysis.keys())
        precisions = [class_analysis[name]['precision'] for name in class_names]
        recalls = [class_analysis[name]['recall'] for name in class_names]
        f1_scores = [class_analysis[name]['f1'] for name in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax2.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax2.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Score')
        ax2.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy per class
        ax3 = plt.subplot(3, 3, 3)
        accuracies = [class_analysis[name]['accuracy'] for name in class_names]
        bars = ax3.bar(class_names, accuracies, color=self.class_colors, alpha=0.8)
        ax3.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. ROC AUC per class (if available)
        ax4 = plt.subplot(3, 3, 4)
        roc_aucs = []
        valid_classes = []
        for i, class_name in enumerate(class_names):
            if metrics['roc_auc_per_class'][i] is not None:
                roc_aucs.append(metrics['roc_auc_per_class'][i])
                valid_classes.append(class_name)
        
        if roc_aucs:
            bars = ax4.bar(valid_classes, roc_aucs, color=self.class_colors[:len(valid_classes)], alpha=0.8)
            ax4.set_title('Per-Class ROC AUC', fontsize=14, fontweight='bold')
            ax4.set_ylabel('ROC AUC')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, auc in zip(bars, roc_aucs):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Sample count vs performance
        ax5 = plt.subplot(3, 3, 5)
        sample_counts = [class_analysis[name]['total_samples'] for name in class_names]
        accuracies = [class_analysis[name]['accuracy'] for name in class_names]
        
        scatter = ax5.scatter(sample_counts, accuracies, c=range(len(class_names)), 
                            s=100, alpha=0.7, cmap='viridis')
        ax5.set_xlabel('Number of Samples')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Sample Count vs Accuracy', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add class labels
        for i, class_name in enumerate(class_names):
            ax5.annotate(class_name, (sample_counts[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 6. Precision-Recall trade-off
        ax6 = plt.subplot(3, 3, 6)
        precisions = [class_analysis[name]['precision'] for name in class_names]
        recalls = [class_analysis[name]['recall'] for name in class_names]
        
        scatter = ax6.scatter(recalls, precisions, c=range(len(class_names)), 
                            s=100, alpha=0.7, cmap='viridis')
        ax6.set_xlabel('Recall')
        ax6.set_ylabel('Precision')
        ax6.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add class labels
        for i, class_name in enumerate(class_names):
            ax6.annotate(class_name, (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 7. Overall metrics summary
        ax7 = plt.subplot(3, 3, 7)
        overall_metrics = ['Accuracy', 'Macro F1', 'Weighted F1', 'Macro Precision', 'Macro Recall']
        overall_values = [
            metrics['accuracy'],
            metrics['macro_f1'],
            metrics['weighted_f1'],
            metrics['macro_precision'],
            metrics['macro_recall']
        ]
        
        bars = ax7.bar(overall_metrics, overall_values, color='skyblue', alpha=0.8)
        ax7.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Score')
        ax7.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, overall_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Error analysis
        ax8 = plt.subplot(3, 3, 8)
        error_rates = []
        for class_name in class_names:
            analysis = class_analysis[class_name]
            error_rate = (analysis['fp'] + analysis['fn']) / analysis['total_samples']
            error_rates.append(error_rate)
        
        bars = ax8.bar(class_names, error_rates, color='lightcoral', alpha=0.8)
        ax8.set_title('Per-Class Error Rate', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Error Rate')
        ax8.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, err in zip(bars, error_rates):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{err:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 9. Model architecture info
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Model info text
        model_info = f"""
        Model Architecture:
        - Type: {self.config.get('model_type', 'lightweight')}
        - Input Size: {self.config.get('input_size', 1080)} samples
        - Number of Leads: {self.config.get('num_leads', 2)}
        - Number of Classes: {self.config.get('num_classes', 5)}
        - Dropout Rate: {self.config.get('training', {}).get('dropout_rate', 0.4)}
        
        Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}
        """
        
        ax9.text(0.1, 0.9, model_info, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comprehensive_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Comprehensive metrics visualization saved")
    
    def run_comprehensive_evaluation(self):
        """Run the comprehensive evaluation."""
        print("=== COMPREHENSIVE MODEL EVALUATION ===")
        
        # Get predictions and probabilities
        predictions, probabilities, targets, subject_ids = self.get_predictions_and_probabilities()
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(predictions, targets, probabilities)
        
        # Analyze per-class performance
        class_analysis = self.analyze_per_class_performance(predictions, targets, probabilities)
        
        # Print summary
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\nPer-Class Performance:")
        for class_name in self.class_names:
            analysis = class_analysis[class_name]
            print(f"  {class_name}: Accuracy={analysis['accuracy']:.3f}, F1={analysis['f1']:.3f}")
        
        # Generate report and visualizations
        self.generate_comprehensive_report(metrics, class_analysis)
        self.plot_comprehensive_metrics(metrics, class_analysis)
        
        print(f"\n✅ Comprehensive evaluation completed!")
        print(f"Results saved to: {self.output_dir}")

def main():
    """Main function."""
    try:
        evaluator = ComprehensiveEvaluator()
        evaluator.run_comprehensive_evaluation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
