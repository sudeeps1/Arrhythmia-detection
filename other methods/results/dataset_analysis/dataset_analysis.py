#!/usr/bin/env python3
"""
Dataset Analysis Script

This script provides comprehensive analysis of the ECG arrhythmia dataset,
including class distribution, data statistics, and other relevant metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
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

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DatasetAnalyzer:
    def __init__(self):
        """Initialize the dataset analyzer."""
        self.output_dir = Path("results/dataset_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Class names and colors
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Paced']
        self.class_colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
    def load_data(self):
        """Load the preprocessed data."""
        print("Loading preprocessed data for analysis...")
        
        self.data_loader = load_preprocessed_data("data/processed")
        
        # Get full dataset
        self.segments, self.labels, self.subject_ids = self.data_loader.get_full_dataset()
        
        print("✅ Data loaded successfully")
    
    def analyze_class_distribution(self):
        """Analyze class distribution in the dataset."""
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Count class occurrences
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        print(f"\nTotal samples: {total_samples:,}")
        print(f"Number of classes: {len(class_counts)}")
        print(f"Number of unique subjects: {len(np.unique(self.subject_ids))}")
        
        print("\nClass Distribution:")
        print("-" * 50)
        print(f"{'Class':<15} {'Count':<10} {'Percentage':<12} {'Subject Count':<15}")
        print("-" * 50)
        
        # Map class names to their string labels
        class_mapping = {
            'normal': 'Normal',
            'supraventricular': 'Supraventricular',
            'ventricular': 'Ventricular',
            'fusion': 'Fusion',
            'paced': 'Paced',
            'unknown': 'Paced'  # 'unknown' maps to 'Paced'
        }
        
        for class_name in self.class_names:
            # Find the string label for this class
            class_label = None
            for label, mapped_name in class_mapping.items():
                if mapped_name == class_name:
                    class_label = label
                    break
            
            if class_label:
                count = class_counts.get(class_label, 0)
                percentage = (count / total_samples) * 100
                
                # Count unique subjects for this class
                class_subject_ids = self.subject_ids[self.labels == class_label]
                unique_subjects = len(np.unique(class_subject_ids))
                
                print(f"{class_name:<15} {count:<10,} {percentage:<11.2f}% {unique_subjects:<15}")
        
        # Calculate imbalance metrics
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        print(f"\nImbalance Analysis:")
        most_frequent = max(class_counts, key=class_counts.get)
        least_frequent = min(class_counts, key=class_counts.get)
        print(f"  - Most frequent class: {most_frequent} ({max_count:,} samples)")
        print(f"  - Least frequent class: {least_frequent} ({min_count:,} samples)")
        print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print(f"  - ⚠️  Dataset is highly imbalanced (ratio > 10:1)")
        elif imbalance_ratio > 5:
            print(f"  - ⚠️  Dataset is moderately imbalanced (ratio > 5:1)")
        else:
            print(f"  - ✅ Dataset is relatively balanced")
    
    def analyze_subject_distribution(self):
        """Analyze distribution across subjects."""
        print("\n" + "="*60)
        print("SUBJECT DISTRIBUTION ANALYSIS")
        print("="*60)
        
        unique_subjects = np.unique(self.subject_ids)
        subject_class_counts = {}
        
        for subject_id in unique_subjects:
            subject_mask = self.subject_ids == subject_id
            subject_labels = self.labels[subject_mask]
            class_counts = Counter(subject_labels)
            subject_class_counts[subject_id] = class_counts
        
        # Calculate statistics
        samples_per_subject = [len(self.labels[self.subject_ids == sid]) for sid in unique_subjects]
        classes_per_subject = [len(counts) for counts in subject_class_counts.values()]
        
        print(f"\nSubject Statistics:")
        print(f"  - Total subjects: {len(unique_subjects)}")
        print(f"  - Average samples per subject: {np.mean(samples_per_subject):.1f}")
        print(f"  - Median samples per subject: {np.median(samples_per_subject):.1f}")
        print(f"  - Min samples per subject: {np.min(samples_per_subject)}")
        print(f"  - Max samples per subject: {np.max(samples_per_subject)}")
        print(f"  - Average classes per subject: {np.mean(classes_per_subject):.1f}")
        
        # Subjects with all classes
        subjects_with_all_classes = sum(1 for counts in subject_class_counts.values() if len(counts) == 5)
        print(f"  - Subjects with all 5 classes: {subjects_with_all_classes}")
        
        # Show top 10 subjects by sample count
        subject_sample_counts = [(sid, len(self.labels[self.subject_ids == sid])) for sid in unique_subjects]
        subject_sample_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Subjects by Sample Count:")
        print("-" * 40)
        for i, (subject_id, count) in enumerate(subject_sample_counts[:10]):
            print(f"  {i+1:2d}. Subject {subject_id}: {count:,} samples")
    
    def analyze_data_statistics(self):
        """Analyze statistical properties of the ECG data."""
        print("\n" + "="*60)
        print("DATA STATISTICS ANALYSIS")
        print("="*60)
        
        print(f"\nData Shape:")
        print(f"  - Segments shape: {self.segments.shape}")
        print(f"  - Labels shape: {self.labels.shape}")
        print(f"  - Subject IDs shape: {self.subject_ids.shape}")
        
        # Segment length and leads
        segment_length = self.segments.shape[1]
        num_leads = self.segments.shape[2]
        print(f"  - Segment length: {segment_length} samples")
        print(f"  - Number of leads: {num_leads}")
        print(f"  - Duration per segment: {segment_length / 360:.1f} seconds (assuming 360 Hz)")
        
        # Statistical properties
        print(f"\nStatistical Properties:")
        print(f"  - Mean amplitude: {np.mean(self.segments):.4f}")
        print(f"  - Standard deviation: {np.std(self.segments):.4f}")
        print(f"  - Min amplitude: {np.min(self.segments):.4f}")
        print(f"  - Max amplitude: {np.max(self.segments):.4f}")
        print(f"  - Range: {np.max(self.segments) - np.min(self.segments):.4f}")
        
        # Per-lead statistics
        for i in range(num_leads):
            lead_data = self.segments[:, :, i]
            print(f"\nLead {i+1} Statistics:")
            print(f"  - Mean: {np.mean(lead_data):.4f}")
            print(f"  - Std: {np.std(lead_data):.4f}")
            print(f"  - Min: {np.min(lead_data):.4f}")
            print(f"  - Max: {np.max(lead_data):.4f}")
        
        # Per-class statistics
        print(f"\nPer-Class Statistical Properties:")
        print("-" * 60)
        
        # Map class names to their string labels
        class_mapping = {
            'normal': 'Normal',
            'supraventricular': 'Supraventricular',
            'ventricular': 'Ventricular',
            'fusion': 'Fusion',
            'paced': 'Paced',
            'unknown': 'Paced'  # 'unknown' maps to 'Paced'
        }
        
        for class_name in self.class_names:
            # Find the string label for this class
            class_label = None
            for label, mapped_name in class_mapping.items():
                if mapped_name == class_name:
                    class_label = label
                    break
            
            if class_label:
                class_mask = self.labels == class_label
                class_segments = self.segments[class_mask]
                
                if len(class_segments) > 0:
                    print(f"\n{class_name}:")
                    print(f"  - Mean amplitude: {np.mean(class_segments):.4f}")
                    print(f"  - Standard deviation: {np.std(class_segments):.4f}")
                    print(f"  - Min amplitude: {np.min(class_segments):.4f}")
                    print(f"  - Max amplitude: {np.max(class_segments):.4f}")
    
    def analyze_data_quality(self):
        """Analyze data quality metrics."""
        print("\n" + "="*60)
        print("DATA QUALITY ANALYSIS")
        print("="*60)
        
        # Check for NaN values
        nan_count = np.isnan(self.segments).sum()
        print(f"\nData Quality Checks:")
        print(f"  - NaN values: {nan_count}")
        
        # Check for infinite values
        inf_count = np.isinf(self.segments).sum()
        print(f"  - Infinite values: {inf_count}")
        
        # Check for zero segments
        zero_segments = np.all(self.segments == 0, axis=(1, 2)).sum()
        print(f"  - Zero segments: {zero_segments}")
        
        # Check for constant segments
        constant_segments = np.all(np.std(self.segments, axis=1) == 0, axis=1).sum()
        print(f"  - Constant segments: {constant_segments}")
        
        # Check for outliers (segments with very high variance)
        segment_variances = np.var(self.segments, axis=(1, 2))
        outlier_threshold = np.percentile(segment_variances, 99)
        outliers = (segment_variances > outlier_threshold).sum()
        print(f"  - High variance outliers (>99th percentile): {outliers}")
        
        # Data completeness
        total_elements = self.segments.size
        valid_elements = total_elements - nan_count - inf_count
        completeness = (valid_elements / total_elements) * 100
        print(f"  - Data completeness: {completeness:.2f}%")
    
    def generate_visualizations(self):
        """Generate visualizations for dataset analysis."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Class distribution plot
        self.plot_class_distribution()
        
        # Subject distribution plot
        self.plot_subject_distribution()
        
        # Data statistics plots
        self.plot_data_statistics()
        
        print(f"\n✅ Visualizations saved to: {self.output_dir}")
    
    def plot_class_distribution(self):
        """Plot class distribution."""
        class_counts = Counter(self.labels)
        
        # Map class names to their string labels
        class_mapping = {
            'normal': 'Normal',
            'supraventricular': 'Supraventricular',
            'ventricular': 'Ventricular',
            'fusion': 'Fusion',
            'paced': 'Paced',
            'unknown': 'Paced'  # 'unknown' maps to 'Paced'
        }
        
        counts = []
        for class_name in self.class_names:
            # Find the string label for this class
            class_label = None
            for label, mapped_name in class_mapping.items():
                if mapped_name == class_name:
                    class_label = label
                    break
            
            if class_label:
                count = class_counts.get(class_label, 0)
                counts.append(count)
            else:
                counts.append(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(self.class_names, counts, color=self.class_colors, alpha=0.8)
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        total = sum(counts)
        percentages = [count/total*100 for count in counts]
        wedges, texts, autotexts = ax2.pie(counts, labels=self.class_names, autopct='%1.1f%%',
                                          colors=self.class_colors, startangle=90)
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Class distribution plot saved")
    
    def plot_subject_distribution(self):
        """Plot subject distribution."""
        unique_subjects = np.unique(self.subject_ids)
        samples_per_subject = [len(self.labels[self.subject_ids == sid]) for sid in unique_subjects]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of samples per subject
        ax1.hist(samples_per_subject, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Samples per Subject', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Samples per Subject')
        ax1.set_ylabel('Number of Subjects')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(samples_per_subject, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        ax2.set_title('Samples per Subject - Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Samples per Subject')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "subject_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Subject distribution plot saved")
    
    def plot_data_statistics(self):
        """Plot data statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Amplitude distribution
        axes[0, 0].hist(self.segments.flatten(), bins=50, alpha=0.7, color='lightcoral')
        axes[0, 0].set_title('Amplitude Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Amplitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Per-class amplitude distributions
        class_mapping = {
            'normal': 'Normal',
            'supraventricular': 'Supraventricular',
            'ventricular': 'Ventricular',
            'fusion': 'Fusion',
            'paced': 'Paced',
            'unknown': 'Paced'  # 'unknown' maps to 'Paced'
        }
        
        for i, class_name in enumerate(self.class_names):
            # Find the string label for this class
            class_label = None
            for label, mapped_name in class_mapping.items():
                if mapped_name == class_name:
                    class_label = label
                    break
            
            if class_label:
                class_mask = self.labels == class_label
                class_segments = self.segments[class_mask]
                if len(class_segments) > 0:
                    axes[0, 1].hist(class_segments.flatten(), bins=30, alpha=0.6, 
                                   label=class_name, color=self.class_colors[i])
        axes[0, 1].set_title('Amplitude Distribution by Class', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Amplitude')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Per-lead statistics
        lead_means = [np.mean(self.segments[:, :, i]) for i in range(self.segments.shape[2])]
        lead_stds = [np.std(self.segments[:, :, i]) for i in range(self.segments.shape[2])]
        
        x = range(len(lead_means))
        width = 0.35
        
        axes[1, 0].bar([i - width/2 for i in x], lead_means, width, label='Mean', alpha=0.8)
        axes[1, 0].bar([i + width/2 for i in x], lead_stds, width, label='Std', alpha=0.8)
        axes[1, 0].set_title('Per-Lead Statistics', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Lead')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'Lead {i+1}' for i in x])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample ECG traces (one from each class)
        time = np.arange(self.segments.shape[1]) / 360  # Convert to seconds
        
        class_mapping = {
            'normal': 'Normal',
            'supraventricular': 'Supraventricular',
            'ventricular': 'Ventricular',
            'fusion': 'Fusion',
            'paced': 'Paced',
            'unknown': 'Paced'  # 'unknown' maps to 'Paced'
        }
        
        for i, class_name in enumerate(self.class_names):
            # Find the string label for this class
            class_label = None
            for label, mapped_name in class_mapping.items():
                if mapped_name == class_name:
                    class_label = label
                    break
            
            if class_label:
                class_mask = self.labels == class_label
                if np.any(class_mask):
                    # Take first sample from this class
                    sample_idx = np.where(class_mask)[0][0]
                    sample = self.segments[sample_idx]
                    
                    for lead in range(sample.shape[1]):
                        axes[1, 1].plot(time, sample[:, lead], 
                                       label=f'{class_name} - Lead {lead+1}', 
                                       color=self.class_colors[i], alpha=0.8)
        
        axes[1, 1].set_title('Sample ECG Traces (One per Class)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "data_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Data statistics plots saved")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        report_file = self.output_dir / "dataset_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ECG ARRHYTHMIA DATASET ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {len(self.labels):,}\n")
            f.write(f"Number of classes: {len(self.class_names)}\n")
            f.write(f"Number of subjects: {len(np.unique(self.subject_ids))}\n")
            f.write(f"Segment length: {self.segments.shape[1]} samples\n")
            f.write(f"Number of leads: {self.segments.shape[2]}\n")
            f.write(f"Duration per segment: {self.segments.shape[1] / 360:.1f} seconds\n\n")
            
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            class_counts = Counter(self.labels)
            total_samples = len(self.labels)
            
            # Map class names to their string labels
            class_mapping = {
                'normal': 'Normal',
                'supraventricular': 'Supraventricular',
                'ventricular': 'Ventricular',
                'fusion': 'Fusion',
                'paced': 'Paced',
                'unknown': 'Paced'  # 'unknown' maps to 'Paced'
            }
            
            for class_name in self.class_names:
                # Find the string label for this class
                class_label = None
                for label, mapped_name in class_mapping.items():
                    if mapped_name == class_name:
                        class_label = label
                        break
                
                if class_label:
                    count = class_counts.get(class_label, 0)
                    percentage = (count / total_samples) * 100
                    f.write(f"{class_name}: {count:,} samples ({percentage:.1f}%)\n")
            
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count
            f.write(f"\nImbalance ratio: {imbalance_ratio:.2f}:1\n\n")
            
            f.write("DATA QUALITY:\n")
            f.write("-" * 15 + "\n")
            nan_count = np.isnan(self.segments).sum()
            inf_count = np.isinf(self.segments).sum()
            total_elements = self.segments.size
            completeness = ((total_elements - nan_count - inf_count) / total_elements) * 100
            f.write(f"Data completeness: {completeness:.2f}%\n")
            f.write(f"NaN values: {nan_count}\n")
            f.write(f"Infinite values: {inf_count}\n\n")
            
            f.write("STATISTICAL PROPERTIES:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Overall mean amplitude: {np.mean(self.segments):.4f}\n")
            f.write(f"Overall standard deviation: {np.std(self.segments):.4f}\n")
            f.write(f"Amplitude range: [{np.min(self.segments):.4f}, {np.max(self.segments):.4f}]\n\n")
            
            f.write("PER-LEAD STATISTICS:\n")
            f.write("-" * 22 + "\n")
            for i in range(self.segments.shape[2]):
                lead_data = self.segments[:, :, i]
                f.write(f"Lead {i+1}: mean={np.mean(lead_data):.4f}, std={np.std(lead_data):.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print("✅ Dataset summary report saved")
    
    def run_complete_analysis(self):
        """Run the complete dataset analysis."""
        print("=== ECG ARRHYTHMIA DATASET ANALYSIS ===")
        
        # Run all analyses
        self.analyze_class_distribution()
        self.analyze_subject_distribution()
        self.analyze_data_statistics()
        self.analyze_data_quality()
        
        # Generate visualizations and report
        self.generate_visualizations()
        self.generate_summary_report()
        
        print(f"\n✅ Complete dataset analysis finished!")
        print(f"Results saved to: {self.output_dir}")

def main():
    """Main function."""
    try:
        analyzer = DatasetAnalyzer()
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
