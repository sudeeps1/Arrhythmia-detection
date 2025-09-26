#!/usr/bin/env python3
"""
Fix severely imbalanced data splits by implementing stratified sampling.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_data_splits():
    """Fix imbalanced data splits with stratified sampling."""
    print("=== Fixing Imbalanced Data Splits ===")
    
    # Load the preprocessed data
    data_file = "data/processed/preprocessed_data_20250901_180354.pkl"
    print(f"Loading data from: {data_file}")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data keys: {list(data.keys())}")
    
    # Get the data
    segments = data['segments']
    labels = data['labels']
    subject_ids = data['subject_ids']
    
    print(f"Total segments: {len(segments)}")
    print(f"Segments shape: {segments.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Subject IDs shape: {subject_ids.shape}")
    
    # Check current class distribution
    print("\n=== Current Class Distribution ===")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples ({100*count/len(labels):.1f}%)")
    
    # Convert string labels to integers if needed
    if labels.dtype.kind in ['U', 'S', 'O']:  # String types
        class_mapping = {
            'normal': 0,
            'supraventricular': 1,
            'ventricular': 2,
            'fusion': 3,
            'paced': 4,
            'unknown': 4
        }
        numeric_labels = np.array([class_mapping.get(str(label), 4) for label in labels])
    else:
        numeric_labels = labels
        class_mapping = None
    
    print(f"\nNumeric labels range: {numeric_labels.min()} to {numeric_labels.max()}")
    
    # Check numeric class distribution
    unique_numeric, counts_numeric = np.unique(numeric_labels, return_counts=True)
    print("\n=== Numeric Class Distribution ===")
    for label, count in zip(unique_numeric, counts_numeric):
        print(f"Class {label}: {count} samples ({100*count/len(numeric_labels):.1f}%)")
    
    # Create stratified splits
    print("\n=== Creating Stratified Splits ===")
    
    # First split: separate test set (20%)
    train_val_indices, test_indices = train_test_split(
        np.arange(len(numeric_labels)),
        test_size=0.2,
        stratify=numeric_labels,
        random_state=42
    )
    
    # Second split: separate validation set from train (20% of remaining = 16% of total)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.2,
        stratify=numeric_labels[train_val_indices],
        random_state=42
    )
    
    print(f"Train set: {len(train_indices)} samples ({100*len(train_indices)/len(numeric_labels):.1f}%)")
    print(f"Val set: {len(val_indices)} samples ({100*len(val_indices)/len(numeric_labels):.1f}%)")
    print(f"Test set: {len(test_indices)} samples ({100*len(test_indices)/len(numeric_labels):.1f}%)")
    
    # Check class distribution in each split
    print("\n=== Train Set Class Distribution ===")
    train_labels = numeric_labels[train_indices]
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        print(f"Class {label}: {count} samples ({100*count/len(train_labels):.1f}%)")
    
    print("\n=== Validation Set Class Distribution ===")
    val_labels = numeric_labels[val_indices]
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    for label, count in zip(unique_val, counts_val):
        print(f"Class {label}: {count} samples ({100*count/len(val_labels):.1f}%)")
    
    print("\n=== Test Set Class Distribution ===")
    test_labels = numeric_labels[test_indices]
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    for label, count in zip(unique_test, counts_test):
        print(f"Class {label}: {count} samples ({100*count/len(test_labels):.1f}%)")
    
    # Create new splits dictionary
    new_splits = {
        'train': {
            'segments': segments[train_indices],
            'labels': labels[train_indices],
            'subject_ids': subject_ids[train_indices] if subject_ids is not None else None
        },
        'val': {
            'segments': segments[val_indices],
            'labels': labels[val_indices],
            'subject_ids': subject_ids[val_indices] if subject_ids is not None else None
        },
        'test': {
            'segments': segments[test_indices],
            'labels': labels[test_indices],
            'subject_ids': subject_ids[test_indices] if subject_ids is not None else None
        }
    }
    
    # Update the data dictionary
    data['splits'] = new_splits
    
    # Save the fixed data
    output_file = "data/processed/preprocessed_data_fixed_splits.pkl"
    print(f"\nSaving fixed data to: {output_file}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print("Data splits fixed successfully!")
    
    # Also save a summary
    summary_file = "data/processed/split_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=== Fixed Data Split Summary ===\n\n")
        f.write(f"Total samples: {len(segments)}\n")
        f.write(f"Train samples: {len(train_indices)} ({100*len(train_indices)/len(numeric_labels):.1f}%)\n")
        f.write(f"Val samples: {len(val_indices)} ({100*len(val_indices)/len(numeric_labels):.1f}%)\n")
        f.write(f"Test samples: {len(test_indices)} ({100*len(test_indices)/len(numeric_labels):.1f}%)\n\n")
        
        f.write("=== Class Distribution ===\n")
        for label, count in zip(unique_numeric, counts_numeric):
            f.write(f"Class {label}: {count} samples ({100*count/len(numeric_labels):.1f}%)\n")
        
        f.write("\n=== Train Set Class Distribution ===\n")
        for label, count in zip(unique_train, counts_train):
            f.write(f"Class {label}: {count} samples ({100*count/len(train_labels):.1f}%)\n")
        
        f.write("\n=== Validation Set Class Distribution ===\n")
        for label, count in zip(unique_val, counts_val):
            f.write(f"Class {label}: {count} samples ({100*count/len(val_labels):.1f}%)\n")
        
        f.write("\n=== Test Set Class Distribution ===\n")
        for label, count in zip(unique_test, counts_test):
            f.write(f"Class {label}: {count} samples ({100*count/len(test_labels):.1f}%)\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = fix_data_splits()
        print(f"\n✅ Data splits fixed successfully!")
        print(f"Use the fixed data file: {output_file}")
        print("Update your training script to use this file instead.")
    except Exception as e:
        print(f"❌ Error fixing data splits: {e}")
        import traceback
        traceback.print_exc()

