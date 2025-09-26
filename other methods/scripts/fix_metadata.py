#!/usr/bin/env python3
"""
Fix metadata in the fixed data splits file.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

def fix_metadata():
    """Fix metadata in the fixed data splits file."""
    print("=== Fixing Metadata in Fixed Data Splits ===")
    
    # Load the fixed data
    data_file = "data/processed/preprocessed_data_fixed_splits.pkl"
    print(f"Loading data from: {data_file}")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data keys: {list(data.keys())}")
    
    # Check if metadata exists
    if 'metadata' in data:
        print(f"Metadata keys: {list(data['metadata'].keys())}")
    else:
        print("No metadata found, creating it...")
        data['metadata'] = {}
    
    # Create stats metadata
    segments = data['segments']
    labels = data['labels']
    
    # Calculate class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_distribution = {}
    for label, count in zip(unique_labels, counts):
        class_distribution[f"class_{label}"] = str(count)
    
    # Create stats
    stats = {
        'class_distribution': class_distribution,
        'total_segments': len(segments),
        'segment_shape': list(segments.shape),
        'num_classes': len(unique_labels)
    }
    
    # Update metadata
    data['metadata']['stats'] = stats
    
    print(f"Updated metadata with stats: {stats}")
    
    # Save the fixed data
    output_file = "data/processed/preprocessed_data_fixed_splits.pkl"
    print(f"\nSaving fixed data to: {output_file}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print("Metadata fixed successfully!")
    return output_file

if __name__ == "__main__":
    try:
        output_file = fix_metadata()
        print(f"\n✅ Metadata fixed successfully!")
        print(f"Fixed data file: {output_file}")
    except Exception as e:
        print(f"❌ Error fixing metadata: {e}")
        import traceback
        traceback.print_exc()

