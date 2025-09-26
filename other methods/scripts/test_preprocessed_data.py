#!/usr/bin/env python3
"""
Test script for preprocessed data loader.

This script tests the preprocessed data loader to ensure it works correctly
before running training.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocessed_data_loader import load_preprocessed_data

def test_preprocessed_data_loader():
    """Test the preprocessed data loader."""
    print("Testing Preprocessed Data Loader...")
    
    try:
        # Try to load preprocessed data
        data_dir = "data/processed"
        
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            print("Please run the preprocessing script first:")
            print("python scripts/preprocess_all_data.py")
            return False
        
        # Check for preprocessed data files
        data_files = list(Path(data_dir).glob("preprocessed_data_*.pkl"))
        if not data_files:
            print(f"❌ No preprocessed data files found in {data_dir}")
            print("Please run the preprocessing script first:")
            print("python scripts/preprocess_all_data.py")
            return False
        
        print(f"✅ Found {len(data_files)} preprocessed data files")
        
        # Load the most recent data file
        loader = load_preprocessed_data(data_dir)
        
        # Print summary
        loader.print_summary()
        
        # Test data access
        print("\nTesting data access...")
        
        # Get full dataset
        segments, labels, subject_ids = loader.get_full_dataset()
        print(f"✅ Full dataset: {segments.shape} segments, {len(np.unique(labels))} classes")
        
        # Test data splits
        try:
            splits = loader.get_data_splits()
            print(f"✅ Data splits available:")
            for split_name, split_data in splits.items():
                if split_name != 'info':
                    print(f"  {split_name}: {len(split_data['segments'])} segments")
        except ValueError as e:
            print(f"⚠️  No data splits found: {e}")
            print("Run preprocessing with --create_splits to create splits")
        
        # Test PyTorch datasets
        print("\nTesting PyTorch datasets...")
        try:
            datasets = loader.create_datasets('splits')
            print(f"✅ Created {len(datasets)} PyTorch datasets")
        except ValueError:
            # Try with full dataset
            datasets = loader.create_datasets('full')
            print(f"✅ Created {len(datasets)} PyTorch datasets (full dataset)")
        
        # Test data loaders
        print("\nTesting PyTorch data loaders...")
        try:
            data_loaders = loader.create_data_loaders(batch_size=16, split_type='splits')
            print(f"✅ Created {len(data_loaders)} PyTorch data loaders")
            
            # Test a batch
            for split_name, data_loader in data_loaders.items():
                if split_name != 'info':
                    batch = next(iter(data_loader))
                    if len(batch) == 3:  # segments, labels, subject_ids
                        segments_batch, labels_batch, subject_ids_batch = batch
                    else:  # segments, labels
                        segments_batch, labels_batch = batch
                    
                    print(f"  {split_name}: batch shape {segments_batch.shape}, labels shape {labels_batch.shape}")
                    break
                    
        except ValueError:
            # Try with full dataset
            data_loaders = loader.create_data_loaders(batch_size=16, split_type='full')
            print(f"✅ Created {len(data_loaders)} PyTorch data loaders (full dataset)")
            
            # Test a batch
            for split_name, data_loader in data_loaders.items():
                batch = next(iter(data_loader))
                if len(batch) == 3:  # segments, labels, subject_ids
                    segments_batch, labels_batch, subject_ids_batch = batch
                else:  # segments, labels
                    segments_batch, labels_batch = batch
                
                print(f"  {split_name}: batch shape {segments_batch.shape}, labels shape {labels_batch.shape}")
                break
        
        # Test class weights
        print("\nTesting class weights...")
        class_weights = loader.get_class_weights()
        print(f"✅ Class weights: {class_weights}")
        
        # Test class distribution
        print("\nTesting class distribution...")
        class_dist = loader.get_class_distribution()
        print(f"✅ Class distribution: {class_dist}")
        
        # Test subject distribution
        print("\nTesting subject distribution...")
        subject_dist = loader.get_subject_distribution()
        print(f"✅ Subject distribution: {len(subject_dist)} subjects")
        
        # Test record data access
        print("\nTesting record data access...")
        unique_subjects = set(loader.data['subject_ids'])
        test_subject = list(unique_subjects)[0]
        record_data = loader.get_record_data(test_subject)
        if record_data:
            print(f"✅ Record data for subject {test_subject}: {list(record_data.keys())}")
        else:
            print(f"⚠️  No record data for subject {test_subject}")
        
        print("\n🎉 All tests passed! Preprocessed data loader is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing preprocessed data loader: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("=== Testing Preprocessed Data Loader ===\n")
    
    success = test_preprocessed_data_loader()
    
    if success:
        print("\n✅ Preprocessed data loader is ready to use!")
        print("\nNext steps:")
        print("1. Run training with preprocessed data:")
        print("   python scripts/train_with_preprocessed.py")
        print("2. Run cross-subject validation:")
        print("   python scripts/train_with_preprocessed.py --cross_subject")
    else:
        print("\n❌ Preprocessed data loader test failed.")
        print("Please ensure you have run the preprocessing script first:")
        print("python scripts/preprocess_all_data.py")

if __name__ == "__main__":
    main()


