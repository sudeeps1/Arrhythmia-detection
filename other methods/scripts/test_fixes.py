#!/usr/bin/env python3
"""
Test script to verify the fixes work correctly.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocessed_data_loader import load_preprocessed_data
from models.ecg_classifier import create_model

def test_fixes():
    """Test that the fixes work correctly."""
    print("=== Testing Fixes ===")
    
    # Test 1: Load fixed data splits
    print("\n1. Testing data loading...")
    try:
        data_loader = load_preprocessed_data('data/processed', 'preprocessed_data_fixed_splits.pkl')
        print("✅ Data loaded successfully")
        
        # Print summary
        data_loader.print_summary()
        
        # Check splits
        datasets = data_loader.create_datasets('splits')
        print(f"\nDatasets created: {list(datasets.keys())}")
        
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        
        # Check validation class distribution
        val_labels = []
        for i in range(min(1000, len(val_dataset))):  # Sample first 1000
            val_labels.append(val_dataset[i][1])
        
        val_labels = np.array(val_labels)
        unique_labels, counts = np.unique(val_labels, return_counts=True)
        print(f"\nValidation set class distribution (first 1000 samples):")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples ({100*count/len(val_labels):.1f}%)")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Test model forward pass
    print("\n2. Testing model forward pass...")
    try:
        model = create_model(
            model_type='lightweight',
            num_leads=2,
            input_size=1080,
            num_classes=5,
            dropout_rate=0.4
        )
        print("✅ Model created successfully")
        
        # Test with correct input shape
        test_input = torch.randn(4, 2, 1080)  # (batch, num_leads, seq_len)
        print(f"Test input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Model forward pass successful")
            print(f"Output logits shape: {output['logits'].shape}")
            print(f"Output probabilities shape: {output['probabilities'].shape}")
            
            # Check predictions
            predictions = output['logits'].argmax(dim=1)
            print(f"Predictions: {predictions}")
            
            # Check if predictions are diverse
            unique_preds = torch.unique(predictions)
            print(f"Unique predictions: {unique_preds}")
            if len(unique_preds) > 1:
                print("✅ Model produces diverse predictions")
            else:
                print("⚠️  Model produces only one prediction class")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Test data shape handling
    print("\n3. Testing data shape handling...")
    try:
        # Get a batch from validation dataset
        val_batch = []
        val_labels = []
        
        for i in range(4):
            item = val_dataset[i]
            if len(item) == 3:  # segments, labels, subject_ids
                data, label, _ = item
            else:  # segments, labels
                data, label = item
            val_batch.append(data)
            val_labels.append(label)
        
        val_batch = torch.stack(val_batch)
        val_labels = torch.tensor(val_labels)
        
        print(f"Original batch shape: {val_batch.shape}")
        print(f"Labels: {val_labels}")
        
        # Check if data shape needs fixing
        if val_batch.size(1) == 1080 and val_batch.size(2) == 2:
            print("Data shape is (batch, seq_len, num_leads) - needs fixing")
            # Fix shape
            val_batch = val_batch.transpose(1, 2)  # (batch, num_leads, seq_len)
            print(f"Fixed batch shape: {val_batch.shape}")
        else:
            print("Data shape is already correct")
        
        # Test model with actual data
        with torch.no_grad():
            output = model(val_batch)
            predictions = output['logits'].argmax(dim=1)
            print(f"✅ Model works with actual data")
            print(f"Predictions: {predictions}")
            print(f"True labels: {val_labels}")
            
            # Calculate accuracy
            correct = (predictions == val_labels).sum().item()
            accuracy = 100 * correct / len(val_labels)
            print(f"Sample accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"❌ Data shape test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_fixes()
    if success:
        print("\n🎉 Fixes are working correctly!")
        print("You can now run the training script with:")
        print("python scripts/train_with_fixed_splits.py --data_file preprocessed_data_fixed_splits.pkl")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
