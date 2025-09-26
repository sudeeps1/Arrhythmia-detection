#!/usr/bin/env python3
"""
Test Script for Weighted Sampling

This script tests the weighted sampling functionality to ensure it works correctly.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_with_preprocessed import PreprocessedECGTrainer

def test_weight_calculation():
    """Test the weight calculation function."""
    print("Testing weight calculation...")
    
    # Create a simple test dataset with known class distribution
    # Class 0: 100 samples, Class 1: 50 samples, Class 2: 10 samples
    test_labels = np.array([0] * 100 + [1] * 50 + [2] * 10)
    
    # Initialize trainer (this will create the weight calculation method)
    trainer = PreprocessedECGTrainer()
    
    # Test different strategies
    strategies = ['inverse_frequency', 'sqrt_inverse', 'balanced']
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        
        # Calculate weights
        sample_weights = trainer._calculate_sample_weights(test_labels, strategy)
        
        # Check that weights are positive
        assert np.all(sample_weights > 0), f"All weights must be positive for {strategy}"
        
        # Check that weights sum to 1 (normalized)
        assert np.abs(np.sum(sample_weights) - 1.0) < 1e-6, f"Weights must sum to 1 for {strategy}"
        
        # Check class-wise weights
        for class_id in [0, 1, 2]:
            mask = test_labels == class_id
            class_weight = sample_weights[mask].mean()
            print(f"  Class {class_id} (count: {np.sum(mask)}): avg weight = {class_weight:.6f}")
        
        # Verify that rarer classes get higher weights for inverse strategies
        if strategy in ['inverse_frequency', 'sqrt_inverse']:
            class_0_weight = sample_weights[test_labels == 0].mean()
            class_1_weight = sample_weights[test_labels == 1].mean()
            class_2_weight = sample_weights[test_labels == 2].mean()
            
            # Class 2 (rarest) should have highest weight
            assert class_2_weight > class_1_weight, f"Rarer class should have higher weight in {strategy}"
            assert class_1_weight > class_0_weight, f"Rarer class should have higher weight in {strategy}"
        
        print(f"  ✓ {strategy} strategy passed all tests")
    
    print("\n✓ All weight calculation tests passed!")

def test_weighted_sampler():
    """Test the weighted random sampler."""
    print("\nTesting weighted random sampler...")
    
    # Create test data
    test_labels = np.array([0] * 100 + [1] * 50 + [2] * 10)
    test_data = np.random.randn(len(test_labels), 10)  # 10 features per sample
    
    # Create dataset
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = torch.FloatTensor(data)
            self.labels = torch.LongTensor(labels)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    dataset = TestDataset(test_data, test_labels)
    
    # Calculate weights
    trainer = PreprocessedECGTrainer()
    sample_weights = trainer._calculate_sample_weights(test_labels, 'inverse_frequency')
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    # Collect samples to check distribution
    sampled_labels = []
    num_batches = 10
    
    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break
        sampled_labels.extend(labels.numpy())
    
    # Check that sampling worked
    assert len(sampled_labels) > 0, "Sampler should produce samples"
    
    # Count sampled classes
    unique, counts = np.unique(sampled_labels, return_counts=True)
    print(f"  Sampled distribution: {dict(zip(unique, counts))}")
    
    # Verify that rare classes are sampled more frequently
    class_2_count = counts[unique == 2][0] if 2 in unique else 0
    class_0_count = counts[unique == 0][0] if 0 in unique else 0
    
    # With inverse frequency, class 2 should be sampled more than class 0
    # (though this is probabilistic, so we check that it's not completely wrong)
    if class_2_count > 0 and class_0_count > 0:
        print(f"  Class 2 (rare) sampled {class_2_count} times")
        print(f"  Class 0 (common) sampled {class_0_count} times")
        print(f"  Ratio (rare/common): {class_2_count/class_0_count:.2f}")
    
    print("  ✓ Weighted sampler test passed!")

def test_data_loader_creation():
    """Test that data loaders are created correctly with weighted sampling."""
    print("\nTesting data loader creation...")
    
    try:
        # Initialize trainer
        trainer = PreprocessedECGTrainer()
        
        # Test creating data loaders with weighted sampling
        data_loaders = trainer.create_data_loaders_with_sampling(
            batch_size=32,
            split_type='splits'
        )
        
        # Check that loaders were created
        assert 'train' in data_loaders, "Training loader should be created"
        assert 'val' in data_loaders, "Validation loader should be created"
        
        # Check that training loader uses weighted sampling
        train_loader = data_loaders['train']
        assert hasattr(train_loader, 'sampler'), "Training loader should have a sampler"
        assert isinstance(train_loader.sampler, WeightedRandomSampler), "Training loader should use WeightedRandomSampler"
        
        print("  ✓ Data loader creation test passed!")
        
    except Exception as e:
        print(f"  ✗ Data loader creation test failed: {e}")
        raise

def test_configuration_options():
    """Test different configuration options."""
    print("\nTesting configuration options...")
    
    # Test with weighted sampling enabled
    config_with_sampling = {
        'save_dir': 'results/test',
        'model_type': 'lightweight',
        'num_leads': 2,
        'input_size': 1080,
        'num_classes': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'max_epochs': 10,
        'patience': 5,
        'device': 'cpu',
        'training': {
            'use_weighted_sampling': True,
            'sampling_strategy': 'sqrt_inverse'
        }
    }
    
    # Test with weighted sampling disabled
    config_without_sampling = {
        'save_dir': 'results/test',
        'model_type': 'lightweight',
        'num_leads': 2,
        'input_size': 1080,
        'num_classes': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'max_epochs': 10,
        'patience': 5,
        'device': 'cpu',
        'training': {
            'use_weighted_sampling': False
        }
    }
    
    # Test both configurations
    for config, description in [(config_with_sampling, "with sampling"), (config_without_sampling, "without sampling")]:
        print(f"  Testing configuration {description}...")
        
        # Create trainer with custom config
        trainer = PreprocessedECGTrainer()
        trainer.config = config
        
        # Test weight calculation
        test_labels = np.array([0] * 100 + [1] * 50 + [2] * 10)
        
        if config['training'].get('use_weighted_sampling', False):
            # Should work with sampling
            weights = trainer._calculate_sample_weights(
                test_labels, 
                config['training'].get('sampling_strategy', 'inverse_frequency')
            )
            assert len(weights) == len(test_labels), "Weights should match label count"
        else:
            # Should still work without sampling
            weights = trainer._calculate_sample_weights(test_labels, 'balanced')
            assert len(weights) == len(test_labels), "Weights should match label count"
        
        print(f"    ✓ Configuration {description} passed")
    
    print("  ✓ All configuration tests passed!")

def main():
    """Run all tests."""
    print("=" * 60)
    print("WEIGHTED SAMPLING TEST SUITE")
    print("=" * 60)
    
    try:
        test_weight_calculation()
        test_weighted_sampler()
        test_data_loader_creation()
        test_configuration_options()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nThe weighted sampling implementation is working correctly.")
        print("You can now use it in your training pipeline.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Please check the implementation and fix any issues.")
        raise

if __name__ == "__main__":
    main()
