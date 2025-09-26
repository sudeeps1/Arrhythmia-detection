#!/usr/bin/env python3
"""
Debug Script for Weight Calculation

This script helps debug the weight calculation issue in the weighted sampling.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_weight_calculation():
    """Debug the weight calculation step by step."""
    print("=== Debugging Weight Calculation ===")
    
    # Simulate the labels from the dataset
    # Based on the error, we have 36192 training samples
    print("Simulating training dataset with 36192 samples...")
    
    # Create a realistic class distribution based on the logs
    # From the logs: normal: 46569, paced: 4004, ventricular: 1547, supraventricular: 1187, fusion: 634, unknown: 11
    # For training set (36192 samples), let's estimate the distribution
    total_samples = 36192
    
    # Estimate training distribution (roughly 67% of total)
    normal_count = int(46569 * 0.67)
    paced_count = int(4004 * 0.67)
    ventricular_count = int(1547 * 0.67)
    supraventricular_count = int(1187 * 0.67)
    fusion_count = int(634 * 0.67)
    unknown_count = int(11 * 0.67)
    
    # Adjust to match total
    remaining = total_samples - (normal_count + paced_count + ventricular_count + 
                                supraventricular_count + fusion_count + unknown_count)
    normal_count += remaining
    
    print(f"Estimated training distribution:")
    print(f"  Normal: {normal_count}")
    print(f"  Paced: {paced_count}")
    print(f"  Ventricular: {ventricular_count}")
    print(f"  Supraventricular: {supraventricular_count}")
    print(f"  Fusion: {fusion_count}")
    print(f"  Unknown: {unknown_count}")
    print(f"  Total: {normal_count + paced_count + ventricular_count + supraventricular_count + fusion_count + unknown_count}")
    
    # Create labels array
    labels = []
    labels.extend([0] * normal_count)  # Normal
    labels.extend([1] * paced_count)   # Paced
    labels.extend([2] * ventricular_count)  # Ventricular
    labels.extend([3] * supraventricular_count)  # Supraventricular
    labels.extend([4] * fusion_count)  # Fusion
    labels.extend([5] * unknown_count)  # Unknown
    
    labels = np.array(labels)
    print(f"\nLabels array shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    
    # Test the weight calculation logic
    print("\n=== Testing Weight Calculation Logic ===")
    
    # Count occurrences of each class
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    print(f"Unique classes: {unique_classes}")
    print(f"Class counts: {class_counts}")
    
    # Test different strategies
    strategies = ['inverse_frequency', 'sqrt_inverse', 'balanced']
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        
        try:
            if strategy == 'inverse_frequency':
                # Inverse frequency weighting
                class_weights = 1.0 / (class_counts + 1e-8)
            elif strategy == 'sqrt_inverse':
                # Square root of inverse frequency (less aggressive)
                class_weights = 1.0 / np.sqrt(class_counts + 1e-8)
            elif strategy == 'balanced':
                # Balanced weighting (equal weight for all classes)
                class_weights = np.ones_like(class_counts, dtype=float)
            
            print(f"Class weights: {class_weights}")
            
            # Create sample weights
            sample_weights = class_weights[labels]
            print(f"Sample weights shape: {sample_weights.shape}")
            print(f"Sample weights sum: {np.sum(sample_weights):.6f}")
            print(f"Sample weights min: {np.min(sample_weights):.6f}")
            print(f"Sample weights max: {np.max(sample_weights):.6f}")
            
            # Check for issues
            if np.any(np.isnan(sample_weights)):
                print("❌ NaN values detected!")
            if np.any(np.isinf(sample_weights)):
                print("❌ Inf values detected!")
            if np.any(sample_weights <= 0):
                print("❌ Non-positive values detected!")
            
            # Normalize to sum to 1.0
            normalized_weights = sample_weights / np.sum(sample_weights)
            print(f"Normalized weights sum: {np.sum(normalized_weights):.6f}")
            
            print(f"✅ {strategy} strategy completed successfully")
            
        except Exception as e:
            print(f"❌ {strategy} strategy failed: {e}")
    
    print("\n=== Weight Calculation Debug Complete ===")

def test_sampler_creation():
    """Test creating a WeightedRandomSampler."""
    print("\n=== Testing Sampler Creation ===")
    
    try:
        from torch.utils.data import WeightedRandomSampler
        
        # Create simple test weights
        test_weights = np.array([0.1, 0.2, 0.3, 0.4])
        print(f"Test weights: {test_weights}")
        print(f"Test weights sum: {np.sum(test_weights):.6f}")
        
        # Normalize to sum to 1.0
        normalized_weights = test_weights / np.sum(test_weights)
        print(f"Normalized weights: {normalized_weights}")
        print(f"Normalized weights sum: {np.sum(normalized_weights):.6f}")
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=normalized_weights,
            num_samples=len(normalized_weights),
            replacement=True
        )
        
        print("✅ Sampler created successfully")
        
        # Test sampling
        sampled_indices = list(sampler)
        print(f"Sampled indices: {sampled_indices}")
        
    except Exception as e:
        print(f"❌ Sampler creation failed: {e}")

def main():
    """Run all debug tests."""
    print("=" * 60)
    print("WEIGHT CALCULATION DEBUG SCRIPT")
    print("=" * 60)
    
    try:
        debug_weight_calculation()
        test_sampler_creation()
        
        print("\n" + "=" * 60)
        print("🎉 DEBUG COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
