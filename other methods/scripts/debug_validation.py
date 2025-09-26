#!/usr/bin/env python3
"""
Debug script to investigate validation accuracy issue.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocessed_data_loader import load_preprocessed_data
from models.ecg_classifier import create_model

def debug_validation():
    """Debug validation accuracy issue."""
    print("=== Debugging Validation Accuracy Issue ===")
    
    # Load data
    data_loader = load_preprocessed_data('data/processed')
    data_loader.print_summary()
    
    # Create datasets
    datasets = data_loader.create_datasets('splits')
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Check a few samples
    print("\n=== Sample Data Inspection ===")
    for i in range(3):
        sample = train_dataset[i]
        print(f"Sample {i}:")
        print(f"  Data shape: {sample[0].shape}")
        print(f"  Label: {sample[1]}")
        print(f"  Data type: {sample[0].dtype}")
        print(f"  Data range: [{sample[0].min():.3f}, {sample[0].max():.3f}]")
        print(f"  Data mean: {sample[0].mean():.3f}")
        print(f"  Data std: {sample[0].std():.3f}")
    
    # Check validation samples
    print("\n=== Validation Data Inspection ===")
    for i in range(3):
        sample = val_dataset[i]
        print(f"Val Sample {i}:")
        print(f"  Data shape: {sample[0].shape}")
        print(f"  Label: {sample[1]}")
        print(f"  Data type: {sample[0].dtype}")
        print(f"  Data range: [{sample[0].min():.3f}, {sample[0].max():.3f}]")
        print(f"  Data mean: {sample[0].mean():.3f}")
        print(f"  Data std: {sample[0].std():.3f}")
    
    # Check class distribution in validation set
    print("\n=== Validation Class Distribution ===")
    val_labels = []
    for i in range(len(val_dataset)):
        val_labels.append(val_dataset[i][1])
    
    val_labels = np.array(val_labels)
    unique_labels, counts = np.unique(val_labels, return_counts=True)
    print("Validation set class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples ({100*count/len(val_labels):.1f}%)")
    
    # Create model and test forward pass
    print("\n=== Model Forward Pass Test ===")
    model = create_model(
        model_type='lightweight',
        num_leads=2,
        input_size=1080,
        num_classes=5,
        dropout_rate=0.4
    )
    
    # Test with a batch
    batch_size = 4
    test_input = torch.randn(batch_size, 2, 1080)
    print(f"Test input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Model output keys: {output.keys()}")
        print(f"Logits shape: {output['logits'].shape}")
        print(f"Probabilities shape: {output['probabilities'].shape}")
        print(f"Features shape: {output['features'].shape}")
        
        # Check logits
        logits = output['logits']
        print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"Logits mean: {logits.mean():.3f}")
        print(f"Logits std: {logits.std():.3f}")
        
        # Check predictions
        predictions = output['logits'].argmax(dim=1)
        print(f"Predictions: {predictions}")
        
        # Check probabilities
        probs = output['probabilities']
        print(f"Probability sum per sample: {probs.sum(dim=1)}")
        print(f"Max probability per sample: {probs.max(dim=1)[0]}")
    
    # Test with actual validation data
    print("\n=== Testing with Actual Validation Data ===")
    val_batch = []
    val_batch_labels = []
    
    for i in range(min(8, len(val_dataset))):
        data, label = val_dataset[i]
        val_batch.append(data)
        val_batch_labels.append(label)
    
    val_batch = torch.stack(val_batch)
    val_batch_labels = torch.tensor(val_batch_labels)
    
    print(f"Val batch shape: {val_batch.shape}")
    print(f"Val batch labels: {val_batch_labels}")
    
    with torch.no_grad():
        val_output = model(val_batch)
        val_logits = val_output['logits']
        val_preds = val_logits.argmax(dim=1)
        val_probs = val_output['probabilities']
        
        print(f"Val logits shape: {val_logits.shape}")
        print(f"Val predictions: {val_preds}")
        print(f"Val probabilities sum: {val_probs.sum(dim=1)}")
        print(f"Val max probabilities: {val_probs.max(dim=1)[0]}")
        
        # Check if all predictions are the same
        unique_preds = torch.unique(val_preds)
        print(f"Unique predictions: {unique_preds}")
        if len(unique_preds) == 1:
            print("WARNING: All predictions are the same class!")
            print(f"Predicted class: {unique_preds[0]}")
        
        # Calculate accuracy
        correct = (val_preds == val_batch_labels).sum().item()
        accuracy = 100 * correct / len(val_batch_labels)
        print(f"Sample accuracy: {accuracy:.2f}%")
    
    # Check if there's a gradient issue
    print("\n=== Gradient Flow Test ===")
    model.train()
    test_input = torch.randn(2, 2, 1080, requires_grad=True)
    output = model(test_input)
    loss = output['logits'].sum()
    loss.backward()
    
    print(f"Input gradients exist: {test_input.grad is not None}")
    if test_input.grad is not None:
        print(f"Input gradient norm: {test_input.grad.norm():.6f}")
    
    # Check model parameters
    print("\n=== Model Parameter Inspection ===")
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            zero_params += (param == 0).sum().item()
            print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
            print(f"  Range: [{param.min():.6f}, {param.max():.6f}]")
            print(f"  Mean: {param.mean():.6f}, Std: {param.std():.6f}")
    
    print(f"\nTotal parameters: {total_params}")
    print(f"Zero parameters: {zero_params}")
    print(f"Zero parameter percentage: {100*zero_params/total_params:.2f}%")

if __name__ == "__main__":
    debug_validation()
