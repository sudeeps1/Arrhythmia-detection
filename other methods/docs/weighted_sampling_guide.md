# Weighted Sampling Guide for ECG Arrhythmia Classification

This guide explains how to use the weighted sampling functionality in the ECG arrhythmia training pipeline to handle class imbalance effectively.

## Overview

Class imbalance is a common problem in medical datasets where certain conditions (classes) are much more frequent than others. In ECG arrhythmia classification, normal beats are typically much more common than rare arrhythmia types. Weighted sampling helps address this by ensuring that rare classes are sampled more frequently during training.

## Weighting Strategies

The implementation provides three different weighting strategies:

### 1. Inverse Frequency (`inverse_frequency`)
- **Most aggressive balancing**
- Gives the highest weight to the rarest classes
- Formula: `weight = 1 / (class_count + ε)`
- Best for severely imbalanced datasets
- May lead to overfitting on rare classes if used too aggressively

### 2. Square Root Inverse (`sqrt_inverse`)
- **Moderate balancing**
- Less aggressive than inverse frequency
- Formula: `weight = 1 / sqrt(class_count + ε)`
- Good balance between addressing imbalance and preventing overfitting
- Recommended for most use cases

### 3. Balanced (`balanced`)
- **Equal weighting**
- Gives equal weight to all classes
- Formula: `weight = 1` for all classes
- Useful when you want to treat all classes equally
- May not be optimal for highly imbalanced datasets

## Configuration

### Basic Configuration

To enable weighted sampling, add these settings to your configuration file:

```yaml
training:
  use_weighted_sampling: true
  sampling_strategy: 'inverse_frequency'  # or 'sqrt_inverse', 'balanced'
```

### Complete Configuration Example

```yaml
# configs/weighted_sampling_config.yaml
save_dir: 'results/preprocessed_training'
model_type: 'lightweight'
num_leads: 2
input_size: 1080
num_classes: 5
batch_size: 32
learning_rate: 0.001
max_epochs: 50
patience: 15
device: 'auto'
num_workers: 4

training:
  scheduler: 'cosine'
  gradient_clip: 1.0
  weight_decay: 1e-4
  dropout_rate: 0.4
  
  # Weighted Sampling Configuration
  use_weighted_sampling: true
  sampling_strategy: 'inverse_frequency'
```

## Usage

### 1. Training with Weighted Sampling

```bash
# Use the provided configuration
python scripts/train_with_preprocessed.py --config configs/weighted_sampling_config.yaml

# Or specify your own config
python scripts/train_with_preprocessed.py --config your_config.yaml
```

### 2. Cross-Subject Validation with Weighted Sampling

```bash
python scripts/train_with_preprocessed.py --config configs/weighted_sampling_config.yaml --cross_subject
```

### 3. Demonstration and Analysis

```bash
# Run the demo script to see how different strategies work
python scripts/demo_weighted_sampling.py
```

## How It Works

### 1. Weight Calculation

The system calculates sample weights based on the chosen strategy:

```python
def _calculate_sample_weights(self, labels, strategy='inverse_frequency'):
    # Count class occurrences
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    
    if strategy == 'inverse_frequency':
        class_weights = 1.0 / (class_counts + 1e-8)
    elif strategy == 'sqrt_inverse':
        class_weights = 1.0 / np.sqrt(class_counts + 1e-8)
    elif strategy == 'balanced':
        class_weights = np.ones_like(class_counts, dtype=float)
    
    # Normalize and create sample weights
    class_weights = class_weights / np.sum(class_weights)
    sample_weights = class_weights[labels]
    
    return sample_weights
```

### 2. DataLoader Creation

The system creates a `WeightedRandomSampler` for the training data:

```python
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,  # Use sampler instead of shuffle
    num_workers=num_workers,
    pin_memory=True
)
```

### 3. Training Process

During training, the sampler ensures that:
- Rare classes are sampled more frequently
- Common classes are sampled less frequently
- The overall distribution becomes more balanced

## Performance Considerations

### Memory Usage
- Weighted sampling requires storing sample weights for all training samples
- Memory overhead is minimal (one float per sample)
- For very large datasets, consider using `sqrt_inverse` strategy to reduce extreme weight values

### Training Speed
- Sampling with replacement may slightly increase training time
- The benefit of better class balance usually outweighs the small performance cost
- Use `num_workers` to parallelize data loading

### Convergence
- Weighted sampling may require adjusting learning rate
- Monitor validation metrics to ensure balanced performance across classes
- Consider using early stopping to prevent overfitting

## Best Practices

### 1. Strategy Selection
- Start with `sqrt_inverse` for most datasets
- Use `inverse_frequency` only for severely imbalanced datasets
- Use `balanced` when classes are roughly equal

### 2. Monitoring
- Track per-class accuracy and loss
- Ensure no single class dominates the training
- Monitor validation performance across all classes

### 3. Hyperparameter Tuning
- Adjust learning rate when using weighted sampling
- Consider reducing batch size if memory is limited
- Experiment with different strategies on a validation set

## Troubleshooting

### Common Issues

1. **Configuration Type Errors**
   - **Error**: `TypeError: '<=' not supported between instances of 'float' and 'str'`
   - **Solution**: The script now automatically converts string config values to proper types
   - **Prevention**: Ensure your YAML config uses proper data types (e.g., `1e-4` not `'1e-4'`)

2. **Overfitting on Rare Classes**
   - Switch to `sqrt_inverse` strategy
   - Reduce learning rate
   - Increase dropout or regularization

3. **Poor Performance on Common Classes**
   - Check if weights are too extreme
   - Consider using `balanced` strategy
   - Verify data quality for common classes

4. **Memory Issues**
   - Reduce batch size
   - Use `sqrt_inverse` instead of `inverse_frequency`
   - Reduce `num_workers`

### Debugging

Test the configuration fixes:

```bash
python scripts/test_fixes.py
```

This will verify that:
- Configuration type conversion works correctly
- Validation catches configuration errors
- Weight calculation functions properly

For dataset analysis, you can also run:

```bash
python scripts/test_weighted_sampling.py
```

This will show:
- Class distribution
- Weight statistics for each strategy
- Sampling effect simulation

## Advanced Usage

### Custom Weighting Strategies

You can implement custom weighting strategies by modifying the `_calculate_sample_weights` method:

```python
def _calculate_sample_weights(self, labels, strategy='custom'):
    if strategy == 'custom':
        # Implement your custom logic here
        class_weights = your_custom_function(labels)
    else:
        # Use existing strategies
        class_weights = super()._calculate_sample_weights(labels, strategy)
    
    return class_weights
```

### Dynamic Weight Adjustment

For adaptive training, you can modify weights during training:

```python
# Update weights based on current performance
if epoch % 10 == 0:
    new_weights = self._calculate_adaptive_weights(current_performance)
    sampler.weights = new_weights
```

## Conclusion

Weighted sampling is a powerful technique for handling class imbalance in ECG arrhythmia classification. By choosing the right strategy and monitoring performance, you can significantly improve the model's ability to detect rare arrhythmia types while maintaining good performance on common classes.

Start with the `sqrt_inverse` strategy and adjust based on your specific dataset characteristics and performance requirements.
