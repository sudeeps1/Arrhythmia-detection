"""
Preprocessed Data Loader

This module provides functionality to load preprocessed ECG data that was
saved by the comprehensive preprocessing script.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class PreprocessedECGDataset(Dataset):
    """
    PyTorch Dataset for preprocessed ECG data.
    
    Loads preprocessed segments and labels for training/validation.
    """
    
    def __init__(self, segments: np.ndarray, labels: np.ndarray, 
                 subject_ids: np.ndarray = None, transform=None, class_mapping: dict = None):
        """
        Initialize the dataset.
        
        Args:
            segments: Preprocessed ECG segments
            labels: Corresponding labels (can be strings or integers)
            subject_ids: Subject IDs for each segment
            transform: Optional transform to apply
            class_mapping: Dictionary mapping string labels to integers
        """
        self.segments = torch.FloatTensor(segments)
        
        # Convert labels to integers if they are strings
        if labels.dtype.kind in ['U', 'S', 'O']:  # String types
            if class_mapping is None:
                # Default mapping if none provided
                class_mapping = {
                    'normal': 0,
                    'supraventricular': 1,
                    'ventricular': 2,
                    'fusion': 3,
                    'paced': 4,
                    'unknown': 4
                }
            
            # Convert string labels to integers
            numeric_labels = np.array([class_mapping.get(str(label), 4) for label in labels])
            self.labels = torch.LongTensor(numeric_labels)
        else:
            self.labels = torch.LongTensor(labels)
        
        self.subject_ids = subject_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        
        if self.transform:
            segment = self.transform(segment)
        
        if self.subject_ids is not None:
            return segment, label, self.subject_ids[idx]
        else:
            return segment, label

class PreprocessedDataLoader:
    """
    Data loader for preprocessed ECG data.
    
    Handles loading of preprocessed data files and provides convenient
    access to train/validation/test splits.
    """
    
    def __init__(self, data_dir: str, data_file: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing preprocessed data
            data_file: Specific data file to load (None for auto-detection)
        """
        self.data_dir = Path(data_dir)
        
        # Find data file
        if data_file is None:
            # Auto-detect the most recent data file
            data_files = list(self.data_dir.glob("preprocessed_data_*.pkl"))
            if not data_files:
                raise FileNotFoundError(f"No preprocessed data files found in {data_dir}")
            data_file = max(data_files, key=lambda x: x.stat().st_mtime)
        else:
            data_file = self.data_dir / data_file
        
        self.data_file = data_file
        self.metadata_file = data_file.with_suffix('.json')
        self.mapping_file = self.data_dir / 'class_mapping.json'
        
        # Load data
        self.data = self._load_data()
        self.metadata = self._load_metadata()
        self.class_mapping = self._load_class_mapping()
        
        logger.info(f"Loaded preprocessed data from {data_file}")
        logger.info(f"Total segments: {len(self.data['segments'])}")
        logger.info(f"Class distribution: {self.metadata['stats']['class_distribution']}")
    
    def _load_data(self) -> dict:
        """Load the main preprocessed data file."""
        with open(self.data_file, 'rb') as f:
            return pickle.load(f)
    
    def _load_metadata(self) -> dict:
        """Load the metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Metadata file not found: {self.metadata_file}")
            # Try to get metadata from the pickle file
            if 'metadata' in self.data:
                logger.info("Using metadata from pickle file")
                return self.data['metadata']
            else:
                logger.warning("No metadata found in pickle file either")
                return {}
    
    def _load_class_mapping(self) -> dict:
        """Load the class mapping file."""
        if self.mapping_file.exists():
            with open(self.mapping_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Class mapping file not found: {self.mapping_file}")
            return {
                'normal': 0,
                'supraventricular': 1,
                'ventricular': 2,
                'fusion': 3,
                'paced': 4,
                'unknown': 4
            }
    
    def get_data_splits(self) -> Dict[str, Dict]:
        """
        Get train/validation/test data splits.
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        if 'splits' not in self.data:
            raise ValueError("No data splits found. Run preprocessing with --create_splits")
        
        return self.data['splits']
    
    def get_full_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the full dataset without splits.
        
        Returns:
            Tuple of (segments, labels, subject_ids)
        """
        return (
            self.data['segments'],
            self.data['labels'],
            self.data['subject_ids']
        )
    
    def create_datasets(self, split_type: str = 'splits') -> Dict[str, PreprocessedECGDataset]:
        """
        Create PyTorch datasets for training.
        
        Args:
            split_type: 'splits' for train/val/test or 'full' for full dataset
            
        Returns:
            Dictionary of PyTorch datasets
        """
        if split_type == 'splits':
            splits = self.get_data_splits()
            datasets = {}
            
            for split_name, split_data in splits.items():
                if split_name == 'info':
                    continue
                
                datasets[split_name] = PreprocessedECGDataset(
                    segments=split_data['segments'],
                    labels=split_data['labels'],
                    subject_ids=split_data['subject_ids'],
                    class_mapping=self.class_mapping
                )
            
            return datasets
        
        elif split_type == 'full':
            segments, labels, subject_ids = self.get_full_dataset()
            return {
                'full': PreprocessedECGDataset(segments, labels, subject_ids, class_mapping=self.class_mapping)
            }
        
        else:
            raise ValueError(f"Unknown split_type: {split_type}")
    
    def create_data_loaders(self, batch_size: int = 32, 
                           split_type: str = 'splits',
                           shuffle_train: bool = True,
                           num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Create PyTorch data loaders for training.
        
        Args:
            batch_size: Batch size for data loaders
            split_type: 'splits' for train/val/test or 'full' for full dataset
            shuffle_train: Whether to shuffle training data
            num_workers: Number of worker processes
            
        Returns:
            Dictionary of PyTorch data loaders
        """
        datasets = self.create_datasets(split_type)
        loaders = {}
        
        for split_name, dataset in datasets.items():
            shuffle = shuffle_train if split_name == 'train' else False
            
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
        
        return loaders
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced training.
        
        Returns:
            Tensor of class weights
        """
        labels = self.data['labels']
        
        # Convert string labels to integers if needed
        if labels.dtype.kind in ['U', 'S', 'O']:  # String types
            numeric_labels = np.array([self.class_mapping.get(str(label), 4) for label in labels])
        else:
            numeric_labels = labels
        
        # Count occurrences of each class
        unique_classes = np.unique(numeric_labels)
        class_counts = np.zeros(len(unique_classes))
        for i, class_id in enumerate(unique_classes):
            class_counts[i] = np.sum(numeric_labels == class_id)
        
        # Calculate weights (inverse frequency)
        class_weights = 1.0 / (class_counts + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Normalize weights
        class_weights = class_weights / np.sum(class_weights)
        
        return torch.FloatTensor(class_weights)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        labels = self.data['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Map numeric labels to class names
        reverse_mapping = {v: k for k, v in self.class_mapping.items()}
        distribution = {}
        
        for label, count in zip(unique_labels, counts):
            class_name = reverse_mapping.get(label, f'class_{label}')
            distribution[class_name] = int(count)
        
        return distribution
    
    def get_subject_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of subjects in the dataset.
        
        Returns:
            Dictionary mapping subject IDs to segment counts
        """
        subject_ids = self.data['subject_ids']
        unique_subjects, counts = np.unique(subject_ids, return_counts=True)
        
        distribution = {}
        for subject, count in zip(unique_subjects, counts):
            distribution[str(subject)] = int(count)
        
        return distribution
    
    def get_record_data(self, record_name: str) -> Optional[Dict]:
        """
        Get record-specific data (QRS features, RR intervals, etc.).
        
        Args:
            record_name: Name of the record
            
        Returns:
            Dictionary with record-specific data or None if not found
        """
        if 'record_data' in self.data and record_name in self.data['record_data']:
            return self.data['record_data'][record_name]
        else:
            return None
    
    def get_config(self) -> Dict:
        """
        Get the configuration used for preprocessing.
        
        Returns:
            Configuration dictionary
        """
        return self.metadata.get('config', {})
    
    def get_stats(self) -> Dict:
        """
        Get preprocessing statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.metadata.get('stats', {})
    
    def print_summary(self):
        """Print a summary of the loaded data."""
        print("\n" + "="*60)
        print("PREPROCESSED DATA SUMMARY")
        print("="*60)
        print(f"Data file: {self.data_file}")
        print(f"Total segments: {len(self.data['segments'])}")
        print(f"Segment shape: {self.data['segments'].shape}")
        print(f"Number of subjects: {len(np.unique(self.data['subject_ids']))}")
        
        print(f"\nClass Distribution:")
        class_dist = self.get_class_distribution()
        for class_name, count in class_dist.items():
            percentage = (count / len(self.data['labels'])) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        if 'splits' in self.data:
            splits = self.data['splits']
            print(f"\nData Splits:")
            for split_name, split_data in splits.items():
                if split_name != 'info':
                    print(f"  {split_name}: {len(split_data['segments'])} segments")
        
        print(f"\nConfiguration:")
        config = self.get_config()
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}: {type(value).__name__}")
            else:
                print(f"  {key}: {value}")
        
        print("="*60)

def load_preprocessed_data(data_dir: str, data_file: str = None) -> PreprocessedDataLoader:
    """
    Convenience function to load preprocessed data.
    
    Args:
        data_dir: Directory containing preprocessed data
        data_file: Specific data file to load (None for auto-detection)
        
    Returns:
        PreprocessedDataLoader instance
    """
    return PreprocessedDataLoader(data_dir, data_file)

# Example usage
if __name__ == "__main__":
    # Example of how to use the preprocessed data loader
    try:
        # Load preprocessed data
        loader = load_preprocessed_data("data/processed")
        
        # Print summary
        loader.print_summary()
        
        # Create data loaders for training
        data_loaders = loader.create_data_loaders(
            batch_size=32,
            split_type='splits',
            shuffle_train=True
        )
        
        print(f"\nData loaders created:")
        for split_name, data_loader in data_loaders.items():
            print(f"  {split_name}: {len(data_loader)} batches")
        
        # Get class weights for imbalanced training
        class_weights = loader.get_class_weights()
        print(f"\nClass weights: {class_weights}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the preprocessing script first:")
        print("python scripts/preprocess_all_data.py")

