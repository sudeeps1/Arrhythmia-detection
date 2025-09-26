#!/usr/bin/env python3
"""
Comprehensive Data Preprocessing Script

This script processes all MIT-BIH Arrhythmia Database records once and saves
the preprocessed data for immediate use in training and evaluation.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import MITBIHDataLoader
from preprocessing.signal_processing import ECGSignalProcessor
from preprocessing.data_augmentation import ECGAugmenter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDataPreprocessor:
    """
    Comprehensive data preprocessor for MIT-BIH Arrhythmia Database.
    
    Processes all records once and saves preprocessed data for immediate use.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_path = config['data_path']
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = MITBIHDataLoader(self.data_path)
        self.signal_processor = ECGSignalProcessor(
            fs=config['sampling_rate'],
            low_freq=config['low_freq'],
            high_freq=config['high_freq']
        )
        
        # Statistics
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'failed_records': [],
            'total_segments': 0,
            'class_distribution': {},
            'processing_time': 0
        }
        
        logger.info(f"Initialized preprocessor with {len(self.data_loader.records)} records")
    
    def process_single_record(self, record_name: str) -> dict:
        """
        Process a single record and return preprocessed data.
        
        Args:
            record_name: Name of the record to process
            
        Returns:
            Dictionary with preprocessed data
        """
        try:
            # Load record
            record_data = self.data_loader.load_record(record_name)
            if not record_data:
                logger.warning(f"Failed to load record {record_name}")
                return None
            
            # Preprocess record
            preprocessed = self.signal_processor.preprocess_record(
                signal_data=record_data['signals'],
                annotations=record_data['beat_annotations'],
                window_size=self.config['window_size'],
                overlap=self.config['overlap'],
                normalize_method=self.config['normalize_method'],
                target_fs=self.config.get('target_fs', self.config['sampling_rate'])
            )
            
            if not preprocessed or len(preprocessed['segments']) == 0:
                logger.warning(f"No segments extracted from record {record_name}")
                return None
            
            # Add record metadata
            result = {
                'record_name': record_name,
                'segments': preprocessed['segments'],
                'labels': preprocessed['labels'],
                'metadata': preprocessed['metadata'],
                'qrs_features': preprocessed['qrs_features'],
                'rr_intervals': preprocessed['rr_intervals'],
                'record_info': {
                    'duration': record_data['duration'],
                    'fs': record_data['fs'],
                    'signal_names': record_data['signal_names'],
                    'num_beats': len(record_data['beat_annotations'])
                }
            }
            
            logger.info(f"Processed record {record_name}: {len(preprocessed['segments'])} segments")
            return result
            
        except Exception as e:
            logger.error(f"Error processing record {record_name}: {str(e)}")
            return None
    
    def process_all_records(self, max_records: int = None) -> dict:
        """
        Process all records in the database.
        
        Args:
            max_records: Maximum number of records to process (None for all)
            
        Returns:
            Dictionary with all preprocessed data
        """
        import time
        start_time = time.time()
        
        records_to_process = self.data_loader.records[:max_records] if max_records else self.data_loader.records
        self.stats['total_records'] = len(records_to_process)
        
        logger.info(f"Starting to process {len(records_to_process)} records...")
        
        all_data = {
            'segments': [],
            'labels': [],
            'subject_ids': [],
            'record_data': {},
            'metadata': {
                'config': self.config,
                'processing_stats': self.stats
            }
        }
        
        # Process records with progress bar
        for record_name in tqdm(records_to_process, desc="Processing records"):
            result = self.process_single_record(record_name)
            
            if result:
                # Add segments and labels
                all_data['segments'].extend(result['segments'])
                all_data['labels'].extend(result['labels'])
                all_data['subject_ids'].extend([record_name] * len(result['segments']))
                
                # Store record-specific data
                all_data['record_data'][record_name] = {
                    'qrs_features': result['qrs_features'],
                    'rr_intervals': result['rr_intervals'],
                    'record_info': result['record_info']
                }
                
                self.stats['processed_records'] += 1
                self.stats['total_segments'] += len(result['segments'])
            else:
                self.stats['failed_records'].append(record_name)
        
        # Convert to numpy arrays
        all_data['segments'] = np.array(all_data['segments'])
        all_data['labels'] = np.array(all_data['labels'])
        all_data['subject_ids'] = np.array(all_data['subject_ids'])
        
        # Calculate class distribution
        unique_labels, counts = np.unique(all_data['labels'], return_counts=True)
        self.stats['class_distribution'] = dict(zip(unique_labels, counts))
        
        # Update processing time
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Processing completed in {self.stats['processing_time']:.2f} seconds")
        logger.info(f"Processed {self.stats['processed_records']}/{self.stats['total_records']} records")
        logger.info(f"Total segments: {self.stats['total_segments']}")
        logger.info(f"Class distribution: {self.stats['class_distribution']}")
        
        return all_data
    
    def apply_data_augmentation(self, data: dict) -> dict:
        """
        Apply data augmentation to the preprocessed data.
        
        Args:
            data: Preprocessed data dictionary
            
        Returns:
            Augmented data dictionary
        """
        if not self.config.get('augmentation', {}).get('enabled', False):
            logger.info("Data augmentation disabled, skipping...")
            return data
        
        logger.info("Applying data augmentation...")
        
        augmenter = ECGAugmenter(fs=self.config['sampling_rate'])
        
        # Apply augmentation
        augmented_segments, augmented_labels = augmenter.augment_dataset(
            data['segments'], 
            data['labels'], 
            self.config['augmentation']
        )
        
        # Update data
        data['segments'] = augmented_segments
        data['labels'] = augmented_labels
        data['subject_ids'] = np.concatenate([
            data['subject_ids'], 
            data['subject_ids'][:len(augmented_segments) - len(data['subject_ids'])]
        ])
        
        # Update statistics
        self.stats['total_segments'] = len(augmented_segments)
        unique_labels, counts = np.unique(augmented_labels, return_counts=True)
        self.stats['class_distribution'] = dict(zip(unique_labels, counts))
        
        logger.info(f"Augmentation completed. New total segments: {len(augmented_segments)}")
        logger.info(f"New class distribution: {self.stats['class_distribution']}")
        
        return data
    
    def save_preprocessed_data(self, data: dict, filename: str = None):
        """
        Save preprocessed data to disk.
        
        Args:
            data: Preprocessed data dictionary
            filename: Optional filename (default: auto-generated)
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"preprocessed_data_{timestamp}.pkl"
        
        filepath = self.save_dir / filename
        
        logger.info(f"Saving preprocessed data to {filepath}...")
        
        # Save main data
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata separately for easy access
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'config': self.config,
                'stats': self.stats,
                'file_info': {
                    'filename': filename,
                    'filepath': str(filepath),
                    'created_at': pd.Timestamp.now().isoformat()
                }
            }, f, indent=2, default=str)
        
        # Save class mapping
        class_mapping = {
            'normal': 0,
            'supraventricular': 1, 
            'ventricular': 2,
            'fusion': 3,
            'paced': 4,
            'unknown': 4
        }
        
        mapping_file = self.save_dir / 'class_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        logger.info(f"Data saved successfully!")
        logger.info(f"Main data: {filepath}")
        logger.info(f"Metadata: {metadata_file}")
        logger.info(f"Class mapping: {mapping_file}")
        
        return filepath
    
    def create_data_splits(self, data: dict, test_subjects: list = None) -> dict:
        """
        Create train/validation/test splits for cross-subject validation.
        
        Args:
            data: Preprocessed data dictionary
            test_subjects: List of subjects to use for testing
            
        Returns:
            Dictionary with data splits
        """
        logger.info("Creating data splits for cross-subject validation...")
        
        unique_subjects = np.unique(data['subject_ids'])
        
        if test_subjects is None:
            # Use 20% of subjects for testing
            n_test = max(1, int(0.2 * len(unique_subjects)))
            test_subjects = np.random.choice(unique_subjects, n_test, replace=False)
        
        # Create masks
        test_mask = np.isin(data['subject_ids'], test_subjects)
        train_val_mask = ~test_mask
        
        # Split train/val from remaining subjects
        train_val_subjects = data['subject_ids'][train_val_mask]
        unique_train_val = np.unique(train_val_subjects)
        n_val = max(1, int(0.2 * len(unique_train_val)))
        val_subjects = np.random.choice(unique_train_val, n_val, replace=False)
        val_mask = np.isin(train_val_subjects, val_subjects)
        
        # Create final splits
        splits = {
            'train': {
                'segments': data['segments'][train_val_mask][~val_mask],
                'labels': data['labels'][train_val_mask][~val_mask],
                'subject_ids': data['subject_ids'][train_val_mask][~val_mask]
            },
            'val': {
                'segments': data['segments'][train_val_mask][val_mask],
                'labels': data['labels'][train_val_mask][val_mask],
                'subject_ids': data['subject_ids'][train_val_mask][val_mask]
            },
            'test': {
                'segments': data['segments'][test_mask],
                'labels': data['labels'][test_mask],
                'subject_ids': data['subject_ids'][test_mask]
            }
        }
        
        # Add split info
        splits['info'] = {
            'train_subjects': list(np.unique(splits['train']['subject_ids'])),
            'val_subjects': list(np.unique(splits['val']['subject_ids'])),
            'test_subjects': list(np.unique(splits['test']['subject_ids'])),
            'split_sizes': {
                'train': len(splits['train']['segments']),
                'val': len(splits['val']['segments']),
                'test': len(splits['test']['segments'])
            }
        }
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {splits['info']['split_sizes']['train']} segments from {len(splits['info']['train_subjects'])} subjects")
        logger.info(f"  Val: {splits['info']['split_sizes']['val']} segments from {len(splits['info']['val_subjects'])} subjects")
        logger.info(f"  Test: {splits['info']['split_sizes']['test']} segments from {len(splits['info']['test_subjects'])} subjects")
        
        return splits
    
    def run_complete_preprocessing(self, max_records: int = None, 
                                 apply_augmentation: bool = True,
                                 create_splits: bool = True) -> dict:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            max_records: Maximum number of records to process
            apply_augmentation: Whether to apply data augmentation
            create_splits: Whether to create train/val/test splits
            
        Returns:
            Dictionary with all preprocessed data and splits
        """
        logger.info("=== Starting Complete Data Preprocessing Pipeline ===")
        
        # Process all records
        data = self.process_all_records(max_records)
        
        # Apply augmentation if requested
        if apply_augmentation:
            data = self.apply_data_augmentation(data)
        
        # Create splits if requested
        if create_splits:
            splits = self.create_data_splits(data)
            data['splits'] = splits
        
        # Save preprocessed data
        filepath = self.save_preprocessed_data(data)
        
        logger.info("=== Data Preprocessing Pipeline Completed ===")
        logger.info(f"All data saved to: {filepath}")
        
        return data

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Comprehensive Data Preprocessing')
    parser.add_argument('--config', type=str, default='configs/improved_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--max_records', type=int, default=None,
                       help='Maximum number of records to process')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Skip data augmentation')
    parser.add_argument('--no_splits', action='store_true',
                       help='Skip creating train/val/test splits')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for preprocessed data')
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'data_path': 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
            'save_dir': args.output_dir,
            'sampling_rate': 360,
            'low_freq': 0.5,
            'high_freq': 40.0,
            'window_size': 3.0,
            'overlap': 0.5,
            'normalize_method': 'zscore',
            'target_fs': 360,
            'augmentation': {
                'enabled': not args.no_augmentation,
                'window_augment': True,
                'synthetic_vbeats': True,
                'augmentation_prob': 0.5,
                'noise_level': [0.01, 0.05],
                'warp_factor': [0.9, 1.1],
                'scale_factor': [0.8, 1.2]
            }
        }
    
    # Initialize preprocessor
    preprocessor = ComprehensiveDataPreprocessor(config)
    
    # Run complete preprocessing
    data = preprocessor.run_complete_preprocessing(
        max_records=args.max_records,
        apply_augmentation=not args.no_augmentation,
        create_splits=not args.no_splits
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total records processed: {preprocessor.stats['processed_records']}")
    print(f"Total segments created: {preprocessor.stats['total_segments']}")
    print(f"Class distribution: {preprocessor.stats['class_distribution']}")
    print(f"Processing time: {preprocessor.stats['processing_time']:.2f} seconds")
    print(f"Data saved to: {config['save_dir']}")
    print("\nYou can now use the preprocessed data for training without reprocessing!")

if __name__ == "__main__":
    main()

