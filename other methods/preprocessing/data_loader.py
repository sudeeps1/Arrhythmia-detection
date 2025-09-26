"""
MIT-BIH Arrhythmia Database Data Loader

This module provides functionality to load and process the MIT-BIH Arrhythmia Database
in WFDB format, including signal data and beat annotations.
"""

import os
import numpy as np
import pandas as pd
import wfdb
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MITBIHDataLoader:
    """
    Data loader for MIT-BIH Arrhythmia Database.
    
    Handles loading of WFDB format files (.dat, .hea, .atr) and provides
    methods to access ECG signals and beat annotations.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the MIT-BIH database directory
        """
        self.data_path = Path(data_path)
        self.records = self._load_records()
        self.annotation_codes = self._get_annotation_codes()
        
    def _load_records(self) -> List[str]:
        """Load the list of available record names."""
        records_file = self.data_path / "RECORDS"
        if records_file.exists():
            with open(records_file, 'r') as f:
                records = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Fallback: scan directory for .hea files
            records = []
            for file in self.data_path.glob("*.hea"):
                records.append(file.stem)
            records.sort()
        
        logger.info(f"Found {len(records)} records in the database")
        return records
    
    def _get_annotation_codes(self) -> Dict[str, str]:
        """
        Get the mapping of annotation codes to beat types.
        
        Returns:
            Dictionary mapping annotation codes to beat type descriptions
        """
        return {
            'N': 'Normal beat',
            'L': 'Left bundle branch block beat',
            'R': 'Right bundle branch block beat',
            'B': 'Bundle branch block beat (unspecified)',
            'A': 'Atrial premature beat',
            'a': 'Aberrated atrial premature beat',
            'J': 'Nodal (junctional) premature beat',
            'S': 'Supraventricular premature beat',
            'V': 'Premature ventricular contraction',
            'r': 'R-on-T premature ventricular contraction',
            'F': 'Fusion of ventricular and normal beat',
            'e': 'Atrial escape beat',
            'j': 'Nodal (junctional) escape beat',
            'n': 'Supraventricular escape beat',
            'E': 'Ventricular escape beat',
            '/': 'Paced beat',
            'f': 'Fusion of paced and normal beat',
            'Q': 'Unclassifiable beat',
            '?': 'Signal quality change',
            '|': 'Isolated QRS-like artifact',
            '~': 'Change in signal quality',
            '+': 'Rhythm change',
            's': 'ST segment change',
            'T': 'T-wave change',
            '*': 'Systole',
            'D': 'Diastole',
            '="': 'Measurement annotation',
            'p': 'P-wave peak',
            't': 'T-wave peak',
            'u': 'U-wave peak',
            '`': 'Non-conducted P-wave',
            "'": 'PQ junction',
            '^': 'J-point',
            '"': 'R-wave peak',
            '[': 'Left bracket',
            ']': 'Right bracket',
            '!': 'Ventricular flutter wave',
            'x': 'Non-conducted P-wave (blocked APB)',
            '(': 'Waveform onset',
            ')': 'Waveform end',
            'k': 'Peak of P-wave',
            'm': 'Peak of T-wave',
            'o': 'Peak of U-wave'
        }
    
    def load_record(self, record_name: str) -> Dict:
        """
        Load a single record with its signal and annotation data.
        
        Args:
            record_name: Name of the record to load
            
        Returns:
            Dictionary containing signal data, annotations, and metadata
        """
        record_path = self.data_path / record_name
        
        try:
            # Load signal data
            record = wfdb.rdrecord(str(record_path))
            
            # Load annotations
            ann = wfdb.rdann(str(record_path), 'atr')
            
            # Extract signal data
            signals = record.p_signal
            signal_names = record.sig_name
            fs = record.fs
            
            # Extract annotation data
            sample_indices = ann.sample
            annotation_codes = ann.symbol
            annotation_types = ann.aux_note
            
            # Create beat annotations DataFrame
            beat_annotations = pd.DataFrame({
                'sample': sample_indices,
                'code': annotation_codes,
                'type': annotation_types,
                'time': sample_indices / fs
            })
            
            # Filter for beat annotations (remove rhythm and signal quality annotations)
            beat_codes = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 
                         'e', 'j', 'n', 'E', '/', 'f', 'Q']
            beat_annotations = beat_annotations[
                beat_annotations['code'].isin(beat_codes)
            ].reset_index(drop=True)
            
            result = {
                'record_name': record_name,
                'signals': signals,
                'signal_names': signal_names,
                'fs': fs,
                'duration': len(signals) / fs,
                'beat_annotations': beat_annotations,
                'all_annotations': pd.DataFrame({
                    'sample': sample_indices,
                    'code': annotation_codes,
                    'type': annotation_types,
                    'time': sample_indices / fs
                })
            }
            
            logger.info(f"Loaded record {record_name}: {len(signals)} samples, "
                       f"{len(beat_annotations)} beats, {fs} Hz")
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading record {record_name}: {str(e)}")
            return None
    
    def get_record_info(self, record_name: str) -> Dict:
        """
        Get basic information about a record without loading full data.
        
        Args:
            record_name: Name of the record
            
        Returns:
            Dictionary with record information
        """
        record_path = self.data_path / record_name
        
        try:
            # Load header information
            header = wfdb.rdheader(str(record_path))
            
            return {
                'record_name': record_name,
                'num_samples': header.n_sig,
                'num_leads': header.n_sig,
                'fs': header.fs,
                'duration': header.sig_len / header.fs,
                'signal_names': header.sig_name,
                'units': header.units
            }
            
        except Exception as e:
            logger.error(f"Error getting info for record {record_name}: {str(e)}")
            return None
    
    def get_all_records_info(self) -> pd.DataFrame:
        """
        Get information about all available records.
        
        Returns:
            DataFrame with information about all records
        """
        records_info = []
        
        for record_name in self.records:
            info = self.get_record_info(record_name)
            if info:
                records_info.append(info)
        
        return pd.DataFrame(records_info)
    
    def get_beat_statistics(self, record_name: str) -> Dict:
        """
        Get beat type statistics for a record.
        
        Args:
            record_name: Name of the record
            
        Returns:
            Dictionary with beat type counts and percentages
        """
        record_data = self.load_record(record_name)
        if record_data is None:
            return None
        
        beat_annotations = record_data['beat_annotations']
        beat_counts = beat_annotations['code'].value_counts()
        beat_percentages = (beat_counts / len(beat_annotations) * 100).round(2)
        
        # Map codes to descriptions
        beat_descriptions = {code: self.annotation_codes.get(code, 'Unknown') 
                           for code in beat_counts.index}
        
        return {
            'record_name': record_name,
            'total_beats': len(beat_annotations),
            'beat_counts': beat_counts.to_dict(),
            'beat_percentages': beat_percentages.to_dict(),
            'beat_descriptions': beat_descriptions
        }
    
    def get_all_beat_statistics(self) -> pd.DataFrame:
        """
        Get beat statistics for all records.
        
        Returns:
            DataFrame with beat statistics for all records
        """
        all_stats = []
        
        for record_name in self.records:
            stats = self.get_beat_statistics(record_name)
            if stats:
                all_stats.append(stats)
        
        return pd.DataFrame(all_stats)
    
    def get_rhythm_annotations(self, record_name: str) -> pd.DataFrame:
        """
        Get rhythm change annotations for a record.
        
        Args:
            record_name: Name of the record
            
        Returns:
            DataFrame with rhythm annotations
        """
        record_data = self.load_record(record_name)
        if record_data is None:
            return None
        
        all_annotations = record_data['all_annotations']
        rhythm_annotations = all_annotations[
            all_annotations['code'] == '+'
        ].copy()
        
        return rhythm_annotations
    
    def export_record_summary(self, output_path: str = "record_summary.csv"):
        """
        Export a summary of all records to CSV.
        
        Args:
            output_path: Path to save the summary CSV
        """
        records_info = self.get_all_records_info()
        beat_stats = self.get_all_beat_statistics()
        
        # Merge information
        summary = records_info.merge(beat_stats, on='record_name', how='left')
        
        # Save to CSV
        summary.to_csv(output_path, index=False)
        logger.info(f"Record summary exported to {output_path}")
        
        return summary


def main():
    """Example usage of the MITBIHDataLoader."""
    # Initialize data loader
    data_path = "mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0"
    loader = MITBIHDataLoader(data_path)
    
    # Get summary of all records
    summary = loader.export_record_summary("data/record_summary.csv")
    print(f"Loaded {len(summary)} records")
    
    # Example: Load a specific record
    record_data = loader.load_record("100")
    if record_data:
        print(f"Record 100: {record_data['duration']:.1f} seconds, "
              f"{len(record_data['beat_annotations'])} beats")
        
        # Show beat type distribution
        beat_stats = loader.get_beat_statistics("100")
        print("Beat type distribution:")
        for code, count in beat_stats['beat_counts'].items():
            desc = beat_stats['beat_descriptions'][code]
            pct = beat_stats['beat_percentages'][code]
            print(f"  {code} ({desc}): {count} ({pct}%)")


if __name__ == "__main__":
    main()

