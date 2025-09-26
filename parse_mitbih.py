#!/usr/bin/env python3
"""
Minimal script to parse MIT-BIH Arrhythmia Database into CSV format for ML training.
"""

import os
import struct
import csv
import numpy as np
from pathlib import Path

def read_header(hea_file):
    """Read header file to get signal info."""
    with open(hea_file, 'r') as f:
        lines = f.readlines()
    
    # Parse first line: record_name num_signals sampling_freq num_samples
    header_line = lines[0].strip().split()
    record_name = header_line[0]
    num_signals = int(header_line[1])
    sampling_freq = int(header_line[2])
    num_samples = int(header_line[3])
    
    return record_name, num_signals, sampling_freq, num_samples

def read_signal(dat_file, num_samples, num_signals):
    """Read binary signal data."""
    with open(dat_file, 'rb') as f:
        data = f.read()
    
    # MIT-BIH uses 16-bit signed integers, 2 bytes per sample
    samples = struct.unpack('<' + 'h' * (len(data) // 2), data)
    samples = np.array(samples)
    
    # Reshape for multiple signals
    if num_signals > 1:
        samples = samples.reshape(-1, num_signals)
    
    return samples

def read_annotations(atr_file, num_samples):
    """Read binary annotation file."""
    annotations = []
    
    with open(atr_file, 'rb') as f:
        while True:
            # Read annotation record (8 bytes)
            record = f.read(8)
            if len(record) < 8:
                break
                
            # Parse annotation record
            sample_num = struct.unpack('<I', record[:4])[0]
            annotation = struct.unpack('<H', record[4:6])[0]
            
            # Extract beat type (lower 4 bits)
            beat_type = annotation & 0x0F
            
            if sample_num < num_samples:
                annotations.append((sample_num, beat_type))
    
    return annotations

def parse_record(record_path):
    """Parse a single MIT-BIH record."""
    record_name = record_path.stem
    
    # Read files
    hea_file = record_path.with_suffix('.hea')
    dat_file = record_path.with_suffix('.dat')
    atr_file = record_path.with_suffix('.atr')
    
    if not all(f.exists() for f in [hea_file, dat_file, atr_file]):
        return None
    
    # Parse header
    _, num_signals, sampling_freq, num_samples = read_header(hea_file)
    
    # Read signal data
    signal_data = read_signal(dat_file, num_samples, num_signals)
    
    # Read annotations
    annotations = read_annotations(atr_file, num_samples)
    
    return {
        'record_name': record_name,
        'signal_data': signal_data,
        'annotations': annotations,
        'sampling_freq': sampling_freq,
        'num_samples': num_samples
    }

def create_csv_data(parsed_records):
    """Convert parsed records to CSV format with better data extraction."""
    csv_data = []
    
    # Beat type mapping (MIT-BIH standard) - focus on main types
    beat_types = {
        0: 'N',  # Normal
        1: 'L',  # Left bundle branch block
        2: 'R',  # Right bundle branch block
        3: 'A',  # Atrial premature
        4: 'a',  # Aberrated atrial premature
        5: 'J',  # Nodal (junctional) premature
        6: 'S',  # Supraventricular premature
        7: 'V',  # Premature ventricular contraction
        8: 'E',  # Ventricular escape
        9: 'F',  # Fusion of ventricular and normal
        10: 'Q', # Unclassifiable
        11: '/', # Paced
        12: 'f', # Fusion of paced and normal
    }
    
    for record in parsed_records:
        if record is None:
            continue
            
        signal_data = record['signal_data']
        annotations = record['annotations']
        
        # Use first signal channel only for simplicity
        if signal_data.ndim > 1:
            signal_data = signal_data[:, 0]
        
        # Create multiple samples per annotation for data augmentation
        for sample_idx, beat_type in annotations:
            if sample_idx < len(signal_data) and beat_type in beat_types:
                # Create multiple window sizes around each beat
                window_sizes = [128, 256]  # Different window sizes
                
                for window_size in window_sizes:
                    half_window = window_size // 2
                    start_idx = max(0, sample_idx - half_window)
                    end_idx = min(len(signal_data), sample_idx + half_window)
                    
                    if end_idx - start_idx >= window_size * 0.8:  # Allow some flexibility
                        signal_segment = signal_data[start_idx:end_idx]
                        
                        # Pad or truncate to exact size
                        if len(signal_segment) < window_size:
                            # Pad with zeros
                            signal_segment = np.pad(signal_segment, (0, window_size - len(signal_segment)), 'constant')
                        else:
                            # Truncate
                            signal_segment = signal_segment[:window_size]
                        
                        # Add some noise for data augmentation
                        noise = np.random.normal(0, 0.01, window_size)
                        signal_segment = signal_segment + noise
                        
                        row = {
                            'record_name': record['record_name'],
                            'sample_index': sample_idx,
                            'beat_type': beat_types[beat_type],
                            'beat_type_code': beat_type,
                            'window_size': window_size
                        }
                        
                        # Add signal values as columns
                        for i, value in enumerate(signal_segment):
                            row[f'signal_{i}'] = value
                        
                        csv_data.append(row)
    
    return csv_data

def main():
    """Main function to parse MIT-BIH database."""
    db_path = Path('mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0')
    
    if not db_path.exists():
        print(f"Database path not found: {db_path}")
        return
    
    # Get all record files
    record_files = [f for f in db_path.glob('*.hea')]
    print(f"Found {len(record_files)} records")
    
    # Parse records
    parsed_records = []
    for hea_file in record_files:
        record_path = hea_file.with_suffix('')
        print(f"Parsing {record_path.name}...")
        
        parsed_record = parse_record(record_path)
        if parsed_record:
            parsed_records.append(parsed_record)
    
    # Convert to CSV format
    csv_data = create_csv_data(parsed_records)
    
    # Write to CSV
    if csv_data:
        output_file = 'mitbih_parsed.csv'
        fieldnames = list(csv_data[0].keys())
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"Created {output_file} with {len(csv_data)} samples")
    else:
        print("No data to write")

if __name__ == '__main__':
    main()
