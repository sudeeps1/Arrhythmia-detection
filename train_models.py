#!/usr/bin/env python3
"""
Minimal ML script to train arrhythmia classification models on MIT-BIH data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import scipy

def load_data(csv_file):
    """Load and prepare the parsed MIT-BIH data."""
    df = pd.read_csv(csv_file)
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col.startswith('signal_')]
    X = df[feature_cols].values
    y = df['beat_type'].values
    
    # Check what classes we actually have
    print(f"Available classes: {np.unique(y)}")
    print(f"Class distribution before filtering:\n{pd.Series(y).value_counts()}")
    
    # Keep all classes that have at least 2 samples
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= 2]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]
    
    # Remove outliers (signals with extreme values) - less aggressive
    if len(X) > 0:
        signal_std = np.std(X, axis=1)
        signal_mean = np.mean(X, axis=1)
        outlier_mask = (np.abs(signal_mean) < 5 * np.std(signal_mean)) & (signal_std < 5 * np.std(signal_std))
        X, y = X[outlier_mask], y[outlier_mask]
    
    print(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    print(f"After filtering to main classes: {len(X)} samples")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")
    
    return X, y

def extract_features(X):
    """Extract comprehensive features from signal segments."""
    features = []
    
    for signal in X:
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Basic statistical features
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        max_val = np.max(signal)
        min_val = np.min(signal)
        range_val = max_val - min_val
        median_val = np.median(signal)
        q25, q75 = np.percentile(signal, [25, 75])
        iqr = q75 - q25
        skewness = scipy.stats.skew(signal)
        kurtosis = scipy.stats.kurtosis(signal)
        
        # Peak detection and analysis
        from scipy.signal import find_peaks
        peaks, peak_props = find_peaks(signal_norm, height=0.05, distance=5)
        num_peaks = len(peaks)
        if num_peaks > 0:
            avg_peak_height = np.mean(peak_props['peak_heights'])
            peak_std = np.std(peak_props['peak_heights'])
            max_peak = np.max(peak_props['peak_heights'])
        else:
            avg_peak_height = 0
            peak_std = 0
            max_peak = 0
        
        # R-R interval features (distance between peaks)
        if num_peaks > 1:
            rr_intervals = np.diff(peaks)
            avg_rr = np.mean(rr_intervals)
            rr_std = np.std(rr_intervals)
            rr_cv = rr_std / (avg_rr + 1e-8)  # Coefficient of variation
        else:
            avg_rr = 0
            rr_std = 0
            rr_cv = 0
        
        # Frequency domain features
        fft = np.fft.fft(signal_norm)
        power_spectrum = np.abs(fft[:len(fft)//2])**2
        dominant_freq = np.argmax(power_spectrum)
        spectral_centroid = np.sum(np.arange(len(power_spectrum)) * power_spectrum) / (np.sum(power_spectrum) + 1e-8)
        spectral_rolloff = np.where(np.cumsum(power_spectrum) >= 0.85 * np.sum(power_spectrum))[0]
        spectral_rolloff = spectral_rolloff[0] if len(spectral_rolloff) > 0 else 0
        
        # Morphological features
        signal_diff = np.diff(signal)
        zero_crossings = len(np.where(np.diff(np.sign(signal_diff)))[0])
        slope_changes = len(np.where(np.diff(np.sign(np.diff(signal))))[0])
        
        # Energy features
        energy = np.sum(signal**2)
        rms = np.sqrt(np.mean(signal**2))
        mav = np.mean(np.abs(signal))  # Mean absolute value
        
        # Waveform complexity
        complexity = np.sum(np.abs(np.diff(signal)))
        hjorth_mobility = np.sqrt(np.var(np.diff(signal)) / (np.var(signal) + 1e-8))
        hjorth_complexity = np.sqrt(np.var(np.diff(np.diff(signal))) / (np.var(np.diff(signal)) + 1e-8)) / (hjorth_mobility + 1e-8)
        
        # Additional time domain features
        waveform_length = np.sum(np.abs(np.diff(signal)))
        variance = np.var(signal)
        
        features.append([
            mean_val, std_val, max_val, min_val, range_val, median_val, iqr, skewness, kurtosis,
            num_peaks, avg_peak_height, peak_std, max_peak, avg_rr, rr_std, rr_cv,
            dominant_freq, spectral_centroid, spectral_rolloff, zero_crossings, slope_changes,
            energy, rms, mav, complexity, hjorth_mobility, hjorth_complexity, waveform_length, variance
        ])
    
    return np.array(features)

def train_models(X, y):
    """Train multiple ML models with advanced techniques."""
    # Extract features instead of using raw signal
    print("Extracting features...")
    X_features = extract_features(X)
    
    # Feature selection - select best features
    print("Selecting best features...")
    selector = SelectKBest(f_classif, k=min(25, X_features.shape[1]))
    X_selected = selector.fit_transform(X_features, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use RobustScaler for better outlier handling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with optimized parameters for high performance
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=3, 
            min_samples_leaf=1, random_state=42, class_weight='balanced',
            max_features='sqrt', bootstrap=True
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=500, max_depth=20, min_samples_split=3,
            min_samples_leaf=1, random_state=42, class_weight='balanced',
            max_features='sqrt', bootstrap=True
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=10, 
            min_samples_split=3, min_samples_leaf=1, random_state=42,
            subsample=0.8
        ),
        'SVM': SVC(
            kernel='rbf', C=100.0, gamma='scale', random_state=42, 
            class_weight='balanced', probability=True
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=5000, C=10.0, class_weight='balanced',
            solver='liblinear'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=3, weights='distance', metric='minkowski', p=2
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for all models
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'accuracy': accuracy, 'f1': f1}
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} F1-Score: {f1:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
        print(f"{name} CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Print detailed classification report for best model
        if accuracy == max([r['accuracy'] for r in results.values()]):
            print(f"\nDetailed report for {name}:")
            print(classification_report(y_test, y_pred))
    
    return results

def main():
    """Main function."""
    csv_file = 'mitbih_parsed.csv'
    
    try:
        # Load data
        X, y = load_data(csv_file)
        
        # Train models
        results = train_models(X, y)
        
        # Print summary
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        for model, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{model}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Run parse_mitbih.py first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
