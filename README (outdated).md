# Interpretable Multi-Subject ECG Analysis for Novel Arrhythmia Insights

## Project Overview

This project implements a clinically meaningful and interpretable machine learning framework for arrhythmia detection using the MIT-BIH Arrhythmia Database. The system combines cross-subject generalization, causal feature attribution, and neuro-symbolic rule extraction to identify previously underreported patterns of arrhythmic activity.

## Key Features

- **Multi-subject ECG Analysis**: Processes 47 subjects with 2-lead ECG data (MLII & V1)
- **Interpretable ML Pipeline**: 1D CNN + Bi-GRU with temporal attention
- **Causal Feature Attribution**: Integrated Gradients and ablation studies
- **Neuro-Symbolic Rule Extraction**: Human-readable decision rules
- **Cross-Subject Validation**: Robust generalization across patients
- **Novel Pattern Discovery**: Identifies previously underreported arrhythmic signatures

## Dataset

- **MIT-BIH Arrhythmia Database**: 47 subjects, 2-lead ECG (MLII & V1)
- **Sampling Rate**: 360 Hz
- **Duration**: ~24 hours per subject
- **Annotations**: Normal, ventricular ectopic, supraventricular ectopic, fusion, unknown

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── configs/                      # Configuration files
│   ├── training_config.yaml     # Basic training configuration
│   └── improved_config.yaml     # Enhanced training configuration
├── data/                         # Data storage
│   ├── raw/                     # Additional raw ECG data (empty)
│   ├── processed/               # Preprocessed ECG segments (generated)
│   └── features/                # Extracted features (generated)
├── mit-bih-arrhythmia-database-1.0.0/  # MIT-BIH dataset
├── models/                       # Model implementations
│   └── ecg_classifier.py        # Main ECG classifier
├── preprocessing/                # Data preprocessing
│   ├── data_loader.py          # WFDB data loading
│   ├── signal_processing.py    # Signal filtering and segmentation
│   └── data_augmentation.py    # Data augmentation techniques
├── interpretability/            # Interpretability analysis
│   ├── feature_attribution.py  # Temporal feature attribution
│   ├── rule_extraction.py      # Neuro-symbolic rule extraction
│   └── advanced_interpretability.py  # Advanced interpretability
├── evaluation/                  # Model evaluation
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Results visualization
├── scripts/                     # Execution scripts
│   ├── preprocess_all_data.py  # Comprehensive data preprocessing
│   ├── train_with_preprocessed.py # Fast training with preprocessed data
│   ├── improved_train.py       # Enhanced training (real-time processing)
│   ├── comprehensive_evaluation.py  # Detailed evaluation
│   ├── run_complete_pipeline.py     # End-to-end pipeline
│   └── test_preprocessed_data.py # Preprocessed data testing
├── results/                     # Output directory
│   └── improved/               # Results from improved training
├── data/                        # Data storage
│   ├── raw/                    # Additional raw ECG data (empty)
│   ├── processed/              # Preprocessed ECG segments (generated)
│   └── features/               # Extracted features (generated)
├── notebooks/                   # Jupyter notebooks (empty)
├── demo.py                      # Quick demonstration
├── requirements.txt             # Python dependencies
├── PROJECT_STRUCTURE.md         # Detailed project documentation
├── QUICK_START.md              # Quick start guide
└── .gitignore                  # Git ignore rules
```

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run demo to test functionality**:
```bash
python demo.py
```

3. **Start with enhanced training**:
```bash
python scripts/improved_train.py --config configs/improved_config.yaml
```

4. **Run complete pipeline**:
```bash
python scripts/run_complete_pipeline.py --config configs/improved_config.yaml
```

For detailed usage instructions, see [QUICK_START.md](QUICK_START.md).

## Key Results

- **Classification Performance**: AUROC, AUPRC, F1-score per subject
- **Temporal Feature Importance**: Critical time windows and leads
- **Causal Validation**: Ablation studies confirming feature importance
- **Neuro-Symbolic Rules**: Human-readable decision rules
- **Novel Discoveries**: Previously underreported arrhythmic patterns

## Novel Contributions

1. **Cross-Subject Temporal Analysis**: First systematic analysis of cross-subject temporal-electrode patterns
2. **Causal Validation**: Integration of causal validation with temporal attribution
3. **Clinical Interpretability**: Human-readable rules linking ECG patterns to arrhythmias
4. **Novel Pattern Discovery**: Identification of previously underreported arrhythmic signatures

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ecg_interpretable_2024,
  title={Interpretable Multi-Subject ECG Analysis for Novel Arrhythmia Insights},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Documentation

- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed project documentation
- [instructions.txt](instructions.txt) - Original project requirements

## License

MIT License - see LICENSE file for details.
