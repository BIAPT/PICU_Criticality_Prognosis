# PICU_Criticality_Prognosis

By: Derek Newman 

EEG Criticality and Complexity machine learning analyses for pediatric ICU (PICU) prognosis of functional outcome/recovery.

This repository contains preprocessing, feature extraction, statistical comparison, and classification workflows focused on criticality-informed EEG features.

## Overview

The analysis pipeline includes:

- EEG preprocessing.
- Feature extraction across entropy, complexity, fractal, DFA, and spectral power.
- Group-difference and normality analyses.
- Outcome classification with bootstrap-averaged logistic regression and comparative models.
- Visualization scripts and figures.

## Repository Layout

```text
PICU_Criticality/
├── figures/                          # Generated figures and comparison tables
├── notebooks/
│   ├── Demos_and_descriptive_statistics.ipynb # cohort and descriptive summaries
│   ├── Group_differences.ipynb                # Recovery, etiology and drug distributed based group analyses.
│   ├── ML_Classification_Bootstrap.ipynb      # bootstrap model training/evaluation.
│   └── Ordinal_analysis.ipynb                 # Data driven identification of group behvaioural recovery score
├── preprocessing/
│   └── Preprocessing_10_ICA.ipynb    # preprocessing workflow
├── scripts/
│   └── feature_extraction/
│       ├── neural_complexity.py       # CLI entrypoint
│       ├── complexity.py              # Main orchestration function
│       ├── feature_analysis.py        # Metric extraction methods
│       ├── parameter_selection.py     # Delay/dimension optimization
│       └── visualize.py               # Diagnostic and summary plots
├── source/                            
├── requirements.txt                   #  Dependencies
└── .gitignore                         # excludes data and source directories.

```

## Installation

1. Create and activate a Python environment.
2. Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Expectations

- The feature extraction CLI accepts either:
	- A path to an MNE epochs file readable by `mne.read_epochs(...)`.
	- An in-memory `mne.Epochs` object.
- Typical shape assumptions in the pipeline are `(n_epochs, n_channels, n_times)`.

## Main Feature Extraction Pipeline

CLI entrypoint:

```bash
python scripts/feature_extraction/neural_complexity.py /path/to/epochs-epo.fif \
	--save \
	--out_dir ./ANALYSIS \
	--condition baseline \
	--inspection \
	--entropy \
	--complexity \
	--fractal \
	--dfa \
	--power \
	--PDF
```

Key behaviors of `Complexity_Feature_Extraction(...)`:

- Optional delay optimization (Fraser method) and embedding dimension optimization (Cao method).
- Extraction families:
	- Entropy metrics (sample entropy, permutation variants, etc.)
	- Complexity metrics (Lempel-Ziv variants, Lyapunov, Hjorth, etc.)
	- Fractal metrics (Higuchi FD, Katz FD, Petrosian FD, etc.)
	- DFA-based metrics
	- Power and spectral metrics
- Diagnostics:
	- Channel traces
	- PSD visualizations
	- Per-metric temporal/channel summaries
	- Heatmaps and correlation maps across metrics

Outputs:

- Pickle file (default): `complex_results.pkl`
- Condition-specific pickle: `{condition}_complex_results.pkl`
- PDF report: `complexity_analysis_{condition}.pdf` (or `complexity_analysis.pdf`)
