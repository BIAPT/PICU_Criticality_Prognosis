import argparse
import time
import mne
import neurokit2 as nk2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from fooof import FOOOF
import ordpy
import pickle
import joblib
import fooof
import seaborn as sns
import os
import glob

from complexity import Complexity_Feature_Extraction

def main():
    parser = argparse.ArgumentParser(description='Perform complexity analysis on neural data.')
    
    parser.add_argument('source', type=str, help='Path to the source data.')
    parser.add_argument('--save', action='store_true', help='Save the output. Defaults to True.')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for saving results.')
    parser.add_argument('--condition', type=str, default=None, help='Specific condition to analyze.')
    parser.add_argument('--standardize', action='store_true', help='Standardize features. Defaults to True.')
    parser.add_argument('--impute', action='store_true', help='Impute missing values. Defaults to True.')
    parser.add_argument('--inspection', action='store_true', help='Perform inspection. Defaults to True.')
    parser.add_argument('--converge', action='store_true', help='Check for convergence. Defaults to True.')
    parser.add_argument('--entropy', action='store_true', help='Calculate entropy. Defaults to True.')
    parser.add_argument('--complexity', action='store_true', help='Calculate complexity. Defaults to True.')
    parser.add_argument('--fractal', action='store_true', help='Calculate fractal dimensions. Defaults to True.')
    parser.add_argument('--dfa', action='store_true', help='Calculate DFA dimensions. Defaults to True.')
    parser.add_argument('--power', action='store_true', help='Calculate power metrics. Defaults to True.')
    parser.add_argument('--delay', type=int, default=None, help='Delay for entropy and complexity calculations.')
    parser.add_argument('--dimension', type=int, default=None, help='Dimension for entropy and complexity calculations.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output. Defaults to True.')
    parser.add_argument('--PDF', action='store_true', help='Generate PDF report. Defaults to True.')

    args = parser.parse_args()

    kwargs = vars(args)
    
    Complexity_Feature_Extraction(**kwargs)

if __name__ == "__main__":
    main()
