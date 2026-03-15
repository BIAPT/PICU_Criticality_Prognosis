# Functions to identitfy and estimate parameters for anlaysis 

import time
import mne
import neurokit2 as nk2
import numpy as np
import scipy as sp
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import fooof as fooof
from fooof import FOOOF
import seaborn as sns
import os
import glob
from visualize import *
import warnings
warnings.filterwarnings("ignore")

def delay_selection(epoch_data):
    '''
    Function to extract time delay.

    Parameters:
    epochs_data: array 
        array form (mne.read_epochs().get_data
    Returns:
        all_delays, all_delays_info
    '''
    delay_max = int(epoch_data.shape[-1]*0.2)
    def process_channel(epoch_index, channel_index):
        signal = epoch_data[epoch_index, channel_index, :]
        delay_info = nk2.complexity_delay(signal, delay_max=delay_max, method="fraser1986", show=False)
        return delay_info

    n_epochs, n_channels, n_samples = epoch_data.shape
    all_delays = []
    all_delay_info = []
    parallel = joblib.Parallel(n_jobs=-1)

    for epoch_index in range(n_epochs):
        epoch_delay = []
        epoch_delay_info = []
        results = parallel(joblib.delayed(process_channel)(epoch_index, channel_index) for channel_index in range(n_channels))
        for result in results:
            epoch_delay.append(result[0])
            epoch_delay_info.append(result)
        all_delays.append(epoch_delay)
        all_delay_info.append(epoch_delay_info)
        
    return all_delays, all_delay_info

def dimension_selection(epoch_data, delay):
    '''
    Function to extract embedding dimension.

    Parameters:
    epochs_data: array 
        array form (mne.read_epochs().get_data/
    delay: int
        The time delay value for the analysis.
    
    Returns:
        all_dim, all_dim_info
    '''
    dim_max = int(epoch_data.shape[-1]/delay)-3
    def process_channel(epoch_index, channel_index, delay, dim_max):
        signal = epoch_data[epoch_index, channel_index, :]
        dimension_info = nk2.complexity_dimension(signal, delay=delay, dimension_max=dim_max, method='afnn', show=False)
        return dimension_info
    
    n_epochs, n_channels, n_samples = epoch_data.shape
    all_dim = []
    all_dim_info = []
    parallel = joblib.Parallel(n_jobs=-1)

    for epoch_index in range(n_epochs):
        epoch_dim = []
        epoch_dim_info = []
        results = parallel(joblib.delayed(process_channel)(epoch_index, channel_index, delay, dim_max) for channel_index in range(n_channels))
        for result in results:
            epoch_dim.append(result[0])
            epoch_dim_info.append(result)
        all_dim.append(epoch_dim)
        all_dim_info.append(epoch_dim_info)

    return all_dim, all_dim_info



