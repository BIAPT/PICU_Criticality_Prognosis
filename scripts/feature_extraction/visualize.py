# Functions to visualize the feature analysis

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
import ordpy
import seaborn as sns
import os
import glob
import warnings
warnings.filterwarnings("ignore")
import feature_analysis
from feature_analysis import *

def filter_invalid_values(data):
    """
    Filters out epoch data that contains any infinite or NaN values.
    
    Parameters:
    data (list of lists): The input data where each sublist represents features of epoch data.
    
    Returns:
    list of lists: The filtered data with no sublist containing infinite or NaN values.
    """
    return [epoch_data for epoch_data in data if not np.isinf(epoch_data).any() and not np.isnan(epoch_data).any()]

def inspect_channels(epoch_data, ch_names,pdf=None):
    '''
    Function to plot channel time series.
    Parameters:
    epochs_data: array 
        array form (mne.read_epochs().get_data
    ch_names: list
        list of channel names
    pdf: matplotlib.backends.backend_pdf.PdfPages
        if passed saves file to open pdf file
    Returns:
        pdf.savefig()
    '''
    # Inspect the data quality across Channels... did we miss something?
    n_epochs, n_channels, n_samples= epoch_data.shape

    long_data = np.concatenate(epoch_data,axis=1)
    n_plots = n_channels
    fig = plt.figure(figsize=(20, n_plots))  
    gs = gridspec.GridSpec(n_plots, 1, hspace=0.05) 
    axs = []
    for i in range(n_plots):
        ax = fig.add_subplot(gs[i, 0])
        normed = sp.stats.zscore(long_data[i])
        ax.plot(normed)
        ax.margins(x=0)
        ax.tick_params(axis='y', labelsize=7) 
        ax.set_ylabel(ch_names[i])
        axs.append(ax)
    axs[0].set_title('All Channel Data Along Whole Recording Duration')
    for ax in axs[:-1]:
        ax.set_xticklabels([]) 
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)  
    else:
        plt.show() 

def visualize_psd(epochs, pdf=None):
    '''
    Function to extract power.
    Parameters:
    epochs: mne.object
        The EEG mne object contains info. 
    pdf: matplotlib.backends.backend_pdf.PdfPages
        if passed saves file to open pdf file
    Returns:
        pdf.savefig()    
    '''
    # View the Power Spectral Density Plots using Welch's method
    
    n_epochs, n_channels, n_samples= epochs.get_data().shape
    samp_freq = epochs.info['sfreq']
    h_pass = epochs.info['highpass']
    l_pass = epochs.info['lowpass']
    
    psds, freqs = mne.time_frequency.psd_array_welch(epochs.get_data(), sfreq=samp_freq, fmin=h_pass, fmax=l_pass)# n_fft=2048)
    psd_mean = np.mean(psds, axis=(0, 1)) 
    
    fig = plt.figure(figsize=(12, 5))  
    for i in range(n_channels):
        plt.plot(freqs, 10 * np.log10(psds[:, i, :].mean(axis=0)), label=f'Chs' if i == 0 else "", color='blue', alpha=0.3)
    plt.plot(freqs, 10 * np.log10(psd_mean), label='Ch_avg', color='red',linestyle='--',linewidth=3)

    plt.title('Power Spectral Density Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.legend()
    if pdf:
        pdf.savefig()
        plt.close()  

    else:
        plt.show() 

def visualize_factor(factors, name, epochs, ch_names, pdf=None, dfa=False):
    '''
    Function to plot factors over epochs, histograms per channel and a topomap of mean standardized values. 

    Parameters:
    factors: array 
        array with shape epochs.get_data[:,:]
    name: str
        name of feature plotting
    epochs: mne.object
        The EEG mne object contains info. 
    ch_names: list
        list of channel names
    pdf: matplotlib.backends.backend_pdf.PdfPages
        if passed saves file to open pdf file
    Returns:
        pdf.savefig()
    '''
    # Let's see how eachchannel factor changes over time 
    epoch_data = epochs.get_data()
    if dfa:
        epoch_data = feature_analysis.concatenate_epochs(epoch_data)
    epoch_shape = epoch_data.shape
    n_epochs, n_channels, n_samples= epoch_shape
    colors = plt.cm.jet(np.linspace(0, 1, n_channels))

    factors = filter_invalid_values(factors) # Add the filtered invalid values for plotting

    plt.figure(figsize=(20, 6))  
    for channel_index, color in zip(range(n_channels), colors):
        channel_factor = [epoch_factor[channel_index] for epoch_factor in factors]
        plt.plot(range(n_epochs), channel_factor, linestyle='-', marker='o', label=f'{ch_names[channel_index]}', alpha = 0.5, color=color)
    plt.title(f'{name} Changes Over Epochs for All Channels')
    plt.xlabel('Epoch Index')
    plt.ylabel(f'{name}')
    plt.grid(True)
    #plt.legend()  
    if pdf:
        pdf.savefig()
        plt.close()  

    else:
        plt.show() 
    
    # View factor per channel to see if anything weird is going on
    fig, axs = plt.subplots(4, 7, figsize=(20, 8))  
    axs = axs.flatten()
    for channel_index, color in zip(range(n_channels), colors):
        channel_factor = [epoch_factor[channel_index] for epoch_factor in factors]
        axs[channel_index].hist(channel_factor, bins=100, alpha=0.7, color=color) 
        axs[channel_index].set_title(f'{ch_names[channel_index]}')
        axs[channel_index].set_xlabel(f'{name}')
        axs[channel_index].set_ylabel('Frequency')
    if n_channels < (4 * 7):
        for i in range(n_channels, 4 * 7):
            fig.delaxes(axs[i])
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)  

    else:
        plt.show() 

    # Let's view the distirbution of factor, if normal we take the mean, And lets look at mean per channel on a topomap 
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(np.array(factors).ravel(), bins=20, alpha=0.7)
    axs[0].set_title(f'Histogram of All {name} Mean {np.round(np.array(factors).ravel().mean(),1)}')
    axs[0].set_xlabel(f'{name}')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True)

    ch_mean_metrics = np.zeros(n_channels)
    for channel_index in range(n_channels):
        channel_factor = [epoch_factor[channel_index] for epoch_factor in factors]
        ch_mean_metrics[channel_index] = np.mean(channel_factor)
    
    sorted_indices_reconstructed = [epochs.ch_names.index(ch) for ch in ch_names]
    pos = np.array([mne.find_layout(epochs.info).pos[i] for i in sorted_indices_reconstructed])
    im, _ = mne.viz.plot_topomap(sp.stats.zscore(ch_mean_metrics), (pos*2.4)-1.15,sensors=False, names = ch_names, res=128, sphere= 1, axes=axs[1], show=False)
    axs[1].set_title(f'Channel Z-Sc. Mean {name}')
    fig.colorbar(im, ax=axs[1], format='%0.2f')

    plt.tight_layout()
    
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)  

    else:
        plt.show() 

def plot_chan_average_metrics_heatmap(All_parameters, feature_key,ch_names, pdf=None):
    metric_type = All_parameters[feature_key]
    epochs_idx = list(metric_type.keys())
    channels_idx = list(metric_type[next(iter(epochs_idx))].keys())
    metrics_idx = list(metric_type[next(iter(epochs_idx))][next(iter(channels_idx))].keys())
    num_metrics = len(metrics_idx)
    average_metrics = np.zeros((len(channels_idx), num_metrics))

    for channel_index in channels_idx:
        for i, metric_index in enumerate(metrics_idx):
            metric_values = [metric_type[epoch][channel_index][metric_index][0] for epoch in epochs_idx]
            average_metrics[channel_index, i] = np.mean(metric_values)

    standardized_values = sp.stats.zscore(average_metrics, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    sns.heatmap(average_metrics, annot=False, cmap='viridis', ax=axs[0], xticklabels=metrics_idx, yticklabels=ch_names)
    axs[0].set_title(f'Average Metrics Across Channels for {feature_key}')
    axs[0].set_xlabel('Metrics')
    axs[0].set_ylabel('Channels')
    axs[0].tick_params(axis='x', rotation=45)

    sns.heatmap(standardized_values, annot=False, cmap='viridis', ax=axs[1], xticklabels=metrics_idx, yticklabels=ch_names)
    axs[1].set_title(f'Standardized Average Metrics Across Channels for {feature_key}')
    axs[1].set_xlabel('Metrics')
    axs[1].set_ylabel('Channels')
    axs[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)

    else:
        plt.show()

def collapse_features_epochs(All_parameters, features_keys, ch_names, pdf=None):
    all_metrics = []
    all_metric_id = []
    for key in features_keys:
        if All_parameters[key] is not  None:
            metric_type = All_parameters[key]
            epochs_idx = list(metric_type.keys())
            channels_idx = list(metric_type[next(iter(epochs_idx))].keys())
            metrics_idx = list(metric_type[next(iter(epochs_idx))][next(iter(channels_idx))].keys())
            num_metrics = len(metrics_idx)
        
            average_metrics = np.zeros((len(channels_idx), num_metrics))
        
            for channel_index in channels_idx:
                for i, metric_index in enumerate(metrics_idx):
                    metric_values = [metric_type[epoch][channel_index][metric_index][0] for epoch in epochs_idx]
                    average_metrics[channel_index, i] = np.mean(metric_values)
        
            all_metrics.append(average_metrics)
            all_metric_id.extend(metrics_idx)
        else:
            print(f"Skipping '{key}' as it is None in All_parameters.")
    
    merged_metrics = np.concatenate(all_metrics, axis=1)
    standard_metrics =  sp.stats.zscore(merged_metrics)

    # Create the heatmap with all merged metrics
    plt.figure(figsize=(10, 20))
    sns.heatmap(standard_metrics.T, annot=False, cmap='viridis', xticklabels=ch_names, yticklabels=all_metric_id)
    plt.title('Merged Average Metrics Across Channels for All Feature Keys')
    plt.xlabel('Metrics')
    plt.ylabel('Channels')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if pdf:
        pdf.savefig()
        plt.close()  
    else:
        plt.show()     
    return standard_metrics, all_metric_id


