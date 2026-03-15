# Functions for Feature extraction 

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
import ordpy
import glob
import warnings
warnings.filterwarnings("ignore")
from visualize import *
from parameter_selection import *

# Function for Fractal metrics can be used for others 
freq_bands = {'delta': (1, 4),
              'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30),
              'low-gamma': (30, 45)}

def bandpass_filter(signal, sfreq, freq_bands):
    """Applies bandpass filters to the given signal for each specified frequency band."""
    filtered_signals = {}
    for band, freq in freq_bands.items():
        filtered_signals[band] = mne.filter.filter_data(signal, sfreq=sfreq, l_freq=freq[0], h_freq=freq[1], verbose=False) 
    return filtered_signals   

def amplitude_envelope(signal):
    analytic_signal = sp.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def filter_and_envelope(signal, sfreq, freq_bands):
    """ Applies bandpass filters to the given signal for each specified frequency band and conputes amplitude envelope """
    filtered_envelopes = {}
    for band, freq in freq_bands.items():
        filtered_signal = mne.filter.filter_data(signal, sfreq, l_freq=freq[0], h_freq=freq[1], verbose=False)
        envelope = amplitude_envelope(filtered_signal)
        filtered_envelopes[band] = envelope
    return filtered_envelopes

def calculate_dfa_window_sizes(signal_length, min_points_per_window=20):
    min_scale = 4
    max_scale = signal_length // 10
    scales = np.unique(np.floor(np.logspace(np.log10(min_scale), np.log10(max_scale), num=max_scale)).astype(int))
    scales = [scale for scale in scales if scale >= signal_length // min_points_per_window]
    return scales

# Functions for Binarizing signal for Entopy and Complexity Signals
def binarize_by_mean(time_series):
    mean_value = np.mean(time_series)
    binarized_series = np.where(time_series <= mean_value, 0, 1)
    return binarized_series
    
# METHODS to extract features 
# ENTROPY FEATURES 
def Extract_Entropy(epochs, delay, dimension):
    '''
    Function to extract entropy.

    Parameters:
    epochs: mne
        The EEG mne object contains info.
    delay: int
        The time delay value for the analysis.
    dimension: int 
        The embedding dimension value for the analysis.

    Returns:
        entropy_metrics
    '''
    epoch_data = epochs.get_data()
    epoch_data = sp.stats.zscore(epoch_data, axis=2)
    
    def process_channel_entropy(epoch_index, channel_index, signal, delay, dimension):
        bin_ts = binarize_by_mean(signal)
        entropy_metrics = {
            'ShanEn': nk2.entropy_shannon(bin_ts, base=2),
            #'MaxEn': nk2.entropy_maximum(signal),
            #'DiffEn': nk2.entropy_differential(signal,base=2), # Same as Scipy, (5 methods to choose from, Vasicek, Vanes, Ebrahimi, correa, Auto is vanes)
            #'CumReEn': nk2.entropy_cumulativeresidual(signal, symbolize='mean'), # Like Diff but Cumulative distribution function not PDF
            #'PowEn': nk2.entropy_power(signal), # WIP
            #'TSEn': nk2.entropy_tsallis(signal, q=3, symbolize='mean'), # As Q=1 is shannon, as Q increases more importance on LRTC
               #'HartEn': nk2.entropy_renyi(signal, alpha=0, symbolize='mean'), 
            #'RenShanEn': nk2.entropy_renyi(signal, alpha=1, symbolize='mean'), #Should be same as ShanEn
            'RenEn': nk2.entropy_renyi(bin_ts, alpha=2),
            'AppEn': nk2.entropy_approximate(signal, delay=delay, dimension=dimension, tolerance='sd', Corrected=True),
            'SampEn': nk2.entropy_sample(signal, delay=delay, dimension=dimension, tolerance='sd'),
            #'QuadEn': nk2.entropy_quadratic(signal, delay=delay, dimension=dimension, tolerance='sd'), Funciton not implemented Cant find
            #'RangeEn': nk2.entropy_range(signal, delay=delay, dimension=dimension, tolerance='sd', approximate=False), # Function returns singular value for all
            'RateEN': nk2.entropy_rate(signal,kmax=10, symbolize='mean'), # Given k histories entropy given history, amount info to describe  slope linear fit history and joint Shannon,  signal. # 2 other metrics Excess Entorpy and Maximum Entropy rate in dictionary    
            'PEn': nk2.entropy_permutation(signal, delay=delay, dimension=dimension, corrected=True, weighted=False, conditional=False),
            'WPEn': nk2.entropy_permutation(signal, delay=delay, dimension=dimension, corrected=True, weighted=True, conditional=False),
            'CPEN': nk2.entropy_permutation(signal, delay=delay, dimension=dimension, corrected=True, weighted=False, conditional=True),
            'CWPEn': nk2.entropy_permutation(signal, delay=delay, dimension=dimension, corrected=True, weighted=True, conditional=True),
            'MSPEn': nk2.entropy_multiscale(signal, dimension=dimension, tolerance='sd', method="MSPEn"),
               #'MSCoSiEn': nk2.entropy_multiscale(signal, dimension=dimension, tolerance='sd', method="MSCoSiEn"),
            'BubbEn': nk2.entropy_bubble(signal, delay=delay, dimension=dimension, alpha=2, tolerance="sd"),
            #'SpecEn': nk2.entropy_spectral(signal, bins=None),
            'SVDEn': nk2.entropy_svd(signal, delay=delay, dimension=dimension), # Info richness based on eigenvectors
            #'KLEn': nk2.entropy_kl(signal, delay=delay, dimension=dimension, norm='euclidean'),
            'AttEn': nk2.entropy_attention(signal),
            #'EnofEn': nk2.entropy_ofentropy(signal, scale=delay, bins=10),
            #'FuzzEn': nk2.entropy_fuzzy(signal, delay=delay, dimension=dimension, tolerance='sd', approximate=False),
             # nk2.entropy_slope(signal, dimension=dimension, thresholds=[0.1,45]) # Cuesta-Frau, D. (2019). Slope entropy: A new time series complexity estimator based on both symbolic patterns and amplitude information. Entropy, 21(12), 1167.
            'DispEn': nk2.entropy_dispersion(signal,delay=delay,dimension=dimension, c=6, symbolize='NCDF'), # another method inesort use rho to calibrate sort, rostagi suggest symbol = 6 Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for time-series analysis. IEEE Signal Processing Letters, 23(5), 610-614.
            #'SyDyEn': nk2.entropy_hierarchical(signal, scale='default', dimension=dimension, tolerance='sd'), # Li, W., Shen, X., & Li, Y. (2019). A comparative study of multiscale sample entropy and hierarchical entropy and its application in feature extraction for ship-radiated noise. Entropy, 21(8), 793.
            #'AngEn': nk2.entropy_angular(signal, delay=delay, dimension=dimension), # Nardelli et al. (2022) applied to EDA signal, phase space angular distance PDF then Quadratic reny
            #'DistEn': nk2.entropy_distribution(signal,delay=delay,dimension=dimension,bins='Sturges',base=2),
            #'KolEn': nk2.entropy_kolmogorov(signal,delay=delay,dimension=dimension,tolerance='sd') 
        }
        return (epoch_index, channel_index, entropy_metrics)

    n_epochs, n_channels, _ = epoch_data.shape
    start_entropy = time.time()

    # Prepare jobs for parallel execution
    jobs = (joblib.delayed(process_channel_entropy)(epoch_index, channel_index, epoch_data[epoch_index, channel_index, :], delay, dimension) 
            for epoch_index in range(n_epochs) 
            for channel_index in range(n_channels))

    # Execute the jobs in parallel
    entropy_results = joblib.Parallel(n_jobs=-1)(jobs)

    entropy_metrics = {}
    for epoch_index, channel_index, metrics in entropy_results:
        if epoch_index not in entropy_metrics:
            entropy_metrics[epoch_index] = {}
        entropy_metrics[epoch_index][channel_index] = metrics
            
    end_time = time.time()
    duration = end_time - start_entropy
    print(f'Entropy Metric Extraction took {(duration / 60):.2f} minutes')
    
    return entropy_metrics
    
# Methods for Phase Shuffled Normalization of LZC
def phase_randomize(signal_segment):
    """Randomize the phase of a signal segment."""
    fourier_transform = np.fft.fft(signal_segment)
    amplitudes = np.abs(fourier_transform)
    random_phases = np.angle(fourier_transform)
    np.random.seed()  # Ensure randomness
    random_phases[1:-1] = np.random.uniform(0, 2*np.pi, len(signal_segment) - 2)  # Randomize phases, excluding the first (DC) and last (Nyquist for even-length signals) attempt to preserve mean value
    randomized_fourier = amplitudes * np.exp(1j * random_phases) # Above the np.random.uniform(0, 2*np.pi...) ensures each pahase is independnelty chosen from a uniform distirbution (rather than shuffle), 0-2pi covers all possible phase angles for a sinusoid
    phase_randomized_segment = np.fft.ifft(randomized_fourier)
    return phase_randomized_segment.real

def PSN_LZC(signal): #https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133532#pone.0133532.s001 from Schartner et al.
    """Compute normalized LZC using phase-randomized surrogate data."""
    analytic_signal = amplitude_envelope(signal)  # Get hilbert transform
    original_lzc = nk2.complexity_lempelziv(analytic_signal, permutation=False, symbolize='mean')[1]['Complexity_Kolmogorov'] 
    # Generate surrogates and compute their LZCs
    surrogate_lzcs = []
    for _ in range(100):  # Generate 100 phase-randomized surrogate data segments
        shuffled_signal = phase_randomize(signal)
        shuffled_analytic_signal = amplitude_envelope(shuffled_signal)
        lz = nk2.complexity_lempelziv(shuffled_analytic_signal, permutation=False, symbolize='mean')[1]['Complexity_Kolmogorov'] 
        surrogate_lzcs.append(lz)
    mean_surrogate_lzc = np.mean(surrogate_lzcs) #Get mean of surrogates
    normalized_lzc = original_lzc / mean_surrogate_lzc #Normalize by mean of surrogates
    return normalized_lzc

# COMPLEXITY FEATURES
def Extract_Complexity(epochs, delay, dimension):
    '''
    Function to extract complexity.

    Parameters:
    epochs: mne
        The EEG mne object contains info.
    delay: int
        The time delay value for the analysis.
    dimension: int 
        The embedding dimension value for the analysis.

    Returns:
        complexity_metrics
    '''
    epoch_data = epochs.get_data()
    epoch_data = sp.stats.zscore(epoch_data, axis=2)
    def process_channel_complexity(epoch_index, channel_index, signal, delay, dimension):
        analytic_signal = amplitude_envelope(signal)
        HC = list(ordpy.complexity_entropy(signal, dx=dimension, taux = delay)) # run statistical complexity
        FS = list(ordpy.fisher_shannon(signal, dx=dimension, taux=delay)) # Get fisher plane info 
        complexity_metrics = {
            'HC_LMC': [HC[1]],
            'HC_PEn': [HC[0]],
            'HC_FI': [FS[1]],
            'LZC_PSN': [PSN_LZC(signal)],
            'LZC' : nk2.complexity_lempelziv(analytic_signal, permutation=False, symbolize='mean'), 
            'PLZC' : nk2.complexity_lempelziv(analytic_signal, delay=delay, dimension=dimension, permutation=True, symbolize='mean'),
            'MSLZC' : nk2.entropy_multiscale(analytic_signal, method="LZC", permutation=False, show=False),
            'MSPLZC' : nk2.entropy_multiscale(analytic_signal, method="LZC", delay=delay, dimension=dimension, permutation=True, show=False),
            'LE' :  nk2.complexity_lyapunov(signal, delay=delay, dimension=dimension, method='rosenstein1993', separation='auto', len_trajectory = int(len(signal)*0.05)),
            # _complexity_lyapunov_rosenstein(signal, delay, dimension,sampling_rate), #nk2.complexity_lyapunov(signal, delay=delay, dimension=dimension, method='rosenstein1993', separation='auto', len_trajectory = int(len(signal)*0.05)),
            'LE2' : nk2.complexity_lyapunov(signal, delay=delay, dimension=dimension, method='makowski'),
            'Hjorth' : nk2.complexity_hjorth(signal),
            'FishInf' : nk2.fisher_information(signal, delay=delay, dimension=dimension),
            'RelRou' : nk2.complexity_relativeroughness(signal),
        }
        # Unblock if you wish to perform RQA analysis
        #rqa_results = nk2.complexity_rqa(signal, delay=delay, dimension=dimension, tolerance='sd', min_linelength=2)
        #for metric, value in rqa_results[0].items():
        #    complexity_metrics[metric] = (value.iloc[0], None) # Do this to maintain dict structure 

        return (epoch_index, channel_index, complexity_metrics)

    n_epochs, n_channels, _ = epoch_data.shape
    start_complexity = time.time()

    # Prepare jobs for parallel execution
    jobs = (joblib.delayed(process_channel_complexity)(epoch_index, channel_index, epoch_data[epoch_index, channel_index, :], delay, dimension) 
            for epoch_index in range(n_epochs) 
            for channel_index in range(n_channels))

    # Execute the jobs in parallel
    complexity_results = joblib.Parallel(n_jobs=-1)(jobs)
    
    complexity_metrics = {}
    for epoch_index, channel_index, metrics in complexity_results:
        if epoch_index not in complexity_metrics:
            complexity_metrics[epoch_index] = {}
        complexity_metrics[epoch_index][channel_index] = metrics
            
    end_time = time.time()
    duration = end_time - start_complexity
    print(f'Complexity Metric Extraction took {(duration / 60):.2f} minutes')

    return complexity_metrics

# FRACTAL FEATURES
def Extract_Fractality(epochs, delay, dimension, freq_bands):
    '''
    Function to extract fractality.
    Parameters:
    epochs: mne
        The EEG mne object contains info.
    delay: int
        The time delay value for the analysis.
    dimension: int 
        The embedding dimension value for the analysis.
    freq_bands: dict
        The specified frequency band and range. 
    Returns:
        fractality_metrics
    '''
    epoch_data = epochs.get_data()
    samp_freq = epochs.info['sfreq']
    epoch_data = sp.stats.zscore(epoch_data, axis=2)
    def process_channel_fractality(epoch_index, channel_index, signal, delay, dimension, samp_freq, freq_bands):
        fractal_metrics = {
            # Fractal Dimensions ++
            'HiguchiFD' : nk2.fractal_higuchi(signal, k_max=int(len(signal)*0.5), show=False),
            'KatzFD' : nk2.fractal_katz(signal),
            'linelength' : nk2.fractal_linelength(signal), #extension Katz 
            'PetrosianFD' : nk2.fractal_petrosian(signal, symbolize='C'),
            'Sevcik' : nk2.fractal_sevcik(signal),
            'nldFD' : nk2.fractal_nld(signal),
            #'CD': nk2.fractal_correlation(signal, delay=delay, dimension=dimension, radius=64, show=False),
            #'tMF':  nk2.fractal_tmf(signal, n=40, show=False), #Used with MFS,  !! ToDo !!
         }     
        return (epoch_index, channel_index, fractal_metrics)

    n_epochs, n_channels, _ = epoch_data.shape
    start_fractality = time.time()

    # Prepare jobs for parallel execution
    jobs = (joblib.delayed(process_channel_fractality)(epoch_index, channel_index, epoch_data[epoch_index, channel_index, :], delay, dimension, samp_freq, freq_bands) 
            for epoch_index in range(n_epochs) 
            for channel_index in range(n_channels))

    # Execute the jobs in parallel
    fractal_results = joblib.Parallel(n_jobs=-1)(jobs)

    fractal_metrics = {}
    for epoch_index, channel_index, metrics in fractal_results:
        if epoch_index not in fractal_metrics:
            fractal_metrics[epoch_index] = {}
        fractal_metrics[epoch_index][channel_index] = metrics
            
    end_time = time.time()
    duration = end_time - start_fractality
    print(f'Fractal Metric Extraction took {(duration / 60):.2f} minutes')

    return fractal_metrics

# METHODS TO CALCULATE DFA 
def concatenate_epochs(epoch_data):
    """
    Concatenate 10-second epochs into longer epochs.
    Parameters:
    epochs_data: numpy.ndarray
        3D array of shape (epochs, channels, timepoints) containing EEG data.
    Returns:
    long_epochs: list of numpy.ndarray
        List containing 2-minute epochs, each as a numpy array of shape (channels, timepoints).
    """
    # Given some of the short data we will do 1.5 min concat 
    # Calculate how many 10-second epochs form a 1.5-minute epoch # Adjust to 2 min if possible
    epochs_to_concat = 9  # 90 seconds / 10 seconds
    # Determine the total number of complete 1.5-minute epochs that can be formed
    total_epochs = epoch_data.shape[0]
    num_long_epochs = total_epochs // epochs_to_concat
    
    long_epochs = []
    for i in range(num_long_epochs):
        # Calculate start and end indices for epochs to concatenate
        start_idx = i * epochs_to_concat
        end_idx = start_idx + epochs_to_concat
        # Concatenate epochs along the timepoints axis to form a 1.5-minute epoch
        long_epoch = np.concatenate(epoch_data[start_idx:end_idx], axis=-1) # last dimension is the time points
        long_epochs.append(long_epoch)
    # If come up with clever way to handle remaining epochs FILL 
    return np.array(long_epochs)

# DFA RELATED METRICS
def Extract_DFA_Fractality(epochs, freq_bands):
    '''
    Function to extract DFA related measures (needs to concatonate signal to a longer timeseries).
    Parameters:
    epochs: mne
        The EEG mne object contains info.
    delay: int
        The time delay value for the analysis.
    dimension: int 
        The embedding dimension value for the analysis.
    freq_bands: dict
        The specified frequency band and range. 
    Returns:
        DFA_metrics
    '''
    epoch_data = epochs.get_data()
    epoch_data = sp.stats.zscore(epoch_data, axis=2)
    concat_epochs = concatenate_epochs(epoch_data)
    samp_freq = epochs.info['sfreq']
    scales = calculate_dfa_window_sizes(concat_epochs.shape[-1], min_points_per_window=20)
    #print(scales)
    def process_channel_DFAs(epoch_index, channel_index, signal, samp_freq, freq_bands):
        filtered_bands =  bandpass_filter(signal, samp_freq, freq_bands)
        envelope_bands = filter_and_envelope(signal, samp_freq, freq_bands)
        envelope_signal = amplitude_envelope(signal)
        # Hurst Self-Similarity related measures
        dfa_metrics = {}

        bands_signals = {
            'dfa': envelope_signal,
            'dfaDelta': envelope_bands['delta'],
            'dfaTheta': envelope_bands['theta'],
            'dfaAlpha': envelope_bands['alpha'],
            # Include additional bands as needed
        }
        for metric, signal in bands_signals.items():
            try:
                dfa_metrics[metric] = nk2.fractal_dfa(signal, scale=scales, q=2)
            except Exception as e:
                print(f"Error calculating {metric}: {e}")
                dfa_metrics[metric] = np.nan

        MFDFA = nk2.fractal_dfa(envelope_signal, scale=scales,q=[-5, -3, -1, 0, 1, 3, 5], multifractal=True, show=False)
        for metric in MFDFA[0].keys():
            dfa_metrics[f'{metric}'] = (MFDFA[0][metric].iloc[0], [0]) # Can MFDFA on Delta, Theta and try to get ALpha.

        return (epoch_index, channel_index, dfa_metrics)

    n_epochs, n_channels, _ = concat_epochs.shape
    start_dfa = time.time()

    # Prepare jobs for parallel execution
    jobs = (joblib.delayed(process_channel_DFAs)(epoch_index, channel_index, concat_epochs[epoch_index, channel_index, :], samp_freq, freq_bands) 
            for epoch_index in range(n_epochs) 
            for channel_index in range(n_channels))

    # Execute the jobs in parallel
    dfa_results = joblib.Parallel(n_jobs=-1)(jobs)

    dfa_metrics = {}
    for epoch_index, channel_index, metrics in dfa_results:
        if epoch_index not in dfa_metrics:
            dfa_metrics[epoch_index] = {}
        dfa_metrics[epoch_index][channel_index] = metrics

    end_time = time.time()
    duration = end_time - start_dfa
    print(f'DFA Metric Extraction took {(duration / 60):.2f} minutes')

    return dfa_metrics
# Function for Power Metrics
def convert_db_to_linear(psd_db):
    psd_linear = 10 ** (psd_db / 10)
    return psd_linear

# Extract Power Metrics
def Extract_Power(epochs, delay, dimension, freq_bands):
    '''
    Function to extract power.

    Parameters:
    epochs: mne
        The EEG mne object contains info.
    delay: int
        The time delay value for the analysis.
    dimension: int
        The embedding dimension value for the analysis.
    freq_bands: dict
        The specified frequency band and range.
    Returns:
        power_metrics
    '''
    epoch_data = epochs.get_data()
    epoch_data = sp.stats.zscore(epoch_data, axis=2)
    # Param for FoooF
    h_pass = epochs.info['highpass']
    l_pass = epochs.info['lowpass']
    freq_range = [h_pass, l_pass]
    samp_freq = epochs.info['sfreq']
    n_fft = int(samp_freq)
    #freq_res = samp_freq / epoch_data.shape[-1]

    def process_channel_power(epoch_index, channel_index, signal, delay, dimension, samp_freq, freq_bands, h_pass, l_pass):
        # mne_sig = mne.io.RawArray(signal[np.newaxis, :], mne.create_info(ch_names=['ch'], sfreq=samp_freq, ch_types='eeg'), verbose=False)

        psds, freqs = mne.time_frequency.psd_array_welch(signal, sfreq=samp_freq, fmin=h_pass, fmax=l_pass, n_fft=n_fft, verbose=False)
        freq_res = freqs[1]-freqs[0]
        fm = fooof.FOOOF(peak_width_limits=[freq_res, 12], min_peak_height=0.1, verbose=False) # Default is 0.5-12 maybe try thinner? 1,12?   
        fm.fit(freqs, psds, freq_range)

        power_metrics = {
            # Power Spectral Slope
            'PSDslope' : nk2.fractal_psdslope(signal, method='voss1988', show=False),
            'FOOOF_Slope' :   [fm.get_params('aperiodic_params')[1]],
            'FOOOF_Offset' :  [fm.get_params('aperiodic_params')[0]],
         }
        #filtered_signals = bandpass_filter(signal, n_fft, freq_bands)
        # psds = np.array([convert_db_to_linear(i) for i in psds]) # need to transform  DB to voltz to get proper area under curve
        # psd_welch, returns in linear do not need to switch.
        #band_powers = {}
        total_power = sp.integrate.simps(psds,dx=freq_res)

        for band, (low,high) in freq_bands.items():
            #mne_sig = mne.io.RawArray(filtered_signal[np.newaxis, :], mne.create_info(ch_names=['ch'], sfreq=samp_freq, ch_types='eeg'), verbose=False)
            #psds, freqs = mne.time_frequency.psd_array_welch(filtered_signal, sfreq=samp_freq, fmin=freq_bands[band][0], fmax=freq_bands[band][1], n_fft=n_fft, verbose=False)
            # https://raphaelvallat.com/bandpower.html#:~:text=In%20order%20to%20compute%20the,signal%2C%20with%20or%20without%20overlapping. (Using this method)
            # Integral of the power spectral density
            idx_band = np.logical_and(freqs >=low, freqs <= high)
            band_power = sp.integrate.simps(psds[idx_band], dx=freq_res)
            # band_powers[band] = band_power
            relative_power = band_power/total_power # Calculate relative powers
            # Absolute power in Db for better interpretability and vis (muV2/Hz is the actual power)# absolute_power_db = band_power #np.abs(10 * np.log10(band_power)) # Put back to  DB if wanted (poor for area)

            power_metrics[f'|{band}|Power'] = [band_power] #[10*np.log10(band_power)] #put back to DB
            power_metrics[f'{band}_RelPower'] = [relative_power]

        return (epoch_index, channel_index, power_metrics)

    n_epochs, n_channels, _ = epoch_data.shape
    start_power = time.time()

    # Prepare jobs for parallel execution
    jobs = (joblib.delayed(process_channel_power)(epoch_index, channel_index, epoch_data[epoch_index, channel_index, :], delay, dimension,  samp_freq, freq_bands, h_pass, l_pass) 
            for epoch_index in range(n_epochs)
            for channel_index in range(n_channels))

    # Execute the jobs in parallel
    power_results = joblib.Parallel(n_jobs=-1)(jobs)

    power_metrics = {}
    for epoch_index, channel_index, metrics in power_results:
        if epoch_index not in power_metrics:
            power_metrics[epoch_index] = {}
        power_metrics[epoch_index][channel_index] = metrics

    end_time = time.time()
    duration = end_time - start_power
    print(f'Power Metric Extraction took {(duration / 60):.2f} minutes')

    return power_metrics

# Post analysis Imputation and correlations
def impute_nan_inf_with_channel_mean(metrics):
    '''
    Imputes NaN and inf values in metrics with the mean of the respective channel.
    Parameters:
    metrics: dict
        The dictionary containing metrics for each epoch and channel.
    Returns:
    dict
        The updated metrics with NaN and inf values imputed.
    '''
    for epoch_index in metrics:
        for channel_index in metrics[epoch_index]:
            channel_metrics = metrics[epoch_index][channel_index]
            for metric in channel_metrics:
                check_value = channel_metrics[metric] if np.isscalar(channel_metrics[metric]) else channel_metrics[metric][0]
                if np.isnan(check_value) or np.isinf(check_value):
                    # Gather values for the metric across all epochs, ensuring the metric exists
                    values = [metrics[ep][channel_index][metric] if np.isscalar(metrics[ep][channel_index][metric]) else metrics[ep][channel_index][metric][0] for ep in metrics if metric in metrics[ep][channel_index]]
                    channel_mean = np.nanmean([value for value in values if not np.isnan(value) and not np.isinf(value)])
                    channel_metrics[metric] = (channel_mean, channel_metrics[metric][1]) if not np.isscalar(channel_metrics[metric]) else channel_mean

    return metrics

def check_and_report_nan_inf(metrics):
    '''
    Checks for NaNs and infs in metrics and reports their count and locations.
    Parameters:
    metrics: dict
        The dictionary containing metrics for each epoch and channel.
    '''
    nan_inf_counts = {}
    for epoch_index in metrics:
        for channel_index in metrics[epoch_index]:
            channel_metrics = metrics[epoch_index][channel_index]
            for metric, value in channel_metrics.items():
                # Check is a scalar or array
                check_value = value if np.isscalar(value) else value[0]
                if np.isnan(check_value) or np.isinf(check_value):
                    nan_inf_counts[(epoch_index, channel_index, metric)] = nan_inf_counts.get((epoch_index, channel_index, metric), 0) + 1

    total_nan_inf = sum(nan_inf_counts.values())
    print(f"Total NaNs/Infs found: {total_nan_inf}")
    for location, count in nan_inf_counts.items():
        print(f"NaNs/Infs in Epoch {location[0]}, Channel {location[1]}, Metric {location[2]}: {count}")

def chan_metric_correlations(standard_metrics,all_metric_id, ch_names, pdf=None):
    corr_metrics = np.corrcoef(standard_metrics.T)
    corr_chan = np.corrcoef(standard_metrics)
    
    mask_met = np.triu(np.ones_like(corr_metrics, dtype=bool))
    mask_ch = np.triu(np.ones_like(corr_chan, dtype=bool))
    
    plt.figure(figsize=(20,15))
    sns.heatmap(corr_metrics, annot=False, cmap='coolwarm',mask=mask_met, xticklabels=all_metric_id, yticklabels=all_metric_id)
    plt.title("Correlation Matrix Across Metrics")
    if pdf:
        pdf.savefig()
        plt.close()  
    else:
        plt.show() 
    
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr_chan, annot=False, cmap='coolwarm', mask = mask_ch, xticklabels=ch_names, yticklabels=ch_names)
    plt.title("Correlation Matrix Across Channels")
    if pdf:
        pdf.savefig()
        plt.close()  
    else:
        plt.show() 

