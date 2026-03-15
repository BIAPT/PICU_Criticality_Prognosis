# Main function to perform complexity analysis on neural data
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
from visualize import *
from parameter_selection import *
from feature_analysis import *
import warnings
warnings.filterwarnings("ignore")

def Complexity_Feature_Extraction(source, save=True, out_dir=None, condition = None, standardize=True, impute = False, inspection=True, converge=False,
                                  entropy=True, complexity=True, fractal=True,dfa=True, power=True, delay = None, dimension = None, 
                                  verbose=True, PDF=True):
    '''
    Function to extract complexity features.

    Parameters:
    file_path: str
        The input file path to the EEG data from which to extract complexity features.
    save: bool, optional
        If True, saves the output results in the specified output directory. 
        Default is True.
    out_dir: str, optional
        The directory where the output files will be saved. This parameter 
        is required if 'save' is set to True. Default is None.
    inspection: bool, optional
        If True, provides descriptive statistics for the data prior to parameter 
        or feature selection. Default is True.
    entropy: bool, optional
        If True, performs entropy feature extraction. Default is True.
    complexity: bool, optional
        If True, performs complexity feature extraction. Default is True.
    fractal: bool, optional
        If True, performs fractal feature extraction. Default is True.
    dfa: bool, optional
        If True, performs DFA feature Extraction. Default is True.
    power: bool, optional
        If True, performs power feature extraction. Default is True.
    delay: int or None, optional
        The time delay value for the analysis. If None, optimal time delay will be 
        estimated using the Fraser (1986) method. Default is None.
    dimension: int or None, optional
        The embedding dimension value for the analysis. If None, optimal embedding dimension will be 
        estimated using the Cao (1997) method. Default is None. 
    verbose:  bool, optional
        If True, prints detailed results and progress updates during processing. 
        Default is True.
    pdf: bool, optional
        If True, saves generated figures and descriptive outputs as PDF files 
        in the output directory. Default is True.
    
    Returns:
    All_features, All_parameters, All_labels
    Example
    >>> Complexity_Feature_Extraction('path/to/eeg_data.fif', save=True, 
                                      out_dir='path/to/output', inspection=True, 
                                      verbose=True, pdf=True)
    '''

    # 
    if PDF == True:
        if out_dir is not None:
            pdf = PdfPages(f'{out_dir}/complexity_analysis_{condition}.pdf')
        else:
            pdf = PdfPages('complexity_analysis.pdf')
    else:
        pdf = None

    if isinstance(source, str):
        epochs = mne.read_epochs(source, preload=True, verbose=False)
    elif isinstance(source, mne.Epochs):
        epochs = source
    else:
        raise ValueError("source must be a filepath or an mne.Epochs object")

    start_time = time.time()
    # Load Data:
    epoch_data = epochs.get_data()
    epoch_shape = epoch_data.shape
    samp_freq = epochs.info['sfreq']
    nyquist = samp_freq/2
    h_pass = epochs.info['highpass']
    l_pass = epochs.info['lowpass']
    bad_ch = epochs.info['bads']
    all_ch = epochs.ch_names
    good_ch = [chan for chan in all_ch if chan not in bad_ch]
    n_epochs, n_channels, n_samples= epoch_data.shape

    all_delay_info = None
    all_dim_info = None
    freq_bands = {'delta': (1, 4),
                  'theta': (4, 8),
                  'alpha': (8, 13),
                  'beta': (13, 30),
                  'low-gamma': (30, 45)}
    
    # Data Inspection: First data descriptives, assumption checks 
    if verbose:
        epoch_duration = np.round(((1 / samp_freq) * epoch_data.shape[-1]), 2)
        total_duration_seconds = np.round(epoch_duration * epoch_data.shape[0], 2)
        total_duration_minutes = np.round(total_duration_seconds / 60, 2)
        text = f'######################################## Data Inspection ######################################## \nYour data has: \nShape: {epoch_shape} (n_epochs, n_channels, n_times) \nSampling Frequency: {samp_freq} Hz and Nyquist Frequency: {nyquist} \nFiltered between {h_pass} Hz and {l_pass} Hz \nEpoch Time: {epoch_duration} seconds \nTotal Time: {total_duration_seconds} secs. = {total_duration_minutes} mins.'
        if isinstance(source, str):
            source_text = f'\nYour file is being processed.\nFile: {source}'
            text += source_text
    
        if not bad_ch:
            nobad_text = f'\nNo bad channels detected, you have {len(good_ch)} EEG channels'
            text += nobad_text
        else:
            bad_text = f'\nOf {len(all_ch)} EEG channels:\nBad channels detected: {bad_ch}\nGood Channels detected: {good_ch}'
            text += bad_text
        print(text)
        if PDF:
            fig = plt.figure(figsize=(8, 11))
            plt.axis('off')
            plt.text(0.5, 0.9, text, fontsize=8, va='top', ha='center', family='monospace', wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
            
    # Sort the channels alphabetically by their index
    sorted_indices = np.argsort([ch[0] + ch[1:] for ch in all_ch])
    epoch_data = epoch_data[:, sorted_indices, :]
    ch_names = [epochs.ch_names[i] for i in sorted_indices]
    print(f'Channel order: {ch_names}')
    
    if standardize:
        epoch_data = sp.stats.zscore(epoch_data, axis=2)

    if inspection:
        # Inspect Channel timeseries
        inspect_channels(epoch_data, ch_names, pdf)
    
        # Visualize psd only if MNE object 
        visualize_psd(epochs, pdf) # add non mne PSD function
    
        end = time.time()
        duration = end-start_time
        print(f'Data inspection took {duration:.2f} seconds = {(duration/60):.2f} minutes')
        
    # Can inspect the ACF and MI over lags for vsual inspection of best Time Delay Use lag of first ACF lag at 0 x2 for the max lag. as a heuristic 
     
    print(f'###################################### Parameter Selection ######################################')
    start_p = time.time()

    if delay is None:
        if converge:
            print(f'Finding the Optimal Time Delay using Fraser (1986) method with a convergence threshold:')
            mean_delay = []
            current_delays = []
            convergence_threshold = 0.1  
            total_epochs = epoch_data.shape[0]
            shuffled_indices = np.random.permutation(total_epochs)
            consecutive_converged = 0   
            
            for segment in range(total_epochs):
                random_index = shuffled_indices[segment]
                new_delays, _ = delay_selection(epoch_data[random_index:random_index+1, :, :])
                current_delays.extend(np.array(new_delays).ravel())
                mean = np.round(np.mean(current_delays), 2)
                mean_delay.append(mean)
                # Convergence check
                if segment > 0 and abs(mean_delay[segment] - mean_delay[segment - 1]) < convergence_threshold:
                    consecutive_converged += 1
                    if consecutive_converged >= 3: 
                        all_mean = np.array(mean_delay).ravel().mean()
                        delay = int(np.round(all_mean))
                        print(f'Time delay determined: {delay}. Convergence reached at segment {segment}, after {mean_delay}')  
                        all_delays, all_delay_info = [delay, mean_delay]
                        break
                else:
                    consecutive_converged = 0
        else:
            print(f'Finding the Optimal Time Delay using Fraser (1986) method:')
            # Calculate the mean Delay 
            all_delays, all_delay_info = delay_selection(epoch_data)
            # visualize the selected delay 
            visualize_factor(all_delays, "Delays",  epochs, ch_names, pdf)
            # Mean delay will be our chosen delay
            delay = int(np.round(np.mean(np.array(all_delays).ravel())))
            print(f'Optimal time delay determined by every epoch is {delay}')
    
            end_time = time.time()
            duration = end_time - start_p
            print(f'Time Delay Optimization took {duration:.2f} seconds = {(duration/60):.2f} minutes')

    else: # Delay is given 
        print(f'Selected time delay for every epoch is {delay}, are you sure you do not want to optimize delay selection?')
        all_delays, all_delay_info = [delay, delay]

    if dimension is None:
        start_d = time.time()

        if converge:
            print(f'Finding the Optimal Embedding Dimension using Cao (1997) method with convergence threshold:')
            mean_dim = []
            current_dim = []
            convergence_threshold = 0.1  
            total_epochs = epoch_data.shape[0]
            shuffled_indices = np.random.permutation(total_epochs)
            consecutive_converged = 0   

            for segment in range(total_epochs):
                random_index = shuffled_indices[segment]
                new_dims, _ = dimension_selection(epoch_data[random_index:random_index+1, :, :], delay)
                current_dim.extend(np.array(new_dims).ravel())
                mean = np.round(np.mean(current_dim), 2)
                mean_dim.append(mean)
                # Convergence check
                if segment > 0 and abs(mean_dim[segment] - mean_dim[segment - 1]) < convergence_threshold:
                    consecutive_converged += 1
                    if consecutive_converged >= 3: 
                        all_mean = np.array(mean_dim).ravel().mean()
                        dimension = int(np.round(all_mean))
                        print(f'Embedding dimension determined: {dimension}. Convergence reached at segment {segment}, after {mean_dim}') 
                        all_dim, all_dim_info = [dimension,mean_dim]
                        break
                else:
                    consecutive_converged = 0
        else:
            print(f'Finding the Optimal Embedding Dimension using Cao (1997) method:')
            # Embedding Dimension Optimization:
            all_dim, all_dim_info = dimension_selection(epoch_data, delay)
            #Visualize the embedding dimensions
            visualize_factor(all_dim, "Dim.",  epochs,  ch_names, pdf)
            # Mean dimension will be our chosen dim
            dimension = int(np.mean(np.array(all_dim).ravel()))
            print(f'Optimal embedding dimension determined by every epoch is {dimension}... Holy Cao!')

            end_time = time.time()
            duration = end_time - start_d
            print(f'Dimension Embedding estimation using Cao Method took {(duration/60):.2f} minutes')

    else:
        print(f'Selected embedding dimension for every epoch is {dimension}, are you sure you do not want to optimize dimension selection?')
        all_dim, all_dim_info = [dimension, dimension]

    end_time = time.time()
    durs = end_time-start_p
    print(f'Total Parameter Selection Optimization took {(durs/60):.2f} minutes')

    ##### ADD attractor reconstruction HERE!!!

    # Feature Selection
    print(f'####################################### Feature Selection #######################################')
    #We will analyze Entropy Metrics, Complexity Metrics, and Fractal Metrics 
    print(f'Entering pandemonium... Let the chaos begin...')  

    # Initialize the outputs, for selection purposes
    entropy_metrics = None
    complexity_metrics = None
    fractal_metrics = None
    dfa_metrics = None
    power_metrics = None

    all_entropies = None
    all_comp = None
    all_frac = None
    all_dfa = None
    all_pow = None

    entropies = None
    complexities = None
    fractalities = None
    dfass = None
    powerss = None

    # ENTROPY
    if entropy:
        print(f'######################################## Entropy Metrics ########################################')
        print(f'Damn Shannon! How Uncertain are we? Hold your Boltzes!')

        entropy_metrics = Extract_Entropy(epochs, delay, dimension)

        # Check and report NaNs and Infs
        check_and_report_nan_inf(entropy_metrics)

        # Impute NaNs and Infs
        if impute:
            entropy_metrics = impute_nan_inf_with_channel_mean(entropy_metrics)

        entropies = list(entropy_metrics[0][0].keys())
        # Make this conditional so that can vis or not and pdf
        for metric in entropies:
            all_entropies = []
            for epoch_key in entropy_metrics.keys():
                epoch = entropy_metrics[epoch_key]
                epoch_entropies = []
                for chan_key in epoch.keys():
                    chan = epoch[chan_key]
                    entropy_value = chan[metric][0]
                    epoch_entropies.append(entropy_value)
                all_entropies.append(epoch_entropies)
            visualize_factor(all_entropies, metric,  epochs, ch_names, pdf)

        print(f'Oh Gibbs, I Carnot believe, it is getting mixxedupness in here, it is irreversible!')

    # COMPLEXITY
    if complexity:
        print(f'###################################### Complexity Metrics #######################################')
        print(f'Have we emerged yet? I dont feel so stable... should we Bak up?')

        complexity_metrics = Extract_Complexity(epochs, delay, dimension)

        # Check and report NaNs and Infs
        check_and_report_nan_inf(complexity_metrics)

        # Impute NaNs and Infs
        if impute:
            complexity_metrics = impute_nan_inf_with_channel_mean(complexity_metrics)

        complexities = list(complexity_metrics[0][0].keys())

        for metric in complexities:
            all_comp = []
            for epoch_key in complexity_metrics.keys():
                epoch = complexity_metrics[epoch_key]
                epoch_comp = []
                for chan_key in epoch.keys():
                    chan = epoch[chan_key]
                    comp_value = chan[metric][0]
                    #print(comp_value)
                    epoch_comp.append(comp_value)
                all_comp.append(epoch_comp)
            visualize_factor(all_comp, metric,  epochs, ch_names, pdf)

        print(f'We are in the depths of chaos, but there is a method to this madness! (Just Yorke-ing around), Lets double down, Feigenbaum style.')
    # FRACTALITY 
    if fractal: 
        print(f'######################################## Fractal Metrics ########################################')
        print(f'MANdelbrot, were getting closer but everything looks similar... I CANTor believe it! As Poincare said, lets inspect these monsters, dont get lost!')

        fractal_metrics = Extract_Fractality(epochs, delay, dimension, freq_bands)

        # Check and report NaNs and Infs
        check_and_report_nan_inf(fractal_metrics)

        # Impute NaNs and Infs
        if impute:
            fractal_metrics = impute_nan_inf_with_channel_mean(fractal_metrics)

        fractalities = list(fractal_metrics[0][0].keys())

        # Visualize metrics
        for metric in fractalities:
            all_frac = []
            for epoch_key in fractal_metrics.keys():
                epoch = fractal_metrics[epoch_key]
                epoch_frac = []
                for chan_key in epoch.keys():
                    chan = epoch[chan_key]
                    frac_value = chan[metric][0]
                    epoch_frac.append(frac_value)
                all_frac.append(epoch_frac)
            visualize_factor(all_frac, metric,  epochs, ch_names, pdf)

    # DFA
    if dfa:
        print(f'######################################### DFA Metrics #########################################')
        print(f'We had to concatonate, it is a Peng and that Hursts!!!')

        dfa_metrics = Extract_DFA_Fractality(epochs, freq_bands)

        # Check and report NaNs and Infs
        check_and_report_nan_inf(dfa_metrics)

        # Impute Nans and Infs
        if impute:
            dfa_metrics = impute_nan_inf_with_channel_mean(dfa_metrics)

        dfass = list(dfa_metrics[0][0].keys())
        for metric in dfass:
            all_dfa = []
            for epoch_key in dfa_metrics.keys():
                epoch = dfa_metrics[epoch_key]
                epoch_dfa = []
                for chan_key in epoch.keys():
                    chan = epoch[chan_key]
                    dfa_value = chan[metric][0]
                    epoch_dfa.append(dfa_value)
                all_dfa.append(epoch_dfa)
            visualize_factor(all_dfa, metric,  epochs, ch_names, pdf, dfa=True)

    # POWER 
    if power:
        print(f'######################################### Power Metrics #########################################')
        print(f'Fourier we are getting close! Have we PARSEval-ed all this complexity yet? Oh VOYtek, this will be the Wiener!!!')

        power_metrics = Extract_Power(epochs, delay, dimension, freq_bands)

        # Check and report NaNs and Infs
        check_and_report_nan_inf(power_metrics)

        # Impute NaNs and Infs
        if impute:
            power_metrics = impute_nan_inf_with_channel_mean(power_metrics)

        powerss = list(power_metrics[0][0].keys())

        for metric in powerss:
            all_pow = []
            for epoch_key in power_metrics.keys():
                epoch = power_metrics[epoch_key]
                epoch_pow = []
                for chan_key in epoch.keys():
                    chan = epoch[chan_key]
                    pow_value = chan[metric][0]
                    epoch_pow.append(pow_value)
                all_pow.append(epoch_pow)
            visualize_factor(all_pow, metric,  epochs, ch_names, pdf)

    All_parameters = {
        'Delays' : all_delay_info,
        'Dimensions' : all_dim_info,
        'Entropy' : entropy_metrics,
        'Complexity': complexity_metrics,
        'Fractal' : fractal_metrics,
        'DFA': dfa_metrics,
        'Power' : power_metrics
    }

    feature_keys = ['Entropy','Complexity','Fractal','DFA','Power']

    for key in feature_keys:
        if All_parameters[key] is not None:
            plot_chan_average_metrics_heatmap(All_parameters, key, ch_names, pdf)

    standardized_metrics, all_metric_ids = collapse_features_epochs(All_parameters, feature_keys, ch_names, pdf)

    chan_metric_correlations(standardized_metrics, all_metric_ids, ch_names, pdf)

    end_time = time.time()
    duration = end_time-start_time

    print(f'Complete Complexity feature extraction took {duration:.2f} seconds = {(duration/60):.2f} minutes')
    print(f'Order has emerged from the randomnsss, spontaneous!')

    if PDF:
        pdf.close()

    results = {'Parameters': All_parameters,
               'Ch_names': ch_names,
               'Standardized Features': standardized_metrics}

    if save:
        save_file_name = "complex_results.pkl"
        if out_dir:
            save_file_path = os.path.join(out_dir, save_file_name)
            if condition:
                save_file_name = f"{condition}_complex_results.pkl"
                save_file_path = os.path.join(out_dir, save_file_name)
        else:
            save_file_path = save_file_name
            if condition:
                save_file_path = f"{condition}_complex_resultss.pkl"

        with open(save_file_path, "wb") as file:
            pickle.dump(results, file)

        print(f"Results saved to {save_file_path}")

    return   results

