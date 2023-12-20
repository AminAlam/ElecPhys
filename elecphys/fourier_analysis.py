import os
import numpy as np
from scipy import signal
from tqdm import tqdm
import json

import utils
import cfc


def stft_numeric_output_from_npz(input_npz_folder, output_npz_folder, window_size, overlap, window_type):
    """ Computes STFT and saves results as NPZ files
        input:
        input_npz_folder: path to input npz folder - type: os.PathLike
        output_npz_folder: path to output npz folder to save STFT results - type: os.PathLike
        window_size: window size in seconds - type: float
        overlap: overlap in seconds - type: float
        window_type: window type - type: str
        output:    
    """

    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    print(f'Computing STFT with window size {window_size} seconds, overlap {overlap} seconds, and window type {window_type}...')
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            f, t, Zxx = stft_from_array(data, fs, window_size, overlap, window_type)
            stft_npz_file_path = os.path.join(output_npz_folder, f'STFT_{npz_file}')
            np.savez(stft_npz_file_path, f=f, t=t, Zxx=Zxx)


def dft_numeric_output_from_npz(input_npz_folder, output_npz_folder, nfft=None):
    """ Computes DFT and saves results as NPZ files
        input:
        input_npz_folder: path to input npz folder - type: os.PathLike
        output_npz_folder: path to output npz folder to save DFT results - type: os.PathLike
        output:    
    """

    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    print(f'Computing DFT...')
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            f, Zxx = dft_from_array(data, fs)
            dft_npz_file_path = os.path.join(output_npz_folder, f'DFT_{npz_file}')
            np.savez(dft_npz_file_path, f=f, Zxx=Zxx)


def stft_from_array(signal_array, fs, window_size, overlap, window_type='hann', nfft=None):
    """ Computes STFT from 1D array
        input:
            signal_array: 1D array of signal - type: np.ndarray
            fs: sampling frequency - type: float
            window_size: window size in seconds - type: float
            overlap: overlap in seconds - type: float
            window_type: window type - type: str
            nfft: number of FFT points - type: int
        output:
            f: frequency vector - type: np.ndarray
            t: time vector - type: np.ndarray
            Zxx: STFT matrix - type: np.ndarray
    """
    if np.ndim(signal_array) == 2:
        signal_array = np.squeeze(signal_array)
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')
    window_length = int(window_size*fs)
    overlap_length = int(overlap*fs)
    if nfft is None:
        nfft = 2*window_length
    if 'kaiser' in window_type:
        window_type = signal.windows.kaiser(window_length, float(window_type.split(' ')[1]))
    [f, t, Zxx] = signal.stft(signal_array, fs, window=window_type, nperseg=window_length, noverlap=overlap_length, nfft=nfft)
    return f, t, Zxx


def dft_from_array(signal_array, fs, nfft=None):
    """ Computes DFT from 1D array
        input:
            signal_array: 1D array of signal - type: np.ndarray
            fs: sampling frequency - type: float
            nfft: number of FFT points - type: int
        output:
            f: frequency vector - type: np.ndarray
            Zxx: DFT vector - type: np.ndarray
    """
    if np.ndim(signal_array) == 2:
        signal_array = np.squeeze(signal_array)
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')
    if nfft is None:
        nfft = len(signal_array)
    f = np.linspace(0, fs, nfft)
    Zxx = np.fft.fft(signal_array, nfft)
    return f, Zxx


def butterworth_filtering_from_array(signal_array, fs: int, _args: dict):
    """ Filters signal array
        input:
            signal_array: 1D array of signal - type: np.ndarray
            fs: sampling frequency - type: float
            _args: dictionary containing filter parameters - type: dict
        output:
            signal_array: filtered signal array - type: np.ndarray
            filter_freq_response: dictionary containing filter frequency response - type: dict
    """
    
    if np.ndim(signal_array) == 2:
        signal_array = np.squeeze(signal_array)
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')
    if _args['filter_type'] == 'LPF':
        b, a = signal.butter(_args['filter_order'], int(_args['freq_cutoff']) / (fs / 2), btype='lowpass')
    elif _args['filter_type'] == 'HPF':
        b, a = signal.butter(_args['filter_order'], int(_args['freq_cutoff']) / (fs / 2), btype='highpass')
    elif _args['filter_type'] == 'BPF':
        _args['freq_cutoff'] = utils.convert_string_to_list(_args['freq_cutoff'])
        _args['freq_cutoff'] = [int(i) for i in _args['freq_cutoff']]
        b, a = signal.butter(_args['filter_order'], [_args['freq_cutoff'][0] / (fs / 2), _args['freq_cutoff'][1] / (fs / 2)], btype='bandpass')
    else:
        raise ValueError(f'Invalid filter type: {_args["filter_type"]}. Must be "LPF", "HPF", or "BPF"')
    
    signal_array = signal.filtfilt(b, a, signal_array)
    return signal_array


def butterworth_filtering_from_npz(input_npz_folder, output_npz_folder, _args: dict):
    """ Filters signal array
        input:
            input_npz_folder: path to input npz folder - type: os.PathLike
            output_npz_folder: path to output npz folder to save filtered results - type: os.PathLike
            fs: sampling frequency - type: float
            _args: dictionary containing filter parameters - type: dict
        output:
            output_npz_folder: no return value
    """
    
    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            data = butterworth_filtering_from_array(data, fs, _args)
            filtered_npz_file_path = os.path.join(output_npz_folder, f'Filtered_{npz_file}')
            np.savez(filtered_npz_file_path, data=data, fs=fs)

    _args['fs'] = int(fs)
    filter_freq_response_json_file_path = os.path.join(output_npz_folder, f'filter_freq_response.json')

    with open(filter_freq_response_json_file_path, 'w') as fp:
        json.dump(_args, fp)


def calc_freq_response(_args):
    """ Calculates filter frequency response
        input:
            _args: dictionary containing filter parameters - type: dict
        output:
            f: frequency vector - type: np.ndarray
            mag: magnitude vector - type: np.ndarray
            phase: phase vector - type: np.ndarray
    """
    
    fs = _args['fs']
    if _args['filter_type'] == 'LPF':
        b, a = signal.butter(_args['filter_order'], int(_args['freq_cutoff'])/ (fs / 2), btype='lowpass')
    elif _args['filter_type'] == 'HPF':
        b, a = signal.butter(_args['filter_order'], int(_args['freq_cutoff']) / (fs / 2), btype='highpass')
    elif _args['filter_type'] == 'BPF':
        _args['freq_cutoff'] = utils.convert_string_to_list(_args['freq_cutoff'])
        _args['freq_cutoff'] = [int(i) for i in _args['freq_cutoff']]
        b, a = signal.butter(_args['filter_order'], [_args['freq_cutoff'][0] / (fs / 2), _args['freq_cutoff'][1] / (fs / 2)], btype='bandpass')
    else:
        raise ValueError(f'Invalid filter type: {_args["filter_type"]}. Must be "LPF", "HPF", or "BPF"')
    
    w, h = signal.freqz(b, a, worN=2000)
    f = w / (2 * np.pi) * fs
    mag = 20 * np.log10(abs(h))
    phase = np.unwrap(np.angle(h)) * 180 / np.pi
    
    return f, mag, phase, _args


def calc_cfc_from_array(signal_array, fs: int, freqs_amp: list, freqs_phase:list, time_interval: list=None):
    """ Calculates CFC matrix
        input:
            signal_array: 1D array of signal - type: np.ndarray
            fs: sampling frequency - type: float
            freqs_amp: amplitude frequencies - type: list
            freqs_phase: phase frequencies - type: list
            time_interval: time interval to calculate CFC over in seconds- type: list
        output:
            MI_mat: CFC matrix - type: np.ndarray
    """
    if np.ndim(signal_array) == 2:
        signal_array = np.squeeze(signal_array)
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')

    if time_interval is None:
        time_interval = [0, len(signal_array)/fs]
    signal_array = signal_array[int(time_interval[0]*fs):int(time_interval[1]*fs)]
    MI_mat = cfc.calc_tf_mvl(signal_array, fs, freqs_phase, freqs_amp)
    return MI_mat


def calc_cfc_from_npz(input_npz_folder, output_npz_folder, freqs_amp: list, freqs_phase:list, time_interval: list=None):
    """ Calculates CFC matrix
        input:
            input_npz_folder: path to input npz folder - type: os.PathLike
            output_npz_folder: path to output npz folder to save CFC results - type: os.PathLike
            freqs_amp: amplitude frequencies - type: list
            freqs_phase: phase frequencies - type: list
            time_interval: time interval to calculate CFC over in seconds- type: list
        output:
            output_npz_folder: no return value
    """
    
    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            MI_mat = calc_cfc_from_array(data, fs, freqs_amp, freqs_phase, time_interval)
            cfc_npz_file_path = os.path.join(output_npz_folder, f'CFC_{npz_file}')
            np.savez(cfc_npz_file_path, MI_mat=MI_mat, freqs_amp=freqs_amp, freqs_phase=freqs_phase, time_interval=time_interval)