import os
import numpy as np
from scipy import signal
from tqdm import tqdm

def apply_notch(_signal_chan, _args):
    """ Applies notch filter to given signal
    input:
        _signal_chan: signal channel - type: numpy.ndarray
        _args: dictionary containing notch filter parameters - type: dict
    output:
        _signal_chan: signal channel with notch filter applied - type: numpy.ndarray
    """
    for f0 in np.arange(_args['f0'],300,_args['f0']):
        b_notch, a_notch = signal.iirnotch(f0, _args['Q'], _args['fs'])
        _signal_chan = signal.filtfilt(b_notch, a_notch, _signal_chan)
    return _signal_chan

def zscore_normalize_npz(input_npz_folder, output_npz_folder):
    """ Z-score normalizes NPZ files
    input:
        input_npz_folder: path to input npz folder - type: os.PathLike
        output_npz_folder: path to output npz folder - type: os.PathLike
    output:
    """
    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    
    print(f'Z-score normalizing NPZ files in {input_npz_folder} and saving to {output_npz_folder}...')
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            data_zscore = zscore_normalize(data)
            np.savez(os.path.join(output_npz_folder, npz_file), data=data_zscore, fs=fs)

def zscore_normalize(data):
    """ Z-score normalizes data
    input:
        data: data to be normalized - type: numpy.ndarray
    output:
        data_zscore: normalized data - type: numpy.ndarray
    """
    data_zscore = (data - np.mean(data))/np.std(data)
    return data_zscore

def normalize_npz(input_npz_folder, output_npz_folder):
    """ Normalizes NPZ files
    input:
        input_npz_folder: path to input npz folder - type: os.PathLike
        output_npz_folder: path to output npz folder - type: os.PathLike
    output:
    """
    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    
    print(f'Normalizing NPZ files in {input_npz_folder} and saving to {output_npz_folder}...')
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            data_normalized = normalize(data)
            np.savez(os.path.join(output_npz_folder, npz_file), data=data_normalized, fs=fs)

def normalize(data):
    """ Normalizes data
    input:
        data: data to be normalized - type: numpy.ndarray
    output:
        data_normalized: normalized data - type: numpy.ndarray
    """
    data_normalized = data/np.max(np.abs(data))
    return data_normalized