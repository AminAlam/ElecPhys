
import os
import numpy as np
from scipy import signal
from tqdm import tqdm

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
    

def stft_from_array(signal_array, fs, window_size, overlap, window_type='hann', nfft=None):
    if np.ndim(signal_array) == 2 and signal_array.shape[0] == 1 or signal_array.shape[1] == 1:
        signal_array = signal_array[0]
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')
    window_length = int(window_size*fs)
    overlap_length = int(overlap*fs)
    if nfft is None:
        nfft = 2*window_length
    if 'kaiser' in window_type:
        window_type = signal.kaiser(window_length, beta=int(window_type.split(' ')[-1]))
    [f, t, Zxx] = signal.stft(signal_array, fs, window=window_type, nperseg=window_length, noverlap=overlap_length, nfft=nfft)
    return f, t, Zxx