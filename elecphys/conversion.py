import os
import numpy as np
import mat73
import preprocessing
import utils
import shutil
from tqdm import tqdm

def convert_rhd_to_mat(folder_path, output_mat_file, ds_factor):
    """Converts RHD files to mat files using RHD to MAT converter written in MATLAB
    input:
        folder_path: path to folder containing RHD files - type: os.PathLike
        output_mat_file: path to output mat file - type: os.PathLike
        ds_factor: downsample factor - type: int
    output:
        output_mat_file: no return value
    """
    output_mat_file_folder = os.path.dirname(output_mat_file)
    # if os is unix, check if MATLAB is installed
    Warning('This functionality needs MATLAB to be installed on your computer. If you do not have MATLAB installed, please install it first.') 
    eng = utils.get_matlab_engine()
    eng.addpath(os.path.join(os.path.dirname(__file__), 'matlab_scripts'))
    eng.convertRHD2Mat(folder_path, output_mat_file, ds_factor, nargout=0)
    eng.quit()


def convert_mat_to_npz(mat_file, output_npz_folder, notch_filter_freq):
    """Converts MAT files to NPZ files
    input:
        mat_file: path to mat file - type: os.PathLike
        output_npz_folder: path to output npz folder - type: os.PathLike
    output:
        output_npz_folder: no return value
    """

    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    
    if notch_filter_freq !=0 and notch_filter_freq != 50 and notch_filter_freq != 60:
        raise ValueError('Notch filter frequency must be 0 (no filtering), 50 (Hz), or 60 (Hz)')
    
    mat_file_contents = mat73.loadmat(mat_file)
    data = mat_file_contents['data']
    fs = mat_file_contents['fs']
    
    for ch_num in tqdm(range(data.shape[0])):
        ch_name = f'Ch{ch_num+1}'
        if notch_filter_freq == 0:
            data_filtered = data[ch_num, :]
        else:
            data_filtered = preprocessing.apply_notch(data[ch_num, :], {'Q':60, 'fs':fs, 'f0':notch_filter_freq})

        np.savez(os.path.join(output_npz_folder, f'{ch_name}.npz'), data=data_filtered, fs=fs)