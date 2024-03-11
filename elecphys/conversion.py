import os
import numpy as np
import mat73
import preprocessing
import utils
import shutil
from tqdm import tqdm
import pandas as pd
import re
from io import StringIO


def convert_rhd_to_mat(
        folder_path: str, output_mat_file: str, ds_factor: int) -> None:
    """ Function that Converts RHD files to mat files using RHD to MAT converter written in MATLAB

        Parameters
        ----------
        folder_path: str
            path to folder containing RHD files
        output_mat_file: str
            path to output mat file
        ds_factor: int
            downsample factor

        Returns
        ----------
    """
    output_mat_file_folder = os.path.dirname(output_mat_file)
    if not os.path.exists(output_mat_file_folder):
        os.makedirs(output_mat_file_folder)
    elif os.path.exists(output_mat_file):
        Warning(f'{output_mat_file} already exists. File will be overwritten.')
    # if os is unix, check if MATLAB is installed
    Warning('This functionality needs MATLAB to be installed on your computer. If you do not have MATLAB installed, please install it first.')
    eng = utils.get_matlab_engine()
    eng.addpath(os.path.join(os.path.dirname(__file__), 'matlab_scripts'))
    eng.convertRHD2Mat(folder_path, output_mat_file, ds_factor, nargout=0)
    eng.quit()


def convert_mat_to_npz(mat_file: str, output_npz_folder: str,
                       notch_filter_freq: int) -> None:
    """ Function that  Converts MAT files to NPZ files

        Parameters
        ----------
        mat_file: str
            path to mat file
        output_npz_folder: str
            path to output npz folder

        Returns
        ----------
    """

    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')

    if notch_filter_freq != 0 and notch_filter_freq != 50 and notch_filter_freq != 60:
        raise ValueError(
            'Notch filter frequency must be 0 (no filtering), 50 (Hz), or 60 (Hz)')

    mat_file_contents = mat73.loadmat(mat_file)
    data = mat_file_contents['data']
    fs = mat_file_contents['fs']

    for ch_num in tqdm(range(data.shape[0])):
        ch_name = f'Ch{ch_num+1}'
        if notch_filter_freq == 0:
            data_filtered = data[ch_num, :]
        else:
            data_filtered = preprocessing.apply_notch(
                data[ch_num, :], {'Q': 60, 'fs': fs, 'f0': notch_filter_freq})

        np.savez(
            os.path.join(
                output_npz_folder,
                f'{ch_name}.npz'),
            data=data_filtered,
            fs=fs)


def convert_OpenBCI_csv_to_npz(file_path: str, output_npz_folder: str, notch_filter_freq: int) -> None:
    """ Function that Converts OpenBCI CSV files to NPZ files

        Parameters
        ----------
        file_path: str
            path to OpenBCI csv file
        output_npz_folder: str
            path to output npz folder
        notch_filter_freq: int
            frequency of notch filter

        Returns
        ----------
    """

    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')

    # read openbci txt file, separate first lines which start with %

    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        header = []
        for line in lines:
            if line.startswith('%'):
                header.append(line)
            else:
                data.append(line)

    for header_line in header:
        if 'Sample Rate' in header_line:
            fs = int(re.findall(r'\d+', header_line)[0])
        elif 'Number of channels' in header_line:
            num_channels = int(re.findall(r'\d+', header_line)[0])
    # convert data to pandas dataframe with columns as the first row
    data = pd.read_csv(StringIO(''.join(data)), sep=',')

    for ch_no in tqdm(range(num_channels)):
        ch_name = f' EXG Channel {ch_no}'
        data_ch = data[ch_name].values

        if notch_filter_freq == 0:
            data_filtered = data_ch
        else:
            data_filtered = preprocessing.apply_notch(
                data_ch, {'Q': 60, 'fs': fs, 'f0': notch_filter_freq})
        ch_name = f'Ch{ch_no+1}'
        np.savez(
            os.path.join(
                output_npz_folder,
                f'{ch_name}.npz'),
            data=data_filtered,
            fs=fs)
