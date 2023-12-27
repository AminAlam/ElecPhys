import mat73
import numpy as np
import os
import utils


def load_mat(mat_file) -> [np.ndarray, int]:
    """ Function that Loads MAT file

        Parameters
        ----------
        mat_file: str
            path to mat file

        Returns
        ----------
        data: numpy.ndarray
            data from MAT file
        fs: int
            sampling frequency (Hz)
    """
    with mat73.loadmat(mat_file) as mat_file_contents:
        data = mat_file_contents['data']
        fs = mat_file_contents['fs']
    return data, fs


def load_npz(npz_file) -> [np.ndarray, int]:
    """ Function that Loads NPZ file

        Parameters
        ----------
        npz_file: path to npz file - type: os.PathLike

        Returns
        ----------
        data: data from NPZ file - type: numpy.ndarray
        fs: sampling frequency - type: float
    """
    with np.load(npz_file) as npz_file_contents:
        data = npz_file_contents['data']
        fs = npz_file_contents['fs']
    return data, fs


def load_all_npz_files(npz_folder: str, ignore_channels: [
                       list, str] = None) -> [np.ndarray, int]:
    """ Function that Loads all NPZ files in a folder

        Parameters
        ----------
        npz_folder: str
            path to npz folder containing NPZ files
        ignore_channels: list, str
            list of channels to be ignored and not loaded. If None, all channels will be loaded. Either a list of channel names or a string of channel names separated by commas.

        Returns
        --------
        data_all: np.ndarray
        fs: int
    """
    files_list = os.listdir(npz_folder)
    for file_name in files_list:
        if not file_name.endswith('.npz'):
            files_list.remove(file_name)
    files_list = utils.sort_file_names(files_list)
    if ignore_channels is not None:
        ignore_channels = utils.convert_string_to_list(ignore_channels)
        for ch_indx in ignore_channels:
            files_list.remove(ch_indx)
    num_channels = len(files_list)
    ch_indx = 0
    for npz_file in files_list:
        npz_file_path = os.path.join(npz_folder, npz_file)
        if ch_indx == 0:
            data, fs = load_npz(npz_file_path)
            data_all = np.zeros((num_channels, len(data)))
        else:
            data, _ = load_npz(npz_file_path)
        data_all[ch_indx, :] = data
    return data_all, fs


def load_npz_stft(npz_file) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Function that Loads NPZ file

        Parameters
        ----------
        npz_file: path to npz file - type: os.PathLike

        Returns
        ----------
        f: frequency array - type: numpy.ndarray
        t: time array - type: numpy.ndarray
        Zxx: STFT array - type: numpy.ndarray
    """
    with np.load(npz_file) as npz_file_contents:
        f = npz_file_contents['f']
        t = npz_file_contents['t']
        Zxx = npz_file_contents['Zxx']
    return f, t, Zxx


def load_npz_dft(npz_file) -> [np.ndarray, np.ndarray]:
    """ Function that Loads NPZ file

        Parameters
        ----------
        npz_file: path to npz file - type: os.PathLike

        Returns
        ----------
        f: frequency array - type: numpy.ndarray
        Zxx: DFT array - type: numpy.ndarray
    """

    with np.load(npz_file) as npz_file_contents:
        f = npz_file_contents['f']
        Zxx = npz_file_contents['Zxx']
    return f, Zxx


def load_npz_mvl(npz_file) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Function that Loads NPZ file

        Parameters
        ----------
        npz_file: path to npz file - type: os.PathLike

        Returns
        ----------
        MVL: MVL array - type: numpy.ndarray
        freqs_amp: amplitude frequencies - type: numpy.ndarray
        freqs_phase: phase frequencies - type: numpy.ndarray
        time_interval: time interval - type: numpy.ndarray
    """
    with np.load(npz_file) as npz_file_contents:
        MI_mat = npz_file_contents['MI_mat']
        freqs_amp = npz_file_contents['freqs_amp']
        freqs_phase = npz_file_contents['freqs_phase']
        time_interval = npz_file_contents['time_interval']
    return MI_mat, freqs_amp, freqs_phase, time_interval


def write_separate_npz_files(
        data: np.ndarray, fs: int, output_npz_folder: str) -> None:
    """ Function that Writes data to NPZ file as separate files for each channel

        Parameters
        ----------
        data: numpy.ndarray
            data to be written to NPZ file. Shape: (num_channels, num_samples)
        fs: int
            sampling frequency (Hz)
        output_npz_folder: str
            path to output npz folder

        Returns
        ----------
    """
    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    for ch_indx in range(data.shape[0]):
        np.savez(os.path.join(output_npz_folder,
                 f'Ch{ch_indx}.npz'), data=data[ch_indx, :], fs=fs)
