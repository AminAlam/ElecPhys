import os
import numpy as np
from scipy import signal
from tqdm import tqdm
import utils
import data_io


def apply_notch(_signal_chan: np.ndarray, _args: dict) -> np.ndarray:
    """ Applies notch filter to given signal

        Parameters
        ----------
        _signal_chan: np.ndarray
            signal channel
        _args: dict
            dictionary containing notch filter parameters

        Returns
        ----------
        _signal_chan: np.ndarray
            signal channel with notch filter applied
    """
    for f0 in np.arange(_args['f0'], 300, _args['f0']):
        b_notch, a_notch = signal.iirnotch(f0, _args['Q'], _args['fs'])
        _signal_chan = signal.filtfilt(b_notch, a_notch, _signal_chan)
    return _signal_chan


def zscore_normalize_npz(input_npz_folder: str,
                         output_npz_folder: str) -> None:
    """ Z-score normalizes NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder

        Returns
        ----------
    """
    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')

    print(
        f'Z-score normalizing NPZ files in {input_npz_folder} and saving to {output_npz_folder}...')
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            data_zscore = zscore_normalize(data)
            np.savez(
                os.path.join(
                    output_npz_folder,
                    npz_file),
                data=data_zscore,
                fs=fs)


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """ Z-score normalizes data

        Parameters
        ----------
        data: np.ndarray
            data to be normalized

        Returns
        ----------
        data_zscore: np.ndarray
            normalized data
    """
    data_zscore = (data - np.mean(data)) / np.std(data)
    return data_zscore


def normalize_npz(input_npz_folder: str, output_npz_folder: str) -> None:
    """ Normalizes NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder

        Returns
        ----------
    """
    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')

    print(
        f'Normalizing NPZ files in {input_npz_folder} and saving to {output_npz_folder}...')
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            data_normalized = normalize(data)
            np.savez(
                os.path.join(
                    output_npz_folder,
                    npz_file),
                data=data_normalized,
                fs=fs)


def normalize(data: np.ndarray) -> np.ndarray:
    """ Normalizes data

        Parameters
        ----------
        data: umpy.ndarray
            data to be normalized

        Returns
        ----------
        data_normalized: numpy.ndarray
            normalized data
    """
    data_normalized = data / np.max(np.abs(data))
    return data_normalized


def re_reference_npz(input_npz_folder: str, output_npz_folder: str, ignore_channels: [
                     list, str] = None, rr_channel: int = None) -> None:
    """ re-references NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder
        ignore_channels: list, str
            list of channels to be ignored. Either a list of channel indexes or a string of channel indexes separated by commas. If None, no channels will be ignored
        rr_channel: int
            channel to be used as reference. If None, average re-referencing will be used

        Returns
        ----------
    """
    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')

    print(
        f'Average re-referencing NPZ files in {input_npz_folder} and saving to {output_npz_folder}...')
    data_all, fs, _ = data_io.load_all_npz_files(input_npz_folder)
    data_all_rereferenced = re_reference(data_all, ignore_channels, rr_channel)
    data_io.write_separate_npz_files(
        data_all_rereferenced, fs, output_npz_folder)


def re_reference(data: np.ndarray, ignore_channels: [
                 list, str] = None, rr_channel: int = None) -> np.ndarray:
    """ Average re-references data

        Parameters
        ----------
        data: numpy.ndarray
            data to be re-referenced. Shape: (n_channels, n_samples)
        ignore_channels: str, list
            list of channels to be ignored. Either a list of channel indexes or a string of channel indexes separated by commas. If None, no channels will be ignored
        rr_channel: int
            channel to be used as reference. If None, average re-referencing will be used

        Returns
        ----------
        data_rereferenced: numpy.ndarray
            re-referenced data. Shape: (n_channels, n_samples)
    """
    ignore_channels = utils.convert_string_to_list(ignore_channels)
    if ignore_channels is not None:
        ignore_channels = [i - 1 for i in ignore_channels]
        channels_list = [
            i for i in range(
                data.shape[0]) if i not in ignore_channels]
    else:
        channels_list = [i for i in range(data.shape[0])]
    rr_channel = rr_channel - 1 if rr_channel is not None else None
    data_rereferenced = data.copy()
    if rr_channel is not None:
        reference = data[rr_channel, :].reshape(1, -1)
        data_rereferenced[channels_list, :] = data[channels_list,
                                                   :] - np.repeat(reference, len(channels_list), axis=0)
    else:
        reference = np.mean(data[channels_list, :], axis=0).reshape(1, -1)
        data_rereferenced[channels_list, :] = data[channels_list,
                                                   :] - np.repeat(reference, len(channels_list), axis=0)
    return data_rereferenced
