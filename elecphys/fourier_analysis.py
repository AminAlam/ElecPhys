import os
import numpy as np
from scipy import signal
from tqdm import tqdm
import json
import csv

import utils
import cfc
import data_io
import visualization


def stft_numeric_output_from_npz(input_npz_folder: str, output_npz_folder: str,
                                 window_size: float, overlap: float, window_type: str = 'hann') -> None:
    """ Computes STFT and saves results as NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder to save STFT results
        window_size: float
            window size in seconds
        overlap: float:
            overlap in seconds
        window_type: str
            window type. Default is 'hann', but can be any window type supported by scipy.signal.get_window()

        Returns
        ----------
    """

    if not os.path.exists(output_npz_folder):
        os.makedirs(output_npz_folder)
    else:
        Warning(f'{output_npz_folder} already exists. Files will be overwritten.')
    print(
        f'Computing STFT with window size {window_size} seconds, overlap {overlap} seconds, and window type {window_type}...')
    for npz_file in tqdm(os.listdir(input_npz_folder)):
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(input_npz_folder, npz_file)
            npz_file_contents = np.load(npz_file_path)
            data = npz_file_contents['data']
            fs = npz_file_contents['fs']
            f, t, Zxx = stft_from_array(
                data, fs, window_size, overlap, window_type)
            stft_npz_file_path = os.path.join(
                output_npz_folder, f'STFT_{npz_file}')
            np.savez(stft_npz_file_path, f=f, t=t, Zxx=Zxx)


def dft_numeric_output_from_npz(
        input_npz_folder: str, output_npz_folder: str, nfft: int = None) -> None:
    """ Computes DFT and saves results as NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder to save DFT results

        Returns
        ----------
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
            dft_npz_file_path = os.path.join(
                output_npz_folder, f'DFT_{npz_file}')
            np.savez(dft_npz_file_path, f=f, Zxx=Zxx)


def stft_from_array(signal_array, fs: int, window_size: float, overlap: float,
                    window_type: str = 'hann', nfft: int = None) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Computes STFT from 1D array

        Parameters
        ----------
            signal_array: np.ndarray
                signal array in time domain
            fs: int
                sampling frequency (Hz)
            window_size: float
                window size in seconds
            overlap: float
                windows overlap in seconds
            window_type: str
                window type. Default is 'hann', but can be any window type supported by scipy.signal.get_window()
            nfft: int
                number of FFT points

        Returns
        ----------
            f: np.ndarray
                frequency vector
            t: np.ndarray
                time vector
            Zxx: np.ndarray
                STFT matrix (complex)
    """
    if np.ndim(signal_array) == 2:
        signal_array = np.squeeze(signal_array)
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(
            f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')
    window_length = int(window_size * fs)
    overlap_length = int(overlap * fs)
    if nfft is None:
        nfft = 2 * window_length
    if 'kaiser' in window_type:
        window_type = signal.windows.kaiser(
            window_length, float(window_type.split(' ')[1]))
    [f,
     t,
     Zxx] = signal.stft(signal_array,
                        fs,
                        window=window_type,
                        nperseg=window_length,
                        noverlap=overlap_length,
                        nfft=nfft)
    return f, t, Zxx


def dft_from_array(signal_array, fs: int,
                   nfft: int = None) -> [np.ndarray, np.ndarray]:
    """ Computes DFT from 1D array

        Parameters
        ----------
            signal_array: np.ndarray
                signal array in time domain
            fs: int
                sampling frequency (Hz)
            nfft: int
                number of FFT points

        Returns
        ----------
            f: np.ndarray
                frequency vector
            Zxx: np.ndarray
                DFT vector (complex)
    """
    if np.ndim(signal_array) == 2:
        signal_array = np.squeeze(signal_array)
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(
            f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')
    if nfft is None:
        nfft = len(signal_array)
    f = np.linspace(0, fs, nfft)
    Zxx = np.fft.fft(signal_array, nfft)
    return f, Zxx


def butterworth_filtering_from_array(
        signal_array, fs: int, _args: dict) -> np.ndarray:
    """ Filters signal array

        Parameters
        ----------
            signal_array: type: np.ndarray
                number of FFT points
            fs: int
                sampling frequency (Hz)
            _args: dict
                dictionary containing filter parameters

        Returns
        ----------
            signal_array: np.ndarray
                filtered signal array in time domain
            filter_freq_response: dict
                dictionary containing filter frequency response
    """

    if np.ndim(signal_array) == 2:
        signal_array = np.squeeze(signal_array)
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(
            f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')
    if _args['filter_type'] == 'LPF':
        b, a = signal.butter(_args['filter_order'], int(
            _args['freq_cutoff']) / (fs / 2), btype='lowpass')
    elif _args['filter_type'] == 'HPF':
        b, a = signal.butter(_args['filter_order'], int(
            _args['freq_cutoff']) / (fs / 2), btype='highpass')
    elif _args['filter_type'] == 'BPF':
        _args['freq_cutoff'] = utils.convert_string_to_list(
            _args['freq_cutoff'])
        _args['freq_cutoff'] = [int(i) for i in _args['freq_cutoff']]
        b, a = signal.butter(_args['filter_order'], [
                             _args['freq_cutoff'][0] / (fs / 2), _args['freq_cutoff'][1] / (fs / 2)], btype='bandpass')
    else:
        raise ValueError(
            f'Invalid filter type: {_args["filter_type"]}. Must be "LPF", "HPF", or "BPF"')

    signal_array = signal.filtfilt(b, a, signal_array)
    return signal_array


def butterworth_filtering_from_npz(
        input_npz_folder: str, output_npz_folder: str, _args: dict) -> None:
    """ Filters signal array

        Parameters
        ----------
            input_npz_folder: str
                path to input npz folder
            output_npz_folder: str
                path to output npz folder to save filtered results
            _args: dict
                dictionary containing filter parameters

        Returns
        ----------
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
            filtered_npz_file_path = os.path.join(
                output_npz_folder, f'Filtered_{npz_file}')
            np.savez(filtered_npz_file_path, data=data, fs=fs)

    _args['fs'] = int(fs)
    filter_freq_response_json_file_path = os.path.join(
        output_npz_folder, f'filter_freq_response.json')

    with open(filter_freq_response_json_file_path, 'w') as fp:
        json.dump(_args, fp)


def calc_freq_response(
        _args: dict) -> [np.ndarray, np.ndarray, np.ndarray, dict]:
    """ Calculates filter frequency response

        Parameters
        ----------
            _args: dict
                dictionary containing filter parameters

        Returns
        ----------
            f: np.ndarray
                frequency vector
            mag: np.ndarray
                magnitude vector
            phase: np.ndarray
                phase vector
            _args: dict
                dictionary containing filter parameters
    """

    fs = _args['fs']
    if _args['filter_type'] == 'LPF':
        b, a = signal.butter(_args['filter_order'], int(
            _args['freq_cutoff']) / (fs / 2), btype='lowpass')
    elif _args['filter_type'] == 'HPF':
        b, a = signal.butter(_args['filter_order'], int(
            _args['freq_cutoff']) / (fs / 2), btype='highpass')
    elif _args['filter_type'] == 'BPF':
        _args['freq_cutoff'] = utils.convert_string_to_list(
            _args['freq_cutoff'])
        _args['freq_cutoff'] = [int(i) for i in _args['freq_cutoff']]
        b, a = signal.butter(_args['filter_order'], [
                             _args['freq_cutoff'][0] / (fs / 2), _args['freq_cutoff'][1] / (fs / 2)], btype='bandpass')
    else:
        raise ValueError(
            f'Invalid filter type: {_args["filter_type"]}. Must be "LPF", "HPF", or "BPF"')

    w, h = signal.freqz(b, a, worN=2000)
    f = w / (2 * np.pi) * fs
    mag = 20 * np.log10(abs(h))
    phase = np.unwrap(np.angle(h)) * 180 / np.pi

    return f, mag, phase, _args


def calc_cfc_from_array(signal_array, fs: int, freqs_amp: list,
                        freqs_phase: list, time_interval: list = None) -> np.ndarray:
    """ Calculates CFC matrix

        Parameters
        ----------
            signal_array: np.ndarray
                1D array of signal
            fs: int
                sampling frequency (Hz)
            freqs_amp: list
                amplitude frequencies (Hz)
            freqs_phase: list
                phase frequencies (Hz)
            time_interval: list
                time interval to calculate CFC over in seconds

        Returns
        ----------
            MI_mat: np.ndarray
                CFC matrix
    """
    if np.ndim(signal_array) == 2:
        signal_array = np.squeeze(signal_array)
        Warning('Signal array is 2D but only one channel. Using first channel.')
    if np.ndim(signal_array) != 1:
        raise ValueError(
            f'Signal array must be 1D array but has {np.ndim(signal_array)} dimensions')

    if time_interval is None:
        time_interval = [0, len(signal_array) / fs]
    signal_array = signal_array[int(
        time_interval[0] * fs):int(time_interval[1] * fs)]
    MI_mat = cfc.calc_tf_mvl(signal_array, fs, freqs_phase, freqs_amp)
    return MI_mat


def calc_cfc_from_npz(input_npz_folder: str, output_npz_folder: str,
                      freqs_amp: list, freqs_phase: list, time_interval: list = None) -> None:
    """ Calculates CFC matrix

        Parameters
        ----------
            input_npz_folder: str
                path to input npz folder
            output_npz_folder: str
                path to output npz folder to save CFC results
            freqs_amp: list
                amplitude frequencies (Hz)
            freqs_phase: list
                phase frequencies (Hz)
            time_interval: list
                time interval to calculate CFC over in seconds

        Returns
        ----------
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
            MI_mat = calc_cfc_from_array(
                data, fs, freqs_amp, freqs_phase, time_interval)
            cfc_npz_file_path = os.path.join(
                output_npz_folder, f'CFC_{npz_file}')
            np.savez(
                cfc_npz_file_path,
                MI_mat=MI_mat,
                freqs_amp=freqs_amp,
                freqs_phase=freqs_phase,
                time_interval=time_interval)


def freq_bands_power_over_time(
        input_npz_folder: str,
        freq_bands: [
            tuple,
            list] = None,
    channels_list: str = None,
    ignore_channels: str = None,
    window_size: float = 1,
    overlap: float = 0.5,
    t_min: float = None,
    t_max: float = None,
    output_csv_file: str = None,
    output_plot_file: str = None,
        plot_type: str = 'average_of_channels') -> None:
    """ Calculates power over time for given frequency bands

        Parameters
        ----------
            input_npz_folder: str
                path to input npz folder containing signal npz files (in time domain)
            freq_bands: tuple, list
                tuple or list of frequency bands to calculate power over time for. It should be a tuple or list of lists, where each list contains two elements: the lower and upper frequency bounds of the band (in Hz). For example, freq_bands = [[1, 4], [4, 8], [8, 12]] would calculate power over time for the delta, theta, and alpha bands.
            channels_list: str
                list of channels to include in analysis
            ignore_channels: str
                list of channels to ignore in analysis
            window_size: float
                window size in seconds to calculate power over time
            overlap: float
                windows overlap in seconds to calculate power over time
            t_min: float
                start of time interval to calculate power over time. Default is None which means start from beginning of signal.
            t_max: float
                end of time interval to calculate power over time. Default is None which means end at end of signal.
            output_csv_file: str
                path to output csv file to save power over time results
            output_plot_file: str
                path to output plot file to save power over time results
            plot_type: str
                type of plot to generate. Options are 'avg' or 'all'. Default is 'avg' which plots average power over time for all channels with an erros cloud. 'all' plots power over time for all channels individually.

        Returns
        ----------
    """
    channels_list = utils.convert_string_to_list(channels_list)
    ignore_channels = utils.convert_string_to_list(ignore_channels)

    data_all, fs, channels_map = data_io.load_all_npz_files(input_npz_folder, ignore_channels, channels_list)
    # if freq_bands only has one list, we should make sure it is a list of lists
    if len(freq_bands) == 2 and isinstance(freq_bands[0], int) and isinstance(freq_bands[1], int):
        freq_bands = [freq_bands]
    freq_bands = [utils.convert_string_to_list(freq_band) for freq_band in freq_bands]
    freq_bands = utils.check_freq_bands(freq_bands, fs)

    if t_min is None:
        t_min = 0
    if t_max is None:
        t_max = data_all.shape[1] / fs

    if t_max <= t_min:
        raise ValueError(
            f'Invalid time interval: [{t_min}, {t_max}]. t_max must be larger than t_min.')

    for freq_band in freq_bands:
        for ch_indx in range(data_all.shape[0]):
            data = data_all[ch_indx, :]
            f, t, Zxx = stft_from_array(data, fs, window_size, overlap)
            Zxx = np.abs(Zxx)
            if ch_indx == 0:
                spectrum_all = np.zeros((data_all.shape[0], len(t), len(f)))
            spectrum_all[ch_indx, :, :] = Zxx.T

        t0 = np.where(t >= t_min)[0][0]
        t1 = np.where(t <= t_max)[0][-1]
        t = t[t0:t1 + 1]
        f0 = np.where(f >= freq_band[0])[0][0]
        f1 = np.where(f <= freq_band[1])[0][-1]
        spectrum_all = spectrum_all[:, :, f0:f1 + 1]
        spectrum_all = spectrum_all[:, t0:t1 + 1, :]
        power_all = np.sum(spectrum_all**2, axis=2)
        avg_power = np.mean(power_all, axis=0)
        avg_power = 10 * np.log10(avg_power)
        power_all = 10 * np.log10(power_all)
        if output_csv_file is not None:
            if 'csv' in output_csv_file:
                output_csv_file = output_csv_file.replace('.csv', '')
            # save to csv file
            with open(f'{output_csv_file}_{freq_band[0]}_{freq_band[1]}.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Channel', 'Time', 'Power'])
                for ch_indx in range(data_all.shape[0]):
                    for t_indx in range(len(t)):
                        csvwriter.writerow([channels_map[ch_indx] + 1, t[t_indx], power_all[ch_indx, t_indx]])
                for t_indx in range(len(t)):
                    csvwriter.writerow(['Avg_channels', t[t_indx], avg_power[t_indx]])
        if output_plot_file is not None:
            output_plot_file_format = output_plot_file.split('.')[-1]
            if output_plot_file_format == '':
                output_plot_file_format = 'pdf'
            output_plot_file_band = f"{output_plot_file.split('.')[0]}_{freq_band[0]}_{freq_band[1]}.{output_plot_file_format}"
        else:
            output_plot_file_band = None
        visualization.plot_power_over_time_from_array(power_all, t, channels_map, plot_type, output_plot_file_band)
