import os
import numpy as np
import matplotlib.pyplot as plt
import json

import data_io
import preprocessing
import utils
import fourier_analysis


def plot_stft_from_npz(input_npz_file: str, output_plot_file: str, f_min: int,
                       f_max: int, t_min: float, t_max: float, db_min: float, db_max: float) -> None:
    """ Plots STFT from NPZ file (STFT must be saved as NPZ file)

        Parameters
        ----------
        input_npz_file: str
            path to input npz file
        output_plot_file: str
            path to save figure. If None, will show figure instead of saving it
        f_min: int
            minimum frequency to plot in Hz
        f_max: int
            maximum frequency to plot in Hz
        t_min: float
            minimum time to plot in seconds
        t_max: float
            maximum time to plot in seconds
        db_min: float
            minimum dB to plot
        db_max: float
            maximum dB to plot

        Returns
        ----------
    """
    f, t, Zxx = data_io.load_npz_stft(input_npz_file)
    Zxx_plot = 10 * np.log10(np.abs(Zxx))

    if f_min is None:
        f_min = np.min(f)
    if f_max is None:
        f_max = np.max(f)
    if t_min is None:
        t_min = np.min(t)
    if t_max is None:
        t_max = np.max(t)
    if db_min is None:
        db_min = np.min(10 * np.log10(np.abs(Zxx)))
    if db_max is None:
        db_max = np.max(10 * np.log10(np.abs(Zxx)))

    plot_stft_from_array(
        Zxx_plot,
        t,
        f,
        f_min,
        f_max,
        t_min,
        t_max,
        db_min,
        db_max,
        output_plot_file)


def plot_avg_stft_from_npz(npz_folder_path: str, output_plot_file: str, f_min: int, f_max: int,
                           t_min: float, t_max: float, db_min: int, db_max: int, channels_list: [str, list] = None) -> None:
    """ Plots average STFT from NPZ files (STFT must be saved as NPZ file)

        Parameters
        ----------
        npz_folder_path: str
            path to input npz folder
        output_plot_file: str
            path to save figure. If None, will show figure instead of saving it
        f_min: int
            minimum frequency to plot in Hz
        f_max: int
            maximum frequency to plot in Hz
        t_min: float
            minimum time to plot in seconds
        t_max: float
            maximum time to plot in seconds
        db_min: float
            minimum dB to plot
        db_max: float
            maximum dB to plot
        channels_list: str
            list of channels to plot (can be a strin of comma-separated values or a list of integers)

        Returns
        ----------
    """

    npz_files = os.listdir(npz_folder_path)
    npz_files = utils.sort_file_names(npz_files)
    channels_list = utils.convert_string_to_list(channels_list)
    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files) + 1))
    else:
        channels_list = sorted(channels_list)

    for channel_index, channel in enumerate(channels_list):
        npz_file = npz_files[channel_index]
        f, t, Zxx = data_io.load_npz_stft(
            os.path.join(npz_folder_path, npz_file))
        Zxx = 10 * np.log10(np.abs(Zxx))
        if channel_index == 0:
            Zxx_list = Zxx
        else:
            Zxx_list = np.dstack((Zxx_list, Zxx))

    Zxx_avg = np.mean(Zxx_list, axis=2)
    if f_min is None:
        f_min = np.min(f)
    if f_max is None:
        f_max = np.max(f)
    if t_min is None:
        t_min = np.min(t)
    if t_max is None:
        t_max = np.max(t)
    if db_min is None:
        db_min = np.min(Zxx_avg)
    if db_max is None:
        db_max = np.max(Zxx_avg)

    plot_stft_from_array(
        Zxx_avg,
        t,
        f,
        f_min,
        f_max,
        t_min,
        t_max,
        db_min,
        db_max,
        output_plot_file)


def plot_stft_from_array(Zxx: np.ndarray, t: np.ndarray, f: np.ndarray, f_min: int, f_max: int,
                         t_min: float, t_max: float, db_min: float, db_max: float, output_plot_file: str = None) -> None:
    """ Plots STFT from 2D array

        Parameters
        ----------
        Zxx: np.ndarray
            STFT array
        t: np.ndarray
            time array
        f: np.ndarray
            frequency array
        f_min: int
            minimum frequency to plot in Hz
        f_max: int
            maximum frequency to plot in Hz
        t_min: float
            minimum time to plot in seconds
        t_max: float
            maximum time to plot in seconds
        db_min: float
            minimum dB to plot
        db_max: float
            maximum dB to plot
        output_plot_file: str
            path to save figure. If None, will show figure instead of saving it

        Returns
        ----------
    """

    desired_freq_index_low = np.where(
        np.min(abs(f - f_min)) == abs(f - f_min))[0][0]
    desired_freq_index_high = np.where(
        np.min(abs(f - f_max)) == abs(f - f_max))[0][0]
    desired_time_index_low = np.where(
        np.min(abs(t - t_min)) == abs(t - t_min))[0][0]
    desired_time_index_high = np.where(
        np.min(abs(t - t_max)) == abs(t - t_max))[0][0]
    Zxx = Zxx[desired_freq_index_low:desired_freq_index_high,
              desired_time_index_low:desired_time_index_high]
    f = f[desired_freq_index_low:desired_freq_index_high]
    t = t[desired_time_index_low:desired_time_index_high]

    plt.figure(figsize=(30, 15))
    pc = plt.pcolormesh(t, f, Zxx, cmap=plt.get_cmap('jet'), shading='auto')
    plt.colorbar(pc)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.clim(db_min, db_max)
    plt.text(
        1.01,
        0.5,
        'Power (dB)',
        va='center',
        rotation=-90,
        transform=plt.gca().transAxes)
    plt.tight_layout()

    if output_plot_file is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(output_plot_file)):
            os.makedirs(os.path.dirname(output_plot_file))
        plt.savefig(output_plot_file, dpi=600)
        plt.close()


def plot_signals_from_npz(
        npz_folder_path: str,
        output_plot_file: str,
        t_min: float,
        t_max: float,
        channels_list: [
            str,
            list] = None,
    normalize: bool = False,
    scale_bar: bool = True,
        _rereference_args: dict = None) -> None:
    """ Plots signals from NPZ file

        Parameters
        ----------
        npz_folder_path: str
            path to input npz folder
        output_plot_file: str
            path to save figure. If None, will show figure instead of saving it
        t_min: float
            minimum time to plot in seconds
        t_max: float
            maximum time to plot in seconds
        channels_list: str
            list of channels to plot (can be a strin of comma-separated values or a list of integers)
        normalize: bool
            whether to normalize the signal
        scale_bar: bool
            whether to plot a scale bar
        _rereference_args: dict
            dictionary containing rereferencing parameters. If None, no rereferencing will be applied

        Returns
        ----------
    """

    npz_files = os.listdir(npz_folder_path)
    # remove non-NPZ files
    npz_files = utils.sort_file_names(npz_files)
    channels_list = utils.convert_string_to_list(channels_list)
    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files) + 1))
    else:
        channels_list = sorted(channels_list)

    if len(channels_list) > 20:
        fig = plt.figure(figsize=(30, len(channels_list)))
    else:
        fig = plt.figure(figsize=(30, 10))

    if normalize:
        scale_bar = False

    if scale_bar:
        ax = fig.subplots(
            len(channels_list), 2, gridspec_kw={
                'width_ratios': [
                    10, 1]})
    else:
        ax = fig.subplots(len(channels_list), 1, sharex=True)

    data_all, fs, channels_map = data_io.load_all_npz_files(npz_folder_path, channels_list=channels_list)
    t = np.arange(0, data_all.shape[1] / fs, 1 / fs)
    if t_min is None:
        t_min = np.min(t)
    if t_max is None:
        t_max = np.max(t)
    if _rereference_args is not None:
        ignore_channels = _rereference_args['ignore_channels']
        rr_channel = _rereference_args['rr_channel']
        ignore_channels = utils.convert_string_to_list(ignore_channels)
        if rr_channel is not None:
            if rr_channel not in channels_list:
                raise ValueError('rr_channel must be in channels_list')
        if ignore_channels is not None:
            if not set(ignore_channels).issubset(set(channels_list)):
                raise ValueError(
                    'All channels in ignore_channels must be in channels_list, as it does not make sense to ignore a channel for re-referencing if it is not in channels_list')

        data_all = preprocessing.re_reference(
            data_all, ignore_channels, rr_channel)

    for row_no, channel_index in enumerate(channels_map):
        signal_chan = data_all[row_no, :]
        if not scale_bar:
            ax[row_no].plot(t, signal_chan, color='k')
            ax[row_no].set_ylabel(f'Channel {channel_index}')
            ax[row_no].spines['top'].set_visible(False)
            ax[row_no].spines['right'].set_visible(False)
            if row_no != len(channels_list) - 1:
                ax[row_no].spines['bottom'].set_visible(False)
            ax[row_no].tick_params(axis='both', which='both', length=0)
            ax[row_no].set_yticks([])
            ax[row_no].set_xlim(t_min, t_max)
            if row_no != len(channels_list) - 1:
                ax[row_no].set_xticks([])
        else:
            scale_bar_size = np.max(signal_chan) - np.min(signal_chan)
            ax[row_no, 0].plot(t, signal_chan, color='k')
            ax[row_no, 0].set_ylabel(f'Channel {channel_index}')
            ax[row_no, 0].spines['top'].set_visible(False)
            ax[row_no, 0].spines['right'].set_visible(False)
            if row_no != len(channels_list) - 1:
                ax[row_no, 0].spines['bottom'].set_visible(False)
            ax[row_no, 0].tick_params(axis='both', which='both', length=0)
            ax[row_no, 0].set_yticks([])
            # symmetric error bar for scale bar
            ax[row_no, 1].errorbar(
                0, 0, yerr=scale_bar_size / 2, color='k', capsize=5)
            ax[row_no, 1].text(
                1 / fs, 0, f'{scale_bar_size:.2f} uV', ha='left')
            ax[row_no, 1].axis('off')
            ax[row_no, 0].set_xlim(t_min, t_max)
            if row_no != len(channels_list) - 1:
                ax[row_no, 0].set_xticks([])
                ax[row_no, 1].set_xticks([])

    if not scale_bar:
        ax[-1].set_xlabel('Time (s)')
        ax[-1].set_xlim(t_min, t_max)
    else:
        ax[-1, 0].set_xlabel('Time (s)')
        ax[-1, 1].set_xlabel('Scale bar')
        ax[-1, 0].set_xlim(t_min, t_max)
        ax[-1, 1].set_ylim(-scale_bar_size / 2, scale_bar_size / 2)
    plt.tight_layout()

    if output_plot_file is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(output_plot_file)):
            os.makedirs(os.path.dirname(output_plot_file))
        plt.tight_layout()
        plt.savefig(output_plot_file, dpi=600)
        plt.close()


def plot_dft_from_npz(npz_folder_path: str, output_plot_file: str, f_min: int, f_max: int,
                      plot_type: str, channels_list: [str, list] = None, conv_window_size: float = None) -> None:
    """ Plots DFT from NPZ file (DFT must be saved as NPZ file)

        Parameters
        ----------
        npz_folder_path: str
            path to input npz folder
        output_plot_file: str
            path to save figure. If None, will show figure instead of saving it
        f_min: int
            minimum frequency to plot in Hz
        f_max: int
            maximum frequency to plot in Hz
        plot_type: str
            whether to plot all channels or average of channels
        channels_list: str
            list of channels to plot (can be a strin of comma-separated values or a list of integers)
        conv_window_size: float
            size of convolution window to smooth the DFT

        Returns
        ----------
    """
    if plot_type not in ['all_channels', 'average_of_channels']:
        raise ValueError(
            'plot_type must be either "all_channels" or "average_of_channels"')

    npz_files = os.listdir(npz_folder_path)
    npz_files = utils.keep_npz_files(npz_files)
    npz_files = utils.sort_file_names(npz_files)
    channels_list = utils.convert_string_to_list(channels_list)
    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files) + 1))
    else:
        channels_list = sorted(channels_list)

    if len(channels_list) > 20:
        fig = plt.figure(figsize=(20, int(len(channels_list) / 2)))
    else:
        fig = plt.figure(figsize=(20, 10))

    ax = fig.subplots(1, 1)

    for channel_index, channel in enumerate(channels_list):
        f, Zxx = data_io.load_npz_dft(os.path.join(
            npz_folder_path, npz_files[channel_index]))
        if f_min is None:
            f_min = np.min(f)
        if f_max is None:
            f_max = np.max(f)
        Zxx_plot = 10 * np.log10(np.abs(Zxx))
        desired_freq_index_low = np.where(
            np.min(abs(f - f_min)) == abs(f - f_min))[0][0]
        desired_freq_index_high = np.where(
            np.min(abs(f - f_max)) == abs(f - f_max))[0][0]
        Zxx_plot = Zxx_plot[desired_freq_index_low:desired_freq_index_high]
        f = f[desired_freq_index_low:desired_freq_index_high]

        if conv_window_size is not None:
            Zxx_plot = np.convolve(
                Zxx_plot,
                np.ones(conv_window_size) /
                conv_window_size,
                mode='same')

        if plot_type == 'all_channels':
            ax.plot(f, Zxx_plot, label=f'Channel {channel}')
        elif plot_type == 'average_of_channels':
            if channel_index == 0:
                Zxx_list = Zxx_plot
            else:
                Zxx_list = np.vstack((Zxx_list, Zxx_plot))

    if plot_type == 'average_of_channels':
        Zxx_plot_avg = np.mean(Zxx_list, axis=0)
        Zxx_plot_std = np.std(Zxx_list, axis=0)
        ax.plot(f, Zxx_plot_avg, label='Average of DFTs', color='k')
        ax.fill_between(
            f,
            Zxx_plot_avg -
            Zxx_plot_std,
            Zxx_plot_avg +
            Zxx_plot_std,
            alpha=0.5,
            color='k')

    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.set_xlim(f_min, f_max)
    ax.legend()

    plt.tight_layout()
    if output_plot_file is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(output_plot_file)):
            os.makedirs(os.path.dirname(output_plot_file))
        plt.savefig(output_plot_file, dpi=600)
        plt.close()


def plot_power_over_time_from_array(power: np.ndarray, t: np.ndarray, channels_map: list,
                                    plot_type: str = 'avg', output_plot_file: str = None) -> None:
    """ Plots power over time from 3D array

            Parameters
            ----------
            power: np.ndarray
                A 2D array containing power values. Shape: (num_channels, num_timepoints)
            t: np.ndarray
                time array (seconds)
            plot_type: str
                whether to plot all channels or average of channels. Must be either 'all' or 'avg'

            Returns
            ----------
        """

    num_channels = power.shape[0]

    if num_channels > 20:
        fig = plt.figure(figsize=(30, int(num_channels / 2)))
    else:
        fig = plt.figure(figsize=(30, 10))

    ax = fig.subplots(1, 1)

    if plot_type == 'avg':
        power_avg = np.mean(power, axis=0)
        power_std = np.std(power, axis=0)
        ax.plot(t, power_avg, color='k')
        ax.fill_between(t, power_avg - power_std, power_avg + power_std, alpha=0.5, color='k')
    elif plot_type == 'all':
        for channel_index in range(num_channels):
            ax.plot(t, power[channel_index, :], label=f'Ch{channels_map[channel_index]+1}')
        ax.legend()
    else:
        raise ValueError('plot_type must be either "avg" or "all"')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (dB)')

    plt.tight_layout()

    if output_plot_file is None:
        plt.show()
    else:
        plt.savefig(output_plot_file, dpi=600)
        plt.close()


def plot_filter_freq_response(filter_args: dict,
                              figure_save_path: str = None) -> None:
    """ Plots filter frequency response

        Parameters
        ----------
        filter_args: dict
            dictionary containing filter parameters
        figure_save_path: str
            path to save figure. If None or not specified, will show figure instead of saving it

        Returns
        ----------
    """

    f, mag, phase, _args = fourier_analysis.calc_freq_response(filter_args)

    _, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(f, mag)
    ax[0].set_ylabel('Magnitude (dB)')
    ax[1].plot(f, phase)
    ax[1].set_ylabel('Phase (rad)')
    ax[1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()

    # plot freq_cutoff as a single line on both plots

    if _args['filter_type'] == 'LPF':
        ax[0].axvline(int(_args['freq_cutoff']), color='k', linestyle='--')
        ax[1].axvline(int(_args['freq_cutoff']), color='k', linestyle='--')
        ax[0].axvspan(0, int(_args['freq_cutoff']), alpha=0.5, color='gray')
        ax[1].axvspan(0, int(_args['freq_cutoff']), alpha=0.5, color='gray')
    elif _args['filter_type'] == 'HPF':
        ax[0].axvline(int(_args['freq_cutoff']), color='k', linestyle='--')
        ax[1].axvline(int(_args['freq_cutoff']), color='k', linestyle='--')
        ax[0].axvspan(int(_args['freq_cutoff']),
                      _args['fs'] / 2, alpha=0.5, color='gray')
        ax[1].axvspan(int(_args['freq_cutoff']),
                      _args['fs'] / 2, alpha=0.5, color='gray')
    elif _args['filter_type'] == 'BPF':
        _args['freq_cutoff'] = utils.convert_string_to_list(
            _args['freq_cutoff'])
        _args['freq_cutoff'] = [int(i) for i in _args['freq_cutoff']]
        ax[0].axvline(_args['freq_cutoff'][0], color='k', linestyle='--')
        ax[0].axvline(_args['freq_cutoff'][1], color='k', linestyle='--')
        ax[1].axvline(_args['freq_cutoff'][0], color='k', linestyle='--')
        ax[1].axvline(_args['freq_cutoff'][1], color='k', linestyle='--')
        ax[0].axvspan(
            _args['freq_cutoff'][0],
            _args['freq_cutoff'][1],
            alpha=0.5,
            color='gray')
        ax[1].axvspan(
            _args['freq_cutoff'][0],
            _args['freq_cutoff'][1],
            alpha=0.5,
            color='gray')

    if figure_save_path is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(figure_save_path)):
            os.makedirs(os.path.dirname(figure_save_path))
        plt.savefig(figure_save_path, dpi=600)
        plt.close()


def plot_filter_freq_response_from_json(
        filter_freq_response_json_file_path: str, figure_save_path: str = None) -> None:
    """ Plots filter frequency response from JSON file

        Parameters
        ----------
        filter_freq_response_json_file_path: str
            path to filter frequency response JSON file
        figure_save_path: str
            path to save figure. If None or not specified, will show figure instead of saving it

        Returns
        ----------
    """

    with open(filter_freq_response_json_file_path, 'r') as f:
        filter_freq_response_dict = json.load(f)

    plot_filter_freq_response(filter_freq_response_dict, figure_save_path)


def plot_mvl_form_array(MI_mat: np.ndarray, freqs_phase: np.ndarray,
                        freqs_amp: np.ndarray, clim: list = None, figure_save_path: str = None) -> None:
    """ Plots MVL from 2D array

        Parameters
        ----------
        MI_mat: np.ndarray
            MVL array
        freqs_phase: np.ndarray
            phase frequencies (Hz)
        freqs_amp: np.ndarray
            amplitude frequencies (Hz)
        clim: list
            colorbar limits. if None or not specified, will use default limits
        figure_save_path: str
            path to save figure. If None or not specified, will show figure instead of saving it

        Returns
        ----------
    """

    MI_mat_plot = MI_mat
    plt.figure(figsize=(24, 12))
    pc = plt.pcolormesh(
        freqs_phase,
        freqs_amp,
        np.transpose(MI_mat_plot),
        cmap=plt.get_cmap('jet'),
        shading='gouraud')
    plt.colorbar(pc)
    plt.xlabel('Phase Freq')
    plt.ylabel('Amp Freq')
    if clim:
        plt.clim(clim[0], clim[1])
    if figure_save_path is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(figure_save_path)):
            os.makedirs(os.path.dirname(figure_save_path))
        plt.savefig(figure_save_path, dpi=600)
        plt.close()


def plot_mvl_from_npz(npz_file_path: str,
                      figure_save_path: str = None) -> None:
    """ Plots MVL from NPZ file

        Parameters
        ----------
        npz_file_path: str
            path to npz file
        figure_save_path: str
            path to save figure. If None or not specified, will show figure instead of saving it

        Returns
        ----------
    """

    MI_mat, freqs_amp, freqs_phase, _ = data_io.load_npz_mvl(npz_file_path)
    plot_mvl_form_array(MI_mat, freqs_phase, freqs_amp,
                        figure_save_path=figure_save_path)
