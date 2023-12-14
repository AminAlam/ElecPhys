import os
import numpy as np
import matplotlib.pyplot as plt
import json

import data_loading
import preprocessing
import utils
import fourier_analysis

def plot_stft_from_npz(input_npz_file, output_plot_file, f_min, f_max, t_min, t_max, db_min, db_max):
    """ Plots STFT from NPZ file (STFT must be saved as NPZ file)
    input:
        input_npz_file: path to input npz file - type: os.PathLike
        output_plot_file: path to output plot file - type: os.PathLike
        f_min: minimum frequency to plot in Hz - type: float
        f_max: maximum frequency to plot in Hz - type: float
        t_min: minimum time to plot in seconds - type: float
        t_max: maximum time to plot in seconds - type: float
        db_min: minimum dB to plot - type: float
        db_max: maximum dB to plot - type: float
    output:
    """
    f, t, Zxx = data_loading.load_npz_stft(input_npz_file)
    Zxx_plot = 10*np.log10(np.abs(Zxx))
    
    if f_min is None:
        f_min = np.min(f)
    if f_max is None:
        f_max = np.max(f)
    if t_min is None:
        t_min = np.min(t)
    if t_max is None:
        t_max = np.max(t)
    if db_min is None:
        db_min = np.min(10*np.log10(np.abs(Zxx)))
    if db_max is None:
        db_max = np.max(10*np.log10(np.abs(Zxx)))

    plot_stft_from_array(Zxx_plot, t, f, f_min, f_max, t_min, t_max, db_min, db_max, output_plot_file)


def plot_avg_stft_from_npz(npz_folder_path, output_plot_file, f_min, f_max, t_min, t_max, db_min, db_max, channels_list=None):
    """ Plots average STFT from NPZ files (STFT must be saved as NPZ file)
    input:
        npz_folder_path: path to input npz folder - type: os.PathLike
        output_plot_file: path to output plot file - type: os.PathLike
        f_min: minimum frequency to plot in Hz - type: float
        f_max: maximum frequency to plot in Hz - type: float
        t_min: minimum time to plot in seconds - type: float
        t_max: maximum time to plot in seconds - type: float
        db_min: minimum dB to plot - type: float
        db_max: maximum dB to plot - type: float
        channels_list: list of channels to plot - type: list
    output:
    """
    
    npz_files = os.listdir(npz_folder_path)
    npz_files = utils.sort_file_names(npz_files)

    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files)+1))
    else:
        channels_list = utils.convert_string_to_list(channels_list)
        channels_list = sorted(channels_list)

    for channel_index, channel in enumerate(channels_list):
        npz_file = npz_files[channel_index]
        f, t, Zxx = data_loading.load_npz_stft(os.path.join(npz_folder_path, npz_file))
        Zxx = 10*np.log10(np.abs(Zxx))
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

    plot_stft_from_array(Zxx_avg, t, f, f_min, f_max, t_min, t_max, db_min, db_max, output_plot_file)

    
def plot_stft_from_array(Zxx, t, f, f_min, f_max, t_min, t_max, db_min, db_max, output_plot_file=None):
    """ Plots STFT from 2D array
    input:
        Zxx: STFT matrix - type: np.ndarray
        t: time vector - type: np.ndarray
        f: frequency vector - type: np.ndarray
        f_min: minimum frequency to plot in Hz - type: float
        f_max: maximum frequency to plot in Hz - type: float
        t_min: minimum time to plot in seconds - type: float
        t_max: maximum time to plot in seconds - type: float
        db_min: minimum dB to plot - type: float
        db_max: maximum dB to plot - type: float
        output_plot_file: path to output plot file - type: os.PathLike
    output:
    """

    desired_freq_index_low = np.where(np.min(abs(f-f_min))==abs(f-f_min))[0][0]
    desired_freq_index_high = np.where(np.min(abs(f-f_max))==abs(f-f_max))[0][0]
    desired_time_index_low = np.where(np.min(abs(t-t_min))==abs(t-t_min))[0][0]
    desired_time_index_high = np.where(np.min(abs(t-t_max))==abs(t-t_max))[0][0]
    Zxx = Zxx[desired_freq_index_low:desired_freq_index_high, desired_time_index_low:desired_time_index_high]
    f = f[desired_freq_index_low:desired_freq_index_high]
    t = t[desired_time_index_low:desired_time_index_high]
    
    pc = plt.pcolormesh(t, f, Zxx, cmap=plt.get_cmap('jet'), shading='auto')
    plt.colorbar(pc)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.clim(db_min, db_max)
    plt.text(1.01, 0.5, 'Power (dB)', va='center', rotation=-90, transform=plt.gca().transAxes)
    plt.tight_layout()

    if output_plot_file is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(output_plot_file)):
            os.makedirs(os.path.dirname(output_plot_file))
        plt.savefig(output_plot_file, dpi=600)
        plt.close()


def plot_signals_from_npz(npz_folder_path, output_plot_file, t_min, t_max, channels_list=None, normalize=False):
    """ Plots signals from NPZ file
    input:
        npz_folder_path: path to input npz folder - type: os.PathLike
        output_plot_file: path to output plot file - type: os.PathLike
        t_min: minimum time to plot in seconds - type: float
        t_max: maximum time to plot in seconds - type: float
        channels_list: list of channels to plot - type: list
    output:
    """
    npz_files = os.listdir(npz_folder_path)
    npz_files = utils.sort_file_names(npz_files)

    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files)+1))
    else:
        channels_list = utils.convert_string_to_list(channels_list)
        channels_list = sorted(channels_list)

    if len(channels_list) > 20:
        fig = plt.figure(figsize=(30, len(channels_list)))
    else:
        fig = plt.figure(figsize=(30, 10))

    ax = fig.subplots(len(channels_list), 1, sharex=True)

    for channel_index, channel in enumerate(channels_list):
        npz_file = npz_files[channel_index]
        signal_chan, fs = data_loading.load_npz(os.path.join(npz_folder_path, npz_file))
        if normalize:
            signal_chan = preprocessing.normalize(signal_chan)
        t = np.linspace(0, len(signal_chan)/fs, len(signal_chan))
        if t_min is None:
            t_min = np.min(t)
        if t_max is None:
            t_max = np.max(t)

        desired_time_index_low = np.where(np.min(abs(t-t_min))==abs(t-t_min))[0][0]
        desired_time_index_high = np.where(np.min(abs(t-t_max))==abs(t-t_max))[0][0]
        signal_chan = signal_chan[desired_time_index_low:desired_time_index_high]
        t = t[desired_time_index_low:desired_time_index_high]
        ax[channel_index].plot(t, signal_chan, color='k')
        ax[channel_index].set_ylabel(f'Channel {channel}')
        ax[channel_index].spines['top'].set_visible(False)
        ax[channel_index].spines['right'].set_visible(False)
        if channel_index != len(channels_list)-1:
            ax[channel_index].spines['bottom'].set_visible(False)
        ax[channel_index].tick_params(axis='both', which='both', length=0)
        ax[channel_index].set_yticks([])
    ax[-1].set_xlabel('Time (s)')
    ax[-1].set_xlim(t_min, t_max)
    plt.tight_layout()

    if output_plot_file is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(output_plot_file)):
            os.makedirs(os.path.dirname(output_plot_file))
        plt.tight_layout()
        plt.savefig(output_plot_file, dpi=600)
        plt.close()


def plot_dft_from_npz(npz_folder_path, output_plot_file, f_min, f_max, plot_type, channels_list=None, conv_window_size=None):
    """ Plots DFT from NPZ file (DFT must be saved as NPZ file)
    input:
        npz_folder_path: path to input npz folder - type: os.PathLike
        output_plot_file: path to output plot file - type: os.PathLike
        f_min: minimum frequency to plot in Hz - type: float
        f_max: maximum frequency to plot in Hz - type: float
        plot_type: type of plot - type: str
        channels_list: list of channels to plot - type: list
        conv_window_size: size of convolution window to make plot smoother - type: int
    output:
    """
    if plot_type not in ['all_channels', 'average_of_channels']:
        raise ValueError('plot_type must be either "all_channels" or "average_of_channels"')

    npz_files = os.listdir(npz_folder_path)
    npz_files = utils.sort_file_names(npz_files)

    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files)+1))
    else:
        channels_list = utils.convert_string_to_list(channels_list)
        channels_list = sorted(channels_list)

    if len(channels_list) > 20:
        fig = plt.figure(figsize=(20, int(len(channels_list)/2)))
    else:
        fig = plt.figure(figsize=(20, 10))

    ax = fig.subplots(1, 1)

    for channel_index, channel in enumerate(channels_list):
        f, Zxx = data_loading.load_npz_dft(os.path.join(npz_folder_path, npz_files[channel_index]))
        if f_min is None:
            f_min = np.min(f)
        if f_max is None:
            f_max = np.max(f)
        Zxx_plot = 10*np.log10(np.abs(Zxx))
        desired_freq_index_low = np.where(np.min(abs(f-f_min))==abs(f-f_min))[0][0]
        desired_freq_index_high = np.where(np.min(abs(f-f_max))==abs(f-f_max))[0][0]
        Zxx_plot = Zxx_plot[desired_freq_index_low:desired_freq_index_high]
        f = f[desired_freq_index_low:desired_freq_index_high]

        if conv_window_size is not None:
            Zxx_plot = np.convolve(Zxx_plot, np.ones(conv_window_size)/conv_window_size, mode='same')

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
        ax.fill_between(f, Zxx_plot_avg-Zxx_plot_std, Zxx_plot_avg+Zxx_plot_std, alpha=0.5, color='k')

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


def plot_filter_freq_response(filter_args, figure_save_path=None):
    """ Plots filter frequency response
    input:
        filter_args: dictionary containing filter parameters - type: dict
        figure_save_path: path to save figure - type: str
        output:
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
        ax[0].axvspan(int(_args['freq_cutoff']), _args['fs']/2, alpha=0.5, color='gray')
        ax[1].axvspan(int(_args['freq_cutoff']), _args['fs']/2, alpha=0.5, color='gray')
    elif _args['filter_type'] == 'BPF':
        _args['freq_cutoff'] = utils.convert_string_to_list(_args['freq_cutoff'])
        _args['freq_cutoff'] = [int(i) for i in _args['freq_cutoff']]
        ax[0].axvline(_args['freq_cutoff'][0], color='k', linestyle='--')
        ax[0].axvline(_args['freq_cutoff'][1], color='k', linestyle='--')
        ax[1].axvline(_args['freq_cutoff'][0], color='k', linestyle='--')
        ax[1].axvline(_args['freq_cutoff'][1], color='k', linestyle='--')
        ax[0].axvspan(_args['freq_cutoff'][0], _args['freq_cutoff'][1], alpha=0.5, color='gray')
        ax[1].axvspan(_args['freq_cutoff'][0], _args['freq_cutoff'][1], alpha=0.5, color='gray')
    
    if figure_save_path is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(figure_save_path)):
            os.makedirs(os.path.dirname(figure_save_path))
        plt.savefig(figure_save_path, dpi=600)
        plt.close()

def plot_filter_freq_response_from_json(filter_freq_response_json_file_path, figure_save_path=None):
    """ Plots filter frequency response from JSON file
    input:
        filter_freq_response_json_file_path: path to filter frequency response JSON file - type: str
        figure_save_path: path to save figure - type: str
    output:
    """

    with open(filter_freq_response_json_file_path, 'r') as f:
        filter_freq_response_dict = json.load(f)

    plot_filter_freq_response(filter_freq_response_dict, figure_save_path)

    
def plot_mvl_form_array(MI_mat, freqs_phase, freqs_amp, clim=None, figure_save_path=None):
    """ Plots MVL from 2D array
    input:
        MI_mat: MVL matrix - type: np.ndarray
        freqs_phase: phase frequency vector - type: np.ndarray
        freqs_amp: amplitude frequency vector - type: np.ndarray
        clim: color limit - type: tuple
        figure_save_path: path to save figure - type: str
    output:
    """
    
    MI_mat_plot = MI_mat
    plt.figure(figsize=(24, 12))
    pc = plt.pcolormesh(freqs_phase, freqs_amp, np.transpose(MI_mat_plot), cmap=plt.get_cmap('jet'), shading='gouraud')
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


def plot_mvl_from_npz(npz_file_path, figure_save_path=None):
    """ Plots MVL from NPZ file
    input:
        npz_file_path: path to NPZ file - type: os.PathLike
        figure_save_path: path to save figure - type: str
    output:
    """
    
    MI_mat, freqs_amp, freqs_phase, _ = data_loading.load_npz_mvl(npz_file_path)
    plot_mvl_form_array(MI_mat, freqs_phase, freqs_amp, figure_save_path=figure_save_path)