import os
import numpy as np
import matplotlib.pyplot as plt

import data_loading
import preprocessing

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

    desired_freq_index_low = np.where(np.min(abs(f-f_min))==abs(f-f_min))[0][0]
    desired_freq_index_high = np.where(np.min(abs(f-f_max))==abs(f-f_max))[0][0]
    desired_time_index_low = np.where(np.min(abs(t-t_min))==abs(t-t_min))[0][0]
    desired_time_index_high = np.where(np.min(abs(t-t_max))==abs(t-t_max))[0][0]
    Zxx_plot = Zxx_plot[desired_freq_index_low:desired_freq_index_high, desired_time_index_low:desired_time_index_high]
    f = f[desired_freq_index_low:desired_freq_index_high]
    t = t[desired_time_index_low:desired_time_index_high]
    pc = plt.pcolormesh(t, f, Zxx_plot, cmap=plt.get_cmap('jet'), shading='auto')
    plt.colorbar(pc)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.clim(db_min, db_max)    
    plt.text(1.01, 0.5, 'Power (dB)', va='center', rotation=-90, transform=plt.gca().transAxes)
    if output_plot_file is None:
        plt.show()
    else:
        if not os.path.exists(os.path.dirname(output_plot_file)):
            os.makedirs(os.path.dirname(output_plot_file))
        plt.tight_layout()
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
    npz_files.sort()
    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files)+1))
    else:
        channels_list = sorted(channels_list)
    fig = plt.figure(figsize=(30, len(channels_list)))
    ax = fig.subplots(len(channels_list), 1, sharex=True)
    for channel in channels_list:
        npz_file = npz_files[channel-1]
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
        ax[channel-1].plot(t, signal_chan, color='k')
        ax[channel-1].set_ylabel(f'Channel {channel}')
        ax[channel-1].spines['top'].set_visible(False)
        ax[channel-1].spines['right'].set_visible(False)
        if channel != channels_list[-1]:
            ax[channel-1].spines['bottom'].set_visible(False)
        ax[channel-1].tick_params(axis='both', which='both', length=0)
        ax[channel-1].set_yticks([])
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
    npz_files.sort()
    if channels_list is None:
        channels_list = tuple(range(1, len(npz_files)+1))
    else:
        channels_list = sorted(channels_list)

    fig = plt.figure(figsize=(20, int(len(channels_list)/2)))
    ax = fig.subplots(1, 1)
    for channel in channels_list:
        f, Zxx = data_loading.load_npz_dft(os.path.join(npz_folder_path, npz_files[channel-1]))
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
            if channel == 1:
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
