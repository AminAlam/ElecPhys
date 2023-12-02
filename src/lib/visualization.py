import os
import numpy as np
import matplotlib.pyplot as plt

import data_loading

def plot_stft_from_npz(input_npz_file, output_plot_file, f_min, f_max, t_min, t_max, db_min, db_max):
    """ Plots STFT from NPZ file
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
        plt.savefig(output_plot_file)
        plt.close()
