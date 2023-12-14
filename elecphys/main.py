import click
import sys
import os

import conversion
import preprocessing
import fourier_analysis
import visualization
import dimensionality_reduction
from handlers import ErrorHandler
error_handler = ErrorHandler().error_handler

@click.group(chain=True, help="ElecPhys is a Python package for electrophysiology data analysis. It provides tools for data loading, conversion, preprocessing, and visualization.")
@click.pass_context
def cli(ctx):
    pass

### Conversion ###
@cli.command('convert_rhd_to_mat', help='Converts RHD files to mat files using RHD to MAT converter (needs MATLAB installed')
@click.option('--folder_path', '-f', help='Path to folder containing RHD files', required=True, type=str)
@click.option('--output_mat_file', '-o', help='Path to output mat file', required=True, type=str, default='output.mat', show_default=True)
@click.option('--ds_factor', '-d', help='Downsample factor', required=False, type=int, default=1, show_default=True)
@click.pass_context
@error_handler
def convert_rhd_to_mat(ctx, folder_path, output_mat_file, ds_factor):
    """ Converts RHD files to mat files using RHD to MAT converter written in MATLAB
    input:
        folder_path: path to folder containing RHD files - type: str
        output_mat_file: path to output mat file - type: str
        ds_factor: downsample factor - type: int
    output:
    """

    Warning('** This command requires MATLAB to be installed on your system.\n')
    print('--- Converting RHD files to MAT files...')
    conversion.convert_rhd_to_mat(folder_path, output_mat_file, ds_factor)
    print('--- Conversion complete.\n\n')


@cli.command('convert_mat_to_npz', help='Converts MAT files to NPZ files using MAT to NPZ converter')
@click.option('--mat_file', '-m', help='Path to mat file', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder', required=True, type=str, default='output_npz', show_default=True)
@click.option('--notch_filter_freq', '-n', help='Notch filter frequency in Hz', required=False, type=int, default=50, show_default=True)
@click.pass_context
@error_handler
def convert_mat_to_npz(ctx, mat_file, output_npz_folder, notch_filter_freq):
    """ Converts MAT files to NPZ files
    input:
        mat_file: path to mat file - type: str
        output_npz_folder: path to output npz folder - type: str
        notch_filter_freq: notch filter frequency - type: int
    output:
    """
    
    print('--- Converting MAT files to NPZ files...')
    conversion.convert_mat_to_npz(mat_file, output_npz_folder, notch_filter_freq)
    print('--- Conversion complete.\n\n')
### Conversion ###


### Preprocessing ###
@cli.command('zscore_normalize_npz', help='Z-score normalizes NPZ files')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder', required=True, type=str, default='output_npz', show_default=True)
@click.pass_context
@error_handler
def zscore_normalize_npz(ctx, input_npz_folder, output_npz_folder):
    """ Z-score normalizes NPZ files
    input:
        input_npz_folder: path to input npz folder - type: str
        output_npz_folder: path to output npz folder - type: str
    output:
    """

    print('--- Z-score normalizing NPZ files...')
    preprocessing.zscore_normalize_npz(input_npz_folder, output_npz_folder)
    print('--- Normalization complete.\n\n')


@cli.command('normalize_npz', help='Normalizes NPZ files')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder', required=True, type=str, default='output_npz', show_default=True)
@click.pass_context
@error_handler
def normalize_npz(ctx, input_npz_folder, output_npz_folder):
    """ Normalizes NPZ files
    input:
        input_npz_folder: path to input npz folder - type: str
        output_npz_folder: path to output npz folder - type: str
    output:
    """

    print('--- Normalizing NPZ files...')
    preprocessing.normalize_npz(input_npz_folder, output_npz_folder)
    print('--- Normalization complete.\n\n')
### Preprocessing ###


### Fourier Analysis ###
@cli.command('stft_numeric_output_from_npz', help='Computes STFT and saves results as NPZ files')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder to save STFT results', required=True, type=str, show_default=True)
@click.option('--window_size', '-w', help='Window size in seconds', required=True, type=float)
@click.option('--overlap', '-ov', help='Overlap in seconds', required=True, type=float)
@click.option('--window_type', '-wt', help='Window type', required=False, type=str, default='hann', show_default=True)
@click.pass_context
@error_handler
def stft_numeric_output(ctx, input_npz_folder, output_npz_folder, window_size, overlap, window_type):
    """ Computes STFT and saves results as NPZ files
    input:
        input_npz_folder: path to input npz folder - type: str
        output_npz_folder: path to output npz folder to save STFT results - type: str
        window_size: window size in seconds - type: float
        overlap: overlap in seconds - type: float
        window_type: window type - type: str
    output:
    """

    print('--- Computing STFT and saving results as NPZ files...')
    fourier_analysis.stft_numeric_output_from_npz(input_npz_folder, output_npz_folder, window_size, overlap, window_type)
    print('--- STFT computation complete.\n\n')


@cli.command('dft_numeric_output_from_npz', help='Computes DFT and saves results as NPZ files')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder to save DFT results', required=True, type=str, show_default=True)
@click.pass_context
@error_handler
def dft_numeric_output(ctx, input_npz_folder, output_npz_folder):
    """ Computes DFT and saves results as NPZ files
    input:
        input_npz_folder: path to input npz folder - type: str
        output_npz_folder: path to output npz folder to save DFT results - type: str
    output:
    """

    print('--- Computing DFT and saving results as NPZ files...')
    fourier_analysis.dft_numeric_output_from_npz(input_npz_folder, output_npz_folder)
    print('--- DFT computation complete.\n\n')


@cli.command('frequncy_domain_filter', help='Filtering in frequency domain using butterworth filter and saves results as NPZ files')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder to save filtered signals', required=True, type=str, show_default=True)
@click.option('--filter_type', '-ft', help='Filter type. LPF (low-pass filter), HPF (high-pass filter), or BPF (band-pass filter)', required=True, type=str, default='LPF', show_default=True)
@click.option('--freq_cutoff', '-fc', help='Frequency cutoff in Hz. If filter_type is LPF or HPF, then freq_cutoff is a single value. If filter_type is BPF, then freq_cutoff is a list of two values', required=True, type=str, default=None, show_default=True)
@click.option('--filter_order', '-fo', help='Filter order', required=True, type=int, default=4, show_default=True)
@click.pass_context
@error_handler
def frequncy_domain_filter(ctx, input_npz_folder, output_npz_folder, filter_type, freq_cutoff, filter_order):
    """ Filtering in frequency domain using butterworth filter and saves results as NPZ files
    input:
        input_npz_folder: path to input npz folder - type: str
        output_npz_folder: path to output npz folder to save filtered signals - type: str
        filter_type: filter type. LPF (low-pass filter), HPF (high-pass filter), or BPF (band-pass filter) - type: str
        freq_cutoff: frequency cutoff in Hz. If filter_type is LPF or HPF, then freq_cutoff is a single value. If filter_type is BPF, then freq_cutoff is a list of two values - type: str
        order: filter order - type: int
    output:
    """

    filter_args = {'filter_type': filter_type, 'freq_cutoff': freq_cutoff, 'filter_order': filter_order}
    print('--- Filtering in frequency domain using butterworth filter and saving results as NPZ files...')
    fourier_analysis.butterworth_filtering_from_npz(input_npz_folder, output_npz_folder,filter_args)
    print('--- Filtering complete.\n\n')
### Fourier Analysis ###


### Visualization ###
@cli.command('plot_stft', help='Plots STFT from NPZ file')
@click.option('--input_npz_file', '-i', help='Path to input npz file', required=True, type=str)
@click.option('--output_plot_file', '-o', help='Path to output plot file', required=True, type=str, default=None, show_default=True)
@click.option('--f_min', '-fmin', help='Minimum frequency to plot in Hz', required=False, type=float, default=None, show_default=True)
@click.option('--f_max', '-fmax', help='Maximum frequency to plot in Hz', required=False, type=float, default=None, show_default=True)
@click.option('--t_min', '-tmin', help='Minimum time to plot in seconds', required=False, type=float, default=None, show_default=True)
@click.option('--t_max', '-tmax', help='Maximum time to plot in seconds', required=False, type=float, default=None, show_default=True)
@click.option('--db_min', '-dbmin', help='Minimum dB to plot', required=False, type=float, default=None, show_default=True)
@click.option('--db_max', '-dbmax', help='Maximum dB to plot', required=False, type=float, default=None, show_default=True)
@click.pass_context
@error_handler
def plot_stft(ctx, input_npz_file, output_plot_file, f_min, f_max, t_min, t_max, db_min, db_max):
    """ Plots STFT from NPZ file
    input:
        input_npz_file: path to input npz file - type: str
        output_plot_file: path to output plot file - type: str
        f_min: minimum frequency to plot in Hz - type: float
        f_max: maximum frequency to plot in Hz - type: float
        t_min: minimum time to plot in seconds - type: float
        t_max: maximum time to plot in seconds - type: float
        db_min: minimum dB to plot - type: float
        db_max: maximum dB to plot - type: float
    output:
    """

    print('--- Plotting STFT...')
    visualization.plot_stft_from_npz(input_npz_file, output_plot_file, f_min, f_max, t_min, t_max, db_min, db_max)
    print('--- Plotting complete.\n\n')


@cli.command('plot_avg_stft', help='Plots average STFT from NPZ files')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder containing STFT NPZ files', required=True, type=str)
@click.option('--output_plot_file', '-o', help='Path to output plot file', required=True, type=str, default=None, show_default=True)
@click.option('--f_min', '-fmin', help='Minimum frequency to plot in Hz', type=float, default=None, show_default=True, required=False)
@click.option('--f_max', '-fmax', help='Maximum frequency to plot in Hz', type=float, default=None, show_default=True, required=False)
@click.option('--t_min', '-tmin', help='Minimum time to plot in seconds', type=float, default=None, show_default=True, required=False)
@click.option('--t_max', '-tmax', help='Maximum time to plot in seconds', type=float, default=None, show_default=True, required=False)
@click.option('--db_min', '-dbmin', help='Minimum dB to plot', type=float, default=None, show_default=True, required=False)
@click.option('--db_max', '-dbmax', help='Maximum dB to plot', type=float, default=None, show_default=True, required=False)
@click.option('--channels_list', '-cl', help='List of channels to plot, if None then all of the channels will be plotted', required=False, type=list, default=None, show_default=True)
@click.pass_context
@error_handler
def plot_avg_stft(ctx, input_npz_folder, output_plot_file, f_min, f_max, t_min, t_max, db_min, db_max, channels_list):
    """ Plots average STFT from NPZ files
    input:
        input_npz_folder: path to input npz folder - type: str
        output_plot_file: path to output plot file - type: str
        f_min: minimum frequency to plot in Hz - type: float
        f_max: maximum frequency to plot in Hz - type: float
        t_min: minimum time to plot in seconds - type: float
        t_max: maximum time to plot in seconds - type: float
        db_min: minimum dB to plot - type: float
        db_max: maximum dB to plot - type: float
        channels_list: list of channels to plot - type: list
    output:
    """

    print('--- Plotting average STFT...')
    visualization.plot_avg_stft_from_npz(input_npz_folder, output_plot_file, f_min, f_max, t_min, t_max, db_min, db_max, channels_list)
    print('--- Plotting complete.\n\n')


@cli.command('plot_signal', help='Plots signals from NPZ file')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder', required=True, type=str)
@click.option('--output_plot_file', '-o', help='Path to output plot file', required=True, type=str, default=None, show_default=True)
@click.option('--t_min', '-tmin', help='Minimum time to plot in seconds', required=False, type=float, default=None, show_default=True)
@click.option('--t_max', '-tmax', help='Maximum time to plot in seconds', required=False, type=float, default=None, show_default=True)
@click.option('--channels_list', '-cl', help='List of channels to plot, if None then all of the channels will be plotted', required=False, type=list, default=None, show_default=True)
@click.option('--normalize', '-n', help='Normalize signals. If true, each channel will be normalized', required=False, type=bool, default=False, show_default=True)
@click.pass_context
@error_handler
def plot_signal(ctx, input_npz_folder, output_plot_file, t_min, t_max, channels_list, normalize):
    """ Plots signals from NPZ file
    input:
        input_npz_folder: path to input npz folder - type: str
        output_plot_file: path to output plot file - type: str
        t_min: minimum time to plot in seconds - type: float
        t_max: maximum time to plot in seconds - type: float
        channels_list: list of channels to plot - type: list
        normalize: normalize signals - type: bool
    output:
    """

    print('--- Plotting signals...')
    visualization.plot_signals_from_npz(input_npz_folder, output_plot_file, t_min, t_max, channels_list, normalize)
    print('--- Plotting complete.\n\n')


@cli.command('plot_dft', help='Plots DFT from NPZ file')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder', required=True, type=str)
@click.option('--output_plot_file', '-o', help='Path to output plot file', required=True, type=str, default=None, show_default=True)
@click.option('--f_min', '-fmin', help='Minimum frequency to plot in Hz', required=False, type=float, default=None, show_default=True)
@click.option('--f_max', '-fmax', help='Maximum frequency to plot in Hz', required=False, type=float, default=None, show_default=True)
@click.option('--channels_list', '-cl', help='List of channels to plot, if None then all of the channels will be plotted', required=False, type=list, default=None, show_default=True)
@click.option('--plot_type', '-pt', help='Plot type. If "all_channels", then all channels will be plotted in one figure. If "average_of_channels", then average of channels will be plotted in one figure with errorbar', required=True, type=str, default='average_of_channels', show_default=True)
@click.option('--conv_window_size', '-cws', help='Convolution window size in seconds', required=False, type=int, default=None, show_default=True)
@click.pass_context
@error_handler
def plot_dft(ctx, input_npz_folder, output_plot_file, f_min, f_max, channels_list, plot_type, conv_window_size):
    """ Plots DFT from NPZ file
    input:
        input_npz_folder: path to input npz folder - type: str
        output_plot_file: path to output plot file - type: str
        f_min: minimum frequency to plot in Hz - type: float
        f_max: maximum frequency to plot in Hz - type: float
        channels_list: list of channels to plot - type: list
        plot_type: plot type - type: str
        conv_window_size: convolution window size in seconds - type: float
    output:
    """

    print('--- Plotting DFT...')
    visualization.plot_dft_from_npz(input_npz_folder, output_plot_file, f_min, f_max, plot_type, channels_list, conv_window_size)
    print('--- Plotting complete.\n\n')


@cli.command('plot_filter_freq_response', help='Plots filter frequency response')
@click.option('--filter_type', '-ft', help='Filter type. LPF (low-pass filter), HPF (high-pass filter), or BPF (band-pass filter)', required=True, type=str, default='LPF', show_default=True)
@click.option('--freq_cutoff', '-fc', help='Frequency cutoff in Hz. If filter_type is LPF or HPF, then freq_cutoff is a single value. If filter_type is BPF, then freq_cutoff is a list of two values', required=True, type=str, default=None, show_default=True)
@click.option('--filter_order', '-fo', help='Filter order', required=True, type=int, default=4, show_default=True)
@click.option('--frequency_sampling', '-fs', help='Frequency sampling in Hz', required=True, type=int, show_default=True)
@click.option('--output_plot_file', '-o', help='Path to output plot file', required=True, type=str, default=None, show_default=True)
@click.pass_context
@error_handler
def plot_filter_freq_response(ctx, filter_type, freq_cutoff, filter_order, frequency_sampling, output_plot_file):
    """ Plots filter frequency response
    input:
        filter_type: filter type. LPF (low-pass filter), HPF (high-pass filter), or BPF (band-pass filter) - type: str
        freq_cutoff: frequency cutoff in Hz. If filter_type is LPF or HPF, then freq_cutoff is a single value. If filter_type is BPF, then freq_cutoff is a list of two values - type: str
        order: filter order - type: int
        output_plot_file: path to output plot file - type: str
    output:
    """
    
    filter_args = {'filter_type': filter_type, 'freq_cutoff': freq_cutoff, 'filter_order': filter_order, 'fs': frequency_sampling}
    print('--- Plotting filter frequency response...')
    visualization.plot_filter_freq_response(filter_args, output_plot_file)
    print('--- Plotting complete.\n\n')
### Visualization ###


### Dimensionality Reduction ###
@cli.command('pca_from_npz', help='Computes PCA from NPZ files')
@click.option('--input_npz_folder', '-i', help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder to save PCA results', required=True, type=str, show_default=True)
@click.option('--n_components', '-n', help='Number of components to keep after applying the PCA', required=True, type=int, default=None, show_default=True)
@click.option('--matrix_whitenning', '-mw', help='Matrix whitening boolean. If true, the singular values are divided by n_samples', required=False, type=bool, default=False, show_default=True)
@click.option('--channels_list', '-cl', help='List of channels to apply PCA, if None then all of the channels will be applied', required=False, type=list, default=None, show_default=True)
@click.pass_context
@error_handler
def pca_from_npz(ctx, input_npz_folder, output_npz_folder, n_components, matrix_whitenning, channels_list):
    """ Computes PCA from NPZ files
    input:
        input_npz_folder: path to input npz folder - type: str
        output_npz_folder: path to output npz folder to save PCA results - type: str
        n_components: number of components to keep after applying the PCA - type: int
        matrix_whitenning: matrix whitening boolean. If true, the singular values are divided by n_samples - type: bool
        channels_list: list of channels to apply PCA, if None then all of the channels will be applied - type: list
    output:
    """

    print('--- Computing PCA and saving results as NPZ files...')
    dimensionality_reduction.pca_from_npz(input_npz_folder, output_npz_folder, n_components, matrix_whitenning, channels_list)
    print('--- PCA computation complete.\n\n')



def main():
    cli()



if __name__ == '__main__':
    main()