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
@click.option('--verbose', '-v', help='Verbose mode', required=False,
              type=bool, default=False, show_default=True, is_flag=True)
@click.option('--debug', '-d', help='Debug mode', required=False,
              type=bool, default=False, show_default=True, is_flag=True)
@click.pass_context
def cli(ctx, verbose: bool = False, debug: bool = False) -> None:
    """ ElecPhys is a Python package for electrophysiology data analysis. It provides tools for data loading, conversion, preprocessing, and visualization.

        Parameters
        ----------
        verbose: bool
            verbose mode. If not specified, the default value is False
        debug: bool
            debug mode. If not specified, the default value is False

        Returns
        ----------
    """

    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    if verbose:
        print('--- Verbose mode is on.\n\n')
        os.environ['ELECPHYS_VERBOSE'] = 'True'

    ctx.obj['DEBUG'] = debug
    if debug:
        print('--- Debug mode is on.\n\n')
        os.environ['ELECPHYS_DEBUG'] = 'True'


### Conversion ###
@cli.command('convert_rhd_to_mat',
             help='Converts RHD files to mat files using RHD to MAT converter (needs MATLAB installed')
@click.option('--folder_path', '-f',
              help='Path to folder containing RHD files', required=True, type=str)
@click.option('--output_mat_file', '-o', help='Path to output mat file',
              required=True, type=str, default='output.mat', show_default=True)
@click.option('--ds_factor', '-d', help='Downsample factor',
              required=False, type=int, default=1, show_default=True)
@click.pass_context
@error_handler
def convert_rhd_to_mat(ctx, folder_path: str,
                       output_mat_file: str = 'output.mat', ds_factor: int = 1) -> None:
    """ Converts RHD files to mat files using RHD to MAT converter written in MATLAB

        Parameters
        ----------
        folder_path: str
            path to folder containing RHD files
        output_mat_file: str
            path to output mat file. If the file already exists, it will be overwritten. If not specified, the default value is 'output.mat'
        ds_factor: int
            downsample factor. If not specified, the default value is 1

        Returns
        ----------
    """

    Warning('** This command requires MATLAB to be installed on your system.\n')
    print('--- Converting RHD files to MAT files...')
    conversion.convert_rhd_to_mat(folder_path, output_mat_file, ds_factor)
    print('--- Conversion complete.\n\n')


@cli.command('convert_mat_to_npz',
             help='Converts MAT files to NPZ files using MAT to NPZ converter')
@click.option('--mat_file', '-m', help='Path to mat file',
              required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder',
              required=True, type=str, default='output_npz', show_default=True)
@click.option('--notch_filter_freq', '-n', help='Notch filter frequency in Hz',
              required=False, type=int, default=50, show_default=True)
@click.pass_context
@error_handler
def convert_mat_to_npz(ctx, mat_file: str, output_npz_folder: str = 'output_npz',
                       notch_filter_freq: int = 50) -> None:
    """ Converts MAT files to NPZ files

        Parameters
        ----------
        mat_file: str
            path to mat file
        output_npz_folder: str
            path to output npz folder. If the folder already exists, it will be overwritten. If not specified, the default value is 'output_npz'
        notch_filter_freq: int
            notch filter frequency in Hz. If not specified, the default value is 50. It should be 0 (no filtering), 50 (Hz), or 60 (Hz)

        Returns
        ----------
    """

    print('--- Converting MAT files to NPZ files...')
    conversion.convert_mat_to_npz(
        mat_file,
        output_npz_folder,
        notch_filter_freq)
    print('--- Conversion complete.\n\n')
### Conversion ###


### Preprocessing ###
@cli.command('zscore_normalize_npz', help='Z-score normalizes NPZ files')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder',
              required=True, type=str, default='output_npz_z_normalized', show_default=True)
@click.pass_context
@error_handler
def zscore_normalize_npz(ctx, input_npz_folder: str,
                         output_npz_folder: str = 'output_npz_z_normalized') -> None:
    """ Z-score normalizes NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder. If the folder already exists, it will be overwritten. If not specified, the default value is 'output_npz_z_normalized'

        Returns
        ----------
    """

    print('--- Z-score normalizing NPZ files...')
    preprocessing.zscore_normalize_npz(input_npz_folder, output_npz_folder)
    print('--- Normalization complete.\n\n')


@cli.command('normalize_npz', help='Normalizes NPZ files')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder',
              required=True, type=str, default='output_npz_normalized', show_default=True)
@click.pass_context
@error_handler
def normalize_npz(ctx, input_npz_folder: str,
                  output_npz_folder: str = 'output_npz_normalized') -> None:
    """ Normalizes NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder. If the folder already exists, it will be overwritten. If not specified, the default value is 'output_npz_normalized'

        Returns
        ----------
    """

    print('--- Normalizing NPZ files...')
    preprocessing.normalize_npz(input_npz_folder, output_npz_folder)
    print('--- Normalization complete.\n\n')


@cli.command('re_reference_npz',
             help='re-references NPZ files and save them as NPZ files')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder',
              required=True, type=str, default='output_npz_avg_rereferenced', show_default=True)
@click.option('--ignore_channels',
              '-ic',
              help='List of channels to ignore (e.g EMG, EOG, etc.). If None, then no channels will be ignored (e.g. --ignore_channels "[1,2,3]").',
              required=False,
              type=str,
              default=None,
              show_default=True)
@click.option('--rr_channel',
              '-rrc',
              help='Channel to re-reference signals to. If None, signals will be re-referenced to the average of all channels',
              required=False,
              type=int,
              default=None,
              show_default=True)
@click.pass_context
@error_handler
def re_reference_npz(ctx, input_npz_folder: str, output_npz_folder: str = 'output_npz_avg_rereferenced',
                     ignore_channels: str = None, rr_channel: int = None) -> None:
    """ Re-references NPZ files and save them as NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder. If the folder already exists, it will be overwritten. If not specified, the default value is 'output_npz_avg_rereferenced'
        ignore_channels: str
            list of channels to ignore (e.g EMG, EOG, etc.). If None, then no channels will be ignored. Either a list of channel indexes or a string of channel indexes separated by commas
        rr_channel: int
            channel to re-reference signals to. If None, signals will be re-referenced to the average of all channels. If not specified, the default value is None

        Returns
        ----------
    """

    print('--- Re-referencing NPZ files...')
    preprocessing.re_reference_npz(
        input_npz_folder,
        output_npz_folder,
        ignore_channels,
        rr_channel)
    print('--- Re-referencing complete.\n\n')
### Preprocessing ###


### Fourier Analysis ###
@cli.command('stft_numeric_output_from_npz',
             help='Computes STFT and saves results as NPZ files')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder to save STFT results',
              required=True, type=str, show_default=True, default='output_npz_stft')
@click.option('--window_size', '-w',
              help='Window size in seconds', required=True, type=float)
@click.option('--overlap', '-ov', help='Overlap in seconds',
              required=True, type=float)
@click.option('--window_type', '-wt', help='Window type',
              required=False, type=str, default='hann', show_default=True)
@click.pass_context
@error_handler
def stft_numeric_output(ctx, input_npz_folder: str, window_size: float, overlap: float,
                        window_type: str = 'hann', output_npz_folder: str = 'output_npz_normalized') -> None:
    """ Computes STFT and saves results as NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder to save STFT results. If the folder already exists, it will be overwritten. If not specified, the default value is 'output_npz_stft'
        window_size: float
            window size in seconds
        overlap: float
            overlap in seconds
        window_type: str
            window type. If not specified, the default value is 'hann'. It should be a window type supported by scipy.signal.get_window()

        Returns
        ----------
    """

    print('--- Computing STFT and saving results as NPZ files...')
    fourier_analysis.stft_numeric_output_from_npz(
        input_npz_folder, output_npz_folder, window_size, overlap, window_type)
    print('--- STFT computation complete.\n\n')


@cli.command('dft_numeric_output_from_npz',
             help='Computes DFT and saves results as NPZ files')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder to save DFT results',
              required=True, type=str, show_default=True, default='output_npz_dft')
@click.pass_context
@error_handler
def dft_numeric_output(ctx, input_npz_folder: str,
                       output_npz_folder: str = 'output_npz_dft') -> None:
    """ Computes DFT and saves results as NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder to save DFT results. If the folder already exists, it will be overwritten. If not specified, the default value is 'output_npz_dft'

        Returns
        ----------
    """

    print('--- Computing DFT and saving results as NPZ files...')
    fourier_analysis.dft_numeric_output_from_npz(
        input_npz_folder, output_npz_folder)
    print('--- DFT computation complete.\n\n')


@cli.command('frequncy_domain_filter',
             help='Filtering in frequency domain using butterworth filter and saves results as NPZ files')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder to save filtered signals',
              required=True, type=str, show_default=True, default='output_npz_filtered')
@click.option('--filter_type',
              '-ft',
              help='Filter type. LPF (low-pass filter), HPF (high-pass filter), or BPF (band-pass filter)',
              required=True,
              type=str,
              default='LPF',
              show_default=True)
@click.option('--freq_cutoff',
              '-fc',
              help='Frequency cutoff in Hz. If filter_type is LPF or HPF, then freq_cutoff is a single value. If filter_type is BPF, then freq_cutoff is a list of two values',
              required=True,
              type=str,
              default=None,
              show_default=True)
@click.option('--filter_order', '-fo', help='Filter order',
              required=True, type=int, default=2, show_default=True)
@click.pass_context
@error_handler
def frequncy_domain_filter(ctx, input_npz_folder: str, output_npz_folder: str = 'output_npz_filtered',
                           filter_type: str = 'LPF', freq_cutoff: str = None, filter_order: int = 2) -> None:
    """ Filtering in frequency domain using butterworth filter and saves results as NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder to save filtered signals. If the folder already exists, it will be overwritten. If not specified, the default value is 'output_npz_filtered'
        filter_type: str
            filter type. If not specified, the default value is 'LPF'. It should be 'LPF' (low-pass filter), 'HPF' (high-pass filter), or 'BPF' (band-pass filter)
        freq_cutoff: str
            frequency cutoff in Hz. If filter_type is LPF or HPF, then freq_cutoff is a single value. If filter_type is BPF, then freq_cutoff is a list of two values. If not specified, the default value is None
        filter_order: int
            filter order. If not specified, the default value is 2

        Returns
        ----------
    """

    filter_args = {
        'filter_type': filter_type,
        'freq_cutoff': freq_cutoff,
        'filter_order': filter_order}
    print('--- Filtering in frequency domain using butterworth filter and saving results as NPZ files...')
    fourier_analysis.butterworth_filtering_from_npz(
        input_npz_folder, output_npz_folder, filter_args)
    print('--- Filtering complete.\n\n')


@cli.command('freq_bands_power_over_time',
             help="Computes signal's power in given frequency bands over time, and saves them as csv files and plots them")
@click.option('--input_npz_folder', '-i', help='Path to input npz folder',
              required=True, type=str)
@click.option('--freq_bands',
              '-fb',
              multiple=True,
              help='Frequency bands to compute power for (in Hz). It should be a list of two values (e.g. "[40,80]"). Also, it accepts multiple values (e.g. --freq_bands "[15,25]" --freq_bands "[45, 80]").',
              required=True,
              type=str)
@click.option('--channels_list',
              '-cl',
              help='List of channels to compute power for. If None, then all of the channels will be used. It should be a string of comma-separated channel numbers (e.g. --channels_list "[1,2,3]").',
              required=False,
              type=str,
              default=None,
              show_default=True)
@click.option('--ignore_channels',
              '-ic',
              help='List of channels to ignore. If None, then no channels will be ignored (e.g. --ignore_channels "[1,2,3]").',
              required=False,
              type=str,
              default=None,
              show_default=True)
@click.option('--window_size', '-w', help='Window size in seconds',
              required=False, type=float, default=1, show_default=True)
@click.option('--overlap', '-ov', help='Overlap in seconds',
              required=False, type=float, default=0.5, show_default=True)
@click.option('--t_min', '-tmin', help='Start of time interval to plot',
              required=False, type=float, default=None, show_default=True)
@click.option('--t_max', '-tmin', help='End of time interval to plot',
              required=False, type=float, default=None, show_default=True)
@click.option('--output_csv_file',
              '-o',
              help='Path to output csv file. If not specified, the default value is None and the csv file will not be saved.',
              required=False,
              type=str,
              default=None,
              show_default=True)
@click.option('--output_plot_file',
              '-op',
              help='Path to output plot file. If not specified, the default value is None and the plot will be displayed.',
              required=False,
              type=str,
              default=None,
              show_default=True)
@click.option('--plot_type',
              '-pt',
              help='Plot type. It can be "avg" or "all". If not specified, the default value is "avg". If "avg" selected, avg of power of the given frequency band of all channels will be plot with an error cloud. If "all" selected, power of the given frequency band of all channels will be plot.',
              required=False,
              type=str,
              default='avg',
              show_default=True)
@click.pass_context
@error_handler
def freq_bands_power_over_time(
        ctx,
        input_npz_folder: str,
        freq_bands: list = None,
        channels_list: str = None,
        ignore_channels: str = None,
        window_size: float = 1,
        overlap: float = 0.5,
        t_min: float = None,
        t_max: float = None,
        output_csv_file: str = None,
        output_plot_file: str = None,
        plot_type: str = None) -> None:
    """ Computes signal's power in given frequency bands over time, and saves them as csv files and plots them

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        freq_bands: list
            frequency bands to compute power for (in Hz). It should be a list of two values (e.g. "[40,80]"). Also, it accepts multiple values (e.g. --freq_bands "[15,25]" --freq_bands "[45, 80]"). If not specified, the default value is None
        channels_list: str
            list of channels to compute power for. If None, then all of the channels will be used. It should be a string of comma-separated channel numbers (e.g. "[1,2,3]"). If not specified, the default value is None
        ignore_channels: str
            list of channels to ignore. If None, then no channels will be ignored. It should be a string of comma-separated channel numbers (e.g. "[1,2,3]"). If not specified, the default value is None
        window_size: float
            window size in seconds. If not specified, the default value is 1 second
        overlap: float
            overlap in seconds. If not specified, the default value is 0.5 seconds
        t_min: float
            start of time interval to plot. If not specified, the default value is None and the minimum time will be 0 seconds
        t_max: float
            end of time interval to plot. If not specified, the default value is None and the maximum time will be the total duration of the signal
        output_csv_file: str
            path to output csv file. If not specified, the default value is None and the csv file will not be saved.
        output_plot_file: str
            path to output plot file. If not specified, the default value is None and the plot will be displayed.
        plot_type: str
            plot type. It can be "avg" or "all". If not specified, the default value is "avg". If "avg" selected, avg of power of the given frequency band of all channels will be plot with an error cloud. If "all" selected, power of the given frequency band of all channels will be plot.

        Returns
        ----------
    """
    print('--- Computing signal\'s power in given frequency bands...')
    fourier_analysis.freq_bands_power_over_time(
        input_npz_folder,
        freq_bands,
        channels_list,
        ignore_channels,
        window_size,
        overlap,
        t_min,
        t_max,
        output_csv_file,
        output_plot_file,
        plot_type)
    print('--- Computation complete.\n\n')
### Fourier Analysis ###


### Visualization ###
@cli.command('plot_stft', help='Plots STFT from NPZ file')
@click.option('--input_npz_file', '-i',
              help='Path to input npz file', required=True, type=str)
@click.option('--output_plot_file', '-o', help='Path to output plot file',
              required=True, type=str, default=None, show_default=True)
@click.option('--f_min', '-fmin', help='Minimum frequency to plot in Hz',
              required=False, type=float, default=None, show_default=True)
@click.option('--f_max', '-fmax', help='Maximum frequency to plot in Hz',
              required=False, type=float, default=None, show_default=True)
@click.option('--t_min', '-tmin', help='Start of time interval to plot',
              required=False, type=float, default=None, show_default=True)
@click.option('--t_max', '-tmax', help='End of time interval to plot',
              required=False, type=float, default=None, show_default=True)
@click.option('--db_min', '-dbmin', help='Minimum dB to plot',
              required=False, type=float, default=None, show_default=True)
@click.option('--db_max', '-dbmax', help='Maximum dB to plot',
              required=False, type=float, default=None, show_default=True)
@click.pass_context
@error_handler
def plot_stft(ctx, input_npz_file: str, output_plot_file: str, f_min: float = None, f_max: float = None,
              t_min: float = None, t_max: float = None, db_min: float = None, db_max: float = None) -> None:
    """ Plots STFT from NPZ file

        Parameters
        ----------
        input_npz_file: str
            path to input npz file
        output_plot_file: str
            path to output plot file. If not specified, the default value is None and the plot will be displayed.
        f_min: float
            minimum frequency to plot in Hz. If not specified, the default value is None and the minimum frequency will be 0 Hz
        f_max: float
            maximum frequency to plot in Hz. If not specified, the default value is None and the maximum frequency will be the Nyquist frequency
        t_min: float
            Start of time interval to plot. If not specified, the default value is None and the minimum time will be 0 seconds
        t_max: float
            End of time interval to plot. If not specified, the default value is None and the maximum time will be the total duration of the signal
        db_min: float
            minimum dB to plot. If not specified, the default value is None and the minimum dB will be the minimum dB of the signal
        db_max: float
            maximum dB to plot. If not specified, the default value is None and the maximum dB will be the maximum dB of the signal

        Returns
        ----------
    """

    print('--- Plotting STFT...')
    visualization.plot_stft_from_npz(
        input_npz_file,
        output_plot_file,
        f_min,
        f_max,
        t_min,
        t_max,
        db_min,
        db_max)
    print('--- Plotting complete.\n\n')


@cli.command('plot_avg_stft', help='Plots average STFT from NPZ files')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder containing STFT NPZ files', required=True, type=str)
@click.option('--output_plot_file', '-o', help='Path to output plot file',
              required=True, type=str, default=None, show_default=True)
@click.option('--f_min', '-fmin', help='Minimum frequency to plot in Hz',
              type=float, default=None, show_default=True, required=False)
@click.option('--f_max', '-fmax', help='Maximum frequency to plot in Hz',
              type=float, default=None, show_default=True, required=False)
@click.option('--t_min', '-tmin', help='Start of time interval to plot',
              type=float, default=None, show_default=True, required=False)
@click.option('--t_max', '-tmax', help='End of time interval to plot',
              type=float, default=None, show_default=True, required=False)
@click.option('--db_min', '-dbmin', help='Minimum dB to plot',
              type=float, default=None, show_default=True, required=False)
@click.option('--db_max', '-dbmax', help='Maximum dB to plot',
              type=float, default=None, show_default=True, required=False)
@click.option('--channels_list', '-cl', help='List of channels to plot, if None then all of the channels will be plotted',
              required=False, type=list, default=None, show_default=True)
@click.pass_context
@error_handler
def plot_avg_stft(
        ctx,
        input_npz_folder: str,
        output_plot_file: str,
        f_min: float = None,
        f_max: float = None,
        t_min: float = None,
        t_max: float = None,
        db_min: float = None,
        db_max: float = None,
        channels_list: str = None) -> None:
    """ Plots average STFT from NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder containing STFT NPZ files
        output_plot_file: str
            path to output plot file. If not specified, the default value is None and the plot will be displayed.
        f_min: float
            minimum frequency to plot in Hz. If not specified, the default value is None and the minimum frequency will be 0 Hz
        f_max: float
            maximum frequency to plot in Hz. If not specified, the default value is None and the maximum frequency will be the Nyquist frequency
        t_min: float
            Start of time interval to plot. If not specified, the default value is None and the minimum time will be 0 seconds
        t_max: float
            End of time interval to plot. If not specified, the default value is None and the maximum time will be the total duration of the signal
        db_min: float
            minimum dB to plot. If not specified, the default value is None and the minimum dB will be the minimum dB of the signal
        db_max: float
            maximum dB to plot. If not specified, the default value is None and the maximum dB will be the maximum dB of the signal
        channels_list: str
            list of channels to plot. either a string of comma-separated channel numbers or a list of integers. If not specified, the default value is None and all of the channels will be plotted.

        Returns
        ----------
    """

    print('--- Plotting average STFT...')
    visualization.plot_avg_stft_from_npz(
        input_npz_folder,
        output_plot_file,
        f_min,
        f_max,
        t_min,
        t_max,
        db_min,
        db_max,
        channels_list)
    print('--- Plotting complete.\n\n')


@cli.command('plot_signal', help='Plots signals from NPZ file')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_plot_file', '-o', help='Path to output plot file',
              required=True, type=str, default=None, show_default=True)
@click.option('--t_min', '-tmin', help='Start of time interval to plot',
              required=False, type=float, default=None, show_default=True)
@click.option('--t_max', '-tmax', help='End of time interval to plot',
              required=False, type=float, default=None, show_default=True)
@click.option('--channels_list',
              '-cl',
              help='List of channels to plot, if None then all of the channels will be plotted (e.g. --channels_list "[1,2,3]").',
              required=False,
              type=str,
              default=None,
              show_default=True)
@click.option('--normalize', '-n', help='Normalize signals. If true, each channel will be normalized',
              required=False, type=bool, default=False, show_default=True)
@click.option('--scale_bar', '-sb', help='Scale bar. If true, a scale bar will be added to the plot',
              required=False, type=bool, default=True, show_default=True)
@click.option('--re_reference',
              '-rr',
              help='Re-reference signals. If -rr_channel not specified, signals will be re-referenced to the average of all channels, unless they will be re-referenced to the given channel. If --ignore_channels is specified, specified channels will not be re referenced or taken into account for avg rereferencing',
              required=False,
              type=bool,
              default=False,
              show_default=True)
@click.option('--ignore_channels',
              '-ic',
              help='List of channels to ignore (e.g EMG, EOG, etc.). If None, then no channels will be ignored (e.g. --ignore_channels "[1,2,3]").',
              required=False,
              type=str,
              default=None,
              show_default=True)
@click.option('--rr_channel',
              '-rrc',
              help='Channel to re-reference signals to. If None, signals will be re-referenced to the average of all channels',
              required=False,
              type=int,
              default=None,
              show_default=True)
@click.pass_context
@error_handler
def plot_signal(
        ctx,
        input_npz_folder: str,
        output_plot_file: str,
        t_min: float = None,
        t_max: float = None,
        channels_list: str = None,
        normalize: bool = False,
        scale_bar: bool = True,
        re_reference: bool = False,
        ignore_channels: str = None,
        rr_channel: int = None) -> None:
    """ Plots signals from NPZ file

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_plot_file: str
            path to output plot file. If not specified, the default value is None and the plot will be displayed.
        t_min: float
            Start of time interval to plot. If not specified, the default value is None and the minimum time will be 0 seconds
        t_max: float
            End of time interval to plot. If not specified, the default value is None and the maximum time will be the total duration of the signal
        channels_list: str
            list of channels to plot. either a string of comma-separated channel numbers or a list of integers. If not specified, the default value is None and all of the channels will be plotted.
        normalize: bool
            normalize signals. If true, each channel will be normalized. If not specified, the default value is False.
        scale_bar: bool
            scale bar. If true, a scale bar will be added to the plot. If not specified, the default value is True.
        re_reference: bool
            re-reference signals. If true, signals will be re-referenced. If not specified, the default value is False.
        ignore_channels: str
            list of channels to ignore (e.g EMG, EOG, etc.). If None, then no channels will be ignored. Either a list of channel indexes or a string of channel indexes separated by commas. If not specified, the default value is None.
        rr_channel: int
            channel to re-reference signals to. If None, signals will be re-referenced to the average of all channels. If not specified, the default value is None.

        Returns
        ----------
    """

    if re_reference:
        _rereference_args = {
            'ignore_channels': ignore_channels,
            'rr_channel': rr_channel}
    else:
        _rereference_args = None

    print('--- Plotting signals...')
    visualization.plot_signals_from_npz(
        input_npz_folder,
        output_plot_file,
        t_min,
        t_max,
        channels_list,
        normalize,
        scale_bar,
        _rereference_args)
    print('--- Plotting complete.\n\n')


@cli.command('plot_dft', help='Plots DFT from NPZ file')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_plot_file', '-o', help='Path to output plot file',
              required=True, type=str, default=None, show_default=True)
@click.option('--f_min', '-fmin', help='Minimum frequency to plot in Hz',
              required=False, type=float, default=None, show_default=True)
@click.option('--f_max', '-fmax', help='Maximum frequency to plot in Hz',
              required=False, type=float, default=None, show_default=True)
@click.option('--channels_list',
              '-cl',
              help='List of channels to plot, if None then all of the channels will be plotted (e.g. --channels_list "[1,2,3]").',
              required=False,
              type=str,
              default=None,
              show_default=True)
@click.option('--plot_type',
              '-pt',
              help='Plot type. If "all_channels", then all channels will be plotted in one figure. If "average_of_channels", then average of channels will be plotted in one figure with errorbar',
              required=True,
              type=str,
              default='average_of_channels',
              show_default=True)
@click.option('--conv_window_size', '-cws', help='Convolution window size in seconds',
              required=False, type=int, default=None, show_default=True)
@click.pass_context
@error_handler
def plot_dft(ctx, input_npz_folder: str, output_plot_file: str, f_min: float = None, f_max: float = None,
             channels_list: str = None, plot_type: str = 'average_of_channels', conv_window_size: int = None) -> None:
    """ Plots DFT from NPZ file

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_plot_file: str
            path to output plot file. If not specified, the default value is None and the plot will be displayed.
        f_min: float
            minimum frequency to plot in Hz. If not specified, the default value is None and the minimum frequency will be 0 Hz
        f_max: float
            maximum frequency to plot in Hz. If not specified, the default value is None and the maximum frequency will be the Nyquist frequency
        channels_list: str
            list of channels to plot. either a string of comma-separated channel numbers or a list of integers. If not specified, the default value is None and all of the channels will be plotted.
        plot_type: str
            plot type. If not specified, the default value is 'average_of_channels'. It should be 'all_channels' or 'average_of_channels'
        conv_window_size: int
            convolution window size in seconds. If not specified, the default value is None.

        Returns
        ----------
    """

    print('--- Plotting DFT...')
    visualization.plot_dft_from_npz(
        input_npz_folder,
        output_plot_file,
        f_min,
        f_max,
        plot_type,
        channels_list,
        conv_window_size)
    print('--- Plotting complete.\n\n')


@cli.command('plot_filter_freq_response',
             help='Plots filter frequency response')
@click.option('--filter_type',
              '-ft',
              help='Filter type. LPF (low-pass filter), HPF (high-pass filter), or BPF (band-pass filter)',
              required=True,
              type=str,
              default='LPF',
              show_default=True)
@click.option('--freq_cutoff',
              '-fc',
              help='Frequency cutoff in Hz. If filter_type is LPF or HPF, then freq_cutoff is a single value. If filter_type is BPF, then freq_cutoff is a list of two values',
              required=True,
              type=str,
              default=None,
              show_default=True)
@click.option('--filter_order', '-fo', help='Filter order',
              required=True, type=int, default=4, show_default=True)
@click.option('--frequency_sampling', '-fs', help='Frequency sampling in Hz',
              required=True, type=int, show_default=True)
@click.option('--output_plot_file', '-o', help='Path to output plot file',
              required=True, type=str, default=None, show_default=True)
@click.pass_context
@error_handler
def plot_filter_freq_response(ctx, filter_type: str = 'LPF', freq_cutoff: str = None,
                              filter_order: int = 4, frequency_sampling: int = None, output_plot_file: str = None) -> None:
    """ Plots filter frequency response

        Parameters
        ----------
        filter_type: str
            filter type. If not specified, the default value is 'LPF'. It should be 'LPF' (low-pass filter), 'HPF' (high-pass filter), or 'BPF' (band-pass filter)
        freq_cutoff: str
            frequency cutoff in Hz. If filter_type is LPF or HPF, then freq_cutoff is a single value. If filter_type is BPF, then freq_cutoff is a list of two values. If not specified, the default value is None
        filter_order: int
            filter order. If not specified, the default value is 4
        frequency_sampling: int
            frequency sampling in Hz. If not specified, the default value is None
        output_plot_file: str
            path to output plot file. If not specified, the default value is None and the plot will be displayed.

        Returns
        ----------
    """

    filter_args = {
        'filter_type': filter_type,
        'freq_cutoff': freq_cutoff,
        'filter_order': filter_order,
        'fs': frequency_sampling}
    print('--- Plotting filter frequency response...')
    visualization.plot_filter_freq_response(filter_args, output_plot_file)
    print('--- Plotting complete.\n\n')
### Visualization ###


### Dimensionality Reduction ###
@cli.command('pca_from_npz', help='Computes PCA from NPZ files')
@click.option('--input_npz_folder', '-i',
              help='Path to input npz folder', required=True, type=str)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder to save PCA results',
              required=True, type=str, show_default=True, default='output_npz_pca')
@click.option('--n_components', '-n', help='Number of components to keep after applying the PCA',
              required=True, type=int, default=None, show_default=True)
@click.option('--matrix_whitenning',
              '-mw',
              help='Matrix whitening boolean. If true, the singular values are divided by n_samples',
              required=False,
              type=bool,
              default=False,
              show_default=True)
@click.option('--channels_list',
              '-cl',
              help='List of channels to apply PCA, if None then all of the channels will be applied (e.g. --channels_list "[1,2,3]").',
              required=False,
              type=list,
              default=None,
              show_default=True)
@click.pass_context
@error_handler
def pca_from_npz(ctx, input_npz_folder: str, output_npz_folder: str = 'output_npz_pca',
                 n_components: int = None, matrix_whitenning: bool = False, channels_list: str = None) -> None:
    """ Computes PCA from NPZ files

        Parameters
        ----------
        input_npz_folder: str
            path to input npz folder
        output_npz_folder: str
            path to output npz folder to save PCA results. If the folder already exists, it will be overwritten. If not specified, the default value is 'output_npz_pca'
        n_components: int
            number of components to keep after applying the PCA. If not specified, the default value is None
        matrix_whitenning: bool
            matrix whitening boolean. If true, the singular values are divided by n_samples. If not specified, the default value is False
        channels_list: str
            list of channels to apply PCA. either a string of comma-separated channel numbers or a list of integers. If not specified, the default value is None and all of the channels will be applied.

        Returns
        ----------
    """

    print('--- Computing PCA and saving results as NPZ files...')
    dimensionality_reduction.pca_from_npz(
        input_npz_folder,
        output_npz_folder,
        n_components,
        matrix_whitenning,
        channels_list)
    print('--- PCA computation complete.\n\n')
### Dimensionality Reduction ###


def main():
    cli()


if __name__ == '__main__':
    main()
