import click
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import conversion
from handlers import ErrorHandler
error_handler = ErrorHandler().error_handler

@click.group(chain=True)
@click.pass_context
def cli(ctx):
    pass

@cli.command('convert_rhd_to_mat')
@click.option('--folder_path', '-f', help='Path to folder containing RHD files', required=True, type=os.PathLike)
@click.option('--output_mat_file', '-o', help='Path to output mat file', required=True, type=os.PathLike, default='output.mat', show_default=True)
@click.option('--ds_factor', '-d', help='Downsample factor', required=False, type=int, default=1, show_default=True)
@click.pass_context
@error_handler
def convert_rhd_to_mat(ctx, folder_path, output_mat_file, ds_factor):
    """ Converts RHD files to mat files using RHD to MAT converter written in MATLAB
    input:
        folder_path: path to folder containing RHD files - type: os.PathLike
        output_mat_file: path to output mat file - type: os.PathLike
        ds_factor: downsample factor - type: int
    output:
        output_mat_file: no return value
    """
    
    print('\n\n--- Converting RHD files to MAT files...')
    conversion.convert_rhd_to_mat(folder_path, output_mat_file, ds_factor)
    print('\n\n--- Conversion complete.')


@cli.command('convert_mat_to_npz')
@click.option('--mat_file', '-m', help='Path to mat file', required=True, type=os.PathLike)
@click.option('--output_npz_folder', '-o', help='Path to output npz folder', required=True, type=os.PathLike, default='output_npz', show_default=True)
@click.option('--notch_filter_freq', '-n', help='Notch filter frequency in Hz', required=False, type=int, default=50, show_default=True)
@click.pass_context
@error_handler
def convert_mat_to_npz(ctx, mat_file, output_npz_folder, notch_filter_freq):
    """ Converts MAT files to NPZ files
    input:
        mat_file: path to mat file - type: os.PathLike
        output_npz_folder: path to output npz folder - type: os.PathLike
        notch_filter_freq: notch filter frequency - type: int
    output:
        output_npz_folder: no return value
    """
    
    print('\n\n--- Converting MAT files to NPZ files...')
    conversion.convert_mat_to_npz(mat_file, output_npz_folder, notch_filter_freq)
    print('\n\n--- Conversion complete.')

