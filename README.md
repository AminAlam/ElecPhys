<div align="center">
  <br/>
<h1>ElecPhys</h1>
  
<br/>
<img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white" alt="built with Python3" />
</div>

[![Documentation Status](https://readthedocs.org/projects/elecphys/badge/?version=latest)](https://elecphys.readthedocs.io) [![Ci testing](https://github.com/AminAlam/elecphys/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/AminAlam/elecphys)


----------

ElecPhys: A Python package for electrophysiology data analysis. It provides tools for data loading, analysis, conversion, preprocessing, and visualization.


----------
## Table of contents			
   * [Overview](https://github.com/AminAlam/ElecPhys#overview)
   * [Installation](https://github.com/AminAlam/ElecPhys#ElecPhys)
   * [Usage](https://github.com/AminAlam/ElecPhys#usage)
   * [Documentation](https://github.com/AminAlam/ElecPhys#documentation)
----------
## Overview
<p align="justify">
 ElecPhys is a Python package for electrophysiology data analysis. It provides tools for data loading, analysis, conversion, preprocessing, and visualization.. ElecPhys can convert .RHD fiels to .mat and .npz, and it can analyze the data in time and fourier domains. Please take a look at <a href="https://github.com/AminAlam/ElecPhys/docs/available_analysis.md">here</a> for the complete list of available tools. 
</p>

----------
## Installation

### Source code
- Clone the repository or download the source code.
- cd into the repository directory.
- Run `python3 setup.py install`

### PyPi
Run `pip3 install elecphys` or `pip install elecphys`

## Usage
It's possible to use ElecPhys as a command line tool or as a Python module.

### Command Line
To use ElecPhys from command line, you need to use the following pattern:

```console
➜ elecphys COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]
```
Where each COMMNAD can be one of the supported commands such as convert_rhd_to_mat, plot_stft, and ... .
To learn more about the commnads, you can use the following command:
```console
➜ elecphys --help
Usage: main.py [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

  ElecPhys is a Python package for electrophysiology data analysis. It
  provides tools for data loading, conversion, preprocessing, and
  visualization.

Options:
  -v, --verbose  Verbose mode
  -d, --debug    Debug mode
  --help         Show this message and exit.

Commands:
  convert_mat_to_npz            Converts MAT files to NPZ files using MAT...
  convert_rhd_to_mat            Converts RHD files to mat files using RHD...
  dft_numeric_output_from_npz   Computes DFT and saves results as NPZ files
  freq_bands_power_over_time    Computes signal's power in given...
  frequncy_domain_filter        Filtering in frequency domain using...
  normalize_npz                 Normalizes NPZ files
  pca_from_npz                  Computes PCA from NPZ files
  plot_avg_stft                 Plots average STFT from NPZ files
  plot_dft                      Plots DFT from NPZ file
  plot_filter_freq_response     Plots filter frequency response
  plot_signal                   Plots signals from NPZ file
  plot_stft                     Plots STFT from NPZ file
  re_reference_npz              re-references NPZ files and save them as...
  stft_numeric_output_from_npz  Computes STFT and saves results as NPZ files
  zscore_normalize_npz          Z-score normalizes NPZ files
```

### Python module

You need to import ElecPhys and use it's modules inside your python code. For example:

```python
import elecphys

# Path to folder containing RHD files
folder_path = "data/rhd_files"
# Path to the output .mat file
output_mat_file = "output.mat"
# Down sample factor
ds_factor = 5
# call rhd to mat conversoin module
elecphys.conversion.convert_rhd_to_mat(folder_path, output_mat_file, ds_factor)
```
## Documentation
Please check [this link](https://elecphys.readthedocs.io/en/stable/) for full documentation