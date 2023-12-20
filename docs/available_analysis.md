### Conversion
#### convert_mat_to_npz
```console
Usage: elecphys convert_mat_to_npz [OPTIONS]

  Converts MAT files to NPZ files using MAT to NPZ converter

Options:
  -m, --mat_file PATHLIKE         Path to mat file  [required]
  -o, --output_npz_folder PATHLIKE
                                  Path to output npz folder  [default:
                                  output_npz; required]
  -n, --notch_filter_freq INTEGER
                                  Notch filter frequency in Hz  [default: 50]
  --help                          Show this message and exit.
```
#### convert_rhd_to_mat
```console
Usage: elecphys convert_rhd_to_mat [OPTIONS]

  Converts RHD files to mat files using RHD to MAT converter (needs MATLAB
  installed

Options:
  -f, --folder_path PATHLIKE      Path to folder containing RHD files
                                  [required]
  -o, --output_mat_file PATHLIKE  Path to output mat file  [default:
                                  output.mat; required]
  -d, --ds_factor INTEGER         Downsample factor  [default: 1]
  --help                          Show this message and exit.
```
### Preprocessing
#### normalize_npz
```console
Usage: elecphys normalize_npz [OPTIONS]

  Normalizes NPZ files

Options:
  -i, --input_npz_folder PATHLIKE
                                  Path to input npz folder  [required]
  -o, --output_npz_folder PATHLIKE
                                  Path to output npz folder  [default:
                                  output_npz; required]
  --help                          Show this message and exit.
```
#### zscore_normalize_npz
```console
Usage: elecphys zscore_normalize_npz [OPTIONS]

  Z-score normalizes NPZ files

Options:
  -i, --input_npz_folder PATHLIKE
                                  Path to input npz folder  [required]
  -o, --output_npz_folder PATHLIKE
                                  Path to output npz folder  [default:
                                  output_npz; required]
  --help                          Show this message and exit.
```
### Visualization
#### plot_dft
```console
Usage: elecphys plot_dft [OPTIONS]

  Plots DFT from NPZ file

Options:
  -i, --input_npz_folder PATHLIKE
                                  Path to input npz folder  [required]
  -o, --output_plot_file PATHLIKE
                                  Path to output plot file  [required]
  -fmin, --f_min FLOAT            Minimum frequency to plot in Hz  [required]
  -fmax, --f_max FLOAT            Maximum frequency to plot in Hz  [required]
  -cl, --channels_list LIST       List of channels to plot, if None then all
                                  of the channels will be plotted  [required]
  -pt, --plot_type TEXT           Plot type  [default: average_of_channels;
                                  required]
  -cws, --conv_window_size FLOAT  Convolution window size in seconds
                                  [required]
  --help                          Show this message and exit.
```
#### plot_signal
```console
Usage: elecphys plot_signal [OPTIONS]

  Plots signals from NPZ file

Options:
  -i, --input_npz_folder PATHLIKE
                                  Path to input npz folder  [required]
  -o, --output_plot_file PATHLIKE
                                  Path to output plot file  [required]
  -tmin, --t_min FLOAT            Minimum time to plot in seconds  [required]
  -tmax, --t_max FLOAT            Maximum time to plot in seconds  [required]
  -cl, --channels_list LIST       List of channels to plot, if None then all
                                  of the channels will be plotted
  -n, --normalize BOOLEAN         Normalize signals. If true, each channel
                                  will be normalized  [default: False]
  --help                          Show this message and exit.
```
#### plot_stft
```console
Usage: elecphys plot_stft [OPTIONS]

  Plots STFT from NPZ file

Options:
  -i, --input_npz_file PATHLIKE   Path to input npz file  [required]
  -o, --output_plot_file PATHLIKE
                                  Path to output plot file  [required]
  -fmin, --f_min FLOAT            Minimum frequency to plot in Hz  [required]
  -fmax, --f_max FLOAT            Maximum frequency to plot in Hz  [required]
  -tmin, --t_min FLOAT            Minimum time to plot in seconds  [required]
  -tmax, --t_max FLOAT            Maximum time to plot in seconds  [required]
  -dbmin, --db_min FLOAT          Minimum dB to plot  [required]
  -dbmax, --db_max FLOAT          Maximum dB to plot  [required]
  --help                          Show this message and exit.
```
#### plot_avg_stft
```console
Usage: elecphys plot_avg_stft [OPTIONS]

  Plots average STFT from NPZ files

Options:
  -i, --input_npz_folder TEXT  Path to input npz folder containing STFT NPZ
                               files  [required]
  -o, --output_plot_file TEXT  Path to output plot file  [required]
  -fmin, --f_min FLOAT         Minimum frequency to plot in Hz
  -fmax, --f_max FLOAT         Maximum frequency to plot in Hz
  -tmin, --t_min FLOAT         Minimum time to plot in seconds
  -tmax, --t_max FLOAT         Maximum time to plot in seconds
  -dbmin, --db_min FLOAT       Minimum dB to plot
  -dbmax, --db_max FLOAT       Maximum dB to plot
  -cl, --channels_list LIST    List of channels to plot, if None then all of
                               the channels will be plotted
  --help                       Show this message and exit.
```
#### plot_filter_freq_response
```console
Usage: elecphys plot_filter_freq_response [OPTIONS]

  Plots filter frequency response

Options:
  -ft, --filter_type TEXT         Filter type. LPF (low-pass filter), HPF
                                  (high-pass filter), or BPF (band-pass
                                  filter)  [default: LPF; required]
  -fc, --freq_cutoff TEXT         Frequency cutoff in Hz. If filter_type is
                                  LPF or HPF, then freq_cutoff is a single
                                  value. If filter_type is BPF, then
                                  freq_cutoff is a list of two values
                                  [required]
  -fo, --filter_order INTEGER     Filter order  [default: 4; required]
  -fs, --frequency_sampling INTEGER
                                  Frequency sampling in Hz  [required]
  -o, --output_plot_file TEXT     Path to output plot file  [required]
  --help                          Show this message and exit.
```
### Fourier Analysis
#### dft_numeric_output_from_npz
```console
Usage: elecphys dft_numeric_output_from_npz [OPTIONS]

  Computes DFT and saves results as NPZ files

Options:
  -i, --input_npz_folder PATHLIKE
                                  Path to input npz folder  [required]
  -o, --output_npz_folder PATHLIKE
                                  Path to output npz folder to save DFT
                                  results  [required]
  --help                          Show this message and exit.
```
#### stft_numeric_output_from_npz
```console
Usage: elecphys stft_numeric_output_from_npz [OPTIONS]

  Computes STFT and saves results as NPZ files

Options:
  -i, --input_npz_folder PATHLIKE
                                  Path to input npz folder  [required]
  -o, --output_npz_folder PATHLIKE
                                  Path to output npz folder to save STFT
                                  results  [required]
  -w, --window_size FLOAT         Window size in seconds  [required]
  -ov, --overlap FLOAT            Overlap in seconds  [required]
  -wt, --window_type TEXT         Window type  [default: hann; required]
  --help                          Show this message and exit.
```
#### frequncy_domain_filter
```console
Usage: elecphys frequncy_domain_filter [OPTIONS]

Filtering in frequency domain using butterworth filter and saves results as
  NPZ files

Options:
  -i, --input_npz_folder TEXT   Path to input npz folder  [required]
  -o, --output_npz_folder TEXT  Path to output npz folder to save filtered
                                signals  [required]
  -ft, --filter_type TEXT       Filter type. LPF (low-pass filter), HPF (high-
                                pass filter), or BPF (band-pass filter)
                                [default: LPF; required]
  -fc, --freq_cutoff TEXT       Frequency cutoff in Hz. If filter_type is LPF
                                or HPF, then freq_cutoff is a single value. If
                                filter_type is BPF, then freq_cutoff is a list
                                of two values  [required]
  -fo, --filter_order INTEGER   Filter order  [default: 4; required]
  --help                        Show this message and exit.
```