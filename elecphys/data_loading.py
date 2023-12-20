import mat73
import numpy as np

def load_mat(mat_file):
    """Loads MAT file
    input:
        mat_file: path to mat file - type: os.PathLike
    output:
        data: data from MAT file - type: numpy.ndarray
        fs: sampling frequency - type: float
    """
    mat_file_contents = mat73.loadmat(mat_file)
    data = mat_file_contents['data']
    fs = mat_file_contents['fs']
    return data, fs

def load_npz(npz_file):
    """Loads NPZ file
    input:
        npz_file: path to npz file - type: os.PathLike
    output:
        data: data from NPZ file - type: numpy.ndarray
        fs: sampling frequency - type: float
    """
    npz_file_contents = np.load(npz_file)
    data = npz_file_contents['data']
    fs = npz_file_contents['fs']
    return data, fs


def load_npz_stft(npz_file):
    """Loads NPZ file
    input:
        npz_file: path to npz file - type: os.PathLike
    output:
        f: frequency array - type: numpy.ndarray
        t: time array - type: numpy.ndarray
        Zxx: STFT array - type: numpy.ndarray
    """
    npz_file_contents = np.load(npz_file)
    f = npz_file_contents['f']
    t = npz_file_contents['t']
    Zxx = npz_file_contents['Zxx']
    return f, t, Zxx


def load_npz_dft(npz_file):
    """ Loads NPZ file
    input:
        npz_file: path to npz file - type: os.PathLike
    output:
        f: frequency array - type: numpy.ndarray
        Zxx: DFT array - type: numpy.ndarray
    """
    npz_file_contents = np.load(npz_file)
    f = npz_file_contents['f']
    Zxx = npz_file_contents['Zxx']
    return f, Zxx


def load_npz_mvl(npz_file):
    """ Loads NPZ file
    input:
        npz_file: path to npz file - type: os.PathLike
    output:
        MVL: MVL array - type: numpy.ndarray
    """
    npz_file_contents = np.load(npz_file)
    MI_mat = npz_file_contents['MI_mat']
    freqs_amp = npz_file_contents['freqs_amp']
    freqs_phase = npz_file_contents['freqs_phase']
    time_interval = npz_file_contents['time_interval']
    return MI_mat, freqs_amp, freqs_phase, time_interval