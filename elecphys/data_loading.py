import mat73
import numpy as np

def load_mat(mat_file) -> [np.ndarray, int]:
    """ Function that Loads MAT file

        Parameters
        ----------
        mat_file: str
            path to mat file
    
        Returns
        ----------
        data: numpy.ndarray
            data from MAT file
        fs: int
            sampling frequency (Hz)
    """
    with mat73.loadmat(mat_file) as mat_file_contents:
        data = mat_file_contents['data']
        fs = mat_file_contents['fs']
    return data, fs

def load_npz(npz_file) -> [np.ndarray, int]:
    """ Function that Loads NPZ file

        Parameters
        ----------
        npz_file: path to npz file - type: os.PathLike
    
        Returns
        ----------
        data: data from NPZ file - type: numpy.ndarray
        fs: sampling frequency - type: float
    """
    with np.load(npz_file) as npz_file_contents:
        data = npz_file_contents['data']
        fs = npz_file_contents['fs']
    return data, fs


def load_npz_stft(npz_file) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Function that Loads NPZ file

        Parameters
        ----------
        npz_file: path to npz file - type: os.PathLike
    
        Returns
        ----------
        f: frequency array - type: numpy.ndarray
        t: time array - type: numpy.ndarray
        Zxx: STFT array - type: numpy.ndarray
    """
    with np.load(npz_file) as npz_file_contents:
        f = npz_file_contents['f']
        t = npz_file_contents['t']
        Zxx = npz_file_contents['Zxx']
    return f, t, Zxx


def load_npz_dft(npz_file) -> [np.ndarray, np.ndarray]:
    """ Function that Loads NPZ file

        Parameters
        ----------
        npz_file: path to npz file - type: os.PathLike
    
        Returns
        ----------
        f: frequency array - type: numpy.ndarray
        Zxx: DFT array - type: numpy.ndarray
    """

    with np.load(npz_file) as npz_file_contents:
        f = npz_file_contents['f']
        Zxx = npz_file_contents['Zxx']
    return f, Zxx


def load_npz_mvl(npz_file) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Function that Loads NPZ file

        Parameters
        ----------
        npz_file: path to npz file - type: os.PathLike
    
        Returns
        ----------
        MVL: MVL array - type: numpy.ndarray
        freqs_amp: amplitude frequencies - type: numpy.ndarray
        freqs_phase: phase frequencies - type: numpy.ndarray
        time_interval: time interval - type: numpy.ndarray
    """
    with np.load(npz_file) as npz_file_contents:
        MI_mat = npz_file_contents['MI_mat']
        freqs_amp = npz_file_contents['freqs_amp']
        freqs_phase = npz_file_contents['freqs_phase']
        time_interval = npz_file_contents['time_interval']
    return MI_mat, freqs_amp, freqs_phase, time_interval