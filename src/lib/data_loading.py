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