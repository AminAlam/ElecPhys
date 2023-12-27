from scipy.special import rel_entr
import numpy as np
from scipy import signal
from tqdm import tqdm
from scipy.fft import fft


def calc_tf_mvl(x, fs: int, freqs_phase: list, freqs_amp: list) -> np.ndarray:
    """ Function that Calculates the tf_MVL matrix for a given signal x

        Parameters
        ----------
            x: np.array
                signal in time domain
            fs: float
                sampling frequency (Hz)
            freqs_phase: list
                phase frequencies (Hz)
            freqs_amp: list
                amplitude frequencies (Hz)

        Returns
        -------
            MI_mat: np.ndarray
                2D tf_MVL matrix (phase x amplitude)
    """

    MI_mat = np.zeros((len(freqs_phase), len(freqs_amp)))
    for phase_counter, freq_phase in enumerate(freqs_phase):
        for amp_counter, freq_amp in enumerate(freqs_amp):
            mvl = band_tfMVL(x, [freq_amp - 1, freq_amp + 1],
                             [freq_phase - 1, freq_phase + 1], fs)
            MI_mat[phase_counter, amp_counter] = mvl
    return MI_mat


def band_tfMVL(x, high_freq, low_freq, fs: int) -> float:
    """ Function that Calculates the tf_MVL for a given signal x

        Parameters
        ----------
            x: np.array
                signal in time domain
            high_freq: list
                high frequency band (Hz)
            low_freq: list
                low frequency band (Hz)
            fs: int
                sampling frequency (Hz)

        Returns
        -----------
            tf_canolty: float
                tf_MVL
    """

    fs = int(fs)
    tfd = rid_rihaczek4(x, fs)
    W = tfd
    W2 = W[1:, :]
    Amp = np.abs(W2[high_freq, :])
    tfd_low = W2[low_freq, :]
    angle_low = np.angle(tfd_low)
    Phase = angle_low
    tf_canolty = calc_MVL(Phase, Amp)
    return tf_canolty


def calc_MVL(phase, amp) -> np.ndarray:
    """ Function that Calculates the MVL for a given signal phasae and amplitude

        Parameters
        ----------
            phase: np.array
                phase values
            amp: np.array
                amplitude values

        Returns
        ----------
            MVL: np.array
                MVL values
    """

    z1 = np.exp(1j * phase)
    z = np.multiply(amp, z1)
    MVL = np.abs(np.mean(z))
    return MVL


def rid_rihaczek4(x, fbins) -> np.ndarray:
    """ Function that Calculates the rid_rihaczek4 for a given signal x

        Parameters
        ----------
            x: np.array
                signal in time domain
            fbins: int
                number of frequency bins

        Returns
        ----------
            tfd: np.ndarray
                2D rid_rihaczek4 matrix
    """

    tbins = len(x)
    amb = np.zeros((tbins, tbins))

    for tau in range(tbins):
        indexes = np.arange(tau, tbins, 1)
        indexes = np.append(indexes, np.arange(0, tau, 1))
        amb[tau, :] = np.multiply(np.conj(x), x[indexes])

    indexes = np.arange(int(tbins / 2), tbins)
    indexes = np.append(indexes, np.arange(0, int(tbins / 2)))

    ambTemp = amb[:, indexes]

    indexes = np.arange(0, int(tbins / 2))
    indexes = np.append(np.arange(int(tbins / 2), tbins), indexes)
    amb1 = ambTemp[indexes, :]

    D_mult = np.linspace(-1, 1, tbins)
    D_mult = np.reshape(D_mult, (-1, len(D_mult)))
    D = np.matmul(np.transpose(D_mult), D_mult)
    L = D

    K = chwi_krn(D, L, 0.01)
    df = K
    ambf = np.multiply(amb1, df)
    A = np.zeros((fbins, tbins))
    tbins = tbins - 1
    if tbins != fbins:
        for tt in range(tbins):
            A[:, tt] = data_wrapper(ambf[:, tt], fbins)
    else:
        A = ambf

    tfd = np.zeros(A.shape, dtype='complex_')
    for col in range(A.shape[1]):
        tfd[:, col] = fft(A[:, col])

    return tfd


def chwi_krn(D, L, A) -> np.ndarray:
    """ Function that Calculates the chwi_krn for a given signal x

        Parameters
        ----------
            D: np.array
                D
            L: np.array
                L
            A: float
                A

        Returns
        ----------
            k: np.ndarray
                chwi_krn
    """

    k = np.exp((-1 / (A**2)) * np.multiply(np.multiply(D, D), np.multiply(L, L)))
    return k


def data_wrapper(x, sec_dim) -> np.ndarray:
    """ Function that Wraps the data for a given signal x

        Parameters
        ----------
            x: np.array
                signal in time domain
            sec_dim: int
                second dimension

        Returns
        ----------
            wrapped_x: np.ndarray
                wrapped signal
    """

    wrapped_x = np.zeros((sec_dim, 1))
    wrapped_x[0:len(x), 0] = x
    wrapped_x[len(x):sec_dim, 0] = x[0:sec_dim - len(x)]
    wrapped_x = np.squeeze(wrapped_x)
    return wrapped_x


def cfc_mi(sig, freqs_phase: list, freqs_amp: list,
           fs: int, nbins: int = 20) -> np.ndarray:
    """ Function that Calculates the MI matrix for a given signal x

        Parameters
        ----------
            sig: np.array
                signal in time domain
            freqs_phase: list
                phase frequencies (Hz)
            freqs_amp: list
                amplitude frequencies (Hz)
            fs: int
                sampling frequency (Hz)
            nbins: int
                number of bins

        Returns
        ----------
            MI_mat: np.ndarray
                2D MI matrix (phase x amplitude)
    """

    uniform_dist = np.ones((nbins - 1,))
    uniform_dist = uniform_dist / np.sum(uniform_dist)
    bins = np.linspace(0, 360, nbins)

    MI_mat = np.zeros((len(freqs_phase), len(freqs_amp)))
    for phase_counter, freq_phase in enumerate(tqdm(freqs_phase)):
        for amp_counter, freq_amp in enumerate(freqs_amp):
            phase_sig_filt, _, _ = butterworth_filter(sig, freq_phase, fs)
            amp_sig_filt, _, _ = butterworth_filter(sig, freq_amp, fs)
            inst_phase = extract_inst_phase(phase_sig_filt)
            inst_amp = extract_inst_amp(amp_sig_filt)

            means_amps_in_bin = []
            for bin_counter in range(len(bins) - 1):
                itemindex = np.where((inst_phase >= bins[bin_counter]) & (
                    inst_phase < bins[bin_counter + 1]))
                amps_in_bin = inst_amp[itemindex[0]]
                mean_amps_in_bin = np.mean(amps_in_bin)
                means_amps_in_bin.append(mean_amps_in_bin)
            means_amps_in_bin = np.array(means_amps_in_bin)
            means_amps_in_bin = means_amps_in_bin / np.sum(means_amps_in_bin)
            MI = np.sum(rel_entr(means_amps_in_bin, uniform_dist))
            MI_mat[phase_counter, amp_counter] = MI / nbins
    return MI_mat


def butterworth_filter(sig, filt_freq: float, fs: int) -> np.ndarray:
    """ Function that Filters signal array

        Parameters
        ----------
            sig: np.array
                signal in time domain
            filt_freq: float
                filter frequency (Hz)
            fs: int
                sampling frequency (Hz)

        Returns
        ----------
            filtered_sig: np.ndarray
                filtered signal
            b: np.ndarray
                filter numerator coefficients
            a: np.ndarray
                filter denominator coefficients
    """

    Q = 5
    b, a = signal.iirpeak(filt_freq, Q, fs)
    filtered_sig = signal.filtfilt(b, a, sig)
    return filtered_sig, b, a


def extract_inst_phase(sig) -> np.ndarray:
    """ Function that Extracts the instantaneous phase for a given signal x

        Parameters
        ----------
            sig: np.array
                signal in time domain

        Returns
        ----------
            inst_phase: np.ndarray
                instantaneous phase
    """

    z = signal.hilbert(sig)
    inst_phase = np.angle(z)  # inst phase
    inst_phase[inst_phase < 0] = inst_phase[inst_phase < 0] + 2 * np.pi
    inst_phase = inst_phase / np.pi * 180
    return inst_phase


def extract_inst_amp(sig) -> np.ndarray:
    """ Function that Extracts the instantaneous amplitude for a given signal x

        Parameters
        ----------
            sig: np.array
                signal in time domain

        Returns
        ----------
            inst_amplitude: np.ndarray
                instantaneous amplitude
    """

    z = signal.hilbert(sig)
    inst_amplitude = np.abs(z)
    return inst_amplitude
