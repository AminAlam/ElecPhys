from scipy.special import rel_entr
import numpy as np
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fft import fft


def calc_tf_mvl(x, fs, freqs_phase, freqs_amp):
    """ Calculates the tf_MVL matrix for a given signal x
        input:
            x: signal - type: np.array
            fs: sampling frequency - type: float
            freqs_phase: phase frequencies - type: list
            freqs_amp: amplitude frequencies - type: list
        output:
            MI_mat: tf_MVL matrix - type: np.ndarray
    """

    MI_mat = np.zeros((len(freqs_phase), len(freqs_amp)))
    for phase_counter, freq_phase in enumerate(freqs_phase):
        for amp_counter, freq_amp in enumerate(freqs_amp):
            mvl = band_tfMVL(x, [freq_amp-1, freq_amp+1], [freq_phase-1, freq_phase+1], fs)
            MI_mat[phase_counter, amp_counter] = mvl        
    return MI_mat


def band_tfMVL(x, high_freq, low_freq, fs):
    """ Calculates the tf_MVL for a given signal x
        input:
            x: signal - type: np.array
            high_freq: high frequency band - type: list
            low_freq: low frequency band - type: list
            fs: sampling frequency - type: float
        output:
            tf_canolty: tf_MVL - type: float
    """

    fs = int(fs)
    tfd = rid_rihaczek4(x,fs)
    W = tfd
    W2 = W[1:,:]
    Amp = np.abs(W2[high_freq,:])
    tfd_low = W2[low_freq,:]
    angle_low = np.angle(tfd_low)
    Phase = angle_low
    tf_canolty = calc_MVL(Phase, Amp)
    return tf_canolty


def calc_MVL(phase, amp):
    """ Calculates the MVL for a given signal x
        input:
            phase: phase - type: np.array
            amp: amplitude - type: np.array
        output:
            MVL: MVL - type: type: np.array
    """

    z1 = np.exp(1j*phase)
    z = np.multiply(amp, z1)
    MVL = np.abs(np.mean(z))
    return MVL


def rid_rihaczek4(x, fbins):
    """ Calculates the rid_rihaczek4 for a given signal x
        input:
            x: signal - type: np.array
            fbins: frequency bins - type: int
        output:
            tfd: rid_rihaczek4 - type: np.ndarray
    """

    tbins = len(x)
    amb = np.zeros((tbins, tbins))

    for tau in range(tbins):
        indexes = np.arange(tau, tbins, 1)
        indexes = np.append(indexes, np.arange(0, tau, 1))
        amb[tau,:] = np.multiply(np.conj(x), x[indexes])

    indexes = np.arange(int(tbins/2), tbins)
    indexes = np.append(indexes, np.arange(0, int(tbins/2)))

    ambTemp = amb[:, indexes]

    indexes = np.arange(0, int(tbins/2))
    indexes = np.append(np.arange(int(tbins/2), tbins), indexes)
    amb1 = ambTemp[indexes, :]

    D_mult = np.linspace(-1,1, tbins)
    D_mult = np.reshape(D_mult, (-1, len(D_mult)))
    D = np.matmul(np.transpose(D_mult), D_mult)
    L = D

    K=chwi_krn(D,L,0.01)
    df = K
    ambf = np.multiply(amb1, df)
    
    A = np.zeros((fbins,tbins))
    tbins = tbins-1
    if tbins != fbins:
        for tt in range(tbins):
            A[:,tt] = data_wrapper(ambf[:,tt], fbins)
    else:
        A = ambf

    tfd = np.zeros(A.shape,dtype = 'complex_')
    for col in range(A.shape[1]):
        tfd[:, col] = fft(A[:, col])

    return tfd


def chwi_krn(D, L, A):
    """ Calculates the chwi_krn for a given signal x
        input:
            D: D - type: np.array
            L: L - type: np.array
            A: A - type: float
        output:
            k: chwi_krn - type: np.ndarray
    """

    k = np.exp((-1/(A**2))*np.multiply(np.multiply(D,D), np.multiply(L,L)))
    return k


def data_wrapper(x, sec_dim):
    """ Wraps the data for a given signal x
        input:
            x: signal - type: np.array
            sec_dim: second dimension - type: int
        output:
            wrapped_x: wrapped signal - type: np.ndarray
    """

    wrapped_x = np.zeros((sec_dim, 1))
    wrapped_x[0:len(x), 0] = x
    wrapped_x[len(x):sec_dim, 0] = x[0:sec_dim-len(x)]
    wrapped_x = np.squeeze(wrapped_x)
    return wrapped_x


def cfc_mi(sig, freqs_phase, freqs_amp, fs, nbins=20):
    """ Calculates the MI matrix for a given signal x
            input:
            sig: signal - type: np.array
            freqs_phase: phase frequencies - type: list
            freqs_amp: amplitude frequencies - type: list
            fs: sampling frequency - type: float
            nbins: number of bins - type: int
        output:
            MI_mat: MI matrix - type: np.ndarray
    """

    uniform_dist = np.ones((nbins-1,))
    uniform_dist = uniform_dist/np.sum(uniform_dist)
    bins = np.linspace(0, 360, nbins)
    
    MI_mat = np.zeros((len(freqs_phase), len(freqs_amp)))
    for phase_counter, freq_phase in enumerate(tqdm(freqs_phase)):
        for amp_counter, freq_amp in enumerate(freqs_amp):
            phase_sig_filt,_,_ = butterworth_filter(sig, freq_phase, fs)
            amp_sig_filt,_,_ = butterworth_filter(sig, freq_amp, fs)
            inst_phase = extract_inst_phase(phase_sig_filt)
            inst_amp = extract_inst_amp(amp_sig_filt)
            
            means_amps_in_bin = []
            for bin_counter in range(len(bins)-1):
                itemindex = np.where((inst_phase>=bins[bin_counter]) & (inst_phase<bins[bin_counter+1]))
                amps_in_bin = inst_amp[itemindex[0]]
                mean_amps_in_bin = np.mean(amps_in_bin)
                means_amps_in_bin.append(mean_amps_in_bin)
            means_amps_in_bin = np.array(means_amps_in_bin)
            means_amps_in_bin = means_amps_in_bin/np.sum(means_amps_in_bin)
            MI = np.sum(rel_entr(means_amps_in_bin, uniform_dist))
            MI_mat[phase_counter, amp_counter] = MI/nbins
    return MI_mat


def butterworth_filter(sig, filt_freq, fs):
    """ Filters signal array
        input:
            sig: signal - type: np.array
            filt_freq: filter frequency - type: float
            fs: sampling frequency - type: float
        output:
            filtered_sig: filtered signal - type: np.ndarray
            b: filter numerator coefficients - type: np.ndarray
            a: filter denominator coefficients - type: np.ndarray
    """

    Q = 5
    b, a = signal.iirpeak(filt_freq, Q, fs)
    filtered_sig = signal.filtfilt(b, a, sig)
    return filtered_sig, b, a


def extract_inst_phase(sig):
    """ Extracts the instantaneous phase for a given signal x
        input:
            sig: signal - type: np.array
        output:
            inst_phase: instantaneous phase - type: np.ndarray
    """

    z = signal.hilbert(sig)
    inst_phase = np.angle(z) #inst phase
    inst_phase[inst_phase<0] = inst_phase[inst_phase<0] + 2*np.pi
    inst_phase = inst_phase/np.pi*180
    return inst_phase


def extract_inst_amp(sig):
    """ Extracts the instantaneous amplitude for a given signal x
        input:
            sig: signal - type: np.array
        output:
            inst_amplitude: instantaneous amplitude - type: np.ndarray
    """

    z = signal.hilbert(sig)
    inst_amplitude = np.abs(z)
    return inst_amplitude