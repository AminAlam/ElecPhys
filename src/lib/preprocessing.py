import numpy as np
from scipy import signal

def apply_notch(_LFP_chan, _args):
    for f0 in np.arange(_args['f0'],300,_args['f0']):
        b_notch, a_notch = signal.iirnotch(f0, _args['Q'], _args['fs'])
        _LFP_chan = signal.filtfilt(b_notch, a_notch, _LFP_chan)
    return _LFP_chan