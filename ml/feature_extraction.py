# feature_extraction.py
import numpy as np
from scipy import signal, stats

def butter_lowpass_filter(data, cutoff=10, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def sliding_windows(X, window_size=200, step=100):
    # X: (T, features)
    windows = []
    for start in range(0, X.shape[0] - window_size + 1, step):
        windows.append(X[start:start+window_size])
    return np.array(windows)

def time_domain_features(win):
    # win shape (T, F)
    feats = []
    # for each signal column
    for i in range(win.shape[1]):
        col = win[:, i]
        feats += [
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col),
            np.percentile(col, 25),
            np.percentile(col, 75),
            stats.skew(col),
            stats.kurtosis(col),
            np.sqrt(np.mean(col**2)), # RMS
            np.ptp(col) # peak to peak
        ]
    # add jerk features (derivative stats)
    jerk = np.diff(win, axis=0)
    for i in range(jerk.shape[1]):
        col = jerk[:, i]
        feats += [np.mean(col), np.std(col)]
    return np.array([0 if np.isnan(x) else x for x in feats])
    
def spectral_features(win, fs=100):
    # compute dominant freq for each column
    feats = []
    T = win.shape[0]
    freqs = np.fft.rfftfreq(T, d=1/fs)
    for i in range(win.shape[1]):
        Xf = np.abs(np.fft.rfft(win[:, i]))
        dom_idx = np.argmax(Xf)
        feats += [freqs[dom_idx], np.sum(Xf**2)]  # dominant frequency, energy
    return np.array(feats)
