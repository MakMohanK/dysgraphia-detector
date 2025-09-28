# build_dataset.py
import os, glob, numpy as np, pandas as pd
from feature_extraction import butter_lowpass_filter, sliding_windows, time_domain_features, spectral_features

DATA_DIR = '../data'  # where serial_logger saved sessions
OUT = 'dataset.npz'

WINDOW = 200  # timesteps (~2s @100Hz)
STEP = 100

X_feats = []
y = []

for f in glob.glob(os.path.join(DATA_DIR, '*.csv')):
    fname = os.path.basename(f)
    # Label extraction from filename convention: e.g. ..._dys.csv or ..._normal.csv
    label = 1 if 'dys' in fname.lower() or 'dysgraphia' in fname.lower() else 0

    df = pd.read_csv(f)
    # columns expected: timestamp_ms,ax,ay,az,gx,gy,gz,p1,p2
    cols = ['ax','ay','az','gx','gy','gz','p1','p2']
    X = df[cols].values.astype(float)
    # lowpass filter
    Xf = butter_lowpass_filter(X, cutoff=20, fs=100, order=4)
    windows = sliding_windows(Xf, window_size=WINDOW, step=STEP)
    for w in windows:
        tf = time_domain_features(w)
        sf = spectral_features(w)
        feats = np.concatenate([tf, sf])
        X_feats.append(feats)
        y.append(label)

X_feats = np.vstack(X_feats)
y = np.array(y)
np.savez(OUT, X=X_feats, y=y)
print("Saved", OUT, "X.shape=", X_feats.shape, "y.shape=", y.shape)
