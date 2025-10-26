# realtime_dysgraphia_predictor.py
import time
import numpy as np
import joblib
from collections import deque
import pandas as pd
from scipy import signal

MODEL_IN = "models/dysgraphia_rf_pipeline.pkl"
WINDOW_SEC = 2.0
STEP_SEC = 1.0

# load saved pipeline
pipe = joblib.load(MODEL_IN)
scaler = pipe.named_steps['scaler']
clf = pipe.named_steps['clf']

# buffer will hold tuples (timestamp, ax,ay,az,gx,gy,gz,p1,p2)
buffer = deque()

# BFS: estimate fs from initial bursts; fallback to 50 Hz
def estimate_fs_from_buffer(buf):
    if len(buf) < 3:
        return 50.0
    ts = np.array([b[0] for b in buf])
    d = np.diff(ts)
    median_dt = np.median(d)
    return 1.0/median_dt if median_dt>0 else 50.0

# same feature extractor as training (must match)
def extract_features_from_window_array(win, fs):
    from scipy import signal
    features = []
    for i in range(win.shape[1]):
        x = win[:, i]
        features += [
            x.mean(), x.std(), np.min(x), np.max(x),
            np.percentile(x,25), np.percentile(x,50), np.percentile(x,75),
            np.mean(np.abs(np.diff(x))),
        ]
    acc = np.linalg.norm(win[:, 0:3], axis=1)
    gyr = np.linalg.norm(win[:, 3:6], axis=1)
    features += [acc.mean(), acc.std(), acc.max(), gyr.mean(), gyr.std(), gyr.max()]
    # frequency domain for ax,ay,az
    for i in range(3):
        x = win[:, i] - np.mean(win[:, i])
        f, Pxx = signal.welch(x, fs=fs, nperseg=min(len(x), 256))
        dom = f[np.argmax(Pxx)] if Pxx.size>0 else 0.0
        bw = np.sum(Pxx > (Pxx.max() * 0.5)) if Pxx.size>0 else 0.0
        features += [dom, bw]
    return np.array(features)

def window_and_predict():
    # build numpy array from buffer
    arr = np.array([list(b)[1:] for b in buffer])  # drop timestamp column for array
    # arr shape (N,8)
    # compute fs
    fs = estimate_fs_from_buffer(buffer)
    n_window = int(round(WINDOW_SEC * fs))
    n_step = int(round(STEP_SEC * fs))
    predictions = []
    idx = 0
    while idx + n_window <= arr.shape[0]:
        win = arr[idx: idx + n_window, :]
        feat = extract_features_from_window_array(win, fs)
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        proba = clf.predict_proba(feat_scaled)[0,1] if hasattr(clf, "predict_proba") else clf.predict(feat_scaled)[0]
        pred = 1 if proba >= 0.5 else 0
        predictions.append((idx, pred, float(proba)))
        idx += n_step
    return predictions

# Example: replace this with your device read (serial, socket, etc.)
def read_sample_from_device():
    """
    Must return a tuple: (timestamp_seconds, ax,ay,az,gx,gy,gz,p1,p2)
    Implement reading from serial or other in your environment.
    This function here generates synthetic data for demonstration.
    """
    t = time.time()
    # replace below with real sensor reads
    ax = np.random.randn() * 0.1
    ay = np.random.randn() * 0.1
    az = 1.0 + np.random.randn() * 0.1
    gx = np.random.randn() * 0.01
    gy = np.random.randn() * 0.01
    gz = np.random.randn() * 0.01
    p1 = np.random.randint(0,2)
    p2 = np.random.randint(0,2)
    return (t, ax, ay, az, gx, gy, gz, p1, p2)

if __name__ == "__main__":
    print("Real-time predictor running. Press Ctrl+C to exit.")
    try:
        while True:
            sample = read_sample_from_device()
            buffer.append(sample)
            # keep buffer size big enough: e.g. 5 seconds worth of data
            # estimate fs roughly: if small, cap buffer length
            fs = estimate_fs_from_buffer(buffer)
            max_len = int(max(5*fs, 200))
            while len(buffer) > max_len:
                buffer.popleft()

            # do prediction whenever we have at least one window
            min_needed = int(round(WINDOW_SEC * fs))
            if len(buffer) >= min_needed:
                preds = window_and_predict()
                if preds:
                    # take last window prediction as current state
                    idx, pred, proba = preds[-1]
                    label = "Dysgraphia" if pred==1 else "Normal"
                    print(f"[{time.strftime('%H:%M:%S')}] Pred: {label}  prob:{proba:.3f}")
            time.sleep(1.0/fs)  # sample rate pacing; in real-case this is driven by incoming data
    except KeyboardInterrupt:
        print("Stopping.")
