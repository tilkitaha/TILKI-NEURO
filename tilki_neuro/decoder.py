"""Windowed spectral baseline. No future windows enter feature computation."""
import numpy as np
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def features(x, sfreq):
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite windows x channels x samples')
    if not np.isfinite(sfreq) or sfreq < 64 or x.shape[-1] < sfreq:
        raise ValueError('Need >=64 Hz and at least one second of samples')
    f, p = welch(x, fs=sfreq, nperseg=min(x.shape[-1], int(sfreq)), axis=-1)
    bands = [(4, 8), (8, 13), (13, 30)]
    values = [np.log(np.maximum(p[..., (f >= lo) & (f < hi)].mean(-1), 1e-12))
              for lo, hi in bands]
    return np.stack(values, axis=-1).reshape(len(x), -1)


def quality(x):
    """Conservative engineering heuristics in uV, not clinically validated."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError('Expected windows x channels x samples')
    finite = np.isfinite(x).all(axis=(1, 2))
    clean = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    flat = (clean.std(-1) < .1).any(-1)
    large = (np.ptp(clean, axis=-1) > 200).any(-1)
    return finite & ~flat & ~large


class Decoder:
    def __init__(self, sfreq):
        self.sfreq = sfreq
        self.pipeline = make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=1000, C=1.0))

    def fit(self, x, y):
        if set(np.unique(y)) != {0, 1, 2}:
            raise ValueError('Training requires idle, left, and right classes')
        self.channels = x.shape[1]
        self.pipeline.fit(features(x, self.sfreq), y)
        return self

    def probabilities(self, x):
        if x.shape[1] != self.channels:
            raise ValueError('Channel count differs from training')
        return self.pipeline.predict_proba(features(x, self.sfreq))

    def predict(self, x, threshold=.8):
        if not .0 <= threshold <= 1.:
            raise ValueError('Threshold must be in [0, 1]')
        ok = quality(x)
        p = np.zeros((len(x), 3))
        p[ok] = self.probabilities(x[ok]) if ok.any() else p[ok]
        pred = np.argmax(p, axis=1)
        pred[(p.max(1) < threshold) | ~ok] = -1
        return pred, p, ok
