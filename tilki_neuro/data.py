"""Dataset contract and deterministic toy signal generator.

Arrays are windows x channels x samples in MICROVOLTS. Labels 0/1/2 mean
idle / left imagery / right imagery; these are NOT decoded words.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sfreq: float
    source: str

    def validate(self):
        if self.x.ndim != 3 or min(self.x.shape) < 1:
            raise ValueError("x must be nonempty windows x channels x samples")
        if self.y.shape != (len(self.x),) or self.groups.shape != self.y.shape:
            raise ValueError("One label and subject group required per window")
        if not np.isfinite(self.sfreq) or self.sfreq < 64 or self.x.shape[2] < self.sfreq:
            raise ValueError("Need >=64 Hz and at least one second per window")
        if not np.isfinite(self.x).all() or not np.isin(self.y, [0, 1, 2]).all():
            raise ValueError("Signals must be finite; labels must be 0, 1, or 2")
        if not self.source:
            raise ValueError("Dataset provenance is required")
        return self

    def save(self, path):
        self.validate()
        np.savez_compressed(path, x=self.x, y=self.y, groups=self.groups,
                            sfreq=self.sfreq, source=self.source)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return Dataset(z['x'], z['y'], z['groups'], float(z['sfreq']),
                       str(z['source'])).validate()


def synthetic(seed=42, subjects=12, per_class=24):
    """Engineered mu suppression; sanity test only, not a physiological model."""
    if subjects < 3 or per_class < 2:
        raise ValueError("Need >=3 subjects and >=2 windows per class")
    rng = np.random.default_rng(seed)
    fs, n = 128., 256
    t = np.arange(n) / fs
    x, y, groups = [], [], []
    for subject in range(subjects):
        gain = rng.uniform(.8, 1.2, (3, 1))
        frequency = rng.uniform(9, 12)
        for label in range(3):
            for _ in range(per_class):
                amplitude = np.array([12., 10., 12.])
                if label == 1:
                    amplitude[2] *= rng.uniform(.25, .65)
                if label == 2:
                    amplitude[0] *= rng.uniform(.25, .65)
                phase = rng.uniform(0, 2*np.pi, (3, 1))
                wave = amplitude[:, None] * np.sin(2*np.pi*frequency*t + phase)
                wave += 4*np.sin(2*np.pi*21*t + phase)
                wave += rng.normal(0, rng.uniform(3, 7), (3, n))
                x.append(wave*gain)
                y.append(label)
                groups.append(subject)
    return Dataset(np.array(x), np.array(y), np.array(groups), fs,
                   'synthetic-engineered-mu-v1').validate()
