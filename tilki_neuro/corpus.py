"""Run-aware EDF import with source provenance. No human data leaves the process."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


@dataclass
class Corpus:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    ids: np.ndarray
    sfreq: float
    channels: tuple
    manifest: list

    def take(self, mask):
        return self.x[mask], self.y[mask]


def load_corpus(root, subjects, protocol):
    import mne
    from mne.datasets.eegbci import standardize
    x, y, ss, rr, ids, manifest = [], [], [], [], [], []
    channels = protocol['channels']
    for subject in subjects:
        for run in protocol['runs']:
            name = f'S{subject:03d}R{run:02d}.edf'
            paths = list(Path(root).rglob(name))
            if len(paths) != 1:
                raise ValueError(f'Expected one {name}, found {len(paths)}')
            path = paths[0]
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            standardize(raw)
            if raw.info['sfreq'] != 160 or len(raw.ch_names) != 64:
                raise ValueError(f'{name}: expected 64 channels at 160 Hz')
            # Fixed 64-channel common average: local in time, no fitted test statistics.
            signal = raw.get_data() * 1e6
            signal -= signal.mean(axis=0, keepdims=True)
            signal = signal[[raw.ch_names.index(c) for c in channels]]
            events = list(raw.annotations)
            for i, a in enumerate(events):
                if a['description'] not in ('T0','T1','T2'):
                    continue
                start = float(a['onset']) + protocol['epoch_start_seconds']
                end = start + protocol['epoch_duration_seconds']
                stop = float(a['onset'] + a['duration'])
                if i+1 < len(events):
                    stop = min(stop, float(events[i+1]['onset']))
                if end > stop or round(end*160) > signal.shape[1]:
                    continue
                window = signal[:,round(start*160):round(end*160)]
                if not np.isfinite(window).all():
                    raise ValueError(f'{name} event {i}: nonfinite input')
                x.append(window); y.append(int(a['description'][1])); ss.append(subject); rr.append(run)
                ids.append(f'{name}:{i}:{round(start*160)}')
            manifest.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if not x:
        raise ValueError('No eligible EEG windows')
    return Corpus(np.array(x),np.array(y),np.array(ss),np.array(rr),np.array(ids),
                  160.,tuple(channels),manifest)
