"""Import local EEGMMIDB EDFs; download is an explicit separate option."""
from pathlib import Path
import numpy as np
from .data import Dataset


def prepare(root, subjects, download=False):
    import mne
    from mne.datasets import eegbci
    subjects = list(subjects)
    if not subjects or len(set(subjects)) != len(subjects) or any(s < 1 or s > 109 for s in subjects):
        raise ValueError('Provide unique subject IDs from 1 to 109')
    x, y, groups = [], [], []
    for subject in subjects:
        for run in (4, 8, 12):  # ONLY left/right motor imagery; no actual motion runs.
            filename = f'S{subject:03d}R{run:02d}.edf'
            if download:
                paths = eegbci.load_data(subject, [run], path=root, update_path=False, verbose=False)
                path = paths[0]
            else:
                paths = list(Path(root).rglob(filename))
                if len(paths) != 1:
                    raise ValueError(f'Expected exactly one {filename} below {root}; found {len(paths)}')
                path = paths[0]
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
            eegbci.standardize(raw)
            # Explicit fixed channel ordering and conversion from volts to uV.
            signal = raw.get_data(picks=['C3', 'Cz', 'C4']) * 1e6
            fs = float(raw.info['sfreq'])
            if fs != 160:
                raise ValueError(f'{filename}: expected 160 Hz EEGMMIDB recording')
            annotations = list(raw.annotations)
            for i, event in enumerate(annotations):
                code = event['description']
                if code not in ('T0', 'T1', 'T2'):
                    continue
                # Exclude transitions; one 2s window per event, never crosses next event.
                start = float(event['onset']) + .5
                end = start + 2
                event_end = float(event['onset']) + float(event['duration'])
                if i+1 < len(annotations):
                    event_end = min(event_end, float(annotations[i+1]['onset']))
                if end > event_end or round(end*fs) > signal.shape[-1]:
                    continue
                window = signal[:, round(start*fs):round(end*fs)]
                if window.shape[-1] != 320:
                    continue
                x.append(window)
                y.append({'T0': 0, 'T1': 1, 'T2': 2}[code])
                groups.append(subject)
    if not x:
        raise ValueError('No usable windows found')
    return Dataset(np.array(x), np.array(y), np.array(groups), 160.,
                   'PhysioNet-EEGMMIDB-1.0.0-runs-4-8-12-C3-Cz-C4').validate()
