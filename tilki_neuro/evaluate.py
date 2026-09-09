"""Subject-disjoint train/validation/test evaluation and threshold selection."""
import platform
import time
import numpy as np
import scipy
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from .decoder import Decoder, quality


def split(data, seed=42):
    data.validate()
    if len(np.unique(data.groups)) < 6:
        raise ValueError('Need at least six subject groups for this protocol')
    ids = np.arange(len(data.y))
    outer = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    remaining, test = next(outer.split(ids, data.y, data.groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed+1)
    a, b = next(inner.split(remaining, data.y[remaining], data.groups[remaining]))
    train, val = remaining[a], remaining[b]
    for part in (train, val, test):
        if set(np.unique(data.y[part])) != {0, 1, 2}:
            raise ValueError('Every partition must contain all three classes')
    return train, val, test


def metrics(y, pred, seconds):
    active = y != 0
    emitted = pred > 0
    accepted = pred >= 0
    idle = y == 0
    return {
        'windows': len(y),
        'coverage': float(accepted.mean()),
        'accepted_accuracy': float((pred[accepted] == y[accepted]).mean()) if accepted.any() else None,
        'correct_command_recall': float(((pred == y) & active).sum()/active.sum()),
        'wrong_commands': int((emitted & (pred != y)).sum()),
        'idle_false_activation_fraction': float(emitted[idle].mean()),
        'idle_false_activations_per_minute': float(emitted[idle].sum()/(idle.sum()*seconds/60)),
        'abstained_windows': int((pred == -1).sum()),
    }


def select_threshold(y, probabilities, good, seconds, target=.05):
    if not 0 <= target <= 1:
        raise ValueError('False activation target must be in [0, 1]')
    candidates = []
    for threshold in np.linspace(1/3, 1, 81):
        pred = probabilities.argmax(1).copy()
        pred[(probabilities.max(1) < threshold) | ~good] = -1
        m = metrics(y, pred, seconds)
        candidates.append((float(threshold), m))
    eligible = [c for c in candidates if c[1]['idle_false_activation_fraction'] <= target]
    # If no threshold qualifies, fail instead of silently violating the target.
    if not eligible:
        raise ValueError('No validation threshold meets the configured target')
    best = max(eligible, key=lambda c: (c[1]['correct_command_recall'], c[1]['coverage'], c[0]))
    return best[0], best[1]


def evaluate(data, seed=42, target=.05):
    train, val, test = split(data, seed)
    clean_train = train[quality(data.x[train])]
    model = Decoder(data.sfreq).fit(data.x[clean_train], data.y[clean_train])
    seconds = data.x.shape[-1]/data.sfreq
    _, vp, good = model.predict(data.x[val], 0)
    threshold, vm = select_threshold(data.y[val], vp, good, seconds, target)
    start = time.perf_counter()
    gated, probabilities, _ = model.predict(data.x[test], threshold)
    elapsed = time.perf_counter() - start
    # Raw model comparator has neither confidence rejection nor quality gating.
    baseline = model.probabilities(data.x[test]).argmax(1)
    stress = data.x[test].copy()
    stress[:, 0, :] = 0
    stress_pred, _, _ = model.predict(stress, threshold)
    report = {
        'version': '0.1.0', 'source': data.source, 'seed': seed,
        'interpretation': 'Offline window classification; not clinical or continuous-use validation.',
        'protocol': 'Subject-disjoint, non-overlapping windows; threshold selected only on validation subjects.',
        'groups': {k: np.unique(data.groups[v]).tolist() for k, v in zip(('train', 'validation', 'test'), (train, val, test))},
        'window_seconds': seconds, 'training_windows_excluded_for_quality': int(len(train)-len(clean_train)),
        'validation_idle_false_activation_target': target,
        'threshold': threshold, 'validation': vm,
        'baseline': metrics(data.y[test], baseline, seconds),
        'baseline_balanced_accuracy': float(balanced_accuracy_score(data.y[test], baseline)),
        'baseline_confusion_matrix_labels_0_1_2': confusion_matrix(data.y[test], baseline, labels=[0,1,2]).tolist(),
        'gated': metrics(data.y[test], gated, seconds),
        'flat_channel_stress': metrics(data.y[test], stress_pred, seconds),
        'batch_wall_ms_per_window': elapsed/len(test)*1000,
        'timing_note': 'Host batch timing, not streaming latency or chip energy; excludes 2-second acquisition where applicable.',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
    }
    replay = []
    for i, idx in enumerate(test[:90]):
        replay.append({'truth': int(data.y[idx]), 'prediction': int(gated[i]),
                       'confidence': float(probabilities[i].max()),
                       'signal_uv': data.x[idx, 0, ::4].round(3).tolist()})
    return report, replay
