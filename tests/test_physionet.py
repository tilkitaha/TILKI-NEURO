"""Importer contract tests using in-memory MNE recordings, not human EEG."""
import numpy as np
import pytest
mne = pytest.importorskip('mne')
from tilki_neuro.physionet import prepare


def test_import_units_order_labels_and_event_boundaries(tmp_path, monkeypatch):
    fs = 160
    signal = np.array([np.full(fs*12, value) for value in (1e-6, 2e-6, 3e-6)])
    raw = mne.io.RawArray(signal, mne.create_info(['C4','C3','Cz'],fs,'eeg'), verbose=False)
    raw.set_annotations(mne.Annotations([0,4,8,11], [4,4,3,1], ['T0','T1','T2','T1']))
    for run in (4,8,12):
        (tmp_path/f'S001R{run:02d}.edf').touch()
    monkeypatch.setattr(mne.io, 'read_raw_edf', lambda *a, **k: raw.copy())
    data = prepare(tmp_path, [1])
    assert data.x.shape == (9,3,320)
    np.testing.assert_array_equal(data.y, [0,1,2]*3)
    np.testing.assert_allclose(data.x[0,:,0], [2,3,1])
    assert set(data.groups) == {1}


def test_missing_edf_fails(tmp_path):
    with pytest.raises(ValueError, match='exactly one'):
        prepare(tmp_path, [1])
