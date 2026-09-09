import numpy as np
import pytest
from tilki_neuro.data import synthetic, load
from tilki_neuro.decoder import Decoder, quality
from tilki_neuro.evaluate import split, evaluate, metrics, select_threshold
from tilki_neuro.communication import Board

@pytest.fixture(scope='module')
def data():
    return synthetic(subjects=8, per_class=8)


def test_subjects_do_not_leak(data):
    train, val, test = split(data)
    groups = [set(data.groups[p]) for p in (train,val,test)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert len(set(train) | set(val) | set(test)) == len(data.y)


def test_determinism_and_serialization(data, tmp_path):
    np.testing.assert_array_equal(data.x, synthetic(subjects=8, per_class=8).x)
    path = tmp_path/'data.npz'
    data.save(path)
    np.testing.assert_array_equal(load(path).x, data.x)


def test_signal_failure_rejects(data):
    model = Decoder(data.sfreq).fit(data.x, data.y)
    x = data.x[:3].copy()
    x[0,0] = 0
    x[1,0,0] = np.nan
    x[2,0,0] = 1000
    pred, p, good = model.predict(x)
    assert (pred == -1).all()
    assert not good.any()
    assert np.isfinite(p).all()


def test_unseen_channel_layout_rejected(data):
    model = Decoder(data.sfreq).fit(data.x, data.y)
    with pytest.raises(ValueError):
        model.predict(data.x[:, :2])


def test_command_requires_idle_and_confirmation():
    b = Board()
    b.step(2)
    b.step(2)
    assert b.history == []
    b.step(0)
    b.step(2)
    assert b.history == ['Yes']
    b.undo()
    assert b.history == []


def test_bad_signal_cancels_and_cannot_rearm():
    b = Board()
    b.step(2)
    b.step(-1)
    assert b.pending is None
    b.step(2)
    assert b.pending is None
    assert not b.history


def test_threshold_selection_has_no_test_input():
    y = np.array([0,0,1,2])
    p = np.array([[.4,.6,0],[.9,.1,0],[.1,.9,0],[.1,0,.9]])
    threshold, report = select_threshold(y,p,np.ones(4,dtype=bool),2,target=0)
    assert threshold > .6
    assert report['idle_false_activation_fraction'] == 0
    assert report['correct_command_recall'] == 1


def test_abstention_metrics_do_not_claim_perfect_accuracy():
    m = metrics(np.array([0,1,2]),np.array([-1,-1,-1]),2)
    assert m['accepted_accuracy'] is None
    assert m['coverage'] == 0
    assert m['correct_command_recall'] == 0


def test_end_to_end(data):
    report, replay = evaluate(data)
    assert report['source'].startswith('synthetic')
    assert report['flat_channel_stress']['coverage'] == 0
    assert report['baseline_balanced_accuracy'] > .6  # Toy signal sanity, not human EEG claim.
    assert len(replay) > 0


def test_invalid_data_rejected(data):
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(data, sfreq=0).validate()
    with pytest.raises(ValueError):
        replace(data, y=np.full(len(data.y), 9)).validate()
