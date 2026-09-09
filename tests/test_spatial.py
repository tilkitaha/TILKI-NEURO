import numpy as np
import pytest
pytest.importorskip('pyriemann')
import mne
from threadpoolctl import threadpool_limits
from tilki_neuro.spatial import make_model,Bandpass,diagnostics
from tilki_neuro.study import choose_gate,apply_gate,bootstrap,assert_partitions
from tilki_neuro.data import synthetic
from tilki_neuro.corpus import Corpus


def test_window_filter_does_not_use_neighboring_windows():
    rng=np.random.default_rng(1);x=rng.normal(size=(3,6,480))
    f=Bandpass().fit(x)
    a=f.transform(x)
    x[2]*=10000
    np.testing.assert_array_equal(a[:2],f.transform(x)[:2])


@pytest.mark.parametrize('name',['bandpower','csp_lda','riemann','filterbank_csp'])
def test_established_models_fit_and_predict_finite_probabilities(name):
    mne.set_log_level('ERROR')
    d=synthetic(subjects=3,per_class=5)
    x=np.tile(d.x,(1,3,1))+np.random.default_rng(7).normal(0,.5,(len(d.x),9,256))
    with threadpool_limits(limits=1):
        model=make_model(name,128.,3).fit(x[:30],d.y[:30])
        p=model.predict_proba(x[30:])
    assert p.shape==(15,3)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1),1)


def test_fail_closed_if_safe_threshold_has_no_utility():
    y=np.array([0,0,1,2]);p=np.tile([.99,.005,.005],(4,1))
    gate=choose_gate(y,p,np.ones(4,bool),3,{'idle_false_activation_target':.05,'minimum_validation_command_recall':.2})
    assert not gate['enabled']
    assert (apply_gate(p,np.ones(4,bool),gate)==-1).all()


def test_quality_gate_cannot_be_bypassed_by_confidence():
    p=np.array([[.01,.99,0]])
    assert apply_gate(p,np.array([False]),{'enabled':True,'threshold':.5})[0]==-1


def test_bootstrap_is_deterministic_and_subject_level():
    a=bootstrap([.5,.7,.8],42,2000)
    assert a==bootstrap([.5,.7,.8],42,2000)
    assert a['subjects']==3
    assert a['subject_bootstrap_95_percentile'][0]<=a['mean']<=a['subject_bootstrap_95_percentile'][1]


def test_subject_leakage_rejected_even_when_windows_are_disjoint():
    c=Corpus(np.ones((3,2,160)),np.array([0,1,2]),np.array([1,1,2]),np.array([4,8,12]),np.array(['a','b','c']),160.,('C3','C4'),[])
    with pytest.raises(ValueError,match='Subject leakage'):
        assert_partitions(c,np.array([1,0,0],bool),np.array([0,1,0],bool),np.array([0,0,1],bool),False)
