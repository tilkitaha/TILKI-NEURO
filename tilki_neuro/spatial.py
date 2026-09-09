"""Established EEG baselines; no invented novelty or test-set adaptation."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .decoder import features


class Bandpass(TransformerMixin, BaseEstimator):
    def __init__(self, sfreq=160., low=8., high=30.):
        self.sfreq=sfreq; self.low=low; self.high=high

    def fit(self, x, y=None):
        if not 0 < self.low < self.high < self.sfreq/2:
            raise ValueError('Invalid passband')
        self.sos_ = butter(4, [self.low,self.high], fs=self.sfreq, btype='bandpass',output='sos')
        return self

    def transform(self, x):
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite EEG')
        # Offline zero-phase filtering WITHIN each window, never across splits.
        # Not a causal streaming implementation; edge effects remain.
        return sosfiltfilt(self.sos_,x,axis=-1)


class Bandpower(TransformerMixin, BaseEstimator):
    def __init__(self, sfreq=160.):
        self.sfreq=sfreq
    def fit(self,x,y=None):
        return self
    def transform(self,x):
        return features(x,self.sfreq)


class FilterbankCSP(TransformerMixin, BaseEstimator):
    def __init__(self, sfreq=160., components=4):
        self.sfreq=sfreq; self.components=components
    def fit(self,x,y):
        from mne.decoding import CSP
        self.filters_=[]; self.csps_=[]
        for lo,hi in ((8,12),(12,16),(16,24),(24,30)):
            f=Bandpass(self.sfreq,lo,hi).fit(x)
            c=CSP(n_components=self.components,reg=.1,log=True,norm_trace=False,rank='full')
            c.fit(f.transform(x),y)
            self.filters_.append(f); self.csps_.append(c)
        return self
    def transform(self,x):
        return np.concatenate([c.transform(f.transform(x)) for f,c in zip(self.filters_,self.csps_)],axis=1)


def make_model(name, sfreq=160., n_classes=3):
    from mne.decoding import CSP
    lr=lambda: LogisticRegression(C=1.,max_iter=1500,class_weight='balanced',solver='lbfgs')
    if name=='bandpower':
        return make_pipeline(Bandpower(sfreq),StandardScaler(),lr())
    if name=='csp_lda':
        return make_pipeline(Bandpass(sfreq),CSP(n_components=6,reg=.1,log=True,
                             norm_trace=False,rank='full'),
                             LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(n_classes)/n_classes))
    if name=='riemann':
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        return make_pipeline(Bandpass(sfreq),Covariances(estimator='oas'),
                             TangentSpace(metric='riemann',tsupdate=False),StandardScaler(),lr())
    if name=='filterbank_csp':
        return make_pipeline(FilterbankCSP(sfreq),StandardScaler(),lr())
    raise ValueError(f'Unknown model {name}')


def diagnostics(x, config):
    std=x.std(axis=-1); ptp=np.ptp(x,axis=-1)
    flat=std<config['minimum_channel_std_uv']; large=ptp>config['maximum_channel_ptp_uv']
    good=np.isfinite(x).all(axis=(1,2)) & ~flat.any(axis=1) & ~large.any(axis=1)
    return good, {'windows':len(x),'flat_windows':int(flat.any(1).sum()),
                  'large_amplitude_windows':int(large.any(1).sum()),
                  'flat_counts_by_channel':flat.sum(0).tolist(),
                  'large_counts_by_channel':large.sum(0).tolist(),
                  'rejected':int((~good).sum())}
