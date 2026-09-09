"""Frozen-protocol comparison with validation-only selection and subject intervals."""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits
from .corpus import load_corpus
from .spatial import make_model, diagnostics
from .evaluate import metrics


def summary(y,p,classes):
    pred=np.asarray(classes)[p.argmax(1)]
    onehot=(y[:,None]==np.asarray(classes)[None,:]).astype(float)
    return {'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'accuracy':float(np.mean(y==pred)),
            'brier_multiclass':float(np.mean(np.sum((p-onehot)**2,axis=1))),
            'confusion':confusion_matrix(y,pred,labels=classes).tolist(),
            'n':len(y)}


def choose_gate(y,p,good,seconds,config):
    candidates=[]
    for threshold in np.linspace(1/3,1,81):
        pred=p.argmax(1)
        pred[(p.max(1)<threshold)|~good]=-1
        m=metrics(y,pred,seconds)
        if m['idle_false_activation_fraction']<=config['idle_false_activation_target'] and m['correct_command_recall']>=config['minimum_validation_command_recall']:
            candidates.append((float(threshold),m))
    if not candidates:
        return {'enabled':False,'threshold':1.,'reason':'No validation threshold meets both idle-error and useful-command requirements.'}
    threshold,m=max(candidates,key=lambda t:(t[1]['correct_command_recall'],t[1]['coverage'],t[0]))
    return {'enabled':True,'threshold':threshold,'validation':m,
            'reason':'Validation requirements met; this is not a test-time guarantee.'}


def apply_gate(p,good,gate):
    pred=p.argmax(1)
    if not gate['enabled']:
        return np.full(len(p),-1,dtype=int)
    pred[(p.max(1)<gate['threshold'])|~good]=-1
    return pred


def bootstrap(values,seed,repeats):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    draws=rng.choice(values,(repeats,len(values)),replace=True).mean(1)
    return {'mean':float(values.mean()),'subject_bootstrap_95_percentile':np.quantile(draws,[.025,.975]).tolist(),
            'subjects':len(values)}


def assert_partitions(corpus,train,val,test,personal):
    if any(np.any(a&b) for a,b in ((train,val),(train,test),(val,test))):
        raise ValueError('Window leakage between partitions')
    if not all(np.any(a) for a in (train,val,test)):
        raise ValueError('Empty partition')
    if not personal:
        groups=[set(corpus.subjects[a]) for a in (train,val,test)]
        if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
            raise ValueError('Subject leakage')
    if len(set(corpus.ids))!=len(corpus.ids):
        raise ValueError('Duplicate source windows')


def experiment(corpus,protocol,train,val,test,task,mode,subject=None):
    labels=protocol['tasks'][task]
    train=train & np.isin(corpus.y,labels); val=val & np.isin(corpus.y,labels); test=test & np.isin(corpus.y,labels)
    assert_partitions(corpus,train,val,test,mode=='personal')
    xtr,ytr=corpus.take(train); xv,yv=corpus.take(val); xt,yt=corpus.take(test)
    good_tr,diag_tr=diagnostics(xtr,protocol['quality'])
    good_v,diag_v=diagnostics(xv,protocol['quality'])
    good_t,diag_t=diagnostics(xt,protocol['quality'])
    if set(ytr[good_tr])!=set(labels) or set(yv)!=set(labels) or set(yt)!=set(labels):
        raise ValueError('Missing class in partition after training quality gate')
    fitted={}; results={}; val_scores={}
    # All selection is finished before any test prediction is computed.
    for name in protocol['models']:
        model=make_model(name,corpus.sfreq,len(labels))
        started=time.perf_counter(); model.fit(xtr[good_tr],ytr[good_tr])
        p=model.predict_proba(xv)
        if list(model.classes_)!=labels:
            raise ValueError('Unexpected class ordering')
        val_scores[name]=summary(yv,p,labels)
        fitted[name]=(model,p,time.perf_counter()-started)
    chosen=max(protocol['models'],key=lambda name:val_scores[name]['balanced_accuracy'])
    gate=choose_gate(yv,fitted[chosen][1],good_v,protocol['epoch_duration_seconds'],protocol) if task=='three_class' else None
    predictions=[]
    for name in protocol['models']:
        model,_,fit_seconds=fitted[name]
        started=time.perf_counter(); p=model.predict_proba(xt); wall=time.perf_counter()-started
        per_subject=[]
        for s in np.unique(corpus.subjects[test]):
            m=corpus.subjects[test]==s
            per_subject.append({'subject':int(s),**summary(yt[m],p[m],labels)})
        results[name]={'validation':val_scores[name],'test':summary(yt,p,labels),'per_subject':per_subject,
                       'fit_seconds':fit_seconds,'batch_ms_per_window':1000*wall/len(xt)}
        for i,idx in enumerate(np.flatnonzero(test)):
            predictions.append({'window_id':str(corpus.ids[idx]),'subject':int(corpus.subjects[idx]),
                                'run':int(corpus.runs[idx]),'truth':int(yt[i]),'model':name,
                                'probabilities':p[i].tolist(),'predicted':int(model.classes_[p[i].argmax()])})
    selected_model=fitted[chosen][0]
    selected_p=selected_model.predict_proba(xt)
    reliability=None
    if gate is not None:
        pred=apply_gate(selected_p,good_t,gate)
        frontier=[]
        for threshold in np.linspace(1/3,1,21):
            pp=selected_p.argmax(1); pp[(selected_p.max(1)<threshold)|~good_t]=-1
            frontier.append({'threshold':float(threshold),**metrics(yt,pp,protocol['epoch_duration_seconds'])})
        stress={}
        rng=np.random.default_rng(protocol['seed'])
        for kind in ('flat_channel','amplitude_x3','noise_30uv'):
            z=xt.copy()
            if kind=='flat_channel': z[:,0,:]=0
            if kind=='amplitude_x3': z*=3
            if kind=='noise_30uv': z+=rng.normal(0,30,z.shape)
            good,_=diagnostics(z,protocol['quality'])
            stress[kind]=metrics(yt,apply_gate(selected_model.predict_proba(z),good,gate),protocol['epoch_duration_seconds'])
        reliability={'gate':gate,'test':metrics(yt,pred,protocol['epoch_duration_seconds']),
                     'stress':stress,'test_frontier_descriptive_only':frontier}
    return {'mode':mode,'task':task,'subject':subject,'selected_on_validation':chosen,
            'partitions':{k:{'windows':int(m.sum()),'subjects':np.unique(corpus.subjects[m]).tolist(),
                              'runs':np.unique(corpus.runs[m]).tolist()} for k,m in (('train',train),('validation',val),('test',test))},
            'quality':{'train':diag_tr,'validation':diag_v,'test':diag_t},'models':results,
            'reliability':reliability},predictions


def run(root,protocol_path,output):
    import mne, sklearn, scipy, pyriemann
    from . import __version__
    mne.set_log_level('ERROR')
    protocol_bytes=Path(protocol_path).read_bytes(); protocol=json.loads(protocol_bytes)
    output=Path(output)
    if (output/'study.json').exists():
        raise ValueError('Study output exists; use a new directory to retain the audit trail')
    corpus=load_corpus(root,protocol['development_subjects']+protocol['new_evaluation_subjects'],protocol)
    all_results=[]; all_predictions=[]
    with threadpool_limits(limits=1):
        for task in protocol['tasks']:
            r,p=experiment(corpus,protocol,np.isin(corpus.subjects,protocol['transfer_train_subjects']),
                           np.isin(corpus.subjects,protocol['transfer_validation_subjects']),
                           np.isin(corpus.subjects,protocol['new_evaluation_subjects']),task,'transfer')
            all_results.append(r);all_predictions.extend([{'mode':'transfer','task':task,**a} for a in p])
            print(f'Transfer {task}: selected {r["selected_on_validation"]}',flush=True)
            for subject in protocol['new_evaluation_subjects']:
                mask=corpus.subjects==subject
                r,p=experiment(corpus,protocol,mask&(corpus.runs==protocol['personal_train_run']),
                               mask&(corpus.runs==protocol['personal_validation_run']),
                               mask&(corpus.runs==protocol['personal_test_run']),task,'personal',subject)
                all_results.append(r);all_predictions.extend([{'mode':'personal','task':task,**a} for a in p])
                print(f'Personal {subject} {task}: selected {r["selected_on_validation"]}',flush=True)
    aggregates={}
    for task in protocol['tasks']:
        for mode in ('transfer','personal'):
            rows=[r for r in all_results if r['task']==task and r['mode']==mode]
            values=[v['balanced_accuracy'] for r in rows for v in r['models'][r['selected_on_validation']]['per_subject']]
            aggregates[f'{mode}/{task}']=bootstrap(values,protocol['seed'],protocol['bootstrap_replicates'])
    report={'version':__version__,'protocol':protocol,'protocol_sha256':hashlib.sha256(protocol_bytes).hexdigest(),
            'data_manifest':corpus.manifest,'n_windows':len(corpus.y),'aggregate_selected_models':aggregates,
            'experiments':all_results,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                        'sklearn':sklearn.__version__,'mne':mne.__version__,'pyriemann':pyriemann.__version__},
            'limits':['Offline, within-window zero-phase filtering; not causal deployment.',
                      'Six new subjects only; subject bootstrap intervals are imprecise.',
                      'Binary imagery excludes idle; it cannot establish autonomous communication.',
                      'Personal runs are from the same day; no long-term stability evidence.',
                      'Validation-selected models; all candidate test results disclosed, no post-test winner selection.',
                      'Raw neural signals are not included in exported reports.']}
    output.mkdir(parents=True,exist_ok=True)
    (output/'study.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'predictions.json').write_text(json.dumps(all_predictions,allow_nan=False)+'\n')
    print(json.dumps(aggregates,indent=2))
    return report
