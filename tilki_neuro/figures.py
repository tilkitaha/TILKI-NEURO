"""Scientific plots directly from stored study results; no hand-entered scores."""
import json
from pathlib import Path
import numpy as np


def figure(report):
    from matplotlib.figure import Figure
    fig=Figure(figsize=(13,8.5),facecolor='#0b1220',layout='constrained')
    axes=fig.subplots(2,2)
    colors=['#69e2c7','#71a7ff']
    for ax in axes.flat:
        ax.set_facecolor('#101f33');ax.tick_params(colors='#dde8f7',labelsize=10)
        for s in ax.spines.values():s.set_color('#37475b')
        ax.title.set_color('#f0f6ff');ax.xaxis.label.set_color('#dde8f7');ax.yaxis.label.set_color('#dde8f7')
    names={'three_class':'Idle + left + right','binary_imagery':'Left vs right (idle excluded)'}
    for col,task in enumerate(('three_class','binary_imagery')):
        ax=axes[0,col]
        for i,mode in enumerate(('transfer','personal')):
            a=report['aggregate_selected_models'][f'{mode}/{task}']
            mean=a['mean']*100;lo,hi=np.array(a['subject_bootstrap_95_percentile'])*100
            ax.bar(i,mean,color=colors[i],width=.5)
            ax.errorbar(i,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='none',ecolor='white',capsize=5)
            ax.text(i,hi+3,f'{mean:.1f}%',color='white',ha='center',fontweight='bold')
        ax.axhline(100/(3 if task=='three_class' else 2),color='#ffbe78',linestyle='--',label='Chance')
        ax.set(xticks=[0,1],xticklabels=['New person\nno personal calibration','Personal calibration\nheld-out run'],ylim=(0,105),ylabel='Subject-mean balanced accuracy (%)',title=names[task])
        ax.legend(facecolor='#101f33',labelcolor='white',loc='lower right',fontsize=9)
    ax=axes[1,0]
    subjects=report['protocol']['new_evaluation_subjects']
    for i,task in enumerate(('three_class','binary_imagery')):
        rows=[r for r in report['experiments'] if r['mode']=='personal' and r['task']==task]
        vals=[r['models'][r['selected_on_validation']]['test']['balanced_accuracy']*100 for r in rows]
        ax.plot(subjects,vals,'o-',color=colors[i],label=names[task])
    ax.set(ylim=(0,105),xticks=subjects,xlabel='Subject',ylabel='Balanced accuracy (%)',title='Every new subject is shown')
    ax.legend(facecolor='#101f33',labelcolor='white',fontsize=9)
    ax=axes[1,1]
    rows=[r for r in report['experiments'] if r['task']=='three_class' and r['mode']=='personal']
    coverage=[r['reliability']['test']['coverage']*100 for r in rows]
    recall=[r['reliability']['test']['correct_command_recall']*100 for r in rows]
    pos=np.arange(len(rows))
    ax.bar(pos-.17,coverage,width=.34,color=colors[0],label='Accepted windows')
    ax.bar(pos+.17,recall,width=.34,color=colors[1],label='Correct command recall')
    ax.set(ylim=(0,105),xticks=pos,xticklabels=subjects,xlabel='Subject (disabled gate = zero output)',ylabel='Percent',title='Reliability includes the cost of rejection')
    ax.legend(facecolor='#101f33',labelcolor='white',fontsize=9)
    fig.suptitle('TILKI NEURO  /  v0.2 evidence dashboard\nValidation-selected models • Six new subjects • Offline EEG\nError bars: 95% subject-bootstrap intervals; transfer and personal test runs differ',color='white',fontsize=17)
    return fig


def export(path, output):
    import matplotlib
    matplotlib.use('Agg')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads(Path(path).read_text())
    f=figure(report)
    f.savefig(output/'evidence.svg',facecolor=f.get_facecolor())
    f.savefig(output/'evidence.png',dpi=150,facecolor=f.get_facecolor())
