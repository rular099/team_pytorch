#!/usr/bin/env python3
"""Previous-meeting versus current-meeting evidence, without any new model run."""
import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

DECK=Path(__file__).resolve().parent
ROOT=DECK.parents[2]
OLD=DECK.with_name('multistation_pga_exchange_20260930')
sys.path[:0]=[str(ROOT),str(OLD)]
os.environ.setdefault('MPLCONFIGDIR',str(DECK/'.matplotlib_cache'))
import numpy as np
from scipy.special import ndtr
from matplotlib import pyplot as plt
from prepare_sample_scatter import load_export,sha256
from prepare_report_assets import scatter,pr_statistics,BLUE,TEAL,CORAL,GRAY
from tools.analyze_rt59_dual_objective_npz import _stack,_target_matrix,_mixture_stats,_point_metrics,_field_metrics

FIG=DECK/'assets/figures';TAB=DECK/'assets/tables'


def write_csv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (TAB/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(FIG/(name+'.'+ext),dpi=240,facecolor='white')
    plt.close(fig)


def read_data(path,config=None,expected_sha=None):
    with np.load(path,allow_pickle=True) as z:
        d={k:_target_matrix(_stack(z,key)) for k,key in [('truth','val_pga_label'),('prediction','val_pga_mu_best')]}
        d['valid']=_stack(z,'val_pga_target_valid').astype(bool).reshape(d['truth'].shape)
        d['station_count']=_stack(z,'val_station_valid_count').reshape(-1).astype(int)
        assert np.array_equal(d['station_count'],_stack(z,'val_station_valid').astype(bool).sum(1))
        for field,key in [('time','val_realtime_requested_elapsed_time'),('event_ids','val_event_id')]:
            if key in z:d[field]=_stack(z,key).reshape(-1)
        if 'val_realtime_target_type' in z:
            d['target_type']=_stack(z,'val_realtime_target_type').reshape(d['truth'].shape)
            d['input']=d['target_type']==0
        if 'val_pga_sigma' in z:d['sigma']=_stack(z,'val_pga_sigma').astype(float).reshape(d['truth'].shape)
        if config:
            cfg=json.loads(Path(config).read_text())['training_params']['pga_target_normalization']
            mdn=_stack(z,'val_pga_pred').astype(float)
            mdn[...,1]=mdn[...,1]*cfg['std']+cfg['mean'];mdn[...,2]*=cfg['std']
            d['weights'],d['means'],d['sigmas'],mean,sigma=_mixture_stats(mdn)
            assert np.allclose(mean[d['valid']],d['prediction'][d['valid']],atol=1e-6,rtol=0)
            d['sigma_moment']=sigma
        if 'val_pga_prob_ge_threshold' in z:
            d['formal_probability']=_stack(z,'val_pga_prob_ge_threshold').astype(float).reshape(d['truth'].shape)
    d['sha256']=sha256(path)
    if expected_sha:assert d['sha256']==expected_sha
    assert np.isfinite(d['truth'][d['valid']]).all() and np.isfinite(d['prediction'][d['valid']]).all()
    return d


def metrics(d,mask):
    r=_point_metrics(d['truth'],d['prediction'],mask)
    r['realtime_rows']=int(mask.any(1).sum())
    r['events']=len(np.unique(d['event_ids'][mask.any(1)])) if 'event_ids' in d else None
    if r['targets']==0:return r
    r['residual_std_ln']=math.sqrt(max(r['rmse']**2-r['bias']**2,0))*math.log(10)
    if 'sigma' in d:
        error=np.abs(d['prediction'][mask]-d['truth'][mask])
        r['coverage1']=float(np.mean(error<=d['sigma'][mask]))
        r['coverage2']=float(np.mean(error<=2*d['sigma'][mask]))
    if 'formal_probability' in d:
        r['brier']=float(np.mean((d['formal_probability'][mask]-(d['truth'][mask]>=-1.2))**2))
    return r


def best_f1(labels,scores):
    labels=np.asarray(labels,bool);scores=np.asarray(scores,float)
    if not labels.any() or labels.all():return None,None
    order=np.argsort(-scores,kind='stable');yy=labels[order];ss=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp=np.cumsum(yy)[ends]
    f1=2*tp/(labels.sum()+ends+1)
    k=int(np.argmax(f1));return float(f1[k]),float(ss[ends[k]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['random-npz','normal-npz','normal-config','rt55-npz','rt55-config','rt55-metrics','old-full-npz','old-knet-npz','old-threshold-npz']:
        p.add_argument('--'+field,required=True,type=Path)
    a=p.parse_args()
    FIG.mkdir(parents=True,exist_ok=True);TAB.mkdir(parents=True,exist_ok=True)
    for folder in ['figures','tables']:
        for path in (OLD/'assets'/folder).iterdir():
            if path.is_file():shutil.copy2(path,DECK/'assets'/folder/path.name)
    s=json.loads((ROOT/'reports/rt61_wave_geometry_validation_20260928/summary.json').read_text())
    data={
        'old_full':read_data(a.old_full_npz),
        'old_knet':read_data(a.old_knet_npz),
        'old_threshold':read_data(a.old_threshold_npz),
        'rt55_normal':read_data(a.rt55_npz,a.rt55_config),
        'rt61_normal':read_data(a.normal_npz,a.normal_config,s['sources']['normal_npz']['sha256']),
        'rt61_random':read_data(a.random_npz,a.normal_config,s['sources']['random_npz']['sha256'])}
    meta=json.loads(a.rt55_metrics.read_text())
    assert meta['checkpoint_metadata']['epoch']==20 and meta['splits']==['val']
    assert meta['waveform_station_permutation']=='none'
    formal=meta['metrics']['val']['pga']
    groups={}
    for key,d in data.items():
        if key=='old_threshold':continue
        for pop in (['all','input','noninput'] if key in ['rt55_normal','rt61_normal'] else ['all']):
            mask=d['valid'].copy()
            if pop=='input':mask&=d['input']
            if pop=='noninput':mask&=~d['input']
            groups[key+'_'+pop]=metrics(d,mask)
    # Historical full-data and best normal MAE representatives are independently checked.
    assert round(groups['old_full_all']['mae'],4)==.2966
    assert round(groups['old_knet_all']['mae'],4)==.2877
    for k in ['targets','mae','rmse','r2','slope','brier']:
        assert np.isclose(groups['rt55_normal_all'][k],formal[k],atol=2e-9,rtol=1e-8),k
    write_csv('meeting_comparison_metrics.csv',[dict(population=k,**v) for k,v in groups.items()])

    # Main progress graphic is meeting-to-meeting, NOT RT57-to-RT61.
    series=[('上次 full-data\nvalidation','old_full_all',GRAY),('上次最佳 KNET\nrt48 validation','old_knet_all',GRAY),
            ('本次 normal\nRT55 ep20','rt55_normal_all',TEAL),('本次最新 normal\nRT61 ep8','rt61_normal_all',BLUE),
            ('本次新增 random\nRT61 ep8','rt61_random_all',CORAL)]
    fig,axs=plt.subplots(1,3,figsize=(15.4,5.2))
    for ax,key,title in zip(axs,['mae','rmse','r2'],['MAE ↓','RMSE ↓','R² ↑']):
        vals=[groups[k][key] for _,k,_ in series]
        bars=ax.bar(range(5),vals,color=[c for _,_,c in series],width=.65)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.015,f'{v:.4f}',ha='center',fontsize=12)
        ax.set(title=title,xticks=range(5),xticklabels=[label for label,_,_ in series],ylim=(0,max(vals)*1.23),ylabel='log10(m/s²)' if key!='r2' else '决定系数')
        ax.tick_params(axis='x',labelsize=10);ax.grid(axis='y',color='#E7EDF1');ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6);save(fig,'meeting_point_progress')

    fig,axs=plt.subplots(1,4,figsize=(16,4.65))
    panels=[('上次 full-data',data['old_full'],data['old_full']['valid'],.30,GRAY),
            ('本次 normal · all',data['rt55_normal'],data['rt55_normal']['valid'],.08,TEAL),
            ('本次 normal · non-input',data['rt55_normal'],data['rt55_normal']['valid']&~data['rt55_normal']['input'],.08,BLUE),
            ('本次新增 random',data['rt61_random'],data['rt61_random']['valid'],.08,CORAL)]
    for ax,(title,d,mask,alpha,color) in zip(axs,panels):scatter(ax,d,mask,title,(-3,1.1),alpha,color)
    fig.subplots_adjust(left=.045,right=.99,bottom=.18,top=.88,wspace=.30);save(fig,'meeting_scatter_comparison')

    # Same latest checkpoint as the random stratified slide; all-target and remote stats separate.
    n=data['rt61_normal'];strata=[]
    bins=[('K=1',n['station_count']==1),('K=2–5',(n['station_count']>=2)&(n['station_count']<=5)),('K≥6',n['station_count']>=6),
          ('1–5 s',n['time']<=5),('10–20 s',(n['time']>=10)&(n['time']<=20)),('40–90 s',n['time']>=40)]
    fig,axs=plt.subplots(2,3,figsize=(14.5,8.5));fig.subplots_adjust(left=.065,right=.99,bottom=.085,top=.95,wspace=.23,hspace=.36)
    for ax,(name,rowmask) in zip(axs.flat,bins):
        mask=n['valid']&rowmask[:,None];scatter(ax,n,mask,name,color=TEAL)
        for pop,m in [('all',mask),('noninput',mask&~n['input'])]:strata.append(dict(group=name,population=pop,**metrics(n,m)))
    save(fig,'normal_geometry_stratified_scatter');write_csv('normal_geometry_strata.csv',strata)

    # Compute the same field statistic from actual historical/current predictions.
    field_rows=[]
    for key in ['old_full','old_knet','rt55_normal','rt61_normal','rt61_random']:
        d=data[key]
        for pop in (['all','noninput'] if 'input' in d and key!='rt61_random' else ['all']):
            mask=d['valid'] if pop=='all' else d['valid']&~d['input']
            field_rows.append(dict(population=key+'_'+pop,**_field_metrics(d['truth'],d['prediction'],mask,d['station_count'])))
    write_csv('meeting_field_metrics.csv',field_rows)
    field_map={r['population']:r for r in field_rows}
    selected=['old_full_all','rt55_normal_all','rt55_normal_noninput','rt61_random_all']
    labels=['上次 full-data','本次 normal all','本次 normal non-input','本次 random']
    fig,ax=plt.subplots(figsize=(10.7,4.6));vals=[field_map[k]['range_ratio_mean'] for k in selected]
    ax.bar(range(4),vals,color=[GRAY,TEAL,BLUE,CORAL],width=.6)
    ax.axhline(1,c='#677683',ls='--',label='理想范围比 1（不是充分条件）')
    for i,(v,k) in enumerate(zip(vals,selected)):
        ax.text(i,v+.04,f"{v:.3f}\nn={field_map[k]['fields_ge5']:,} fields",ha='center',fontsize=13)
    ax.set(xticks=range(4),xticklabels=labels,ylabel='预测 P95−P05 / 真实 P95−P05',ylim=(0,max(1,max(vals))*1.32))
    ax.grid(axis='y',color='#E7EDF1');ax.set_axisbelow(True);ax.legend(frameon=False);fig.tight_layout();save(fig,'meeting_spatial_range')

    # Old meeting calibration was the KNET rt44–51 family. Show an actual old candidate,
    # not an RT57 reference, and explicitly distinguish data populations.
    selected=['old_knet_all','rt55_normal_all','rt61_normal_all','rt61_random_all']
    labels=['上次 rt48 KNET','本次 RT55 normal','本次 RT61 normal','本次 RT61 random']
    fig,axs=plt.subplots(1,2,figsize=(11.4,4.5))
    for ax,key,title,ref in zip(axs,['coverage1','coverage2'],['均值±1σ 经验覆盖率','均值±2σ 经验覆盖率'],[.6827,.9545]):
        vals=[groups[k][key] for k in selected]
        ax.bar(range(4),vals,color=[GRAY,TEAL,BLUE,CORAL],width=.65)
        for i,v in enumerate(vals):ax.text(i,v+.022,f'{v:.1%}',ha='center',fontsize=13)
        ax.axhline(ref,c=GRAY,ls='--',label='高斯参考')
        ax.set(title=title,xticks=range(4),xticklabels=labels,ylim=(0,1.10));ax.tick_params(axis='x',labelsize=10)
        ax.grid(axis='y',color='#E7EDF1');ax.set_axisbelow(True)
    axs[0].legend(frameon=False);fig.tight_layout();save(fig,'meeting_calibration_progress')

    thresholds=[]
    for key in ['old_threshold','rt55_normal','rt61_normal','rt61_random']:
        d=data[key]
        for pop in (['all','noninput'] if 'input' in d and key not in ['old_threshold','rt61_random'] else ['all']):
            mask=d['valid']&(d['time']==20)[:,None]
            if pop=='noninput':mask&=~d['input']
            for percent in [1,2,5]:
                cutoff=math.log10(percent/100*9.80665);labels=d['truth'][mask]>=cutoff
                point_stats=pr_statistics(labels,d['prediction'][mask]);star,point_cut=best_f1(labels,d['prediction'][mask])
                row=dict(model=key,population=pop,time_seconds=20,pga_percent_g=percent,
                         targets=int(mask.sum()),positive_targets=int(labels.sum()),
                         point_f1_star=star,point_f1_star_cutoff=point_cut,
                         point_pr_auc=point_stats['pr_auc_trapezoid'],point_ap=point_stats['average_precision'])
                if 'weights' in d:
                    probs=(d['weights']*ndtr((d['means']-cutoff)/d['sigmas'])).sum(-1)[mask]
                    stats=pr_statistics(labels,probs)
                    row.update(probability_f1_fixed_0p5=stats['f1_at_alpha_0p5'],probability_pr_auc=stats['pr_auc_trapezoid'])
                thresholds.append(row)
    old_threshold=[r for r in thresholds if r['model']=='old_threshold']
    for r,f1,auc in zip(old_threshold,[.620,.545,.438],[.642,.536,.372]):
        assert round(r['point_f1_star'],3)==f1 and round(r['point_pr_auc'],3)==auc
    write_csv('meeting_threshold_comparison.csv',thresholds)
    info={'groups':groups,'fields':field_map,'thresholds':thresholds,
          'normal_representative':'RT55 ep20 validation, lowest overall and remote normal MAE among the reviewed formal candidates; not a new test-based checkpoint selection',
          'normal_latest':'RT61 ep8 validation; normal layer split explicitly shown',
          'historical_point_baseline':'July Slide11 full-data validation; July Slide26 RT48 KNET diagnostic also retained',
          'historical_calibration_baseline':'actual July RT48 KNET validation export, not the later RT57 gamma reference',
          'threshold_policy':'20s point-score F1* is retrospective validation-oracle optimum to reproduce July; never deploy this cutoff or claim independent test performance',
          'historical_change_is_nonpaired':True,
          'source_sha256':{k:d['sha256'] for k,d in data.items()},
          'rt55_metadata_sha256':sha256(a.rt55_metrics),'script_sha256':sha256(__file__)}
    (TAB/'revision_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':'PASS','groups':groups,'fields':field_map,'thresholds':thresholds},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
