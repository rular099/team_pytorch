#!/usr/bin/env python3
"""Build reproducible scientific assets for the native September report.

No image-model output is used. Trusted user-supplied NPZ exports are checked
against the formal report; old scatter metrics are checked against the July CSV.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

DECK = Path(__file__).resolve().parent
ROOT = DECK.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DECK))
os.environ.setdefault('MPLCONFIGDIR', str(DECK / '.matplotlib_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr
from prepare_sample_scatter import load_export, sha256, contour_thresholds
from tools.analyze_rt59_dual_objective_npz import _stack, _target_matrix, _point_metrics, _mixture_stats

REPORT = ROOT / 'reports/rt61_wave_geometry_validation_20260928'
FIG = DECK / 'assets/figures'
TAB = DECK / 'assets/tables'
BLUE, TEAL, CORAL, GRAY = '#5B7FA3', '#55A6A0', '#D66F63', '#647381'
REGULAR = Path('/home/zhangb/.fonts/noto/NotoSansCJKsc-Regular.otf')
FONT_FAMILY = 'Noto Sans CJK SC'
for font in (REGULAR, REGULAR.with_name('NotoSansCJKsc-Bold.otf')):
    if font.exists():
        font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family': FONT_FAMILY, 'axes.unicode_minus': False,
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'figure.facecolor': 'white',
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': '#A3B3BF'})


def write_csv(name, rows):
    if not rows:
        return
    with (TAB / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(FIG / (name + '.' + extension), dpi=240, facecolor='white')
    plt.close(fig)


def scatter(ax, data, mask, title, extent=(-2.1, 1.1), alpha=.08, color=BLUE):
    y, p = data['truth'][mask], data['prediction'][mask]
    met = _point_metrics(data['truth'], data['prediction'], mask)
    if not len(y):
        raise ValueError('No valid points: ' + title)
    order = np.random.default_rng(20260930).permutation(len(y))
    ax.scatter(y[order], p[order], s=5, alpha=alpha, c=color, linewidths=0, rasterized=True)
    hist, xe, ye = np.histogram2d(y, p, bins=95, range=(extent, extent))
    smooth = gaussian_filter(hist, 1.3)
    ax.contour((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2, smooth.T,
               levels=contour_thresholds(smooth), colors=[color], linewidths=.75, alpha=.8)
    ax.plot(extent, extent, '--', c=GRAY, lw=1.2)
    ax.plot(extent, np.asarray(extent)*met['slope']+met['intercept'], c=CORAL, lw=1.4)
    ax.set(xlim=extent, ylim=extent, aspect='equal', xlabel='真实 PGA [log10(m/s²)]',
           ylabel='预测 PGA [log10(m/s²)]')
    ax.set_title(title, loc='left', weight='bold', color='#303C49', pad=10)
    ax.text(.04, .96, f"n={len(y):,}\nMAE={met['mae']:.4f}  R²={met['r2']:.4f}\nslope={met['slope']:.4f}",
        transform=ax.transAxes, va='top', fontsize=12.4,
        bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.92, 'pad':4})
    ax.grid(color='#E7EDF1', lw=.6)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(True)
    return met


def pr_statistics(labels, scores):
    """Tie-aware PR-AUC (trapezoid) and AP, independent of sklearn."""
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return {'positive_targets': positives, 'targets':len(labels), 'f1_at_alpha_0p5':None,
                'pr_auc_trapezoid':None, 'average_precision':None}
    order = np.argsort(-scores, kind='stable')
    yy, ss = labels[order], scores[order]
    end = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp = np.cumsum(yy)[end].astype(float)
    precision = tp / (end+1)
    recall = tp / positives
    recall, precision = np.r_[0.,recall], np.r_[1.,precision]
    area = float(np.trapz(precision, recall))
    ap = float(np.dot(np.diff(recall), precision[1:]))
    selected = scores >= .5
    true_positive = int((labels & selected).sum())
    denominator = int(labels.sum()+selected.sum())
    return {'positive_targets': positives, 'targets':len(labels),
            'f1_at_alpha_0p5':2*true_positive/denominator if denominator else 0.,
            'pr_auc_trapezoid':area, 'average_precision':ap}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-npz', type=Path, required=True)
    parser.add_argument('--normal-npz', type=Path, required=True)
    parser.add_argument('--old-npz', type=Path, required=True)
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    s = json.loads((REPORT/'summary.json').read_text())
    data = {}
    for name, path in [('random',args.random_npz),('normal',args.normal_npz)]:
        d = load_export(path, s['sources'][name+'_npz']['sha256'])
        with np.load(path, allow_pickle=True) as z:
            d['station_count'] = _stack(z,'val_station_valid_count').reshape(-1)
            station_mask = _stack(z,'val_station_valid').astype(bool).reshape(len(d['station_count']),-1)
            assert np.array_equal(d['station_count'], station_mask.sum(1))
            d['time'] = _stack(z,'val_realtime_requested_elapsed_time').reshape(-1)
            d['magnitude'] = _stack(z,'val_mag_label').reshape(-1)
            d['target_type'] = _stack(z,'val_realtime_target_type').reshape(d['truth'].shape)
            mdn = _stack(z,'val_pga_pred').astype(float)
            norm = s['sources'][name+'_metrics']['normalization']
            mdn[...,1] = mdn[...,1]*norm['std']+norm['mean']
            mdn[...,2] *= norm['std']
            d['weights'],d['means'],d['sigmas'], mean,_ = _mixture_stats(mdn)
            if not np.allclose(mean[d['valid']],d['prediction'][d['valid']],atol=2e-6,rtol=0):
                raise ValueError('MDN mean / export mismatch')
        data[name] = d
    # The old NPZ is the asset explicitly referenced by July Slide 11.
    with np.load(args.old_npz, allow_pickle=True) as z:
        old = {k:_target_matrix(_stack(z,key)) for k,key in
               [('truth','val_pga_label'),('prediction','val_pga_mu_best')]}
        old['valid'] = _stack(z,'val_pga_target_valid').astype(bool).reshape(old['truth'].shape)
    old_metrics = _point_metrics(old['truth'],old['prediction'],old['valid'])
    assert old['truth'].shape[0] == 101 and old_metrics['targets'] == 1393
    assert round(old_metrics['mae'],4) == .2966 and round(old_metrics['r2'],4) == .5221
    assert round(old_metrics['rmse'],4) == .3875 and round(old_metrics['slope'],4) == .5931
    comparison = [('2026-07 旧版 · validation',old,old['valid'],.30,GRAY),
        ('RT61 · normal all',data['normal'],data['normal']['valid'],.08,BLUE),
        ('RT61 · normal non-input',data['normal'],data['normal']['valid'] & ~data['normal']['input'],.08,TEAL)]
    fig,axs = plt.subplots(1,3,figsize=(15.5,5.3))
    fig.subplots_adjust(left=.055,right=.99,bottom=.16,top=.89,wspace=.24)
    for ax,(label,d,mask,alpha,c) in zip(axs,comparison):
        scatter(ax,d,mask,label,(-3.,1.1),alpha,c)
    save(fig,'old_vs_rt61_scatter')
    write_csv('old_vs_rt61_metrics.csv',[dict(population=label,**_point_metrics(d['truth'],d['prediction'],mask))
        for label,d,mask,_,_ in comparison])

    # Separate panels by actual K and elapsed time. Every valid point retained.
    r = data['random']
    selections = [('K=1',r['station_count']==1),('K=2–5',(r['station_count']>=2)&(r['station_count']<=5)),
                  ('K≥6',r['station_count']>=6),('1–5 s',r['time']<=5),
                  ('10–20 s',(r['time']>=10)&(r['time']<=20)),('40–90 s',r['time']>=40)]
    fig,axs = plt.subplots(2,3,figsize=(14.5,8.5))
    fig.subplots_adjust(left=.065,right=.99,bottom=.085,top=.95,wspace=.23,hspace=.36)
    strata=[]
    for ax,(name,rowmask) in zip(axs.flat,selections):
        mask = r['valid'] & rowmask[:,None]
        met=scatter(ax,r,mask,name)
        met.update({'group':name,'events':len(np.unique(r['event_ids'][mask.any(1)]))})
        strata.append(met)
    save(fig,'rt61_stratified_scatter')
    write_csv('rt61_scatter_strata.csv',strata)

    # Formal paired comparisons, include CIs for differences (not individual estimates).
    groups=list(csv.DictReader((REPORT/'group_metrics.csv').open()))
    metrics={(v['population'],v['model']):v for v in groups}
    models=['historical_rt57','reference_rt59','candidate_rt61']
    colors=[GRAY,BLUE,TEAL]
    fig,axs=plt.subplots(1,3,figsize=(14,4.9))
    for ax,pop,label in zip(axs,['random','normal_all','normal_noninput'],['Random non-input','Normal all','Normal non-input']):
        vals=[float(metrics[pop,m]['mae']) for m in models]
        ax.plot(range(3),vals,c='#B6C3CC',lw=2,zorder=1)
        for i,(v,c) in enumerate(zip(vals,colors)):
            ax.scatter(i,v,c=c,s=120,zorder=2)
            ax.annotate(f'{v:.4f}',(i,v),xytext=(0,14),textcoords='offset points',ha='center',fontsize=15)
        delta=(vals[0]-vals[-1])/vals[0]*100
        ax.set_title(label+f'\nMAE 降低 {delta:.2f}%',weight='bold')
        ax.set(xticks=range(3),xticklabels=['RT57 γ=1.66','RT59','RT61'],ylabel='MAE [log10(m/s²)]')
        lo,hi=min(vals),max(vals)
        ax.set_ylim(lo-(hi-lo)*.55,hi+(hi-lo)*.85)
        ax.grid(axis='y',color='#E7EDF1')
    fig.tight_layout(pad=1.5)
    save(fig,'paired_mae_progress')

    # Error strata from raw targets. No artificial aggregation across empty bins.
    fig,axs=plt.subplots(2,2,figsize=(13,7.6))
    all_strata=[]
    definitions=[('实际输入台站数',[('1',r['station_count']==1),('2–5',(r['station_count']>=2)&(r['station_count']<=5)),
          ('6–10',(r['station_count']>=6)&(r['station_count']<=10)),('≥11',r['station_count']>=11)],'row'),
       ('目录震级',[('M<4',r['magnitude']<4),('4–5',(r['magnitude']>=4)&(r['magnitude']<5)),
          ('5–6',(r['magnitude']>=5)&(r['magnitude']<6)),('6–7',(r['magnitude']>=6)&(r['magnitude']<7)),('≥7',r['magnitude']>=7)],'row'),
       ('真实 PGA 分桶（log10）',[('−2～−1.5',(r['truth']>=-2)&(r['truth']<-1.5)),
          ('−1.5～−1',(r['truth']>=-1.5)&(r['truth']<-1)),('−1～−0.5',(r['truth']>=-1)&(r['truth']<-.5)),('≥−0.5',r['truth']>=-.5)],'target'),
       ('目标是否已触发',[('已触发非输入',r['target_type']==1),('尚未触发',r['target_type']==2)],'target')]
    for ax,(title,definitions_,kind) in zip(axs.flat,definitions):
        values=[]
        for name,mask in definitions_:
            if kind=='row': mask=mask[:,None]
            mask=r['valid']&mask
            met=_point_metrics(r['truth'],r['prediction'],mask)
            met.update({'dimension':title,'group':name,'events':len(np.unique(r['event_ids'][mask.any(1)]))})
            all_strata.append(met)
            values.append(met)
        bars=ax.bar(range(len(values)),[v['mae'] for v in values],color=BLUE,width=.6)
        for bar,met in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,
                f"{met['mae']:.3f}\nn={met['targets']:,}",ha='center',fontsize=10.7)
        ax.set(title=title,xticks=range(len(values)),xticklabels=[d[0] for d in definitions_],
               ylabel='MAE [log10(m/s²)]',ylim=(0,.76))
        ax.grid(axis='y',color='#E7EDF1'); ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6,h_pad=2)
    save(fig,'rt61_error_strata')
    write_csv('rt61_error_strata.csv',all_strata)

    fields=s['random_fields']
    fig,axs=plt.subplots(1,3,figsize=(14.6,4.7))
    conditions=[('多目标空间范围比','all','range_ratio_mean',.55,True),
                ('单输入台站空间范围比','one_station','range_ratio_mean',.25,True),
                ('单台站 pairwise-delta MAE','one_station','pairwise_delta_mae',.33,False)]
    for ax,(title,pop,metric,threshold,higher) in zip(axs,conditions):
        vals=[fields[pop][m][metric] for m in ['historical','reference','candidate']]
        ax.bar(range(3),vals,color=colors,width=.58)
        ax.axhline(threshold,c=CORAL,ls='--',lw=1.5,label=f'门槛 {threshold}')
        for i,v in enumerate(vals):ax.text(i,v+.007,f'{v:.4f}',ha='center',fontsize=13)
        ax.set(title=title+'\n'+('越高越好' if higher else '越低越好'),xticks=range(3),
               xticklabels=['RT57 γ','RT59','RT61'],ylim=(0,max(threshold,max(vals))*1.22))
        ax.legend(frameon=False,fontsize=11,loc='upper right')
        ax.grid(axis='y',color='#E7EDF1');ax.set_axisbelow(True)
    fig.tight_layout(pad=1.5)
    save(fig,'field_gate_progress')

    # Moment coverage: Gaussian reference lines are benchmarks, not exact MDN interval claims.
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    for ax,col,title,reference in zip(axs,['coverage1','coverage2'],['均值±1σ 的经验覆盖率','均值±2σ 的经验覆盖率'],[.6827,.9545]):
        for offset,(model,color,label) in enumerate(zip(models,colors,['RT57 γ','RT59','RT61'])):
            ax.bar(np.arange(4)+(offset-1)*.23,[float(metrics[p,model][col]) for p in
                ['random','normal_all','normal_input','normal_noninput']],width=.22,color=color,label=label)
        ax.axhline(reference,c=CORAL,ls='--',lw=1.2,label='高斯参考')
        ax.set(title=title,xticks=range(4),xticklabels=['Random','Normal all','Input','Non-input'],ylim=(.60,1.01))
        ax.grid(axis='y',color='#E7EDF1');ax.set_axisbelow(True)
    axs[0].legend(ncol=2,frameon=False,fontsize=10)
    fig.tight_layout(pad=1.4)
    save(fig,'coverage_comparison')

    threshold_rows=[]
    for protocol,d in data.items():
        populations=['random'] if protocol=='random' else ['normal_all','normal_noninput']
        for seconds in (1,5,20):
            for pop in populations:
                mask=d['valid'] & (d['time']==seconds)[:,None]
                if pop=='normal_noninput':mask &= ~d['input']
                for percent in (1,2,5):
                    cut=math.log10(percent/100*9.80665)
                    prob=(d['weights']*ndtr((d['means']-cut)/d['sigmas'])).sum(-1)
                    row=dict(population=pop,requested_time_seconds=seconds,pga_percent_g=percent,
                        threshold_log10_mps2=cut,events=len(np.unique(d['event_ids'][mask.any(1)])),
                        **pr_statistics(d['truth'][mask]>=cut,prob[mask]))
                    threshold_rows.append(row)
    write_csv('rt61_threshold_metrics.csv',threshold_rows)
    # Residual standard deviation, not RMSE: bias removed before ln(10) conversion.
    literature=[]
    for pop in ('random','normal_all','normal_input','normal_noninput'):
        v=metrics[pop,'candidate_rt61']
        sigma=math.sqrt(float(v['rmse'])**2-float(v['bias'])**2)*math.log(10)
        literature.append(dict(population=pop,targets=int(v['targets']),r2=float(v['r2']),
                               residual_std_ln=sigma,coordinate_conversion='std(error_log10) * ln(10)'))
    write_csv('rt61_literature_residual_metrics.csv',literature)
    provenance={'repo':'rular099/team_pytorch','branch':'exp/v01-velocity-prep-padding-control',
        'base_commit':'8f54477ed11bf2623f40121648f7fcc756ada1a2', 'date':'2026-09-30',
        'backend':'matplotlib scientific figures; native python-pptx slide composition',
        'old_npz':{'sha256':sha256(args.old_npz),'original_july_slide':11,'rows':101,**old_metrics},
        'rt61_npz_sha256':{k:v['sha256'] for k,v in data.items()},
        'checks':{'all_points_no_subsampling':True,'old_metrics_match_july':True,
                  'station_mask_counts_match':True,'mdn_mean_matches_export':True,
                  'common_old_new_axis_limits':[-3.,1.1],'source_axes_equal_aspect':True},
        'old_new_comparison':'historical context only: unmatched datasets, times and target protocols',
        'threshold_protocol':{'g_mps2':9.80665,'time':'requested elapsed seconds',
            'probability':'MDN mixture exceedance probability, denormalized component means/sigmas',
            'decision_cutoff':.5,'cutoff_optimization':False,'pr_auc':'tie-aware trapezoidal PR area',
            'ap':'separately exported, not substituted for PR-AUC','split':'validation',
            'no_warning_time_evaluation':True},
        'script_sha256':sha256(__file__),
        'figure_sha256':{p.name:sha256(p) for p in FIG.glob('*.png')}}
    (TAB/'report_asset_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':'PASS','old':old_metrics,'scatter_strata':strata,
        'threshold_20s':[v for v in threshold_rows if v['requested_time_seconds']==20],
        'residual_ln':literature},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
