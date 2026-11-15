"""Static scientific figures and numerical reading tables, without model-ranking prose."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES=['DiTing','PhaseNet','EQT','TEAM']
COLORS=['#32689b','#d67b2d','#4b8a57','#9870a2']


def markdown(frame):
    def item(v):
        if pd.isna(v): return 'NA'
        if isinstance(v,(float,np.floating)): return f'{v:.6f}'
        return str(v).replace('|','/')
    rows=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(item(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)


def main(review):
    r=Path(review);figdir=r/'figures';figdir.mkdir(exist_ok=False)
    sources={}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name,inputs):
        fig.savefig(figdir/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(figdir/(name+'.svg'),bbox_inches='tight')
        svg=figdir/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        sources[name]=inputs
        plt.close(fig)
    s=pd.read_csv(r/'derived/ttum_support_recomputed.csv')
    central=s[s.budget_dex.eq(.2)&s.gate.eq('field_pass')&s.horizon.eq(20)]
    scenes=['ideal_cutoff_replay','measured_compute_replay','measured_rendered_map_replay']
    fig,axs=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for ax,panel in zip(axs,['fixed_all_targets','fixed_never_input_targets']):
        f=central[central.geometry_protocol.eq('normal')&central.panel.eq(panel)]
        for j,scene in enumerate(scenes):
            part=f[f.scene.eq(scene)].set_index('family').reindex(FAMILIES)
            x=np.arange(4)+(j-1)*.23
            ax.bar(x,100*part.fraction_lower,.22,color=COLORS[j],label=['Cutoff clock','Numeric FIFO clock','Numeric + shared render'][j])
            ax.errorbar(x,100*part.fraction_lower,yerr=np.vstack([np.zeros(4),100*(part.fraction_upper-part.fraction_lower)]),fmt='none',ecolor='#333333',capsize=2,lw=.7)
        ax.set_xticks(range(4),FAMILIES)
        ax.set_title(panel+'\nnormal; eligible n='+str(int(f.events.iloc[0]))+' / cohort 1310')
        ax.set_ylim(0,100);ax.set_ylabel('Confirmed by first-P + 20 s (%)')
    axs[0].legend(fontsize=8,loc='upper right')
    fig.suptitle('Validation / seed42 / selected epochs12,12,10,11; field gate, budget 0.20 dex')
    fig.text(.5,-.025,'Bars: confirmed lower fraction. Whiskers: unknown-status bounds, not confidence intervals. Minimum 5 fixed targets.',ha='center',fontsize=8)
    fig.tight_layout()
    save(fig,'ttum_by_20s_b020',['derived/ttum_support_recomputed.csv; normal; field_pass; budget .2; horizon20; both panels'])
    latency=pd.read_csv(r/'derived/latency_recomputed.csv')
    fig,ax=plt.subplots(figsize=(8,3.6))
    for j,scope in enumerate(['station_only','whole_map']):
        f=latency[latency.scope.eq(scope)].set_index('family').reindex(FAMILIES)
        x=np.arange(4)+(j-.5)*.3
        ax.bar(x,f.p50_seconds,.28,color=COLORS[j],label=scope+' P50')
        ax.scatter(x,f.p95_seconds,color='black',marker='_',s=80,zorder=4,label='P95' if j==0 else None)
    ax.axhline(1,color='#999999',ls=':',lw=.8,label='1 s update interval')
    ax.set_yscale('log');ax.set_xticks(range(4),FAMILIES);ax.set_ylabel('Warm numeric service (s; log axis)')
    ax.legend(fontsize=8,ncol=2);ax.set_title('108 repeats/scope/model; 6 metadata-selected events; 36 conditions x 3 repeats')
    fig.tight_layout();save(fig,'warm_latency',['derived/latency_recomputed.csv; P50/P95 distributions, not CIs'])
    legacy=pd.read_csv(r/'source_tables/analyze/0/legacy_with_physical_covariates/cross_encoder_metrics.csv')
    primary=legacy[legacy.checkpoint_policy.eq('selected')&legacy.endpoint.eq('primary_10cell')&legacy.target_group.eq('noninput')].set_index('family').reindex(FAMILIES)
    fig,ax=plt.subplots(figsize=(6.6,3.8))
    for family,color in zip(FAMILIES,COLORS):
        row=primary.loc[family];ax.scatter(row.equal_cell_mae,100*row.equal_cell_covered95,color=color,s=48)
        ax.annotate(family,(row.equal_cell_mae,100*row.equal_cell_covered95),xytext=(5,3),textcoords='offset points')
    ax.axhline(95,color='#777777',ls='--',lw=.8,label='Nominal 95%')
    ax.set_ylim(90,96);ax.set_xlim(.219,.25)
    ax.set_xlabel('10-cell equal MAE [log10(m/s^2)]');ax.set_ylabel('10-cell equal 95% interval coverage (%)')
    ax.set_title('Legacy snapshot; validation / seed42 / selected\n1310 events, 139440 noninput requests; fixed 10-cell weighting')
    ax.legend(fontsize=8);fig.tight_layout();save(fig,'legacy_point_and_coverage',['source_tables/analyze/0/legacy_with_physical_covariates/cross_encoder_metrics.csv; selected, noninput, primary_10cell'])
    queue=pd.read_csv(r/'derived/queue_quantiles.csv')
    fig,ax=plt.subplots(figsize=(7.5,3.8))
    for j,metric in enumerate(['numeric_ready_seconds','rendered_ready_seconds']):
        f=queue[queue.cohort.eq('cutoff_20s')&queue.metric.eq(metric)].set_index('family').reindex(FAMILIES)
        y=np.arange(4)+(j-.5)*.18
        ax.scatter(f.p50,y,label=['Numerical readiness within render-inclusive queue','After shared rendering'][j],s=25,color=COLORS[j])
        ax.hlines(y,f.p50,f.p95,color=COLORS[j],lw=1.2)
        ax.scatter(f.p95,y,s=20,marker='|',color=COLORS[j])
    ax.axvline(20,color='#999999',ls=':',lw=.8,label='Data cutoff: first-P + 20 s')
    ax.set_yticks(range(4),FAMILIES);ax.set_xlabel('Delivery clock of the 20th frame, seconds after first P')
    ax.set_title('Normal full-map shared-comparison-render FIFO scenario\n1310 events/model; dot=P50, right tick=P95, not a CI or deployment measurement')
    ax.legend(fontsize=7,loc='center left',bbox_to_anchor=(1.02,.5));fig.tight_layout();save(fig,'queue_20th_frame',['derived/queue_quantiles.csv; cutoff_20s; render-inclusive shared FIFO scenario'])
    pairs=pd.read_csv(r/'derived/legacy_primary_pairwise_recheck.csv')
    view=[]
    for (geo,panel,family),g in central.groupby(['geometry_protocol','panel','family']):
        if panel not in ['fixed_all_targets','fixed_never_input_targets']:continue
        row=dict(geometry=geo,panel=panel,family=family,eligible_events=int(g.events.iloc[0]),cohort_events=1310)
        for scene,label in zip(scenes,['ideal_20s','numeric_20s','rendered_20s']):
            f=g[g.scene.eq(scene)]
            row[label]=(f"{int(f.achieved.iloc[0])}/{int(f.events.iloc[0])}; unknown={int(f.unknown.iloc[0])}" if len(f) else 'not measured')
        view.append(row)
    shadow=pd.read_csv(r/'derived/shadow_rendered_summary.csv')
    shadow=shadow[shadow.threshold_logpga.eq(-1.2)&shadow.p_alarm.eq(.5)&shadow.tau.eq(3)]
    text='''# FE01 ON V4 数值阅读表（不含 Codex 判断）

仓库 rular099/team_pytorch；整理基线99d9b1c4bb5c47fc26125df7eea2648917e90462。
全部为validation、seed42、ON；selected依次DiTing12/PhaseNet12/EQT10/TEAM11。
PGA坐标为log10(m/s^2)。原始归档SHA与复算范围见verification.json、local_numerical_checks.json。

旧snapshot主端点：1310事件、139440非输入目标请求；normal/random × 1/3/5/10/20秒的10 cell先各自目标等权，再cell等权。
下面仅列原表数值；这是legacy_snapshot_v1，不是新逐秒固定query人口。pooled R2/slope与equal-cell MAE权重不同，保留原列名。

'''
    cols=['family','epoch','events','requests','equal_cell_mae','equal_cell_rmse','equal_cell_nll','equal_cell_brier','equal_cell_predictive_sigma','equal_cell_covered68','equal_cell_covered95','pooled_r2','pooled_prediction_on_truth_slope']
    text+=markdown(primary.reset_index()[cols])+'\n\n'
    text+='对应的全目标主端点（194265请求）见同一源表的target_group=all；非输入和全目标须分别报告。\n\n'
    text+='主MAE配对差为DiTing减对照，单位dex。2000次固定模型事件cluster bootstrap、seed20261007，未校正多重比较。\n\n'+markdown(pairs)+'\n\n'
    text+='新固定query TTUM的预定预算0.20 dex、field_pass、20秒数值例子如下。全预算0.15/0.20/0.25、两个gate、三个panel及5/10/20秒均保留在完整CSV中。下列unknown为支持状态界限，不是置信区间。\n\n'+markdown(pd.DataFrame(view))+'\n\n'
    text+='warm数值时延：每模型/每scope六个元数据预选事件，normal/random × 1/10/20秒共36条件，每条件2 warmup、3 measured重复（108记录）。硬件记录同名不证明物理运行条件严格相同。单位秒。\n\n'
    text+=markdown(latency[['family','scope','repeats','events','p50_seconds','p95_seconds','maximum_seconds','cold_load_seconds','device_names']])+'\n\n'
    text+='下表是渲染交付shadow的一个预定网格例子：normal、PGA阈值−1.2 log10(m/s^2)、p_alarm=0.5、tau=3秒。所有阈值/概率/提前量组合仍保留。pooled为目标加权；paired原表为事件等权，二者不能混作同一估计量。\n\n'
    text+=markdown(shadow[['family','panel','events','events_with_affected_targets','affected_targets','timely','failures','unknown','pooled_failure_lower','pooled_failure_upper','complete_negative_targets','false_positives','pooled_false_positive_rate']])+'\n\n'
    text+='来源：source_tables/analyze/0/legacy_with_physical_covariates/；derived/ttum_support_recomputed.csv；derived/latency_recomputed.csv；derived/shadow_rendered_summary.csv。复算和估计量定义见build_review.py与DATA_DICTIONARY.md。\n'
    with (r/'KEY_RESULTS.md').open('x') as f:f.write(text)
    with (figdir/'FIGURE_SOURCES.json').open('x') as f:
        json.dump(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=sources),f,indent=2);f.write('\n')
    print('PASS: four PNG/SVG scientific figures and neutral numerical reading table')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',required=True)
    main(p.parse_args().review)
