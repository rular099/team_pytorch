"""Reproduce paper assets on CPU exclusively from archived, pinned result tables."""
import gzip
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/fe01_paper_matplotlib')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

from common import *

font_manager.fontManager.addfont('/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
    'pdf.fonttype':42,'savefig.dpi':300,'axes.unicode_minus':False})

E=Evidence()
D={k:E.eid(k) for k in ['E03','E04','E05','E06','E07','E08','E09','E10','E11','E12',
                            'E13','E14','E15','E16','E17','E18','E19','E30','E35']}
META=json.loads((PAPER/'evidence/complete_model_settings.json').read_text())
ENV=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,
         matplotlib=matplotlib.__version__,platform=platform.platform(),cpu_threads=1)
ASSETS=[]
CHECKS=[]

def check(name, condition, **details):
    CHECKS.append(dict(check=name,passed=bool(condition),**details))
    if not condition:raise AssertionError((name,details))

def select(frame,**keys):
    for k,v in keys.items():
        frame=frame[frame[k].isin(v) if isinstance(v,(list,tuple)) else frame[k].eq(v)]
    return frame.copy()

def source_file(asset,frame,ids,filters,unit,weighting,population,claims,uncertainty='none',notes=''):
    dest=PAPER/'assets/source_data'/f'{asset}.csv'
    frame.to_csv(dest,index=False,float_format='%.17g',lineterminator='\n')
    index=json.loads((PAPER/'evidence/blueprint_v2/05_EVIDENCE_INDEX.json').read_text())['evidence']
    sources=[]
    for eid in ids:
        row=next(v for v in index if v['id']==eid)
        if not any(v['full_commit']==row['commit'] and v['path']==row['path'] for v in E.records):
            E.eid(eid)
        rec=next(v for v in E.records if v['full_commit']==row['commit'] and v['path']==row['path'])
        sources.append(dict(evidence_id=eid,repository=rec['repository'],full_commit=rec['full_commit'],
            path=rec['path'],sha256=rec['actual_sha256'],rows_before=len(D[eid]) if eid in D else None))
    record=dict(asset_id=asset.split('_')[0],panel_id=asset,claim_ids=claims,sources=sources,
        filter=filters,rows_after=len(frame),cohort='validation; original event partition',
        population=population,policy='selected unless explicitly labeled epoch12',
        checkpoint_epochs=EPOCHS,seed=42,unit=unit,weighting=weighting,uncertainty_type=uncertainty,
        effective_replicates=sorted(frame.effective_replicates.dropna().unique().tolist()) if 'effective_replicates' in frame else None,
        unknown_policy='preserved; never replaced with success/zero; bounds are not confidence intervals',
        script='scripts/build_assets.py',script_sha256=sha(Path(__file__).read_bytes()),environment=ENV,
        source_data_path=str(dest.relative_to(PAPER)),source_data_sha256=sha(dest.read_bytes()),
        visual_check='PENDING',notes=notes,outputs=[])
    ASSETS.append(record)
    return frame

def savefig(fig,stem,record=None):
    for ext in ['svg','pdf','png']:
        path=PAPER/'assets'/f'{stem}.{ext}'
        fig.savefig(path,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
        if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        if record is not None:record['outputs'].append(dict(path=str(path.relative_to(PAPER)),sha256=sha(path.read_bytes())))
    plt.close(fig)

def group(fid,title,panels,ncols=2,height=2.65):
    rows=math.ceil(len(panels)/ncols)
    fig,axes=plt.subplots(rows,ncols,figsize=(7.35,rows*height+.5),squeeze=False,layout='constrained')
    for i,(name,fn) in enumerate(panels):
        ax=axes.flat[i];fn(ax);ax.set_title(f'({chr(97+i)}) {name}',loc='left',pad=8)
        one,a=plt.subplots(figsize=(3.65,height+.25),layout='constrained')
        fn(a);a.set_title(f'({chr(97+i)}) {name}',loc='left',pad=8)
        record=next(v for v in ASSETS if v['panel_id']==f'{fid}_{chr(97+i)}')
        savefig(one,f'panels/{fid}_{chr(97+i)}',record)
    for ax in list(axes.flat)[len(panels):]:ax.axis('off')
    fig.suptitle(f'图{fid[1:]}  {title}',fontsize=12)
    rec=dict(asset_id=fid,panel_id='assembled',source_panels=[f'{fid}_{chr(97+i)}' for i in range(len(panels))],
        visual_check='PENDING',script='scripts/build_assets.py',script_sha256=sha(Path(__file__).read_bytes()),outputs=[])
    ASSETS.append(rec);savefig(fig,f'figures/{fid}',rec)

def leg(ax,**kw):ax.legend(frameon=False,**kw)
def point(ax,x,y,fam,**kw):
    return ax.plot(x,y,color=COLORS[fam],marker=MARKERS[fam],label=fam,**kw)

def forest(ax,frame,labels,unit):
    for i,(_,r) in enumerate(frame.iterrows()):
        col=COLORS[r.get('comparator','DiTing')]
        ax.errorbar(float(r.estimate),i,xerr=[[float(r.estimate-r.ci_low)],[float(r.ci_high-r.estimate)]],fmt='o',color=col,capsize=2)
    ax.axvline(0,color='.5',lw=.8);ax.set_yticks(range(len(frame)),labels)
    ax.invert_yaxis();ax.set_xlabel(f'DiTing − 对照  ({unit})');ax.grid(axis='x',alpha=.15)

# Full reproducible supplement tables, not hand-selected favorable bins.
for sid,ids in {'S2':['E05','E06','E07'],'S3':['E03','E04'],'S4':['E09'],
                'S5':['E10','E11','E13'],'S6':['E17','E18','E19']}.items():
    for eid in ids:source_file(f'{sid}_{eid}',D[eid],[eid],'all archived rows','see column dictionary',
        'unchanged archived estimand','all archived strata/policies/populations',[],
        'original archived event-cluster intervals where present')
for sid,relative in [('S2_interactions','legacy_with_physical_covariates/interaction_effects.csv'),
    ('S2_spatial_full','legacy_with_physical_covariates/spatial_paired_effects.csv'),
    ('S3_width68','additional_endpoints/width68_metrics.csv'),
    ('S3_width68_effects','additional_endpoints/width68_paired_effects.csv'),
    ('S2_interaction_contrasts','interaction_contrasts/interaction_contrasts.csv'),
    ('S2_interaction_support','interaction_contrasts/interaction_contrast_support.csv')]:
    path=ROOT+'source_tables/analyze/0/'+relative
    d=E.csv(path);d.to_csv(PAPER/f'assets/source_data/{sid}.csv',index=False,float_format='%.17g')
E.csv(ROOT+'source_tables/analyze/0/ttum_paired_summary.csv').to_csv(PAPER/'assets/source_data/S5_original_paired.csv',index=False,float_format='%.17g')

# Geometry/time population ledger from the five saved non-input snapshots.
geometry_rows=[]
for (family,policy,geometry,time),part in D['E30'].groupby(['family','checkpoint_policy','geometry_protocol','elapsed_time']):
    n=int(part.targets.sum())
    geometry_rows.append(dict(family=family,checkpoint_policy=policy,geometry_protocol=geometry,elapsed_time=time,
        events=len(part[['dataset_id','event_id']].drop_duplicates()),requests=n,
        mae=float(part.abs_error_sum.sum()/n),rmse=float(np.sqrt(part.squared_error_sum.sum()/n))))
pd.DataFrame(geometry_rows).to_csv(PAPER/'assets/source_data/F2_geometry_time_support.csv',index=False,float_format='%.17g')

# Fig 1 uses configuration/source evidence, not invented observations.
for c,description in zip('abc',['有效波形掩码与物理量','实际 PGA 前向路径','人口定义与三种时钟']):
    frame=pd.DataFrame([dict(family=f,station_context=META[f]['config']['model_params']['station_context_mode'],
        pga_readout=META[f]['config']['model_params']['pga_readout_mode'],
        readout_layers=META[f]['config']['model_params']['pga_readout_layers'],
        event_context=META[f]['config']['model_params']['pga_use_event_context'],
        native_samples=META[f]['config']['native_n_samples'],sampling_rate=100,
        scale_dimensions=11) for f in FAMILIES])
    source_file('F1_'+c,frame,['E01','E31','E32','E33'],description,'schematic / samples',
        'not a metric','actual complete models',[],'none','Schematic only; source trace in executed_model_path.md; inactive station self-attention excluded')

def boxes(ax,items):
    ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    for i,(text,color) in enumerate(items):
        y=.89-i*.21
        ax.text(.5,y,text,ha='center',va='center',fontsize=9,
            bbox=dict(boxstyle='round,pad=.45',fc=color,ec='.6',lw=.7))
        if i<len(items)-1:ax.annotate('',xy=(.5,y-.14),xytext=(.5,y-.07),arrowprops=dict(arrowstyle='->',color='.4'))

schematic_t=np.linspace(-5,5,201)
schematic=pd.DataFrame(dict(time_seconds=schematic_t,storage_supported=(schematic_t>=-4)&(schematic_t<=4),cutoff_supported=schematic_t<=2))
for j in range(3):schematic[f'synthetic_component_{j}']=np.sin((schematic_t+5)*(j+3))*np.where(schematic_t<0,.08,.3*np.exp(-schematic_t.clip(0)/4))
schematic.to_csv(PAPER/'assets/source_data/F1_schematic.csv',index=False,float_format='%.17g')
next(r for r in ASSETS if r['panel_id']=='F1_a')['notes']+='; additional generated values F1_schematic.csv; deterministic synthetic sinusoids, NOT observations'
def masks(a):
    valid=schematic.storage_supported&schematic.cutoff_supported
    for j in range(3):
        y=schematic[f'synthetic_component_{j}']+j*1.0
        a.plot(schematic_t,y,color='.8',lw=.7)
        a.plot(schematic_t,np.where(valid,y,np.nan),color=COLORS['DiTing'],lw=1)
    a.axvspan(2,5,color='.93');a.axvspan(-5,-4,color='#fff4dd')
    a.axvline(0,color='.5',ls=':',lw=1);a.axvline(2,color='#b35806',ls='--',lw=1)
    a.text(.98,.97,'SCHEMATIC  合成波形\n灰色为不可用样本',ha='right',va='top',transform=a.transAxes,fontsize=8)
    a.set_yticks([0,1,2],['Z','E','N']);a.set_xlim(-5,5);a.set_ylim(-.6,3.1)
    a.set_xticks([-5,-4,0,2,5],['−5','存储起点','首P','当前截止','5'])
    a.set_xlabel('相对首 P 时间 (s)\n有效样本去均值 / 联合峰值归一化 + 11 维幅值时长')
group('F1','预测任务与模型流程',[
    ('有效样本',masks),
    ('实际 PGA 路径',lambda a:boxes(a,[('冻结 DiTing / PhaseNet / EQT\n或可训练 TEAM CNN','#eef4fa'),('可训练适配器 + 尺度 + 输入坐标\n台站 memory 直接进入读出','#f2f2f2'),('目标坐标与 query token\n4 层 target cross-attention','#f2f2f2'),('3 分量高斯混合\nlog10 PGA 概率分布','#fff4dd')])),
    ('人口与交付',lambda a:boxes(a,[('旧 5 时点 × normal / random\nnon-input 随时点变化','#eef4fa'),('新固定 query 1–20 秒\nnever-input 全时段排除输入','#f2f2f2'),('cutoff → 数值 FIFO → 渲染 FIFO\n三个时钟分别报告','#f2f2f2'),('三帧通过的第三帧才确认\n之后仍可能再次不通过','#fff4dd')]))],ncols=1,height=2.5)

base=dict(checkpoint_policy='selected')
m=D['E03'];p=D['E04']
def accuracy(groupname,endpoints):
    return select(m,**base,target_group=groupname,endpoint=endpoints)
endpoints=['primary_10cell','first_second_2cell','early_6cell'];endlabels=['10 cell','1 秒','1/3/5 秒']
f2a=source_file('F2_a',accuracy('noninput',endpoints),['E03'],'selected/noninput/three endpoints','dex','targets within cell; cells equal','dynamic non-input',['C01','C02'])
f2b=source_file('F2_b',accuracy('all',endpoints),['E03'],'selected/all/three endpoints','dex','targets within cell; cells equal','all targets including input',['C01'])
f2c=source_file('F2_c',select(p,**base,target_group='noninput',endpoint=endpoints[:2],metric='mae'),['E04','E07'],'selected/noninput/MAE primary and first second','dex','paired equal-cell','dynamic non-input',['C01','C02'],'95% archived event-cluster percentile bootstrap; 2000 draws')
f2d=source_file('F2_d',select(D['E05'],**base,comparator='PhaseNet',factor='time'),['E05','E06'],'selected/PhaseNet/factor=time','dex','targets within geometry cell; geometry cells equal','dynamic non-input',['C02'],'95% archived event-cluster bootstrap')
source_file('F2_e',f2a,['E03'],'selected/noninput/RMSE/three endpoints','dex','root of equal-cell MSE','dynamic non-input',['C01','C02'])
source_file('F2_f',f2b,['E03'],'selected/all/RMSE/three endpoints','dex','root of equal-cell MSE','all targets',['C01'])
def accplot(a,d,metric):
    for j,f in enumerate(FAMILIES):
        y=[unique(d,family=f,endpoint=k)[metric] for k in endpoints]
        point(a,np.arange(3)+(j-1.5)*.10,y,f,linestyle='none')
    a.set_xticks(range(3),endlabels);a.set_ylabel('MAE (dex)' if metric.endswith('mae') else 'RMSE (dex)')
    lo,hi=a.get_ylim();a.set_ylim(lo,hi+(hi-lo)*.24)
    leg(a,ncol=2,loc='upper right');a.grid(axis='y',alpha=.2)
def effecttime(a):
    dd=f2d.assign(t=f2d.stratum.astype(float)).sort_values('t')
    a.errorbar(dd.t,dd.estimate,yerr=[dd.estimate-dd.ci_low,dd.ci_high-dd.estimate],fmt='o',color=COLORS['DiTing'],capsize=3)
    a.axhline(0,color='.5',lw=.8);a.set_xticks([1,3,5,10,20]);a.set_xlabel('首 P 后 cutoff (s)');a.set_ylabel('DiTing − PhaseNet MAE (dex)')
group('F2','非输入目标预测精度',[
    ('非输入目标',lambda a:accplot(a,f2a,'equal_cell_mae')),
    ('全部目标',lambda a:accplot(a,f2b,'equal_cell_mae')),
    ('配对 MAE 差值',lambda a:forest(a,f2c,[r.comparator+' '+('1 秒' if r.endpoint.startswith('first') else '10 cell') for _,r in f2c.iterrows()],'dex')),
    ('五个已有时点',effecttime),
    ('非输入目标 RMSE',lambda a:accplot(a,f2a,'equal_cell_rmse')),
    ('全部目标 RMSE',lambda a:accplot(a,f2b,'equal_cell_rmse'))],height=2.5)
# RMSE companion remains a directly reproducible independent panel.
for label,d in [('noninput',f2a),('all',f2b)]:
    fig,a=plt.subplots(figsize=(3.65,3),layout='constrained');accplot(a,d,'equal_cell_rmse');a.set_title(label+' RMSE');savefig(fig,'panels/F2_RMSE_'+label)

# Fig 3 independent score units; coverage is not calibration-bin reliability.
prob_panels=[]
for c,metric,title,unit in zip('abc',['nll','crps','brier'],['NLL 差值','CRPS 差值','Brier 差值'],['log-density score','dex','probability²']):
    dd=source_file('F3_'+c,select(p,**base,target_group='noninput',endpoint=endpoints[:2],metric=metric),['E04','E07'],f'selected/noninput/{metric}/primary+first second',unit,'paired equal-cell','dynamic non-input',['C03'],'95% archived event-cluster bootstrap')
    prob_panels.append((title,lambda a,d=dd,u=unit:forest(a,d,[r.comparator+' '+('1 秒' if r.endpoint.startswith('first') else '10 cell') for _,r in d.iterrows()],u)))
for c,metric,title,unit in [('d','coverage','区间覆盖率','fraction'),('e','equal_cell_width95','95% 区间宽度','dex'),('f','equal_cell_predictive_sigma','预测标准差','dex')]:
    dd=source_file('F3_'+c,accuracy('noninput',endpoints[:2]),['E03'],'selected/noninput/primary+first second',unit,'equal-cell','dynamic non-input',['C03'])
    def draw(a,d=dd,metric=metric,unit=unit):
        for j,f in enumerate(FAMILIES):
            if metric=='coverage':
                for off,col,style in [(-.14,'equal_cell_covered68','-'),(.14,'equal_cell_covered95','--')]:
                    y=[unique(d,family=f,endpoint=k)[col] for k in endpoints[:2]]
                    point(a,np.arange(2)+(j-1.5)*.05+off,y,f,linestyle=style,lw=.7)
                a.axhline(.68,color='.5',ls=':',lw=.6);a.axhline(.95,color='.5',ls=':',lw=.6)
                a.set_ylim(.45,1.02);a.text(.01,.02,'实线 68%   虚线 95%',transform=a.transAxes,fontsize=8)
            else:point(a,np.arange(2)+(j-1.5)*.08,[unique(d,family=f,endpoint=k)[metric] for k in endpoints[:2]],f,linestyle='none')
        a.set_xticks([0,1],['10 cell','1 秒']);a.set_ylabel(unit)
        if metric!='coverage':leg(a,ncol=2)
    prob_panels.append((title,draw))
group('F3','概率预测质量',prob_panels,height=2.5)

# Fig 4 effect intervals include neutral/adverse results.
spatial=[]
for c,metrics,g,target,title,unit in [
    ('a',['level_mse','shape_mse','field_mse'],'noninput','all','非输入空间平方误差','dex²'),
    ('b',['range_error'],'noninput','all','非输入空间幅差','dex'),
    ('c',['equal_distance_delta_mae'],'noninput','all','等距对差值误差','dex'),
    ('d',['shape_mse','field_mse'],['all','noninput'],'PhaseNet','PhaseNet 对照的目标人口','dex²'),
    ('e',['level_mse','shape_mse','field_mse'],'all','all','全目标空间平方误差','dex²'),
    ('f',['range_error','equal_distance_delta_mae'],'all','all','全目标空间幅差与等距差值','dex')]:
    keys=dict(checkpoint_policy='selected',target_group=g,metric=metrics)
    if target!='all':keys['comparator']=target
    dd=source_file('F4_'+c,select(D['E08'],**keys),['E08','E07'],str(keys),unit,
        'eligible fields equal within cell; cells equal','eligible fields; population specified per row',['C05'],'95% archived event-cluster bootstrap')
    short={'level_mse':'level','shape_mse':'shape','field_mse':'field','range_error':'range','equal_distance_delta_mae':'pair'}
    spatial.append((title,lambda a,d=dd,u=unit:forest(a,d,[r.comparator+' '+short[r.metric]+(' '+r.target_group if len(d)==4 else '') for _,r in d.iterrows()],u)))
group('F4','空间预测能力',spatial,height=2.5)

# Fig 5 unknown support envelopes, not bootstrap intervals.
scenes=['ideal_cutoff_replay','measured_compute_replay','measured_rendered_map_replay']
f5=[]
for c,scene,title in zip('abc',scenes,['理想截止时钟','完整数值时钟','含渲染时钟']):
    dd=source_file('F5_'+c,select(D['E10'],geometry_protocol='normal',panel='fixed_never_input_targets',budget_dex=.2,gate='field_pass',scene=scene),['E10','E12'],f'normal/never/field/0.20/{scene}','fraction','eligible events equal','fixed never-input; normal; >=5 targets',['C07'],'unknown lower/upper bounds, NOT confidence intervals')
    def tt(a,d=dd):
        for j,f in enumerate(FAMILIES):
            q=select(d,family=f).sort_values('horizon');x=q.horizon.to_numpy()+(j-1.5)*.12
            point(a,x,q.fraction_lower,f,linestyle='none')
            a.vlines(x,q.fraction_lower,q.fraction_upper,color=COLORS[f],lw=2,alpha=.65)
        a.set_xticks([5,10,20]);a.set_ylim(-.015,.56);a.set_xlabel('首 P 后期限 (s)');a.set_ylabel('适格事件中确认比例');leg(a,ncol=2,loc='upper left')
    f5.append((title,tt))
for c,geometry,panel,title in [('d','normal','fixed_all_targets','normal 全目标'),('e','random','fixed_never_input_targets','random 理想场景')]:
    dd=source_file('F5_'+c,select(D['E10'],geometry_protocol=geometry,panel=panel,budget_dex=.2,gate='field_pass',horizon=20,scene=scenes if geometry=='normal' else scenes[0]),['E10'],f'{geometry}/{panel}/field/0.20/20s','fraction','eligible events equal',panel,['C07'],'unknown bounds')
    def b(a,d=dd):
        ss=d.scene.unique()
        for j,f in enumerate(FAMILIES):
            q=[unique(d,family=f,scene=s).fraction_lower for s in ss];point(a,np.arange(len(ss))+(j-1.5)*.09,q,f,linestyle='none')
        a.set_xticks(range(len(ss)),['ideal','numeric','rendered'][:len(ss)]);a.set_ylim(0,1);a.set_ylabel('20 秒内确认比例');leg(a,ncol=2)
    f5.append((title,b))
dd=source_file('F5_f',select(D['E13'],geometry_protocol='normal',panel='fixed_never_input_targets',budget_dex=.2,gate='field_pass',scene=scenes[0]),['E13'],'normal/never/field/0.20/ideal','events / confirmed events','each model conditional confirmed set','model-specific confirmed subsets',['C08'],'descriptive counts; no paired causal interpretation')
def deg(a):
    for i,f in enumerate(FAMILIES):
        r=unique(dd,family=f);ratio=r.post_confirmation_known_nonpassing_events/r.stable_confirmed
        a.bar(i,ratio,color=COLORS[f],width=.55);a.text(i,ratio+.025,f'{int(r.post_confirmation_known_nonpassing_events)}/{int(r.stable_confirmed)}',ha='center',fontsize=8)
    a.set_xticks(range(4),FAMILIES,rotation=15);a.set_ylim(0,1);a.set_ylabel('已确认后再现不通过的比例')
f5.append(('确认后仍可能退化',deg));group('F5','场质量确认与交付时钟',f5,height=2.5)

# Fig 6 warm benchmarks, single cold record, full replay cost kept separate.
for c,eid,dd,fil,unit,pop in [
    ('a','E14',D['E14'],'all scopes; 6 events; 108 measured repeats/scope','s','6-event pilot'),
    ('b','E14',select(D['E14'],scope='whole_map'),'whole_map/cold load single record','s','one cold worker/model'),
    ('c','E15',select(D['E15'],cohort='all_frames',metric=['service_seconds','render_seconds']),'full normal replay','s','104800 frames; 1310 events × 4 models × 20'),
    ('d','E15',select(D['E15'],cohort='cutoff_20s',metric='rendered_ready_seconds'),'20th frame rendered-ready','s after first P','1310 events/model')]:
    source_file('F6_'+c,dd,[eid],'warm/cold kept separate; '+fil,unit,'empirical quantiles; not confidence intervals',pop,['C09'])
def warm(a):
    for i,f in enumerate(FAMILIES):
        for j,scope in enumerate(['station_only','whole_map']):
            r=unique(D['E14'],family=f,scope=scope);x=i+(j-.5)*.25
            a.plot([x,x],[r.p50_seconds,r.p95_seconds],color=COLORS[f],lw=1.4)
            a.scatter([x,x],[r.p50_seconds,r.p95_seconds],color=COLORS[f],marker=MARKERS[f],s=[12,35],facecolors='none' if j==0 else COLORS[f])
    a.axhline(1,color='.4',ls='--',lw=.8);a.set_yscale('log');a.set_xticks(range(4),FAMILIES,rotation=15);a.set_ylabel('warm 完整数值服务 (s)');a.set_ylim(.05,12)
    a.text(.98,.98,'空心 station  实心 whole-map\n小点 P50  大点 P95',transform=a.transAxes,va='top',ha='right',fontsize=8)
def cold(a):
    for i,f in enumerate(FAMILIES):a.bar(i,unique(D['E14'],family=f,scope='whole_map').cold_load_seconds,color=COLORS[f],width=.6)
    a.set_xticks(range(4),FAMILIES,rotation=15);a.set_ylabel('单次 cold load (s)')
def service(a):
    for i,f in enumerate(FAMILIES):
        q=unique(D['E15'],family=f,cohort='all_frames',metric='service_seconds')
        r=unique(D['E15'],family=f,cohort='all_frames',metric='render_seconds')
        a.bar(i,q.p50,color=COLORS[f],width=.6);a.bar(i,r.p50,bottom=q.p50,color='.8',width=.6)
    a.set_xticks(range(4),FAMILIES,rotation=15);a.set_ylabel('全 normal 帧 P50 成本 (s)');a.text(.01,.97,'彩色 数值  灰色 共享比较图渲染\n堆叠是分项 P50，非总耗时分位数',transform=a.transAxes,va='top',fontsize=8);a.set_ylim(0,6)
def queue(a):
    for i,f in enumerate(FAMILIES):
        q=unique(D['E15'],family=f,cohort='cutoff_20s',metric='rendered_ready_seconds')
        a.plot([i,i],[q.p50,q.p95],color=COLORS[f],lw=2);a.scatter([i,i],[q.p50,q.p95],color=COLORS[f],marker=MARKERS[f],s=[15,40])
    a.axhline(20,color='.5',ls=':');a.set_xticks(range(4),FAMILIES,rotation=15);a.set_ylabel('第 20 帧交付时钟 (s)');a.text(.98,.96,'小点 P50  大点 P95',transform=a.transAxes,va='top',ha='right',fontsize=8)
group('F6','计算时延与交付成本',[('warm P50 与 P95',warm),('cold 单列',cold),('全量回放分项成本',service),('第 20 帧含渲染交付',queue)],height=3.0)

# Fig 7 all pre-specified probability thresholds shown, no optimized operating point.
f7=[]
for c,panel,thr,title in [('a','fixed_all_targets',-1.2,'全目标  阈值 −1.2'),('b','fixed_never_input_targets',-1.2,'never-input  阈值 −1.2'),('c','fixed_all_targets',0.,'全目标  阈值 0.0'),('d','fixed_never_input_targets',0.,'never-input  阈值 0.0')]:
    dd=source_file('F7_'+c,select(D['E17'],geometry_protocol='normal',scene=scenes[2],panel=panel,threshold_logpga=thr,tau=3),['E17','E35'],f'normal/rendered/{panel}/threshold={thr}/tau3; all p_alarm','fraction','pooled affected targets; complete negative FPR',panel,['C10'],'unknown failure bounds; not statistical CI')
    def shadow(a,d=dd):
        for f in FAMILIES:
            q=select(d,family=f).sort_values('p_alarm')
            a.plot(q.pooled_false_positive_rate,q.pooled_failure_lower,color=COLORS[f],marker=MARKERS[f],label=f,lw=.8)
            a.fill_between(q.pooled_false_positive_rate.to_numpy(),q.pooled_failure_lower.to_numpy(),q.pooled_failure_upper.to_numpy(),color=COLORS[f],alpha=.13)
            if f=='DiTing':
                for j,(_,r) in enumerate(q.iterrows()):
                    a.annotate(f'p={r.p_alarm:g}',(r.pooled_false_positive_rate,r.pooled_failure_lower),fontsize=7,xytext=(12,9 if j%2 else -17),textcoords='offset points',color=COLORS[f],arrowprops=dict(arrowstyle='-',color=COLORS[f],lw=.4))
        xmax=min(1.,max(.12,float(d.pooled_false_positive_rate.max())*1.5))
        a.set_xlim(0,xmax);a.set_ylim(0,1);a.set_xlabel('完整观测负例的 FPR');a.set_ylabel('affected 目标的失败比例');leg(a,ncol=2,loc='lower left')
    f7.append((title,shadow))
for c,panel,title in [('e','fixed_all_targets','全目标事件等权差值界'),('f','fixed_never_input_targets','never-input 事件等权差值界')]:
    dd=source_file('F7_'+c,select(D['E18'],panel=panel),['E18'],'rendered/normal/threshold−1.2/p0.5/tau3; archived examples','fraction','events equal; DiTing lower−comparator upper and vice versa',panel,['C10'],'two separately bootstrapped bound intervals; not one effect CI')
    f7.append((title,lambda a,d=dd:forest(a,d,[r.comparator+' '+('下界' if 'lower' in r.metric else '上界') for _,r in d.iterrows()],'fraction')))
group('F7','预警失败与误报',f7,height=2.5)

# Tables: split broad result table into legible point/probability/calibration panels.
t1=[]
for f in FAMILIES:
    a=META[f]['audit'];c=META[f]['config']
    t1.append(dict(model=f,pretraining='STEAD v2' if f in ['PhaseNet','EQT'] else ('MAE 1200M; corpus/license unresolved' if f=='DiTing' else 'scratch'),
        encoder='frozen' if f!='TEAM' else 'trainable',native_samples=c['native_n_samples'],
        post_P_capacity_seconds=(c['native_n_samples']-501)/100,
        readout='4-layer target cross-attention; 3-Gaussian MDN',total_parameters=a['total_parameters'],
        trainable_parameters=a['trainable_parameters'],encoder_parameters=a['encoder_parameters'],
        epochs_trained=12,epoch_selected=EPOCHS[f],seed=42,global_batch=128,microbatch=8,
        lr=.001,optimizer='Adam',scheduler='cosine_fixed_updates',amplitude_duration_dims=11,
        count_provenance='archived interface audit; no weight loading in paper run'))
t1=pd.DataFrame(t1);t1.to_csv(PAPER/'assets/tables/T1.csv',index=False)
source_file('T1',t1,['E31','E32','E33'],'archived resolved config + interface audit', 'parameters / samples / seconds','descriptive','complete models',[])
tm=select(m,**base,target_group=['all','noninput'],endpoint=endpoints)
tm.to_csv(PAPER/'assets/tables/T2_full.csv',index=False,float_format='%.17g')
for name,cols in {
    'point':['family','epoch','target_group','endpoint','events','requests','supported_cells','equal_cell_mae','equal_cell_rmse','equal_cell_bias'],
    'probability':['family','target_group','endpoint','equal_cell_nll','equal_cell_crps','equal_cell_brier'],
    'calibration':['family','target_group','endpoint','equal_cell_covered68','equal_cell_covered95','equal_cell_width95','equal_cell_predictive_sigma'],
    'pooled':['family','target_group','endpoint','pooled_valid_targets','pooled_r2','pooled_prediction_on_truth_slope']}.items():
    tm[cols].to_csv(PAPER/f'assets/tables/T2_{name}.csv',index=False,float_format='%.17g')
source_file('T2',tm,['E03'],'selected; all and noninput; primary/first second/early','dex; dimensionless scores; NLL log density','equal-cell; explicitly pooled R2/slope companion','dynamic all/noninput',['C01','C02','C03'])
ledger=[]
for g in ['all','noninput']:
    r=unique(m,**base,family='DiTing',target_group=g,endpoint='primary_10cell')
    ledger.append(dict(protocol='legacy 5 times × 2 geometries',population=g,total_events=int(r.events),eligible_events=int(r.events),ineligible_events=0,requests_or_fields=int(r.requests),clock='cutoff',unknown='not silently excluded; source metric status retained'))
for geom in ['normal','random']:
    for panel in ['fixed_all_targets','fixed_never_input_targets']:
        r=unique(D['E10'],family='DiTing',geometry_protocol=geom,panel=panel,budget_dex=.2,gate='field_pass',scene=scenes[0],horizon=20)
        ledger.append(dict(protocol='fixed query '+geom,population=panel,total_events=int(r.total_cohort_events),eligible_events=int(r.events),ineligible_events=int(r.ineligible_fewer_than_5),requests_or_fields=None,clock='ideal; normal also numeric/rendered',unknown=int(r.unknown)))
for panel in ['fixed_all_targets','fixed_never_input_targets']:
    r=unique(D['E17'],family='DiTing',geometry_protocol='normal',panel=panel,threshold_logpga=-1.2,p_alarm=.5,tau=3,scene=scenes[2])
    ledger.append(dict(protocol='shadow normal rendered −1.2 p.5 tau3',population=panel,total_events=int(r.events),eligible_events=int(r.events_with_affected_targets),ineligible_events=None,requests_or_fields=int(r.affected_targets),clock='rendered',unknown=int(r.unknown)))
t3=pd.DataFrame(ledger);t3.to_csv(PAPER/'assets/tables/T3.csv',index=False)
source_file('T3',t3,['E03','E10','E17'],'population ledger; do not merge different eligibility rules','events / requests / targets','see protocol','different populations explicitly separated',[])

# Cases: exactly original preselection, both archived keyframes unchanged.
case_index=E.eid('E20');(PAPER/'assets/cases').mkdir(exist_ok=True)
case_manifest=[]
for relative in re.findall(r'\]\((cases/[^)]+\.png)\)',case_index):
    dest=PAPER/'assets'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    data=E.data(ROOT+relative,use='original_preselected_case');dest.write_bytes(data)
    case_manifest.append(dict(path=str(dest.relative_to(PAPER)),source_path=ROOT+relative,commit=RESULT,sha256=sha(data)))
for directory in sorted({str(Path(x['source_path']).parent) for x in case_manifest}):
    rel=directory+'/frame_sources.json';dest=PAPER/'assets/cases'/Path(directory).name/'frame_sources.json'
    dest.write_bytes(E.data(rel,use='case_metadata'))
(PAPER/'assets/cases/CASE_INDEX.md').write_text(case_index.replace('](cases/',']('))
write_json(PAPER/'assets/cases/manifest.json',case_manifest)
check('all original 12 cases with 5/20 second frames',len(case_manifest)==24,frames=len(case_manifest))

# Independent arithmetic rechecks of existing endpoints and population accounting.
for f in FAMILIES:
    d=select(D['E30'],checkpoint_policy='selected',family=f)
    cell=d.groupby(['geometry_protocol','elapsed_time'])[['targets','abs_error_sum','squared_error_sum']].sum()
    mae=(cell.abs_error_sum/cell.targets).mean();rmse=np.sqrt((cell.squared_error_sum/cell.targets).mean())
    r=unique(m,**base,family=f,target_group='noninput',endpoint='primary_10cell')
    check('legacy primary MAE '+f,abs(mae-r.equal_cell_mae)<1e-12,value=float(mae),source_value=float(r.equal_cell_mae))
    check('legacy primary RMSE '+f,abs(rmse-r.equal_cell_rmse)<1e-12,value=float(rmse))
q=D['E17'];check('shadow target outcome conservation',bool((q.timely+q.failures+q.unknown==q.affected_targets).all()))
check('shadow complete negative FPR arithmetic',bool(np.allclose(q.false_positives/q.complete_negative_targets,q.pooled_false_positive_rate,equal_nan=True)))
tt=D['E10'];check('TTUM eligibility ledger',bool((tt.events+tt.ineligible_fewer_than_5==tt.total_cohort_events).all()))
check('TTUM lower arithmetic',bool(np.allclose(tt.achieved/tt.events,tt.fraction_lower,equal_nan=True)))
check('normal queue complete frame count',len(D['E16'])==104800,frames=len(D['E16']))
check('legacy event source only five saved times',set(D['E30'].elapsed_time)=={1.,3.,5.,10.,20.})

write_json(PAPER/'qa/numerical_checks.json',dict(status='PASS',checks=CHECKS,environment=ENV,new_inference=False))
write_json(PAPER/'assets/asset_manifest.json',dict(assets=ASSETS,environment=ENV,style='4 fixed colors plus shape; units separate; no smoothed sparse curves',missing=['PIT','new fixed-query MDN reliability bins']))
write_json(PAPER/'evidence/environment.json',ENV)
E.save()
print(f'PASS: 7 figure groups, {sum(x["panel_id"]!="assembled" and x["asset_id"].startswith("F") for x in ASSETS)} main panels, 3 tables, 24 unchanged case frames; {len(CHECKS)} numerical checks')
