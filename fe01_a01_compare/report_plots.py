"""Saved-table plots; exploratory intervals and every predefined stratum retained."""
from pathlib import Path
import numpy as np
import pandas as pd
from .io import csv_read


def render_tables(directory,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    directory,out=Path(directory),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    effects=csv_read(directory/'paired_effects.csv')
    subset=effects[effects.target_group.eq('noninput')&effects.metric.eq('mae')]
    endpoints=['primary_10cell','first_second_2cell','early_6cell']
    colors={'PhaseNet':'#0072B2','EQT':'#D55E00','TEAM':'#009E73'}
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True,layout='constrained')
    for ax,endpoint in zip(axes,endpoints):
        rows=subset[subset.endpoint.eq(endpoint)]
        for i,r in enumerate(rows.itertuples()):
            ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',
                        color=colors[r.comparator],capsize=3)
        ax.set_yticks(range(len(rows)),[r.comparator+' / '+r.checkpoint_policy for r in rows.itertuples()])
        ax.axvline(0,color='gray',linewidth=.8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set(title=endpoint,xlabel='DiTing - comparator MAE (dex)')
    fig.suptitle('Japan validation seed42 | event-cluster 95% CI | exploratory, fixed models')
    fig.savefig(out/'paired_primary_early.png',dpi=150);plt.close(fig)
    strata=csv_read(directory/'stratified_effects.csv')
    for factor,group in strata.groupby('factor',sort=True):
        levels=list(dict.fromkeys(group.stratum))
        fig,axes=plt.subplots(1,2,figsize=(12,max(3,.38*len(levels)+1.5)),sharey=True,layout='constrained')
        for ax,policy in zip(axes,('selected','epoch12')):
            for offset,(comparator,color) in zip((-.17,0,.17),colors.items()):
                part=group[group.checkpoint_policy.eq(policy)&group.comparator.eq(comparator)].set_index('stratum').reindex(levels)
                for i,r in enumerate(part.itertuples()):
                    if np.isfinite(r.estimate):
                        ax.hlines(i+offset,r.ci_low,r.ci_high,color=color,linewidth=1)
                        ax.plot(r.estimate,i+offset,marker='o',mfc='white' if r.descriptive_only else color,color=color,ms=4)
            ax.axvline(0,color='gray',linewidth=.7)
            ax.set(title=policy,xlabel='DiTing - comparator MAE (dex)')
        sample=group[group.checkpoint_policy.eq('selected')&group.comparator.eq('PhaseNet')].set_index('stratum')
        axes[0].set_yticks(range(len(levels)),[str(l)+' | events='+str(int(sample.loc[l,'events'])) for l in levels])
        fig.suptitle(f'{factor}: all fixed strata, Unknown retained; hollow <20 events\nBlue PhaseNet; orange EQT; green TEAM; exploratory uncorrected event CIs')
        fig.savefig(out/('strata_'+factor+'.png'),dpi=135);plt.close(fig)
    # Every positive and negative contribution is retained.
    contrib=csv_read(directory/'first_second_contributions.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,layout='constrained')
    for ax,(comparator,color) in zip(axes,colors.items()):
        sub=contrib[contrib.comparator.eq(comparator)&contrib.checkpoint_policy.eq('selected')]
        for geometry,style in [('normal','o-'),('random','s--')]:
            part=sub[sub.geometry_protocol.eq(geometry)].sort_values('elapsed_time')
            ax.plot(part.elapsed_time,part.contribution_to_primary,style,color=color,label=geometry)
        ax.axhline(0,color='gray',linewidth=.7);ax.legend()
        ax.set(title=comparator,xlabel='seconds after first P')
    axes[0].set_ylabel('Signed contribution to 10-cell MAE difference (dex)')
    fig.suptitle('Algebraic decomposition, not causal attribution or percentage MAE reduction')
    fig.savefig(out/'first_second_signed_contributions.png',dpi=150);plt.close(fig)
    (out/'CAPTIONS.md').write_text('Source: paired_effects.csv, stratified_effects.csv, first_second_contributions.csv.\n'
        'All real saved Japan validation, seed42; fixed selected/epoch12 conditions,2000 event-cluster draws.\n'
        'Negative error difference favors DiTing at this endpoint; intervals omit seed/selection uncertainty, exploratory uncorrected.\n'
        'Unknown physical SNR/missing/origin strata require HPC sidecars. Sparse support and effective draws are in CSV, not inferred from interval sign.\n')
