"""Diagnostics rendered only from saved small tables; no repeated raw CSV reads."""
from pathlib import Path
import numpy as np
import pandas as pd


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    output=Path(output);figs=output/'figures';figs.mkdir(exist_ok=True)
    density=pd.read_csv(output/'density_bins.csv.gz')
    pit=pd.read_csv(output/'pit_bins.csv.gz')
    reliability=pd.read_csv(output/'reliability_bins.csv.gz')
    metrics=pd.read_csv(output/'metrics_by_time_geometry_role.csv')
    fields=pd.read_csv(output/'field_summary.csv')
    density_max=max(2,int(density.targets.max()))
    captions=[]
    for family,all_rows in metrics.groupby('model_family'):
        seed=int(all_rows.seed.min());run=all_rows[all_rows.seed.eq(seed)].run_id.iloc[0]
        for protocol in ('normal','random'):
            fig,axes=plt.subplots(2,5,figsize=(14,6),layout='constrained')
            for col,time in enumerate((1,3,5,10,20)):
                d=density[(density.run_id==run)&(density.geometry_protocol==protocol)&(density.elapsed_time==time)]
                mat=np.zeros((50,50))
                for r in d.itertuples(): mat[int(round((r.truth_left+4)/.1)),int(round((r.prediction_left+4)/.1))]=r.targets
                artist=axes[0,col].imshow(mat.T,origin='lower',extent=[-4,1,-4,1],norm=LogNorm(vmin=1,vmax=density_max),cmap='viridis',aspect='equal')
                axes[0,col].plot([-4,1],[-4,1],'--',color='white',lw=.7)
                axes[0,col].set(title=f'{time}s; N={int(d.total_denominator.iloc[0])}',xlabel='Final truth log10 PGA',ylabel='Mixture mean log10 PGA')
                p=pit[(pit.run_id==run)&(pit.geometry_protocol==protocol)&(pit.elapsed_time==time)&(pit.target_group=='noninput')]
                axes[1,col].bar(p.lower,p.targets/p.denominator/.05,width=.05,align='edge',color='#4477aa');axes[1,col].axhline(1,color='gray',ls='--')
                axes[1,col].set(xlabel='PIT',ylabel='Density')
            fig.colorbar(artist,ax=axes[0].tolist(),label='Bin count (common scale)')
            name=f'{family}_{protocol}_density_pit.png';fig.savefig(figs/name,dpi=140);plt.close(fig)
            captions.append(f'{name}: validation fixed selected checkpoint seed{seed}, noninput, {protocol}, first_p_pick. Top: fixed [-4,1] dex density axes and diagonal; outside-range counts in density_bins. Bottom: PIT, uniform line is calibrated reference. Counts are repeated target records, not independent events. Sources: density_bins.csv.gz, pit_bins.csv.gz. No geological attribution.')
        fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
        r=reliability[(reliability.run_id==run)&(reliability.target_group=='noninput')]
        for time,part in r.groupby('elapsed_time'):
            # Explicitly pool reliability counts over two geometries only for this display.
            g=part.groupby('probability_bin')[['targets','probability_sum','exceedance_count']].sum();g=g[g.targets>0]
            axes[0].plot(g.probability_sum/g.targets,g.exceedance_count/g.targets,'o-',label=f'{time}s')
        axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].legend(fontsize=7);axes[0].set(xlabel='P(y > -1.2)',ylabel='Observed proportion')
        m=all_rows[(all_rows.seed==seed)&(all_rows.target_group=='noninput')]
        for protocol,part in m.groupby('geometry_protocol'):
            axes[1].plot(part.width95,part.covered95,'o-',label=protocol)
            axes[2].plot(part.elapsed_time,part.valid_targets,'o-',label=protocol)
        axes[1].axhline(.95,color='gray',ls='--');axes[1].set(xlabel='95% interval width (dex)',ylabel='95% quantile coverage');axes[1].legend()
        axes[2].set(xlabel='Seconds since first P pick',ylabel='Noninput target records');axes[2].legend()
        name=f'{family}_calibration_counts.png';fig.savefig(figs/name,dpi=140);plt.close(fig)
        captions.append(f'{name}: seed{seed}; reliability pools raw bin counts over geometries, while coverage-width and counts retain geometry. Calibration is unmodified. Sources: reliability_bins.csv.gz, metrics_by_time_geometry_role.csv. Mean ± sigma coverage is a separate metric, not this quantile coverage.')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    selected=fields[(fields.target_group=='noninput')]
    for family,g in selected.groupby('model_family'):
        for ax,metric in zip(axes,('level_mse_field_mean','shape_mse_field_mean','equal_distance_delta_mae_field_mean')):
            means=g.groupby('elapsed_time')[metric].mean();ax.plot(means.index,means,'o-',label=family.replace('_pretrained_frozen','').replace('_original_scratch',''))
            ax.set(xlabel='Seconds since first P pick',ylabel=metric.replace('_field_mean',''))
    axes[0].legend(fontsize=8);fig.savefig(figs/'level_shape_equal_distance.png',dpi=150);plt.close(fig)
    captions.append('level_shape_equal_distance.png: each eligible event field has >=5 targets; cells and fixed seeds displayed with equal weight. MSE components use the same field and satisfy level + shape = field MSE. Near-distance pairs use <=5 km epicentral radius difference; errors averaged within field before cells. Sources: field_summary.csv and event_fields.csv.gz. These are spatial diagnostics; station repeatability requires the missing train-only reference.')
    if (output/'input_decision_counts.csv').exists():
        counts=pd.read_csv(output/'input_decision_counts.csv')
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for protocol,g in counts.groupby('geometry_protocol'):
            if 'run_id' in g:g=g[g.run_id==g.run_id.iloc[0]]
            for count,p in g.groupby('input_count'):axes[0].plot(p.elapsed_time,p.decisions,label=f'{protocol}: {count} input')
        m=metrics[(metrics.target_group=='untriggered_noninput')&(metrics.run_id==metrics.run_id.iloc[0])]
        for protocol,g in m.groupby('geometry_protocol'):axes[1].plot(g.elapsed_time,g.valid_targets,'o-',label=protocol)
        axes[0].set(xlabel='Seconds since first P',ylabel='Decisions (not query records)');axes[0].legend(fontsize=5,ncol=2)
        axes[1].set(xlabel='Seconds since first P',ylabel='Untriggered target records');axes[1].legend()
        fig.savefig(figs/'input_and_untriggered_counts.png',dpi=150);plt.close(fig)
        captions.append('input_and_untriggered_counts.png: deduplicated event/time/geometry decisions on left, repeated target rows on right. Original common population equal across nine systems. Sources: input_decision_counts.csv, metrics_by_time_geometry_role.csv. Neither denominator is a count of independent station samples.')
    if (output/'station_recovery_summary.csv').exists():
        table=pd.read_csv(output/'station_recovery_summary.csv')
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for key,g in table[table.status=='supported'].groupby(['model_family','anchor','target_group','geometry_protocol']):
            mean=g.groupby('elapsed_time').station_equal_recovery_mae.mean();ax.plot(mean.index,mean,'o-',label='/'.join(key))
        ax.set(xlabel='Seconds since first P pick',ylabel='Station-equal recovery MAE (dex)');ax.legend(fontsize=5)
        fig.savefig(figs/'station_recovery.png',dpi=150);plt.close(fig)
        captions.append('station_recovery.png: station >=10 unique validation events; centered within each event/time/geometry/group, station means recomputed under event bootstrap. Train-only reference is oracle metadata, never neural input. Source: station_recovery_summary.csv. No spatial holdout claim.')
    (figs/'CAPTIONS.md').write_text('\n\n'.join(captions)+'\n')
