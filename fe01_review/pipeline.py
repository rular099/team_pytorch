"""Real existing-result diagnostics and explicit incomplete evidence gates."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fe01.metrics import summarize
from .provenance import ROOT,require,sha256,write_json,read_json,provenance,seal,training_identity,evaluation_identity
from .requests import read_frame,population_audit
from .diagnostics import (group_scores,fields,field_summary,fixed_panels,stratified,calibration_data,
    paired_fields,station_recovery)
from .statistics import event_losses,interval,fixed_model_mean_loss


def analyze(source_root,output,reference_path=None,verification_root=None,draws=5000):
    training_identity()
    published=read_frame(ROOT/'reports/fe01_hpc_results_20261004/run_summary.csv')
    stored={};scores=[];panels=[];audits=[];moves=[];strata=[];fs=[];fl=[];losses=[];ci=[];reused=[];reliability=[];pits=[];density=[]
    station=[];station_summary=[];stability=[];controls=[];input_counts=[]
    reference=read_json(reference_path) if reference_path else None
    exposure=None;exposure_status={}
    if reference_path:
        p=Path(reference_path).parent/'actual_training_station_exposure.csv'
        status=Path(reference_path).parent/'training_exposure_identity.json'
        if status.is_file():exposure_status={r['run_id']:r['status'] for r in read_json(status)}
        if p.is_file() and p.stat().st_size>1:exposure=read_frame(p)
    for r in published.itertuples():
        print('ANALYZE',r.run_id,flush=True)
        path=Path(source_root)/r.selected_source
        require(sha256(path)==r.selected_source_sha256,'Source CSV byte SHA differs: '+r.run_id)
        frame=read_frame(path);identity=population_audit(frame)
        require(frame.seed.eq(r.seed).all() and frame.model_family.eq(r.model_family).all(),'CSV model identity differs')
        require(path.name==f'validation_epoch{r.selected_epoch}.csv.gz','Frozen selected CSV source differs')
        require(np.isfinite(frame[['prediction','nll','crps','brier','predictive_sigma']]).all().all(),'Nonfinite output; no row dropping')
        tag=dict(run_id=r.run_id,model_family=r.model_family,seed=r.seed,epoch=r.selected_epoch)
        from .requests import DECISION_KEYS
        count=frame[DECISION_KEYS+['input_count']].drop_duplicates().groupby(['geometry_protocol','elapsed_time','input_count']).size().rename('decisions').reset_index()
        input_counts.append(count.assign(**tag))
        def tagged(table): return table.assign(**tag)
        score=tagged(group_scores(frame));scores.append(score)
        table=tagged(fields(frame));fl.append(table);fs.append(tagged(field_summary(table)));stored[r.run_id]=table
        p,a,m=fixed_panels(frame);panels.append(tagged(p));audits.append(tagged(a));moves.append(tagged(m))
        strata.append(tagged(stratified(frame,r.native_n_samples)))
        rel,pit,den=calibration_data(frame);reliability.append(tagged(rel));pits.append(tagged(pit));density.append(tagged(den))
        loss=event_losses(frame);losses.append(tagged(loss))
        for role,g in loss.groupby('target_group'):
            for endpoint,times in [('primary10cell',(1,3,5,10,20)),('early6cell',(1,3,5)),('one_second',(1,))]:
                ci.append(dict(**tag,target_group=role,endpoint=endpoint,**interval(g,times=times,draws=draws)))
        n=score[score.target_group=='noninput'];primary=float(n.mae.mean())
        require(abs(primary-r.validation_equal_time_protocol_noninput_mae)<1e-12,'Historical primary score changed')
        gate_path=Path(verification_root or '/__missing__')/r.run_id/'verification.json'
        gate=read_json(gate_path) if gate_path.is_file() else {}
        certified=(gate.get('status')=='PASS' and gate.get('reused_csv_sha256')==r.selected_source_sha256
            and gate.get('evaluation_module_sha')==evaluation_identity()['evaluation_module_sha'] and gate.get('checkpoint_epoch')==r.selected_epoch)
        reused.append(dict(**tag,csv_sha256=r.selected_source_sha256,**identity,source='original selected CSV',
            forward_status='PASS' if certified else 'NOT_RUN',reuse_status='CERTIFIED' if certified else 'CSV_IDENTITY_PASS_FORWARD_PENDING',
            primary_noninput_mae=primary,pooled_noninput_rmse=summarize(frame[frame.target_role!='observed_input'])['rmse'],
            mean_cell_rmse=float(n.rmse.mean())))
        if reference is not None:
            a,b,c=station_recovery(frame,reference,draws=draws)
            authenticated=exposure_status.get(r.run_id)=='PASS'
            if exposure is not None and authenticated:
                station_exposure=exposure[exposure.run_id==r.run_id].drop(columns=['run_id'])
                a=a.merge(station_exposure,on='station_id',how='left',validate='many_to_one')
                for col in ('input_training_events','query_training_events'):a[col]=a[col].fillna(0).astype(int)
                for col in ('station_seen_as_input','station_seen_as_query'):a[col]=a[col].fillna(False)
            a['model_training_exposure_status']='AUTHENTICATED_COMMITTED_JOURNALS' if authenticated else 'UNKNOWN'
            a['spatial_holdout']=False
            station.append(tagged(a));station_summary.append(tagged(b));stability.append(tagged(c))
            from fe01.spatial import reference_design
            for anchor,beta in ([] if controls else [('no_site','no_site_coefficients'),('train_site','site_reference_coefficients')]):
                baseline=frame.copy();baseline['prediction']=reference_design(frame)@np.asarray(reference[beta])
                if anchor=='train_site': baseline['prediction']+=frame.station_id.map(reference['train_site_effects']).fillna(0)
                controls.append(fields(baseline).assign(anchor=anchor,model_family=anchor+'_oracle_reference'))
    all_scores=pd.concat(scores,ignore_index=True);all_scores.to_csv(output/'metrics_by_time_geometry_role.csv',index=False)
    files={'fixed_panel_metrics.csv':panels,'fixed_panel_population.csv':audits,'role_migrations.csv':moves,
        'stratified_metrics.csv.gz':strata,'field_summary.csv':fs,'event_fields.csv.gz':fl,'event_losses.csv.gz':losses,
        'reliability_bins.csv.gz':reliability,'pit_bins.csv.gz':pits,'density_bins.csv.gz':density,'input_decision_counts.csv':input_counts}
    for name,tables in files.items():pd.concat(tables,ignore_index=True).to_csv(output/name,index=False)
    pd.DataFrame(ci).to_csv(output/'event_confidence_intervals.csv',index=False)
    pd.DataFrame(reused).to_csv(output/'reused_result_manifest.csv',index=False)
    families=[];family_ci=[];all_loss=pd.concat(losses,ignore_index=True)
    for family,g in pd.DataFrame(reused).groupby('model_family'):
        families.append(dict(model_family=family,seeds=len(g),primary_mean=float(g.primary_noninput_mae.mean()),
            primary_seed_sample_std=float(g.primary_noninput_mae.std(ddof=1)),interpretation='mean fixed-model losses; no averaged predictions or ensemble'))
        avg=fixed_model_mean_loss([t.drop(columns=['run_id','model_family','seed','epoch']) for _,t in all_loss[all_loss.model_family==family].groupby('run_id')])
        for role,g in avg.groupby('target_group'):
            family_ci.append(dict(model_family=family,target_group=role,**interval(g,draws=draws),
                conditioning='three fixed selected models; excludes training randomness and selection uncertainty'))
    pd.DataFrame(families).to_csv(output/'family_scores.csv',index=False);pd.DataFrame(family_ci).to_csv(output/'family_fixed_loss_ci.csv',index=False)
    paired=[];point_paired=[]
    for seed in (42,43,44):
        ids=published[published.seed==seed].run_id.tolist()
        for left,right in itertools.combinations(ids,2):
            print('PAIRED_FIELDS',left,right,flush=True)
            paired.append(paired_fields(stored[left],stored[right],draws=draws).assign(left=left,right=right,seed=seed))
            a,b=all_loss[all_loss.run_id==left],all_loss[all_loss.run_id==right]
            for role in ('noninput','untriggered_noninput','actual_single_input_noninput'):
                for metric in ('abs_error','nll'):
                    point_paired.append(dict(left=left,right=right,target_group=role,metric=metric,
                        **interval(a[a.target_group==role],right=b[b.target_group==role],metric=metric,draws=draws)))
    pd.concat(paired,ignore_index=True).to_csv(output/'paired_field_event_bootstrap.csv',index=False)
    pd.DataFrame(point_paired).to_csv(output/'paired_point_event_bootstrap.csv',index=False)
    if reference is not None:
        for name,tables in [('station_recovery.csv.gz',station),('station_recovery_summary.csv',station_summary),('station_stability.csv.gz',stability),('reference_field_controls.csv.gz',controls)]:
            pd.concat(tables,ignore_index=True).to_csv(output/name,index=False)
    from .plots import render
    render(output)
    site_complete=bool(reference is not None and station_summary and pd.concat(station_summary).status.eq('supported').all())
    gates=[dict(gate='fixed_CSV_population',status='PASS',reason='all nine exact original byte SHAs, population and clock/labels pins'),
        dict(gate='historical_primary_preserved',status='PASS',reason='each selected-epoch score matches prior report to 1e-12'),
        dict(gate='production_forward_reuse',status='PASS' if all(r['forward_status']=='PASS' for r in reused) else 'NOT_RUN',reason='requires original selected checkpoints and Japan HDF5 on DCU'),
        dict(gate='legacy_same_population_comparison',status='NOT_RUN',reason='RT55 ep32 and RT61 ep8 production forward missing'),
        dict(gate='train_reference_site_recovery',status='PASS' if site_complete else 'INCOMPLETE',reason='common train-only reference and sufficient station populations; not a geological success criterion' if site_complete else 'missing reference or insufficient station cohort'),
        dict(gate='random_long_replay',status='NOT_RUN',reason='manual HPC stages pending')]
    pd.DataFrame(gates).to_csv(output/'gates.csv',index=False)
    write_json(output/'summary.json',dict(split='val',models=9,events_per_run=1310,rows_per_run=194265,noninput_rows_per_run=139440,
        historical_primary_preserved=True,reference_available=bool(reference),bootstrap_draws=draws,production_forward_completed=False,
        site_evidence='diagnostic evidence; no geological identification' if site_complete else 'INCOMPLETE',hpc_status='NOT_SUBMITTED_THIS_TASK'))
    write_json(output/'provenance.json',provenance(raw_csv_sha256={r['run_id']:r['csv_sha256'] for r in reused},
        reference_sha256=sha256(reference_path) if reference else None,training_source_sha='73ae9dec50713805c878b5e4efeb8264c01d0c76',
        epistemic_status='actual CSV diagnostics; checkpoint/forward certification pending'))
    seal(output)
