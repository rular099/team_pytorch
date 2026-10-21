#!/usr/bin/env python3
"""Summarize a verified A01 review archive using saved validation predictions.

No models, waveforms, training, held-out test or scheduler access. The input is
the regular-file tree produced by review_archive.py. Output must be new.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fe01.config import fingerprint,sha256
from fe01.metrics import summarize,score_rows
from fe01_review.requests import population_audit,PAIR_KEYS,IDENTITY_KEYS,DECISION_KEYS
from fe01_review.diagnostics import field_summary

GROUPS=('all','noninput','triggered_noninput','untriggered_noninput','K1_noninput','Kmulti_noninput')
MEANS=('mae','bias','nll','crps','brier','covered68','covered95','predictive_sigma','width95')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def csv(path,frame):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0} if str(path).endswith('.gz') else 'infer')


def require(ok,message):
    if not ok:raise ValueError(message)


def subsets(frame):
    return dict(all=frame,noninput=frame[frame.target_role.ne('observed_input')],
        triggered_noninput=frame[frame.target_role.eq('triggered_noninput')],
        untriggered_noninput=frame[frame.target_role.eq('untriggered_noninput')],
        K1_noninput=frame[frame.input_count.eq(1)&frame.target_role.ne('observed_input')],
        Kmulti_noninput=frame[frame.input_count.gt(1)&frame.target_role.ne('observed_input')])


def export_tag_check(frame,family,mode,reused):
    expected=dict(seed=42,model_family=family,split='val',units='log10(m/s^2)',
                  absolute_amplitude_mode='physical_scale_keep_duration' if mode=='ON' else 'no_absolute_scale_keep_duration',
                  comparison_kind='reused_trained_ON_frozen_epoch' if reused else 'trained_ablation')
    for name,value in expected.items():require(name in frame and frame[name].eq(value).all(),'Export condition tag differs: '+name)
    return dict(status='PASS',rows=len(frame),expected=expected)


def mdn_check(frame):
    ids=np.unique(np.r_[np.linspace(0,len(frame)-1,64,dtype=int),np.argmax(frame.nll.to_numpy())])
    sample=frame.iloc[ids]
    arrays={key:np.array([json.loads(v) for v in sample['mdn_'+key]],dtype=np.float64)
            for key in ('logits','weights','mu','sigma')}
    logits,w,mu,sigma=(arrays[k] for k in ('logits','weights','mu','sigma'))
    logw=logits-logsumexp(logits,axis=-1,keepdims=True)
    require(np.isfinite(logits).all() and (sigma>0).all(),'Invalid saved MDN')
    values=score_rows(sample.truth.to_numpy(),w,mu,sigma)
    values['nll']=-logsumexp(logw-np.log(sigma*np.sqrt(2*np.pi))-
                          (sample.truth.to_numpy()[:,None]-mu)**2/(2*sigma*sigma),axis=-1)
    errors={name:float(np.max(np.abs(sample[name].to_numpy(dtype=float)-value.astype(float))))
            for name,value in values.items() if name in sample}
    errors['softmax_weights']=float(np.max(np.abs(w-np.exp(logw))))
    require(all(value<2e-8 for value in errors.values()),'MDN sample recalculation differs: '+str(errors))
    return dict(rows=len(sample),selection='64 evenly spaced rows plus maximal saved NLL; deterministic, no metric selection',
                max_abs_errors=errors,status='PASS')


def diagnostics(raw,out):
    stages=[];sensitivities=[];checks=[];gate_rows=[];nested_rows=[]
    for index in range(5):
        base=raw/'training'/f'index{index}'/'diagnostics';meta=read(base/'diagnostics.json')
        tag=dict(index=index,evidence=meta['evidence'],probe_count=meta['probe_count'])
        table=pd.read_csv(base/'query_stages.csv',float_precision='round_trip')
        table['control']=table.control.fillna('natural')
        table['input_group']=np.where(table.input_count.eq(1),'K1','Kmulti')
        for key,part in table.groupby(['control','input_group','stage']):
            multiple=part[part.effective_queries.ge(2)]
            stages.append(dict(**tag,control=key[0],input_group=key[1],stage=key[2],rows=len(part),
                decisions=part.decision.nunique(),at_least_two_queries=int(part.effective_queries.ge(2).sum()),
                query_rms_median=part.query_rms.median(),query_rms_max=part.query_rms.max(),
                cross_query_max_abs_median=part.cross_query_max_abs.median(),cross_query_max_abs_max=part.cross_query_max_abs.max(),
                multiple_query_rms_median=multiple.query_rms.median(),multiple_query_cross_max_median=multiple.cross_query_max_abs.median()))
        table=pd.read_csv(base/'query_sensitivity.csv',float_precision='round_trip')
        table['control']=table.control.fillna('natural');table['input_group']=np.where(table.input_count.eq(1),'K1','Kmulti')
        for key,part in table.groupby(['control','input_group','step_km','axis']):
            sensitivities.append(dict(**tag,control=key[0],input_group=key[1],step_km=key[2],axis=key[3],rows=len(part),
                decisions=part.decision.nunique(),gradient_median=part.gradient_mean_max_abs.median(),
                gradient_max=part.gradient_mean_max_abs.max(),gradient_min=part.gradient_mean_max_abs.min(),
                fd_median=part.finite_difference_mean_max_abs.median(),
                fd_gradient_error_max=part.mean_fd_gradient_max_error.max()))
        for kind in ('scale_audit','future_audit','station_controls'):
            table=pd.read_csv(base/(kind+'.csv'),float_precision='round_trip')
            record=dict(**tag,kind=kind,rows=len(table),upstream_status=meta['upstream_status'])
            if 'status' in table:record['statuses']=json.dumps(table.status.value_counts().to_dict(),sort_keys=True)
            for col in ('downstream_max_tolerance_ratio','log_shift_max_error','joint_max_abs','coords_only_max_abs',
                        'delivered_amplitude','normalized','duration','mdn','probabilities'):
                if col in table:record[col+'_max']=table[col].max()
            for col in ('complete_inputs_exact','prefix_statistics_exact','selected_ids_equal'):
                if col in table:record[col+'_all']=bool(table[col].dropna().all())
            checks.append(record)
        table=pd.read_csv(base/'gates.csv',float_precision='round_trip')
        for col in table:
            if col=='waveform_scale_gate' or col.endswith(('.attn_gate','.ffn_gate')):
                gate_rows.append(dict(**tag,gate=col,min=table[col].min(),median=table[col].median(),max=table[col].max(),
                                      interpretation='raw saved learned scalar, not necessarily a probability'))
        for record in read(base/'nested_support.json'):
            for k in (1,3,5,8):nested_rows.append(dict(**tag,decision=record['decision'],K=k,
                supported=k not in record['unsupported'],available=record['available'],common_queries=record['common_queries']))
    for name,rows in (('diagnostic_query_stages.csv',stages),('diagnostic_query_sensitivity.csv',sensitivities),
                      ('diagnostic_invariance_checks.csv',checks),('diagnostic_gates.csv',gate_rows),('diagnostic_nested_support.csv',nested_rows)):
        csv(out/name,pd.DataFrame(rows))


def paired_spatial_tables(raw,out):
    """Absolute ON/OFF field levels on exactly the pair's eligible population."""
    rows=[];support=[];nested=[]
    families={0:'TEAM',1:'PhaseNet',2:'EQT',4:'DiTing'}
    metrics=('level_mse','shape_mse','field_mse','range_error','equal_distance_delta_mae',
             'truth_range','prediction_range','range_ratio','pairwise_delta_mae')
    for index,family in families.items():
        for label in ('selected','epoch12'):
            base=raw/'results'/f'index{index}'/'comparisons'/label
            frame=pd.read_csv(base/'spatial_fields.csv.gz',float_precision='round_trip',dtype={'dataset_id':str,'event_id':str})
            keys=DECISION_KEYS+['a01_target_group']
            left=frame[frame.model_mode.eq('ON')].sort_values(keys)
            right=frame[frame.model_mode.eq('OFF')].sort_values(keys)
            require(left[keys+['targets']].reset_index(drop=True).equals(right[keys+['targets']].reset_index(drop=True)),
                    'Paired spatial population differs')
            for (role,mode,geometry,time),part in frame.groupby(['a01_target_group','model_mode','geometry_protocol','elapsed_time']):
                row=dict(family=family,checkpoint_policy=label,target_group=role,mode=mode,
                         geometry_protocol=geometry,elapsed_time=time,fields=len(part),targets=int(part.targets.sum()),
                         events=part[['dataset_id','event_id']].drop_duplicates().shape[0])
                row.update({metric:part[metric].mean() for metric in metrics})
                row.update({metric+'_fields':int(part[metric].notna().sum()) for metric in metrics})
                rows.append(row)
            interval=pd.read_csv(base/'paired_event_bootstrap.csv',float_precision='round_trip')
            support.append(interval.assign(family=family,checkpoint_policy=label)[['family','checkpoint_policy','target_group',
                'metric','status','events','targets','fields','supported_cells','missing_cells','draws','effective_replicates','empty_cell_fraction']])
    cells=pd.DataFrame(rows);summary=[]
    for key,part in cells.groupby(['family','checkpoint_policy','target_group','mode']):
        row=dict(zip(('family','checkpoint_policy','target_group','mode'),key),supported_cells=len(part),
                 fields=int(part.fields.sum()),targets=int(part.targets.sum()))
        row.update({'equal_cell_'+metric:part[metric].mean() for metric in metrics});summary.append(row)
    summary=pd.DataFrame(summary)
    intervals=pd.read_csv(out/'paired_event_bootstrap.csv',float_precision='round_trip')
    for key,part in summary.groupby(['family','checkpoint_policy','target_group']):
        part=part.set_index('mode')
        for metric in metrics[:5]:
            actual=part.loc['OFF','equal_cell_'+metric]-part.loc['ON','equal_cell_'+metric]
            saved=intervals[intervals.family.eq(key[0])&intervals.checkpoint_policy.eq(key[1])&
                            intervals.target_group.eq(key[2])&intervals.metric.eq(metric)].iloc[0]
            require(abs(actual-saved.estimate)<2e-10,'Absolute spatial levels disagree with paired contrast')
    csv(out/'paired_field_by_cell.csv',cells);csv(out/'paired_field_levels.csv',summary)
    csv(out/'bootstrap_support_audit.csv',pd.concat(support,ignore_index=True))
    for index in range(5):
        base=raw/'training'/f'index{index}'/'diagnostics';evidence=read(base/'diagnostics.json')['evidence']
        table=pd.read_csv(base/'query_stages.csv',float_precision='round_trip')
        table=table[table.control.eq('nested_same_queries')&table.effective_queries.ge(2)]
        for (k,stage),part in table.groupby(['input_count','stage']):
            nested.append(dict(index=index,evidence=evidence,K=k,stage=stage,decisions=part.decision.nunique(),
                query_rms_median=part.query_rms.median(),cross_query_max_abs_median=part.cross_query_max_abs.median(),
                scope='same event/time/common query stations; >=2 effective queries; inference control'))
    csv(out/'diagnostic_nested_stages.csv',pd.DataFrame(nested))


def matched_nested_table(raw,out):
    rows=[]
    for index in range(5):
        base=raw/'training'/f'index{index}'/'diagnostics';evidence=read(base/'diagnostics.json')['evidence']
        table=pd.read_csv(base/'query_stages.csv',float_precision='round_trip')
        table=table[table.control.eq('nested_same_queries')&table.effective_queries.ge(2)]
        support=table[table.stage.eq('prediction')].groupby('decision').input_count.agg(lambda value:set(value))
        complete=support[support.apply(lambda value:value=={1,3,5,8})].index
        table=table[table.decision.isin(complete)]
        require(table.groupby('decision').effective_queries.nunique().eq(1).all(),'Nested common query counts changed with K')
        for (k,stage),part in table.groupby(['input_count','stage']):
            require(part.decision.nunique()==len(complete),'Matched nested stage missing')
            rows.append(dict(index=index,evidence=evidence,K=k,stage=stage,matched_decisions=len(complete),
                query_rms_median=part.query_rms.median(),cross_query_max_abs_median=part.cross_query_max_abs.median(),
                scope='same complete decision cohort for all K=1/3/5/8; fixed event/time/query IDs; dispersion is not accuracy'))
    csv(out/'diagnostic_nested_matched.csv',pd.DataFrame(rows))


def main(raw,out):
    raw=Path(raw).resolve();out=Path(out).resolve();require(not out.exists(),'Output already exists')
    transport=read(raw.parent/'archive_verification.json');require(transport['status']=='PASS','Archive not verified')
    for name,record in transport['files'].items():
        require(sha256(raw/name)==record['sha256'],'Extracted member changed: '+name)
    plan=read(raw/'evaluation_plan.json');source=transport['source']
    require(source['a01_code_sha256']==plan['source_core_sha256'],'Archive core identity differs')
    for name,digest in source['a01_files_sha256'].items():
        require(sha256(ROOT/name)==digest,'Local frozen core differs: '+name)
    original=source['original_training']
    require(fingerprint(original['files_sha256'])==original['code_sha256'],'Original training source inventory fingerprint differs')
    original_present=[];original_missing=[]
    for name,digest in original['files_sha256'].items():
        path=ROOT/name
        if path.is_file():
            require(sha256(path)==digest,'Local original training source differs: '+name)
            original_present.append(name)
        else:
            require(name.startswith('dtbench/'),'Unexpected missing local source: '+name)
            original_missing.append(name)
    out.mkdir(parents=True)
    meta=out/'source_metadata';meta.mkdir()
    # Small original evidence, already redacted by the HPC packer. Large raw
    # predictions, request tables and model weights stay outside Git.
    for name in transport['files']:
        path=raw/name
        if path.suffix in ('.json','.csv','.md','.tsv','.png') and not path.name.endswith('.csv.gz'):
            target=meta/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    inventory=[dict(path=name,**record) for name,record in transport['files'].items()]
    csv(out/'source_inventory.csv',pd.DataFrame(inventory))
    seal_checks=[];revisions=[]
    for index in range(5):
        gate=read(raw/'results'/f'index{index}'/'EVAL_COMPLETE.json')
        require(gate['status']=='PASS' and gate['index']==index and gate['split']=='validation','Evaluation seal failed')
        require(gate['source_core_sha256']==source['a01_code_sha256'],'Evaluation core identity differs')
        for name,digest in gate['files_sha256'].items():
            path=raw/'results'/f'index{index}'/name
            require(path.is_file(),'Seal artifact missing: '+str(path))
            same=sha256(path)==digest
            transformed_private_json=(path.suffix=='.json' and
                (path.name=='checkpoint_identity.json' or name.startswith('paired_input_checks/')))
            require(same or transformed_private_json,'Original seal mismatch outside documented private JSON: '+name)
            seal_checks.append(dict(index=index,path=str(path.relative_to(raw)),original_sha256=digest,
                archive_sha256=transport['files'][str(path.relative_to(raw))]['sha256'],
                original_bytes_match=same,scope='same original bytes' if same else 'JSON transformed by documented pack redaction/serialization'))
        proof=gate.get('operations_revision')
        if proof:
            path=raw/'evaluation_execution'/Path(proof['path']).parent.name/'revision.json'
            require(sha256(path)==proof['sha256'],'Revision JSON differs from evaluation proof')
            revision=read(path);require(index in revision['allowed_eval_indices'],'Revision index differs')
            require(revision['operations_sha256']==gate['execution_operations_sha256'],'Execution source differs')
            commit='f29ecdcb4dcc4bdce9eaf560bcbc9453c2b4506e' if index==0 else '59f3eb30d6c955ffe2904aee40693af71dc0a440'
            import hashlib
            for name,digest in revision['operations_sha256'].items():
                data=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':scripts/fe01_a01_ops/'+name])
                require(hashlib.sha256(data).hexdigest()==digest,'Revision source differs from delivered Git commit: '+name)
        else:require(gate['execution_operations_sha256']==plan['operations_sha256'],'Original execution source differs')
        revisions.append(dict(index=index,kind='revision' if proof else 'original',
            revision_path=proof['path'] if proof else '',operations_sha256=json.dumps(gate['execution_operations_sha256'],sort_keys=True)))
    csv(out/'source_seal_checks.csv',pd.DataFrame(seal_checks));csv(out/'evaluation_execution_sources.csv',pd.DataFrame(revisions))
    summaries=[];cells=[];curves=[];model_checks=[];field_cells=[];field_averages=[];event_cells=[];boots=[];train_audits=[]
    reference=None;frames_read=0
    for entry in plan['runs']:
        index=entry['index'];training=raw/'training'/f'index{index}'
        cfg=read(training/'resolved_config.json');lock=read(training/'protocol.lock.json')
        require(lock==read(training/'audits/protocol.lock.json'),'Training/audit lock differs')
        require(lock['status']=='AUDIT_PASS' and lock['a01_code_sha256']==source['a01_code_sha256'],'Training audit differs')
        curve=pd.read_csv(training/'training_curves.csv',float_precision='round_trip')
        require(curve.epoch.tolist()==list(range(1,13)) and curve.updates.tolist()==[212*i for i in range(1,13)],'Training budget differs')
        require(cfg['seed']==42 and lock['world_size']==16 and cfg['training']['global_batch']==128,'Training protocol differs')
        curves.append(curve.assign(index=index,family=entry['family'],mode=entry['mode'],run_id=entry['run_id']))
        train_audits.append(dict(index=index,family=entry['family'],mode=entry['mode'],seed=cfg['seed'],world_size=lock['world_size'],
            epochs=12,updates=2544,native_n_samples=cfg['native_n_samples'],reuse_status=lock['reuse_gate']['status'],
            common_initial_state_sha256=lock['initial_state']['common_sha256'],initial_encoder_sha256=lock['initial_state']['encoder_sha256'],
            full_initial_state_sha256=lock['initial_state']['state_sha256'],sampling_fingerprint=fingerprint(lock['sampling']),
            data_identity_fingerprint=fingerprint(lock['data_identity']),diagnostic_evidence=read(training/'diagnostics/diagnostics.json')['evidence']))
        result=raw/'results'/f'index{index}'
        for label in ('selected','epoch12'):
            if index!=3:
                boot=pd.read_csv(result/'comparisons'/label/'paired_event_bootstrap.csv',float_precision='round_trip')
                boots.append(boot.assign(family=entry['family'],checkpoint_policy=label,source_index=index))
            versions=[(entry['mode'],entry['run_id'],False)]+([('ON',entry['run_id']+'_reused_ON',True)] if index<3 else [])
            for mode,run_id,reused in versions:
                directory=result/'evaluation'/run_id/label
                path=directory/'predictions.csv.gz'
                columns=set(IDENTITY_KEYS+['split','units','status','prediction','nll','crps','brier','covered68','covered95',
                    'seed','model_family','absolute_amplitude_mode','comparison_kind',
                    'covered_mean_sigma','covered_mean_2sigma','width95','predictive_sigma','pit','q025','q975','q16','q84',
                    'error','exceedance_probability','linear_mixture_mean_mps2','linear_mixture_median_mps2',
                    'mdn_logits','mdn_weights','mdn_mu','mdn_sigma'])
                frame=pd.read_csv(path,usecols=lambda col:col in columns,float_precision='round_trip',
                                  dtype={'dataset_id':str,'event_id':str,'station_id':str})
                pop=population_audit(frame)
                condition=export_tag_check(frame,cfg['model_family'],mode,reused)
                identity=frame[IDENTITY_KEYS].sort_values(PAIR_KEYS).reset_index(drop=True)
                if reference is None:reference=identity
                require(identity.equals(reference),'Prediction identity mismatch across exports')
                cp=read(directory/'checkpoint_identity.json');epoch=cp.get('epoch',cp.get('internal_epoch'))
                require(epoch==12 if label=='epoch12' else 0<epoch<=12,'Wrong checkpoint epoch')
                require(cp['updates']==epoch*212,'Checkpoint updates differ')
                if not reused:
                    best=curve.sort_values(['validation_selection','epoch']).iloc[0]
                    require(epoch==(int(best.epoch) if label=='selected' else 12),'Selected epoch differs from curve')
                    require(cp['world_size']==16 and cp['a01_code_sha256']==source['a01_code_sha256'],'Checkpoint metadata differs')
                tag=dict(family=entry['family'],mode=mode,checkpoint_policy=label,epoch=epoch,seed=42,source_index=index,
                         run_id=run_id,reused_old_ON=reused,split='validation')
                scores=[]
                for role,subset in subsets(frame).items():
                    for geometry in ('normal','random'):
                        for time in (1,3,5,10,20):
                            part=subset[subset.geometry_protocol.eq(geometry)&subset.elapsed_time.eq(time)]
                            record=summarize(part) if len(part) else dict(status='N/A',events=0,valid_targets=0)
                            scores.append(dict(target_group=role,geometry_protocol=geometry,elapsed_time=time,**record))
                calculated=pd.DataFrame(scores)
                stored_path=directory/'scores.csv' if not reused else result/'comparisons'/label/'ON_scores.csv'
                stored=pd.read_csv(stored_path,float_precision='round_trip')
                keys=['target_group','geometry_protocol','elapsed_time']
                merged=calculated.merge(stored,on=keys,suffixes=('_recalculated','_saved'),validate='one_to_one')
                require(len(merged)==len(calculated)==len(stored)==60,'Score cell set differs')
                differences={name:float(np.nanmax(np.abs(merged[name+'_recalculated']-merged[name+'_saved'])))
                             for name in ('mae','rmse','nll','crps','brier','covered68','covered95','predictive_sigma')}
                require(all(value<2e-9 for value in differences.values()),'Saved score table differs: '+str(differences))
                mdn=mdn_check(frame)
                model_checks.append(dict(**tag,source=str(path.relative_to(raw)),population=pop,
                    condition_tags=condition,
                    checkpoint_sha256=cp.get('file_sha256',cp.get('checkpoint_sha256')),checkpoint_metadata_source='authenticated HPC export; weights excluded from tar',
                    score_max_abs_errors=differences,mdn_recalculation=mdn,status='PASS'))
                cells.append(calculated.assign(**tag))
                for role,subset in subsets(frame).items():
                    part=calculated[calculated.target_group.eq(role)]
                    pooled=summarize(subset)
                    record=dict(**tag,target_group=role,targets=len(subset),events=subset[['dataset_id','event_id']].drop_duplicates().shape[0],
                        decisions=subset[DECISION_KEYS].drop_duplicates().shape[0],supported_cells=int(part.valid_targets.gt(0).sum()),
                        equal_cell_rmse_of_mse=float(np.sqrt((part.rmse**2).mean())))
                    record.update({'equal_cell_'+name:float(part[name].mean()) for name in MEANS})
                    record.update({'pooled_'+name:pooled[name] for name in ('mae','rmse','r2','prediction_on_truth_slope')})
                    summaries.append(record)
                if not reused and label=='selected':
                    primary=next(row for row in summaries[::-1] if row['run_id']==run_id and row['target_group']=='noninput')
                    require(abs(primary['equal_cell_mae']-cp['best_selection_score'])<2e-9,'Selection endpoint differs from checkpoint')
                subset=frame[frame.target_role.ne('observed_input')].copy()
                subset['abs_error']=(subset.prediction-subset.truth).abs();subset['squared_error']=(subset.prediction-subset.truth)**2
                event=subset.groupby(DECISION_KEYS).agg(targets=('truth','size'),abs_error_sum=('abs_error','sum'),
                    squared_error_sum=('squared_error','sum'),nll_sum=('nll','sum'),crps_sum=('crps','sum'),
                    brier_sum=('brier','sum'),covered95_sum=('covered95','sum')).reset_index()
                event_cells.append(event.assign(**tag))
                fields_path=directory/'fields.csv.gz'
                if fields_path.exists():
                    fields=pd.read_csv(fields_path,float_precision='round_trip',dtype={'dataset_id':str,'event_id':str})
                    valid=fields[fields.status.eq('supported')]
                    require(np.max(np.abs(valid.field_mse-valid.level_mse-valid.shape_mse))<1e-10,'Field decomposition differs')
                    table=field_summary(fields)
                    field_cells.append(table.assign(**tag))
                    for role,part in table.groupby('target_group'):
                        row=dict(**tag,target_group=role,supported_cells=int(part.fields_eligible.gt(0).sum()),
                                 fields_attempted=int(part.fields_attempted.sum()),fields_eligible=int(part.fields_eligible.sum()),
                                 eligible_targets=int(part.eligible_targets.sum()))
                        for name in ('level_mse','shape_mse','field_mse','range_error','range_ratio','pairwise_delta_mae','equal_distance_delta_mae'):
                            row['equal_cell_'+name]=part[name+'_field_mean'].mean()
                        field_averages.append(row)
                frames_read+=1
                print('PREDICTION_VERIFIED '+str(frames_read)+'/16 '+entry['family']+' '+mode+' '+label+' epoch='+str(epoch),flush=True)
                del frame,identity,subset
    require(frames_read==16,'Incomplete export set')
    train=pd.DataFrame(train_audits);a,b=train[train['index'].eq(3)].iloc[0],train[train['index'].eq(4)].iloc[0]
    for key in ('common_initial_state_sha256','initial_encoder_sha256','full_initial_state_sha256','sampling_fingerprint','data_identity_fingerprint'):
        require(a[key]==b[key],'DiTing matched pair differs: '+key)
    csv(out/'training_audit_summary.csv',train);csv(out/'training_curves.csv',pd.concat(curves,ignore_index=True))
    summary=pd.DataFrame(summaries);csv(out/'model_summary.csv',summary)
    csv(out/'validation_by_time_geometry_role.csv',pd.concat(cells,ignore_index=True))
    csv(out/'field_by_time_geometry_role.csv',pd.concat(field_cells,ignore_index=True))
    csv(out/'field_summary.csv',pd.DataFrame(field_averages))
    events=pd.concat(event_cells,ignore_index=True);csv(out/'event_cells_noninput.csv.gz',events)
    boot=pd.concat(boots,ignore_index=True);csv(out/'paired_event_bootstrap.csv',boot)
    # Independently reproduce every primary noninput MAE estimate and CI using
    # saved per-event sums; no rerunning model inference or choosing epochs.
    paired=[];bootstrap_checks=[]
    for (family,label),part in summary.groupby(['family','checkpoint_policy']):
        for role in GROUPS:
            group=part[part.target_group.eq(role)].set_index('mode');on,off=group.loc['ON'],group.loc['OFF']
            interval=boot[boot.family.eq(family)&boot.checkpoint_policy.eq(label)&boot.target_group.eq(role)&boot.metric.eq('abs_error')].iloc[0]
            delta=off.equal_cell_mae-on.equal_cell_mae
            require(abs(delta-interval.estimate)<2e-10,'Paired primary estimate differs')
            paired.append(dict(family=family,checkpoint_policy=label,target_group=role,ON_epoch=on.epoch,OFF_epoch=off.epoch,
                ON_mae=on.equal_cell_mae,OFF_mae=off.equal_cell_mae,delta_OFF_minus_ON=delta,
                relative_increase_percent=100*delta/on.equal_cell_mae,ci_low=interval.ci_low,ci_high=interval.ci_high,
                events=interval.events,targets=interval.targets,supported_cells=interval.supported_cells,
                missing_cells=interval.missing_cells,interval_status=interval.status,
                draws=interval.draws,single_seed=True,split='validation'))
        group=events[events.family.eq(family)&events.checkpoint_policy.eq(label)]
        index=['dataset_id','event_id'];columns=['geometry_protocol','elapsed_time']
        on=group[group['mode'].eq('ON')].set_index(DECISION_KEYS)
        off=group[group['mode'].eq('OFF')].set_index(DECISION_KEYS)
        require(on.index.equals(off.index) and on.targets.equals(off.targets),'Event cell populations differ')
        delta=(off.abs_error_sum-on.abs_error_sum).reset_index(name='delta')
        delta['targets']=on.targets.to_numpy()
        numerator=delta.pivot(index=index,columns=columns,values='delta').fillna(0).to_numpy()
        denominator=delta.pivot(index=index,columns=columns,values='targets').fillna(0).to_numpy()
        rng=np.random.default_rng(20261007);sampled=[]
        for start in range(0,2000,100):
            ids=rng.integers(0,len(numerator),size=(min(100,2000-start),len(numerator)))
            sampled.extend((numerator[ids].sum(1)/denominator[ids].sum(1)).mean(1).tolist())
        actual=np.quantile(sampled,[.025,.975]);stored=boot[boot.family.eq(family)&boot.checkpoint_policy.eq(label)&boot.target_group.eq('noninput')&boot.metric.eq('abs_error')].iloc[0]
        require(np.max(np.abs(actual-[stored.ci_low,stored.ci_high]))<2e-10,'Paired bootstrap CI differs')
        bootstrap_checks.append(dict(family=family,checkpoint_policy=label,ci_low=float(actual[0]),ci_high=float(actual[1]),draws=2000,status='PASS'))
    csv(out/'paired_mae_summary.csv',pd.DataFrame(paired));diagnostics(raw,out);paired_spatial_tables(raw,out);matched_nested_table(raw,out)
    # source_seal JSONs are intentionally assessed separately from the archive manifest.
    evidence=dict(status='PASS',archive_sha256=transport['archive_sha256'],archive_members=transport['regular_members'],
        archive_manifest_members=transport['manifest_members'],all_archive_members_verified=True,source_core_sha256=source['a01_code_sha256'],
        local_core_files_verified=len(source['a01_files_sha256']),prediction_exports=frames_read,paired_comparisons=8,
        original_training_source=dict(inventory_sha256=original['code_sha256'],manifest_files=len(original['files_sha256']),
            local_files_verified=len(original_present),local_dependency_files_unavailable=len(original_missing),
            limitation='Unavailable dtbench source checked as saved inventory SHA only; no dependency download or model load'),
        identical_production_population=read(raw/'results/index4/paired_input_checks/selected.json')['strict_population_audit'],
        source_seal_checks=dict(same_original_bytes=sum(row['original_bytes_match'] for row in seal_checks),
            redacted_or_reserialized_JSONs=sum(not row['original_bytes_match'] for row in seal_checks),
            note='Original unredacted plan/weights are absent. Manifest verifies exported JSONs, not reconstruction of private originals.'),
        bootstrap_reproduction=bootstrap_checks,prediction_checks=model_checks,held_out_test_accessed=False,
        weights_or_waveforms_loaded=False,split='validation',training_seeds=[42],
        jobs=read_jobs(raw/'jobs.tsv'))
    write(out/'evidence_checks.json',evidence)
    print('SUMMARY_COMPLETE '+str(out),flush=True)


def read_jobs(path):
    return [dict(stage=stage,indices=indices,job_id=job) for stage,indices,job in
            (line.split('\t') for line in path.read_text().splitlines() if line)]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.source,args.output)
