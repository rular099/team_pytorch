#!/usr/bin/env python
"""Collect real FE02 evidence, certifying paired initialization/encoder/budget."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
from fe01.config import fingerprint, write_json
from fe01.analysis import training_histograms
from fe02.analysis import compare


def collect(root, output, stage='formal', seeds=(42,)):
    root=Path(root);output=Path(output)
    if output.exists(): raise FileExistsError(output)
    evaluations=[];records=[];paired={};encoder=None;code=None;population=None
    for seed in seeds:
        for variant in ('R0','A','B','C','M'):
            run=root/f'fe02__{stage}__{variant}__seed{seed}'
            cfg=json.loads((run/'resolved_config.json').read_text())
            lock=json.loads((run/'protocol.lock.json').read_text())
            interface=json.loads((run/'model_interface_audit.json').read_text())
            if cfg['variant_id']!=variant or cfg['seed']!=seed or lock['status']!='AUDIT_PASS':
                raise ValueError('Run identity/audit differs')
            curves=pd.read_csv(run/'training_curves.csv')
            if list(curves.epoch)!=list(range(1,cfg['training']['epochs']+1)):
                raise ValueError('Missing/duplicate committed epochs')
            current=fingerprint(curves[['epoch','updates','global_batch','planned_samples','actual_samples','dropped_for_equal_update_budget']].to_dict('records'))
            shared=(current,interface['unchanged_downstream_initial_sha256'],interface['adapter_initial_sha256'])
            if seed in paired and paired[seed]!=shared: raise ValueError('Budget/shared initial values differ')
            paired[seed]=shared
            for label,expected,actual in [('encoder',encoder,lock['encoder_checkpoint_sha256']),
                                          ('source',code,lock['code_sha256']),
                                          ('population',population,lock['validation_population_sha256'])]:
                if expected is not None and expected!=actual: raise ValueError('Different '+label)
            encoder=lock['encoder_checkpoint_sha256'];code=lock['code_sha256'];population=lock['validation_population_sha256']
            freeze=json.loads((root/'audits'/cfg['run_id']/'freeze_gradient_audit.json').read_text())
            if freeze['status']!='PASS': raise ValueError('Freeze audit failed')
            evaluations.append(run/'evaluation_fixed_val')
            records.append(dict(seed=seed,variant_id=variant,run_id=cfg['run_id'],epochs=int(curves.epoch.iloc[-1]),
                updates=int(curves.updates.iloc[-1]),total_parameters=interface['total_parameters'],
                trainable_parameters=interface['trainable_parameters'],
                audit_active_gradient_parameters=freeze['active_gradient_parameters'],
                audit_nonzero_gradient_parameters=freeze['nonzero_gradient_parameters'],
                total_epoch_wall_seconds=float(curves.elapsed_seconds.sum()),encoder_checkpoint_sha256=encoder))
    compare(evaluations,output,seeds)
    pd.DataFrame(records).to_csv(output/'runs_manifest.csv',index=False)
    for record in records:
        training_histograms(root/record['run_id'],output/(record['run_id']+'__training_histograms.csv'))
    write_json(output/'runset_identity.json',dict(status='PASS',encoder_checkpoint_sha256=encoder,
        code_sha256=code,validation_population_sha256=population,shared_initial_and_budget_by_seed=paired,
        evidence_scope='real recorded artifacts; Slurm State/ExitCode require separate sacct export'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['formal','pilot'],default='formal');p.add_argument('--seeds',nargs='+',type=int,default=[42])
    a=p.parse_args();collect(a.root,a.output,a.stage,tuple(a.seeds))
