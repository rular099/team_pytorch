#!/usr/bin/env python
"""Collect completed run metadata and validate common initialization/budget pairing."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fe01.config import fingerprint,sha256,write_json

MAIN={'diting_pretrained_frozen','team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen'}

def collect(runs,destination,allow_incomplete=False):
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()): raise FileExistsError(destination)
    records=[];initials={};budgets={};population=None
    for run in map(Path,runs):
        cfg=json.loads((run/'resolved_config.json').read_text())
        lock=json.loads((run/'protocol.lock.json').read_text())
        interface=json.loads((run/'model_interface_audit.json').read_text())
        current=fingerprint(dict(cohorts=lock['effective_config']['audited_cohorts'],
            normalization=lock['effective_config']['target_normalization'],population=lock['validation_population_sha256'],
            stage=cfg.get('stage'),window=cfg['window']['protocol'],code=lock['code_sha256']))
        if population is not None and current!=population: raise ValueError('Run cohort/statistics/protocol/code differs')
        population=current
        curves=pd.read_csv(run/'training_curves.csv')
        if curves.epoch.duplicated().any(): raise ValueError('Duplicate committed epoch counter')
        seed=str(cfg['seed']);family=cfg['model_family']
        if family in initials.setdefault(seed,{}): raise ValueError('Duplicate family/seed')
        initials[seed][family]=dict(state=interface['common_initial_state_sha256'],structure=interface['common_structure_sha256'])
        budget=fingerprint(curves[['epoch','updates','global_batch','planned_samples','actual_samples','dropped_for_equal_update_budget']].to_dict('records'))
        if seed in budgets and budgets[seed]!=budget: raise ValueError('Optimizer update/sample exposure budget differs')
        budgets[seed]=budget
        provenance=run/'evaluation_fixed_val/provenance.json'
        evaluation=json.loads(provenance.read_text()) if provenance.is_file() else {}
        records.append(dict(run_id=cfg['run_id'],family=family,seed=int(seed),stage=cfg.get('stage'),
            config_sha256=fingerprint(cfg),lock_sha256=sha256(run/'protocol.lock.json'),
            checkpoint_epoch=evaluation.get('checkpoint_epoch'),checkpoint_sha256=evaluation.get('checkpoint_sha256'),
            updates=int(curves.updates.iloc[-1]),epochs=int(curves.epoch.iloc[-1]),
            train_wall_seconds=float(curves.elapsed_seconds.sum()),
            validation_evaluation_status='present' if evaluation else 'NOT_PROVIDED'))
    for seed,models in initials.items():
        if not allow_incomplete and not MAIN.issubset(models): raise ValueError('Missing primary family at seed '+seed)
        if len({v['state'] for v in models.values()})!=1 or len({v['structure'] for v in models.values()})!=1:
            raise ValueError('Common downstream initialization differs at seed '+seed)
    destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(destination/'runs_manifest.csv',index=False)
    write_json(destination/'downstream_initial_state_fingerprints.json',initials)
    write_json(destination/'runset_identity.json',dict(population_protocol_sha256=population,
        budget_fingerprints_by_seed=budgets,initialization='PASS',complete_primary_matrix=not allow_incomplete))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args();collect(a.runs,a.output,a.allow_incomplete)
