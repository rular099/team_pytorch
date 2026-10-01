#!/usr/bin/env python
"""Explicit small CPU synthetic smoke, separate from production training/evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tests.fe01_helpers import config,synthetic_archive
from fe01.config import write_json
from fe01.engine import audit,causal_audit,load_checkpoint,train,export_evaluation
from fe01.data import split_catalog
from fe01.model import build_model,model_audit
from fe01.analysis import fit_train_reference,summarize_run,training_histograms
from fe01.replay import replay,select_cases
from scripts.fe01.render_maps import render
from scripts.fe01.prepare_runs import base_config
from fe01.windows import CAPABILITIES
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output).resolve()
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True);synthetic_archive(root,events=9)
    cfg=config(root);cfg['training'].update(epochs=1,max_updates=1,global_batch=3,microbatch=1)
    write_json(root/'source_config.json',cfg)
    audit(cfg,root/'audit');train(cfg,root/'audit')
    run=root/cfg['run_id'];effective=copy.deepcopy(cfg)
    load_checkpoint(effective,run/'best.pth','cpu')
    export_evaluation(copy.deepcopy(cfg),run/'best.pth',run/'val')
    training_histograms(run,run/'actual_training_time_histograms.csv')
    fit_train_reference(effective,run/'reference.json')
    summarize_run(run/'val',run/'analysis',run/'reference.json')
    select_cases(effective,run/'case_selection_manifest.json',count=3)
    replay(copy.deepcopy(cfg),run/'best.pth',run/'case_selection_manifest.json',run/'replay',start=1,end=2,spacing_km=100)
    render(run/'replay',run/'reference.json',run/'maps')
    # Use the actual RT55 1000-dimensional downstream without optimizer updates.
    interfaces={};causal={}
    for family in ['team_original_scratch','phasenet_pretrained_frozen','eqt_pretrained_frozen','amplitude_only','coords_only']:
        full=config(root,family);full['model_params']=base_config()['model_params']
        model=build_model(full,'cpu')
        interfaces[family]=model_audit(model)
        causal[family]=causal_audit(model,full,split_catalog(full,'train').iloc[0],'cpu')
        del model
    common={item['common_initial_state_sha256'] for item in interfaces.values()}
    if len(common)!=1: raise AssertionError('Actual RT55-width common initialization differs')
    write_json(root/'full_width_interfaces.json',interfaces)
    write_json(root/'full_width_causal.json',causal)
    write_json(root/'SMOKE_SCOPE.json',dict(status='PASS',data='generated synthetic; 9 events, 6 stations; no Japan production reads',
        training='one CPU update, TEAM frontend/common width20; not formal training',
        full_width='RT55 common width1000 synthetic forward + initialization/causal checks for five available families',
        picking_weights='real offline STEAD v2',diting_1200m='NOT_RUN: actual weights absent',
        hpc='NOT_SUBMITTED',replay='3 validation cases, seconds1..2, actual grid queries and GIF'))
    print('LOCAL_SMOKE_PASS',root)

if __name__=='__main__': main()
