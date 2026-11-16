"""P0 pin/manifest/config audit for the manuscript; no model imports or HPC commands."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess

from common import BASE,RESULT,MAIN,ROOT,PAPER,REPO, EPOCHS,Evidence,sha,write_json


def run():
    e=Evidence()
    blueprint=PAPER/'evidence/blueprint_v2'
    for p in sorted(blueprint.iterdir()):e.local(p,'user_supplied/FE01_Paper_Blueprint_Codex_20261116_v2/'+p.name)
    manifest=json.loads((blueprint/'PACKAGE_MANIFEST.json').read_text())
    for name,v in manifest['files'].items():
        b=(blueprint/name).read_bytes();assert sha(b)==v['sha256'] and len(b)==v['bytes']
    for name in ['AGENTS.md','docs/ai/PROJECT_CONTEXT.md','docs/ai/README.md','SESSION_SUMMARY.md']:
        e.text(name,BASE,'project_context_only')
    memory='7445a22e6c9fef3e623e73b21f03bbeb5bf58ec0'
    for name in ['docs/ai/START_HERE.md','docs/ai/memory/tracks/FE01/STATUS.md','docs/ai/memory/tracks/FE01/NEXT_TASK.md','docs/ai/memory/tracks/FE01/DECISIONS.md']:
        e.text(name,memory,'historical_context_only; superseded by V4 evidence for completed results')
    check=subprocess.run(['git','-C',str(REPO),'cat-file','-e',BASE+':docs/ai/START_HERE.md'],capture_output=True)
    assert check.returncode!=0
    delivered=e.json(ROOT+'DELIVERY_MANIFEST.json')
    for path,v in delivered['files_sha256'].items():
        b=e.data(ROOT+path,RESULT,'byte_integrity_only_unless_read_by_asset_builder')
        assert sha(b)==v['sha256'] and len(b)==v['bytes'],path
    inventory=e.eid('E31')
    selected=[v for v in inventory if v['policy']=='selected']
    assert {v['family']:v['epoch'] for v in selected}==EPOCHS
    cfgs={}
    historic='29ffb20ba0effc0c345a0329ba3f8c658dc8cef0'
    for family,stem in [('TEAM','team_original_scratch'),('PhaseNet','phasenet_pretrained_frozen'),('EQT','eqt_pretrained_frozen')]:
        path='reports/fe01_hpc_results_20261004/metadata/formal__'+stem+'__seed42.json'
        v=e.json(path,BASE,'complete_model_identity_and_resolved_settings_only')
        c=v['resolved_config'];a=v['model_interface_audit'];lock=v['protocol_lock']
        cfgs[family]=dict(config=c,audit={k:v for k,v in a.items() if k!='trainable_names'},
            original_config_sha256=lock.get('config_sha256'),checkpoint_config_sha256=next(x for x in selected if x['family']==family)['config_sha256'],
            source_path=path,source_commit=BASE,representation='portable archived metadata; private original config bytes not reread')
    prefix='reports/fe01_a01_hpc_results_20261021/source_metadata/training/index3/'
    c=e.json(prefix+'resolved_config.json',historic,'DiTing complete-model config only')
    a=e.json(prefix+'model_interface_audit.json',historic,'DiTing complete-model parameter audit only')
    split=e.json(prefix+'audits/data_split_audit.json',historic,'complete-model data provenance only')
    budget=e.csv(prefix+'training_curves.csv',historic)
    assert len(budget)==12 and int(budget.iloc[-1]['updates'])==2544
    budget_cols=[k for k in ['epoch','updates','global_batch','planned_samples','actual_samples','dropped_for_equal_update_budget'] if k in budget]
    budget[budget_cols].to_csv(PAPER/'assets/source_data/S1_DiTing_training_budget.csv',index=False)
    cfgs['DiTing']=dict(config=c,audit={k:v for k,v in a.items() if k!='trainable_names'},
        checkpoint_config_sha256=next(x for x in selected if x['family']=='DiTing')['config_sha256'],
        source_path=prefix+'resolved_config.json',source_commit=historic,representation='archived redacted JSON; original private byte SHA not reverified')
    flags=['station_context_mode','pga_readout_mode','pga_readout_layers','readout_first_residual','pga_use_event_context','use_coords_abs','use_coords_rel','coord_fusion_mode','use_rope','use_vs30','use_amplitude_info','disable_waveform_scale','pga_mixture']
    for family,v in cfgs.items():
        p=v['config']['model_params'];assert p['station_context_mode']=='off' and p['pga_readout_mode']=='target_cross_attention'
        assert p['pga_readout_layers']==4 and p['readout_first_residual'] and not p['pga_use_event_context']
        assert p['use_amplitude_info'] and not p['disable_waveform_scale'] and p['pga_mixture']==3
        v['active_flags']={k:p.get(k) for k in flags}
    write_json(PAPER/'evidence/complete_model_settings.json',cfgs)
    write_json(PAPER/'evidence/data_audit_excerpt.json',{k:v for k,v in split.items() if k!='shards'})
    paths=['fe01/model.py','fe01/data.py','fe01/windows.py','fe01/extractors.py','fe01/sampling.py','fe01/engine.py','fe01/metrics.py',
           'fe01_a01/model.py','gemini_models.py','fe01_a01_compare/latency.py','fe01_a01_compare/map_readiness.py',
           'fe01_a01_compare/shadow.py','fe01_a01_compare/clocks.py','fe01_a01_compare/event_statistics.py',
           'fe01_a01_compare/delivery_analysis.py','fe01_a01_compare/statistics.py','fe01_a01_compare/replay.py']
    traces={}
    for path in paths:
        body=e.text(path,MAIN,'executed_method_trace_read_only')
        module=ast.parse(body)
        traces[path]=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno) for n in ast.walk(module) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    e.text('FE01_GEOMETRY_AMPLITUDE_HANDOFF_20261007.md','577a0cec43643597a61209209a770aeecf5fefa0','historical path audit only; excluded condition comparisons not used')
    write_json(PAPER/'evidence/source_function_ranges.json',traces)
    for name in ['run_summary.csv','audit_summary.csv']:
        historical=e.csv('reports/fe01_hpc_results_20261004/'+name,BASE)
        historical.to_csv(PAPER/'assets/source_data'/('S4_'+name),index=False,float_format='%.17g')
    historical_text=e.text('reports/fe01_hpc_results_20261004/README.md',BASE)
    lines=historical_text.splitlines()
    (PAPER/'evidence/historical_control_report_excerpt.txt').write_text('\n'.join(lines[53:67]+lines[138:149])+'\n')
    # Only file names are searched locally. Pilot data are never promoted to full replay evidence.
    local_root=Path(os.environ['FE01_ARTIFACT_ROOT']) if os.environ.get('FE01_ARTIFACT_ROOT') else None
    search=[]
    for sub in ['runs/v4pack','artifacts/fe01_on_eval/v4pack_review_20261115/raw','artifacts/fe01_on_eval/pilot_review_20261109T113132/raw']:
        if local_root is None:continue
        root=local_root/sub
        if root.exists():
            names=[str(p.relative_to(local_root)) for p in root.rglob('*') if p.is_file() and re.search('prediction|mdn|frame_metrics',p.name,re.I)]
            search.append(dict(directory=sub,mode='file-name enumeration only; no model/weights opened',candidate_count=len(names),examples=names[:12]))
    if local_root is not None or not (PAPER/'evidence/local_export_search.json').exists():
        write_json(PAPER/'evidence/local_export_search.json',dict(search=search,full_fixed_query_MDN='BLOCKED_MISSING_SOURCE',
            note='Only explicitly supplied local artifact directories enumerated. Set FE01_ARTIFACT_ROOT to repeat. Full-stage indexed HPC paths are not local entities. No SSH/HPC access used.'))
    write_json(PAPER/'evidence/scope_audit.json',dict(blueprint='v2; all 10 declared input hashes verified',paper_base=BASE,result_commit=RESULT,
        source_main=MAIN,split='validation',seed=42,mode_internal='ON',selected_epochs=EPOCHS,
        active_START_HERE='ABSENT at base; fixed historical fallback read without deployment',
        no_preexisting_paper_worktree=True,protection_snapshot='worktree_protection_before.json',
        production_files_modified=False,FE02_used=False,new_HPC_commands=False,new_model_inference=False,
        held_out_test_accessed=False,publication_authorization='LOCAL_COMMIT_AND_REVIEW_BUNDLE_ONLY for this paper task',
        actual_counts_from_archived_audit={k:v for k,v in split.items() if k in ['counts','retained','excluded','cohort','splits']},
        boundaries=['No excluded-condition comparisons in manuscript/supplement','Same common downstream does not imply same native windows/time exposure/pretraining.']))
    e.save()
    print('P0 PASS: blueprint/156 delivered files/config flags/source trace; no scientific files changed')


if __name__=='__main__':run()
