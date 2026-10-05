"""CPU, one-checkpoint-at-a-time inventory. Unknown identity never becomes PASS."""
import gc
from pathlib import Path

from fe01.config import fingerprint, sha256
from .provenance import expand, read_json, require, write_json, training_identity


def load_cpu(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def structure(state):
    return fingerprint({k: [str(v.dtype), list(v.shape)] for k, v in sorted(state.items())})


def inspect_checkpoint(path, expected_epoch, kind='fe01', config_path=None, run_dir=None):
    path = Path(path)
    result = dict(path=str(path), expected_epoch=expected_epoch, status='BLOCKED',
                  reason='checkpoint missing', internal_epoch='unknown', checkpoint_sha256='unknown')
    for key in ('bytes','updates','best_epoch','checkpoint_format','state_structure_sha256','state_content_sha256',
        'source_config_sha256','config_sha256','training_source','encoder_source','task_id','parent_lock_sha256'):
        result[key]='unknown'
    result['config_file_sha256']=sha256(config_path) if config_path and Path(config_path).is_file() else 'unknown'
    if not path.is_file():
        return result
    result.update(bytes=path.stat().st_size, checkpoint_sha256=sha256(path))
    checkpoint = None
    try:
        checkpoint = load_cpu(path)
        require(isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint, 'Unrecognized checkpoint schema')
        result.update(internal_epoch=checkpoint.get('epoch', 'unknown'),
                      updates=checkpoint.get('updates', 'unknown'), best_epoch=checkpoint.get('best_epoch', 'unknown'),
                      checkpoint_format=checkpoint.get('checkpoint_format', 'fe01_full_state' if kind == 'fe01' else 'unknown'),
                      state_structure_sha256=structure(checkpoint['model_state_dict']), state_tensors=len(checkpoint['model_state_dict']),
                      source_config_sha256=checkpoint.get('source_config_sha256', 'unknown'),
                      training_source=checkpoint.get('runtime', {}).get('git_commit', 'unknown'),
                      encoder_source=checkpoint.get('encoder_source', 'unknown'), task_id=checkpoint.get('task_id', 'unknown'),
                      excluded_prefixes=checkpoint.get('excluded_prefixes', []))
        require(checkpoint.get('epoch') == expected_epoch, 'Internal epoch differs from frozen selected epoch')
        from fe01.model import state_fingerprint
        result['state_content_sha256']=state_fingerprint(checkpoint['model_state_dict'])
        if kind == 'fe01':
            require(config_path and Path(config_path).is_file(), 'Original resolved config missing')
            cfg = read_json(config_path)
            require(checkpoint.get('config') == cfg, 'Checkpoint config differs from original resolved config')
            lock_path = Path(run_dir or path.parent) / 'protocol.lock.json'
            lock = read_json(lock_path)
            require(lock['code_sha256'] == training_identity()['code_sha256'], 'Checkpoint parent code pin differs')
            require(checkpoint.get('source_config_sha256') == lock['config_sha256'], 'Checkpoint source config differs from lock')
            worlds = [world for world in (1, 2, 4, 8, 16) if fingerprint(dict(
                config=cfg, audit_lock=lock, world_size=world)) == checkpoint.get('identity')]
            require(len(worlds) == 1, 'Checkpoint identity does not authenticate config/lock/world')
            require(not checkpoint.get('rank_rng') or len(checkpoint['rank_rng']) == worlds[0], 'Checkpoint rank RNG/world mismatch')
            result['authenticated_world_size'] = worlds[0]
            audit=read_json(Path(run_dir or path.parent)/'model_interface_audit.json')
            initial=checkpoint.get('common_initial_state_sha256')
            require(initial==audit['common_initial_state_sha256'],'Common initial identity differs from original model audit')
            if expected_epoch==0:
                from fe01.model import common_state
                common={k:v for k,v in checkpoint['model_state_dict'].items() if not k.startswith('waveform_model.')}
                require(state_fingerprint(common)==initial,'Actual init common-state fingerprint differs')
            result['common_initial_state_sha256']=initial
            runtime_path=Path(run_dir or path.parent)/'runtime.json'
            runtime=read_json(runtime_path) if runtime_path.is_file() else {}
            result.update(config_sha256=fingerprint(cfg), parent_lock_sha256=sha256(lock_path),
                          training_source=checkpoint.get('runtime', {}).get('git_commit', runtime.get('git_commit', 'unknown')),
                          committed_journals_count=len(checkpoint.get('committed_journals', [])),
                          committed_journals_present=all((path.parent / p).is_file() for p in checkpoint.get('committed_journals', [])))
        else:
            require(config_path and Path(config_path).is_file(), 'Original legacy resolved config missing')
            cfg = read_json(config_path)
            if isinstance(cfg, list):
                require(len(cfg) == 1, 'Legacy config list must contain one resolved config')
                cfg = cfg[0]
            norm = cfg.get('training_params', {}).get('pga_target_normalization', {})
            require(norm.get('enabled') and isinstance(norm.get('mean'), (int, float)) and isinstance(norm.get('std'), (int, float)) and norm['std'] > 0, 'Legacy target normalization unresolved')
            result.update(config_sha256=fingerprint(cfg), original_normalization=norm,
                          strict_model_load='NOT_RUN: verification required')
            if kind == 'rt61':
                payload = checkpoint.get('rt61_reference')
                require(isinstance(payload, dict) and 'readout_state_dict' in payload, 'RT61 immutable reference payload missing')
                require(checkpoint.get('task_id') == '20260925-rt61-wave-geometry-residual-conditioning', 'RT61 task identity mismatch')
                parent = cfg['training_params']['rt61_parent']
                require(payload.get('schema_version')==2 and payload.get('parent_epoch')==parent['epoch'],'RT61 parent payload schema/epoch mismatch')
                from tools.rt61_wave_geometry import tensor_mapping_sha256
                require(tensor_mapping_sha256(payload['readout_state_dict'])==payload['readout_sha256'],'RT61 reference readout content SHA differs')
                for key in ('parent_checkpoint_sha256', 'readout_sha256'):
                    target = 'checkpoint_sha256' if key.startswith('parent_') else key
                    require(payload.get(key) == parent[target], 'RT61 parent/reference mismatch: ' + key)
                result['parent_identity'] = {k: v for k, v in payload.items() if k != 'readout_state_dict'}
        result.update(status='IDENTITY_PASS', reason='selected epoch and accessible metadata verified; forward gate still required')
    except Exception as exc:
        result['reason'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        del checkpoint
        gc.collect()
    return result


def inventory(manifest, output):
    data = expand(read_json(manifest))
    rows = []
    for model in data['models']:
        candidates=[]
        supplied=Path(model['checkpoint'])
        if model['kind']!='fe01' and supplied.is_dir():
            # Identify the preselected epoch by contents only, within its known
            # original experiment directory. Never rank candidates by loss.
            for path in sorted(supplied.glob('*.pth')):
                candidates.append(inspect_checkpoint(path,model['epoch'],model['kind'],model.get('config')))
            matching=[c for c in candidates if c['status']=='IDENTITY_PASS']
            if matching and len({c['state_content_sha256'] for c in matching})==1:
                model['checkpoint']=matching[0]['path']
        selected = inspect_checkpoint(model['checkpoint'], model['epoch'], model['kind'], model.get('config'), model.get('run_dir'))
        row = dict(run_id=model['run_id'], kind=model['kind'], selected=selected,
                   init=dict(status='NOT_PROVIDED'), last=dict(status='NOT_PROVIDED'))
        if candidates:row['candidates']=candidates
        if model['kind'] == 'fe01':
            for role, epoch in (('init', 0), ('last', 12)):
                path = Path(model['run_dir']) / (role + '.pth')
                row[role] = inspect_checkpoint(path, epoch, 'fe01', model['config'], model['run_dir'])
            csv = Path(model['run_dir']) / f"validation_epoch{model['epoch']}.csv.gz"
            row['selected_csv'] = dict(path=str(csv), sha256=sha256(csv) if csv.is_file() else 'unknown')
        rows.append(row)
        model['inventory_status']=selected['status']
        model['checkpoint_sha256']=selected['checkpoint_sha256']
        if model['kind']!='fe01' and selected['status']=='IDENTITY_PASS':
            encoder=Path(model.get('encoder',''))
            recorded=selected.get('encoder_source','unknown')
            if encoder.is_file() and isinstance(recorded,str) and str(encoder.resolve())==str(Path(recorded).resolve()):
                model['encoder_sha256']=sha256(encoder);model['encoder_source_recorded']=recorded
            else:model['inventory_status']='BLOCKED';row['selected']['status']='BLOCKED';row['selected']['reason']='Original external encoder missing'
        print('INVENTORY', model['run_id'], selected['status'], selected['reason'], flush=True)
    write_json(Path(output) / 'checkpoint_inventory.json', rows)
    write_json(Path(output) / 'frozen_checkpoint_manifest.json',data)
    return rows
