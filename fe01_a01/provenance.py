"""Old training identity stays immutable; A01 has a separate source lock."""
import json
import subprocess
from pathlib import Path

from fe01.config import fingerprint, sha256
from fe01_review.provenance import training_identity, require, read_json, write_json
from . import ROOT, TASK_ID, BASE_COMMIT


def source_identity():
    old = training_identity()
    files = {}
    for directory in ('fe01_a01', 'scripts/fe01_a01', 'configs/fe01_a01'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.json', '.sh', '.sbatch'):
                files[str(path.relative_to(ROOT))] = sha256(path)
    # Evaluation helpers are dependencies, not part of the original 115 files.
    for path in sorted((ROOT / 'fe01_review').glob('*.py')):
        files[str(path.relative_to(ROOT))] = sha256(path)
    result = dict(task_id=TASK_ID, base_commit=BASE_COMMIT,
                  original_training=old, a01_files_sha256=files,
                  a01_code_sha256=fingerprint(files))
    evidence={}
    for pattern in ('tests/test_fe01_a01_*.py','docs/ai/FE01_A01_*.md'):
        for path in sorted(ROOT.glob(pattern)):
            evidence[str(path.relative_to(ROOT))]=sha256(path)
    result['a01_tests_docs_sha256']=evidence
    result['a01_tests_docs_manifest_sha256']=fingerprint(evidence)
    packaged = ROOT / 'FE01_A01_SOURCE_IDENTITY.json'
    if packaged.exists():
        declared = read_json(packaged)
        require(declared['a01_code_sha256'] == result['a01_code_sha256'],
                'A01 source package changed; invalidate gates, use a new run ID')
        require(declared['original_training'] == old, 'Original training snapshot changed')
        require(declared['a01_tests_docs_sha256']==evidence, 'Packaged A01 tests/docs changed')
        result.update(source_commit=declared['commit'],source_dirty=False,source_kind='committed source archive')
    else:
        commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
        dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',
            'fe01_a01','scripts/fe01_a01','configs/fe01_a01','tests/test_fe01_a01_*','docs/ai/FE01_A01_*']).strip())
        result.update(source_commit=commit,source_dirty=dirty,source_kind='worktree; code content SHA is authoritative')
    return result


def new_output(path):
    path = Path(path).resolve()
    require(not path.exists() or not any(path.iterdir()), 'Output exists; choose a NEW run ID: ' + str(path))
    import os
    protected = os.environ.get('FE01_TRAIN_OUTPUT_ROOT')
    require(not protected or Path(protected).resolve() not in (path, *path.parents),
            'A01 cannot write inside original training results')
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate(cfg, production=True):
    from fe01.config import validate as validate_fe01
    from . import ON, OFF, FAMILIES
    validate_fe01(cfg)
    import os
    original=os.environ.get('FE01_TRAIN_OUTPUT_ROOT')
    if original:
        out=Path(cfg['output_root']).resolve();old=Path(original).resolve()
        require(old not in (out,*out.parents),'Output overlaps original results')
    require(cfg['model_family'] in FAMILIES, 'Unknown/unsupported A01 family')
    require(cfg['absolute_amplitude_mode'] in (ON, OFF), 'Unknown absolute amplitude policy')
    p = cfg['model_params']
    require(p.get('use_amplitude_info') is True and not p.get('disable_waveform_scale'),
            'A01 retains the duration branch; do not disable amplitude projection')
    for flag in ('use_target_temporal_pooling', 'use_pga_temporal_residual',
                 'pga_layerwise_refinement', 'use_vs30', 'use_rope', 'pga_use_event_context',
                 'use_pga_anchor_transfer','use_pga_anchor_residual_transport',
                 'use_rt60_contrast_readout','use_rt61_wave_geometry_adapter','station_distinctive_adapter'):
        require(not p.get(flag, False), 'Untraced bypass: ' + flag)
    require(p.get('dpk_checkpoint_path') is None, 'DPK bypass forbidden in main A01')
    for key in ('station_token_weight_mode', 'temporal_token_weight_mode'):
        require(p.get(key, 'none') == 'none', 'Energy/token weighting bypass: ' + key)
    require(p.get('station_residual_mode', 'none') == 'none', 'Untraced station residual')
    require(p.get('station_context_mode') == 'off', 'Main A01 keeps original station context')
    require(cfg['seed'] == 42 and cfg['sampling_seed'] == 42, 'First round is seed42 only')
    if production:
        t = cfg['training']
        require(t['epochs'] == 12 and t['max_updates'] is None and t['global_batch'] == 128
                and t['microbatch'] == 8 and cfg['realtime']['draws_per_event'] == 3,
                'Formal A01 budget must match original FE01')
        require(not cfg.get('audit_limits'), 'Formal A01 cannot shrink the cohort')
    return cfg
