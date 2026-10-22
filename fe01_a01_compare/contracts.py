"""Frozen scientific contract and separate execution identity."""
from pathlib import Path
from . import BASE_COMMIT, TASK_ID
from .io import fingerprint, read_json, require, sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/fe01_a01_compare'
FAMILIES = {'DiTing': 'diting_pretrained_frozen', 'PhaseNet': 'phasenet_pretrained_frozen',
            'EQT': 'eqt_pretrained_frozen', 'TEAM': 'team_original_scratch'}
EPOCHS = {'DiTing': 12, 'PhaseNet': 12, 'EQT': 10, 'TEAM': 11}
ON = 'physical_scale_keep_duration'
TIMES = (1, 3, 5, 10, 20)
KEYS = ['dataset_id', 'event_id', 'elapsed_time', 'station_id', 'geometry_protocol']
EVENT = ['dataset_id', 'event_id']
DECISION = EVENT + ['geometry_protocol', 'elapsed_time']


def contract():
    c = read_json(CONFIG / 'DITING_ON_V4_RESEARCH_CONTRACT.json')
    require(c['base_commit'] == BASE_COMMIT and c['task_id'] == TASK_ID, 'Wrong contract identity')
    require({m['family']: m['selected_epoch'] for m in c['models']} == EPOCHS, 'Wrong frozen epochs')
    require(c['split'] == 'validation' and all(m['mode'] == ON and m['seed'] == 42 for m in c['models']), 'Wrong scope')
    return c


def source_identity():
    paths = []
    for d in ('fe01_a01_compare', 'scripts/fe01_a01_compare', 'configs/fe01_a01_compare'):
        paths += [p for p in (ROOT / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}
    result=dict(files_sha256=files, execution_sha256=fingerprint(files),
                contract_sha256=fingerprint(contract()), base_commit=BASE_COMMIT, task_id=TASK_ID)
    result['frozen_core_verification']=verify_frozen_core()
    marker=ROOT/'FE01_ON_EVAL_SOURCE_RELEASE.json'
    if marker.is_file():
        declared=read_json(marker)
        require(declared['execution']['execution_sha256']==result['execution_sha256'],'Source archive content changed')
        result.update(result_commit=declared['commit'],source_kind='committed source archive')
    else:
        import subprocess
        try:
            commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            commit='UNKNOWN_LOCAL_SNAPSHOT'
        result.update(result_commit=commit,source_kind='worktree snapshot; content SHA authoritative')
    return result


def verify_frozen_core():
    original = read_json(ROOT / 'reports/fe01_a01_hpc_results_20261021/source_metadata/A01_REVIEW_MANIFEST.json')['source']
    require(fingerprint(original['original_training']['files_sha256']) ==
            '3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea','Original115 manifest changed')
    require(fingerprint(original['a01_files_sha256']) ==
            '3e1e4398b01d724f793f88b19be2a2c03ca67a71e881d876c173a8fbc5b8177f','A01 core manifest changed')
    checked, absent = [], []
    tables = [original['original_training']['files_sha256'], original['a01_files_sha256']]
    for table in tables:
        for name, digest in table.items():
            if name.startswith('dtbench/'):
                absent.append(name)
                continue
            require(sha256(ROOT / name) == digest, 'Frozen core changed: ' + name)
            checked.append(name)
    # Old A01 tests/docs are also immutable, despite not affecting model tensors.
    for name, digest in original.get('a01_tests_docs_sha256', {}).items():
        require(sha256(ROOT / name) == digest, 'Frozen A01 evidence source changed: ' + name)
    return dict(status='PASS_LOCAL_FILES', checked=len(set(checked)),
                dtbench_not_locally_verified=absent, runtime_dependency_gate='required before model use')
