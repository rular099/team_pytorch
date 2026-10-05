import importlib.util
import json
import os
import subprocess
import sys
import importlib.metadata
from pathlib import Path

from fe01.config import fingerprint, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = 'b2756825190f93034ece436fab778e8a69adb41c'
TRAIN_CODE = '3e537ff6693971d92b139e517f370b998237bd8b29989194e9ddcc06cc9dcfea'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        value = os.path.expandvars(os.path.expanduser(value))
        require('$' not in value and '__FILL' not in value, 'Unresolved setting: ' + value)
    return value


def new_output(path):
    path = Path(path).resolve()
    require(not path.exists() or not any(path.iterdir()), 'Output is nonempty: ' + str(path))
    protected = os.environ.get('FE01_TRAIN_OUTPUT_ROOT')
    require(not protected or Path(protected).resolve() not in [path, *path.parents], 'Output overlaps original training results')
    path.mkdir(parents=True, exist_ok=True)
    return path


def training_identity():
    spec = importlib.util.find_spec('dtbench')
    require(spec is not None, 'dtbench source snapshot missing')
    dependency = Path(next(iter(spec.submodule_search_locations)))
    paths = list((ROOT / 'fe01').glob('*.py'))
    paths += [ROOT / n for n in ('gemini_models.py', 'gemini_util_light.py', 'train_light.py')]
    paths += list((ROOT / 'tools').rglob('*.py')) + list((ROOT / 'diting/config').glob('*.yml'))
    files = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    files.update({'dtbench/' + str(p.relative_to(dependency)): sha256(p) for p in dependency.rglob('*.py')})
    require(fingerprint(files) == TRAIN_CODE, 'Frozen training code identity mismatch; do not rewrite old lock')
    return dict(code_sha256=TRAIN_CODE, files_sha256=files)


def evaluation_identity():
    paths = []
    for name in ('fe01_review', 'scripts/fe01_review', 'configs/fe01_review'):
        paths += [p for p in (ROOT / name).rglob('*') if p.is_file()
                  and p.suffix in ('.py', '.sh', '.sbatch', '.json') and '__pycache__' not in p.parts]
    files = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    packaged=ROOT/'FE01_REVIEW_SOURCE_IDENTITY.json'
    if packaged.is_file():
        declared=read_json(packaged)['evaluation']
        require(declared['evaluation_module_sha']==fingerprint(files),'Packaged evaluation code SHA differs; do not reuse stale verification')
        return dict(evaluation_source_sha=declared['evaluation_source_sha'],evaluation_source_dirty=declared.get('evaluation_source_dirty','unknown'),
            evaluation_module_sha=fingerprint(files),files_sha256=files,legacy_loader_sha256=sha256(ROOT/'eval_checkpoint.py'))
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        commit = read_json(ROOT / 'FE01_REVIEW_SOURCE_IDENTITY.json').get('commit', 'unknown') if (ROOT / 'FE01_REVIEW_SOURCE_IDENTITY.json').exists() else 'unknown'
    try:
        dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--','fe01_review','scripts/fe01_review','configs/fe01_review'],stderr=subprocess.DEVNULL).strip())
    except (OSError,subprocess.CalledProcessError):dirty='unknown'
    return dict(evaluation_source_sha=commit,evaluation_source_dirty=dirty,evaluation_module_sha=fingerprint(files),files_sha256=files,
        legacy_loader_sha256=sha256(ROOT/'eval_checkpoint.py'))


def provenance(**values):
    from . import TASK_ID, VERSION
    versions={}
    for name in ('torch','numpy','pandas','scipy','h5py','seisbench'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='unavailable'
    torch=sys.modules.get('torch')
    runtime=dict(python=sys.version,versions=versions)
    if torch is not None:
        runtime.update(torch_version=torch.__version__,rocm=torch.version.hip,
            accelerator_available=torch.cuda.is_available(),accelerator_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
    return dict(task_id=TASK_ID, analysis_version=VERSION, base_commit=BASE, split='val',
                training_code_sha=TRAIN_CODE,runtime=runtime, **evaluation_identity(), **values)


def seal(output):
    output = Path(output)
    files = [p for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.sha256']
    (output / 'artifact_manifest.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(output)}\n' for p in files))


def manifest_entry(path, run_id):
    data = read_json(path)
    matches = [r for r in data['models'] if r['run_id'] == run_id]
    require(len(matches) == 1, 'Model run ID must identify one entry')
    return expand(matches[0])
