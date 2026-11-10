"""Development machine only: build the committed external compute-submit bundle."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile
from support import ROOT, MAIN, EXECUTION, digest, require, write


def build(output):
    require(not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip(),
            'Commit only task files; build from a clean worktree')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    top = 'v4_compute_submit_' + commit[:12]
    paths = ['scripts/fe01_v4_compute_submit', 'tests/test_fe01_v4_compute_submit.py',
             'docs/ai/FE01_ON_EVAL_RUNBOOK.md', 'reports/fe01_v4_compute_submit_20261110']
    files = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / paths[0]).rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    marker = dict(commit=commit, main_commit=MAIN, main_execution_sha256=EXECUTION, files_sha256=files,
                  scope='Shell-only login submission; compute-only verification/scheduler dispatch; original logical tasks unchanged')
    raw = output / 'source.tar'
    with raw.open('xb') as f:
        subprocess.run(['git', '-C', str(ROOT), 'archive', '--format=tar', '--prefix=' + top + '/', commit, *paths],
                       stdout=f, check=True)
    archive = output / ('fe01_v4_compute_submit_' + commit[:12] + '.tar.gz')
    with tarfile.open(raw, 'r:') as src, tarfile.open(archive, 'w:gz') as dst:
        for m in src:
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
        payload = json.dumps(marker, indent=2, allow_nan=False).encode()
        m = tarfile.TarInfo(top + '/COMPUTE_SUBMIT_RELEASE.json')
        m.size = len(payload); m.mode = 0o444
        dst.addfile(m, io.BytesIO(payload))
    raw.unlink()
    (output / (archive.name + '.sha256')).write_text(digest(archive) + '  ' + archive.name + '\n')
    write(output / 'SOURCE_RELEASE.json', dict(marker, archive=archive.name, archive_sha256=digest(archive),
                                              top_directory=top, hpc_status='NOT_SUBMITTED'))
    print(str(archive))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    build(p.parse_args().output)
