"""Build a small committed operational release without copying the V4 source."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile
from fix_support import ROOT, ORIGINAL_COMMIT, ORIGINAL_EXECUTION, digest, write, require


def build(output):
    output = Path(output)
    require(not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip(),
            'Commit task files and use a clean worktree before release')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    paths = ['scripts/fe01_v4_team_smoke_fix', 'docs/ai/FE01_V4_TEAM_SMOKE_FIX.md',
             'reports/fe01_v4_team_smoke_fix_20261102', 'reports/fe01_v4_team_smoke_shell_fix_20261105',
             'tests/test_fe01_v4_team_smoke_fix.py']
    output.mkdir(parents=True, exist_ok=False)
    top = 'v4_team_smoke_fix_' + commit[:12]
    raw = output / 'source.tar'
    with raw.open('xb') as f:
        subprocess.run(['git', '-C', str(ROOT), 'archive', '--format=tar', '--prefix=' + top + '/', commit, *paths],
                       stdout=f, check=True)
    files = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / paths[0]).rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    marker = dict(commit=commit, original_commit=ORIGINAL_COMMIT, original_execution_sha256=ORIGINAL_EXECUTION,
                  files_sha256=files, scope='TEAM model_smoke task 3 only; unchanged original runner and gates')
    archive = output / ('fe01_v4_team_smoke_fix_' + commit[:12] + '.tar.gz')
    with tarfile.open(raw, 'r:') as src, tarfile.open(archive, 'w:gz') as dst:
        for member in src:
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
        payload = json.dumps(marker, indent=2, allow_nan=False).encode()
        info = tarfile.TarInfo(top + '/TEAM_SMOKE_FIX_RELEASE.json')
        info.size = len(payload); info.mode = 0o444
        dst.addfile(info, io.BytesIO(payload))
    raw.unlink()  # This build's newly created intermediate, never an input artifact.
    (output / (archive.name + '.sha256')).write_text(digest(archive) + '  ' + archive.name + '\n')
    write(output / 'SOURCE_RELEASE.json', dict(**marker, archive=archive.name,
          archive_sha256=digest(archive), top_directory=top, hpc_repair_status='NOT_SUBMITTED'))
    print(str(archive))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    build(p.parse_args().output)
