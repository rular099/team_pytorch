"""Verify/extract the supplied review archive without running models or cluster jobs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ARCHIVE_SHA = '7a5c19ad82d45c8c3a40a00a00b29856f4fd8720634fdbdf371e350f4bc32add'
BASE = '99d9b1c4bb5c47fc26125df7eea2648917e90462'
MAIN = '50672e131d876a604432b283162d5b0189d40cf0'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(archive, output):
    archive, output = Path(archive), Path(output)
    require(not output.exists(), 'Preserve existing review: choose a new output directory')
    require(digest(archive) == ARCHIVE_SHA, 'Unexpected archive SHA')
    require(Path(str(archive) + '.sha256').read_text().split()[0] == ARCHIVE_SHA, 'SHA sidecar differs')
    with gzip.open(archive, 'rb') as stream:
        while stream.read(4 * 1024 * 1024):
            pass  # Read the footer too: independently verify gzip integrity.
    output.mkdir(parents=True)
    files = {}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(len({m.name for m in members}) == len(members), 'Duplicate tar names')
        for member in members:
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and '..' not in name.parts
                    and name.parts[0] == 'fe01_on_eval_v4_review', 'Unsafe member')
            require(member.isdir() or member.isfile(), 'Links/special members prohibited')
            if member.isdir():
                continue
            relative = str(PurePosixPath(*name.parts[1:]))
            dest = output / 'raw' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, dest.open('xb') as target:
                shutil.copyfileobj(source, target)
            files[relative] = digest(dest)
    raw = output / 'raw'
    manifest = read(raw / 'REVIEW_MANIFEST.json')
    require(manifest['completion_scope'] == 'REQUIRED_SCOPE_COMPLETE' and not manifest['partial_reason'], 'Partial pack')
    require(manifest['files_sha256'] == {p: s for p, s in files.items() if p != 'REVIEW_MANIFEST.json'}, 'Review manifest differs')
    plan = read(raw / 'plan.json')
    require(plan['run_id'] == 'on_v4_20261025_inventory_fix', 'Wrong run')
    require(plan['source_identity']['result_commit'] == MAIN, 'Wrong main source')
    require(manifest['execution']['execution_sha256'] == plan['source_identity']['execution_sha256'], 'Execution identity differs')
    index = read(raw / 'full_artifact_index.json')
    original = {row['path']: row for row in index}
    require(len(index) == len(original), 'Duplicate original artifact paths')
    same, redacted = [], []
    for name, actual in files.items():
        if name.startswith(('analyze/', 'compare_existing/', 'inventory/', 'latency/')):
            row = original.get('stages/' + name)
            if row is None and name.endswith('/SEAL.json'):
                continue  # Original seals are not their own payloads in the full index.
            require(row is not None, 'Copied stage file absent from original index: ' + name)
            if actual == row['sha256']:
                same.append(name)
            else:
                require(name.endswith('.json'), 'Non-JSON stage payload differs from original SHA: ' + name)
                redacted.append(name)
    visuals = read(raw / 'full_visuals_index.json')
    require(len(visuals) == 1310 and len({(v['dataset_id'], v['event_id']) for v in visuals}) == 1310, 'Incomplete gallery index')
    by_case = {str(PurePosixPath(v['path']).parent.name): str(PurePosixPath(v['path']).parent) for v in visuals}
    checked_case_files = 0
    for name, actual in files.items():
        if name.startswith('cases/'):
            parts = PurePosixPath(name).parts
            expected = original[by_case[parts[1]] + '/' + str(PurePosixPath(*parts[2:]))]
            require(actual == expected['sha256'], 'Case payload differs from original index: ' + name)
            checked_case_files += 1
    source = manifest['execution']['files_sha256']
    repo = Path(__file__).resolve().parents[2]
    checked_sources, absent_sources = [], []
    for name, expected in source.items():
        p = repo / name
        if p.is_file():
            require(digest(p) == expected, 'Current method source differs from HPC pin: ' + name)
            checked_sources.append(name)
        else:
            absent_sources.append(name)
    status = read(raw / 'status_summary.json')
    stage_status = {}
    for row in status:
        stage_status.setdefault(row['stage'], Counter())[row['status']] += 1
    for stage in ['compare_existing', 'analyze', 'render_normal', 'latency']:
        require(stage_status[stage] == {'COMPLETE_SEAL_VERIFIED': len(plan['tasks'][stage])}, 'Required stage not verified in HPC snapshot: ' + stage)
    seals = {}
    for p in sorted(raw.rglob('SEAL.json')):
        v = read(p)
        seals[str(p.relative_to(raw))] = dict(status=v['status'], stage=v.get('stage'), task=v.get('task'),
                included_payload_sha_matches=sum(digest(p.parent / n) == s for n, s in v['files_sha256'].items() if (p.parent / n).is_file()),
                declared_payloads=len(v['files_sha256']))
    result = dict(repository='rular099/team_pytorch', review_base_commit=BASE,
        review_branch='review/fe01-v4-results-20261115', sync_branch='exp/fe01-on-warning-map-evaluation',
        archive=dict(name=archive.name, bytes=archive.stat().st_size, sha256=ARCHIVE_SHA, sha_sidecar='PASS', gzip='PASS',
                     members=len(members), regular_files=len(files), manifest_payloads_verified=len(files)-1),
        completion_scope=manifest['completion_scope'], main_source_commit=MAIN,
        execution_sha256=manifest['execution']['execution_sha256'],
        original_index_records=len(index), copied_original_payload_sha_matches=len(same),
        copied_redacted_json_differences=redacted, selected_case_payloads_sha_verified=checked_case_files,
        stage_snapshot={k:dict(v) for k,v in stage_status.items()}, copied_seal_summary=seals,
        source_files_sha_verified=checked_sources, source_files_unavailable=absent_sources,
        limitations=['The review manifest verifies actual received bytes; original private JSON seals may differ after redaction.',
            'Full replay/render payloads and model weights remain on HPC. Full index/status are HPC evidence, not local full-payload verification.',
            'Pack RUNNING in its own status snapshot was captured inside packaging; supplied pack stdout subsequently reports exit0.',
            'No models, training, HPC jobs or held-out test were run during this review.'])
    with (output / 'verification.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['copied_seal_summary','source_files_sha_verified','copied_redacted_json_differences','stage_snapshot']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.archive, args.output)
