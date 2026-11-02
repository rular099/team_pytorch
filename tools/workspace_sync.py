#!/usr/bin/env python3
"""Read-only sync checks and external-artifact verification; never add, push or delete."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

DEFAULT_POLICY = Path(__file__).resolve().parents[1] / 'docs/ai/sync/SYNC_POLICY.json'
SECRET_RULES = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'literal_credential': re.compile(
        r'(?im)^\s*(?:export\s+)?["\']?(?:API_KEY|ACCESS_TOKEN|SECRET_KEY|AWS_SECRET_ACCESS_KEY|'
        r'HINET_PASSWORD|GITHUB_TOKEN|PASSWORD)["\']?\s*[:=]\s*["\']?([A-Za-z0-9+/=_-]{12,})["\']?\s*,?\s*$'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(repo, *args):
    process = subprocess.run(['git', '-C', str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise ValueError(process.stderr.decode(errors='replace').strip())
    return process.stdout


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe relative path')
    return path


def inspect_content(path, raw, policy):
    """Do not echo matching credential values."""
    errors = []
    relative = safe_relative(path)
    if any(part in policy['private_path_parts'] for part in relative.parts):
        errors.append('private_path')
    if relative.name in {'.env', 'credentials', 'credentials.json'} or relative.name.endswith('.private.env'):
        errors.append('private_environment')
    suffix = relative.suffix.lower()
    if suffix not in policy['text_extensions'] + policy['binary_extensions'] and relative.name not in policy['extensionless_files']:
        errors.append('non_allowlisted_type')
    if len(raw) > policy['max_file_bytes']:
        errors.append('size_limit_exceeded')
    if suffix in policy['binary_extensions']:
        if suffix != '.png' or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            errors.append('invalid_binary_format')
    else:
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            errors.append('non_utf8_text')
        else:
            for name, rule in SECRET_RULES.items():
                if rule.search(text):
                    errors.append('credential_pattern:' + name)
    return errors


def check_staged(repo, policy):
    entries = git(repo, 'diff', '--cached', '--name-status', '-z', '--no-renames').decode().split('\0')
    errors, count, total = [], 0, 0
    i = 0
    while i < len(entries) and entries[i]:
        status, path = entries[i:i+2]
        i += 2
        if status == 'D':
            errors.append(path + ': deletion requires a separate reviewed change')
            continue
        mode = git(repo, 'ls-files', '--stage', '--', path).decode().split()[0]
        if mode not in {'100644', '100755'}:
            errors.append(path + ': symlink/submodule is not a portable file export')
            continue
        raw = git(repo, 'show', ':' + path)
        count += 1
        total += len(raw)
        errors.extend(path + ': ' + e for e in inspect_content(path, raw, policy))
    if total > policy['max_batch_bytes']:
        errors.append('batch_size_exceeded')
    return {'files': count, 'bytes': total, 'errors': errors,
            'note': 'No upload performed; policy checks do not certify all scientific claims.'}


def inventory(repo, policy):
    names = git(repo, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    rows = []
    for name in filter(None, names):
        safe_relative(name)
        path = repo / name
        if path.is_symlink():
            rows.append({'path': name, 'classification': 'external_symlink_not_followed'})
        elif path.is_file():
            size = path.stat().st_size
            if size > policy['max_file_bytes']:
                errors = ['size_limit_exceeded']
            else:
                errors = inspect_content(name, path.read_bytes(), policy)
            rows.append({'path': name, 'bytes': size, 'classification': 'local_only' if errors else 'review_candidate',
                         'reasons': errors})
    return rows


def verify_artifacts(root, manifest):
    """A manifest is not a backup: only actual bytes under a supplied root can verify."""
    root = root.resolve()
    results = []
    for record in manifest['files']:
        value = record['workspace_relative_path']
        safe_relative(value)
        path = root / value
        resolved = path.resolve()
        if root not in resolved.parents or path.is_symlink():
            state = 'EXTERNAL_PATH_NOT_VERIFIED'
        elif not path.is_file():
            state = 'MISSING'
        elif path.stat().st_size != record['bytes']:
            state = 'SIZE_MISMATCH'
        elif sha256(path) != record['sha256']:
            state = 'HASH_MISMATCH'
        else:
            state = 'VERIFIED'
        results.append({'path': value, 'state': state})
    for record in manifest.get('unavailable_external_dependencies', []):
        results.append({'path': record['logical_path'], 'state': 'NO_SOURCE_HASH_AVAILABLE'})
    complete = bool(results) and all(r['state'] == 'VERIFIED' for r in results)
    return {'complete': complete, 'results': results,
            'note': 'Verification only covers this manifest; no copy, cleanup or source deletion performed.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inventory', 'check-staged'):
        sub = commands.add_parser(name)
        sub.add_argument('--repo', type=Path, default=Path.cwd())
    verify = commands.add_parser('verify-artifacts')
    verify.add_argument('--root', type=Path, required=True)
    verify.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify-artifacts':
            result = verify_artifacts(args.root, read_json(args.manifest))
            status = 0 if result['complete'] else 2
        elif args.command == 'inventory':
            result = inventory(args.repo.resolve(), read_json(args.policy))
            status = 0
        else:
            result = check_staged(args.repo.resolve(), read_json(args.policy))
            status = int(bool(result['errors']))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return status
    except (ValueError, OSError, KeyError) as exc:
        print('ERROR ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
