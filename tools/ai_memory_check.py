#!/usr/bin/env python3
"""Validate memory schema, provenance, generated views, scope and write namespaces."""
import argparse
import datetime as dt
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
from ai_memory_lib import (BEGIN, END, MEMORY, SHA, digest, evidence_available, generated, git,
                           global_views, json_text, load, merge_risk, resource_errors,
                           root_path, safe_path, timestamp, track_views, validate_record)


def check(root, max_age_days=45):
    mem = root / MEMORY
    errors, warnings = [], []
    registry = load(mem / 'registry.json')['track_ids']
    if len(registry) != len(set(registry)):
        errors.append('Duplicate track_id in registry')
    remote = load(mem / 'audit/remote_refs.json')
    remote_heads = {b['branch']: b['commit_sha'] for b in remote['branches']}
    snapshot = load(mem / 'snapshot.json')
    records = snapshot['tracks']
    if [r['track_id'] for r in records] != registry:
        errors.append('Snapshot does not match registry')
    commit = snapshot['metadata']['generated_from_commit']
    git(root, 'cat-file', '-e', commit + '^{commit}')
    # A memory-only descendant is expected after committing the rendered snapshot.
    since = git(root, 'diff', '--name-only', commit, 'HEAD').splitlines()
    if any(not (p.startswith('docs/ai/') or p.startswith('tools/ai_memory') or p.startswith('tests/test_ai_memory') or p == 'AGENTS.md') for p in since):
        warnings.append('Coordinator HEAD contains newer non-memory changes; review snapshot freshness')
    all_records = []
    for track in registry:
        directory = mem / 'tracks' / track
        own = validate_record(load(directory / 'record.json'))
        observed = validate_record(load(mem / 'observations' / (track + '.json')))
        all_records.extend([own, observed])
        if own['track_id'] != track or observed['track_id'] != track:
            errors.append('Directory/track ID mismatch: ' + track)
        selected = next((r for r in records if r['track_id'] == track), None)
        if observed != selected or snapshot['sync'][track]['record_sha256'] != digest(observed):
            errors.append('Contradictory global/observation snapshot: ' + track)
        for name, body in track_views(own).items():
            path = directory / name
            if not path.exists() or generated(path.read_text(), track + ' ' + name[:-3], body) != path.read_text():
                errors.append('Missing/stale generated track view: ' + str(path.relative_to(root)))
    for collection, key in (('experiments', 'experiment_id'), ('decisions', 'decision_id')):
        seen = set()
        for r in records:
            for item in r[collection]:
                if item[key] in seen:
                    errors.append('Duplicate ' + key + ': ' + item[key])
                seen.add(item[key])
    checked = set()
    for r in all_records:
        sha = r['commit_sha']
        if not SHA.fullmatch(sha):
            errors.append('Invalid commit ' + sha)
        try:
            git(root, 'cat-file', '-e', sha + '^{commit}')
        except ValueError:
            if remote_heads.get(r['branch']) != sha:
                errors.append('Unverified commit: ' + sha)
        try:
            local = git(root, 'rev-parse', '--verify', 'refs/heads/' + r['branch']).strip()
            if local != sha:
                warnings.append(r['track_id'] + ': local ref differs from verified source; do not treat unsynced as incomplete')
        except ValueError:
            warnings.append(r['track_id'] + ': branch absent locally; recorded source retained')
        age = (dt.datetime.now(dt.timezone.utc) - timestamp(r['last_verified_at'])).days
        if age > max_age_days:
            warnings.append(r['track_id'] + ': verification older than {} days'.format(max_age_days))
        if age < -1:
            errors.append('Future last_verified_at: ' + r['track_id'])
        for evidence in r['evidence_paths']:
            key = (evidence['commit_sha'], evidence['path'])
            if key in checked:
                continue
            checked.add(key)
            try:
                evidence_available(root, evidence, remote)
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
    for name, body in global_views(snapshot).items():
        path = mem / name
        if not path.exists() or generated(path.read_text(), name[:-3], body) != path.read_text():
            errors.append('Missing/stale generated global view: ' + name)
    for name in ('PROJECT_MEMORY.md', 'CHAT_HANDOFF.md', 'CURRENT_STATUS.md', 'DECISION_LOG.md',
                 'EXPERIMENT_INDEX.md', 'README.md', 'HISTORY_IMPORT_TEMPLATE.md', 'PARALLEL_WORKFLOW.md'):
        if not (mem / name).is_file():
            errors.append('Missing required Markdown: ' + name)
    for path in mem.rglob('*.md'):
        if 'audit/sources' in str(path):
            continue  # byte-exact source snapshots keep their original branch-relative links
        for link in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)', path.read_text()):
            if urlsplit(link).scheme or link.startswith('#'):
                continue
            target = (path.parent / unquote(link.split('#')[0])).resolve()
            if root not in target.parents or not target.exists():
                errors.append('Broken internal link: ' + str(path.relative_to(root)) + ' -> ' + link)
    return sorted(set(errors)), sorted(set(warnings))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--max-age-days', type=int, default=45)
    parser.add_argument('--strict-warnings', action='store_true')
    parser.add_argument('--resources', help='JSON list of track_id/write_paths/readonly_paths; read only')
    parser.add_argument('--scope', help='Reject committed/uncommitted changes outside tracks/<ID>/')
    parser.add_argument('--base', default='HEAD')
    parser.add_argument('--merge-risk', nargs=2, metavar=('LEFT_REF', 'RIGHT_REF'))
    args = parser.parse_args(argv)
    try:
        root = root_path(args.root)
        if args.merge_risk:
            print(json_text(merge_risk(root, *args.merge_risk)))
            return 0
        errors, warnings = check(root, args.max_age_days)
        if args.resources:
            errors.extend(resource_errors(load(args.resources)))
        if args.scope:
            base = git(root, 'rev-parse', '--verify', args.base + '^{commit}').strip()
            paths = set(git(root, 'diff', '--name-only', base).splitlines())
            paths.update(git(root, 'ls-files', '--others', '--exclude-standard').splitlines())
            prefix = str(MEMORY / 'tracks' / args.scope) + '/'
            errors.extend('Outside track ownership: ' + p for p in sorted(paths) if not p.startswith(prefix))
        for message in warnings:
            print('WARN ' + message)
        for message in errors:
            print('ERROR ' + message)
        print('Memory check: {} error(s), {} warning(s)'.format(len(errors), len(warnings)))
        return int(bool(errors) or (args.strict_warnings and bool(warnings)))
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        print('ERROR ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
