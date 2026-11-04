#!/usr/bin/env python3
"""Render reviewed records; only a coordinator may aggregate global memory."""
import argparse
from pathlib import Path
import sys
from ai_memory_lib import (MEMORY, TRACK, atomic_write, digest, generated, git, global_views,
                           json_text, load, lock, root_path, track_views, utcnow, validate_record,
                           evidence_available, preserve_history, handoff_registry, handoff_provenance)


def plan(root, args):
    mem = root / MEMORY
    registry = load(mem / 'registry.json')['track_ids']
    config = handoff_registry(root)
    if args.coordinator and config:
        actual = git(root, 'symbolic-ref', '--short', 'HEAD').strip()
        if actual not in config['coordinator_branches']:
            raise ValueError('Only the designated coordinator branch may aggregate global memory')
    writes, warnings = {}, []
    if args.track:
        if args.track not in registry or not TRACK.fullmatch(args.track):
            raise ValueError('Unknown track')
        if args.source or args.bootstrap:
            raise ValueError('--source/--bootstrap require --coordinator')
        directory = mem / 'tracks' / args.track
        record = validate_record(load(Path(args.record) if args.record else directory / 'record.json'))
        if record['track_id'] != args.track:
            raise ValueError('Record belongs to a different track')
        previous = validate_record(load(directory / 'record.json'))
        preserve_history(previous, record)
        if args.track in config.get('tracks', {}):
            handoff_provenance(root, record, config)
            git(root, 'merge-base', '--is-ancestor', previous['commit_sha'], record['commit_sha'])
        branch = git(root, 'symbolic-ref', '--short', 'HEAD').strip()
        if branch != record['branch']:
            raise ValueError('Track update must run on its recorded branch; use an independent worktree')
        git(root, 'merge-base', '--is-ancestor', record['commit_sha'], 'HEAD')
        for evidence in record['evidence_paths']:
            evidence_available(root, evidence, load(mem / 'audit/remote_refs.json'))
        writes[directory / 'record.json'] = json_text(record)
        for name, body in track_views(record).items():
            path = directory / name
            writes[path] = generated(path.read_text() if path.exists() else '', args.track + ' ' + name[:-3], body)
        return writes, warnings
    if args.record:
        raise ValueError('--record is for a single --track update')
    if args.bootstrap and (mem / 'snapshot.json').exists():
        raise ValueError('Bootstrap already exists; coordinator must not rewrite owned track files')
    sources = {}
    for value in args.source:
        track, ref = value.split('=', 1)
        if track not in registry or track in sources:
            raise ValueError('Unknown or duplicated source track: ' + track)
        sources[track] = ref
    records, sync = [], {}
    for track in registry:
        observed = mem / 'observations' / (track + '.json')
        fallback = observed if observed.exists() else mem / 'tracks' / track / 'record.json'
        record = validate_record(load(fallback))
        state, origin, source_sha = 'retained_snapshot', str(fallback.relative_to(root)), None
        if track in sources:
            origin = sources[track]
            try:
                source_sha = git(root, 'rev-parse', '--verify', origin + '^{commit}').strip()
                import json
                candidate = validate_record(json.loads(git(root, 'show', source_sha + ':' + str(MEMORY / 'tracks' / track / 'record.json'))))
                if candidate['track_id'] != track:
                    raise ValueError('Source record has a different track ID')
                if track in config.get('tracks', {}):
                    handoff_provenance(root, candidate, config)
                symbolic = git(root, 'rev-parse', '--symbolic-full-name', origin).strip()
                if symbolic.startswith('refs/heads/'):
                    source_branch = symbolic[len('refs/heads/'):]
                elif symbolic.startswith('refs/remotes/'):
                    source_branch = symbolic[len('refs/remotes/'):].split('/', 1)[1]
                else:
                    source_branch = None  # An explicit immutable commit is also supported.
                if source_branch and candidate['branch'] != source_branch:
                    raise ValueError('Source branch does not own this track record')
                git(root, 'merge-base', '--is-ancestor', candidate['commit_sha'], source_sha)
                preserve_history(record, candidate)
                for evidence in candidate['evidence_paths']:
                    evidence_available(root, evidence, load(mem / 'audit/remote_refs.json'))
            except ValueError as exc:
                state = 'source_unavailable_retained'
                warnings.append(track + ': ' + str(exc) + '; previous verified state retained')
            else:
                record, state = candidate, 'source_imported'
        if args.bootstrap:
            state = 'bootstrap_audited'
            for name, body in track_views(record).items():
                path = mem / 'tracks' / track / name
                writes[path] = generated(path.read_text() if path.exists() else '', track + ' ' + name[:-3], body)
        records.append(record)
        writes[observed] = json_text(record)
        sync[track] = {'state': state, 'source': origin, 'source_commit': source_sha,
                       'record_sha256': digest(record)}
    # Imported snapshots are validated before any write; incomplete sync never erases a line.
    ids = [r['track_id'] for r in records]
    if ids != registry:
        raise ValueError('Registry/record mismatch')
    for collection, key in (('decisions', 'decision_id'), ('experiments', 'experiment_id')):
        values = [item[key] for r in records for item in r[collection]]
        if len(set(values)) != len(values):
            raise ValueError('Duplicate ' + key)
    snapshot = {'metadata': {'generated_at': utcnow(),
                            'generated_from_commit': git(root, 'rev-parse', 'HEAD').strip(),
                            'coordinator_branch': git(root, 'symbolic-ref', '--short', 'HEAD').strip()},
                'tracks': records, 'sync': sync}
    writes[mem / 'snapshot.json'] = json_text(snapshot)
    for name, body in global_views(snapshot).items():
        path = mem / name
        writes[path] = generated(path.read_text() if path.exists() else '', name[:-3], body)
    return writes, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    role = parser.add_mutually_exclusive_group(required=True)
    role.add_argument('--track')
    role.add_argument('--coordinator', action='store_true')
    parser.add_argument('--record', help='Human-authored complete record; no metrics inference')
    parser.add_argument('--source', action='append', default=[], metavar='TRACK=REF', help='Explicit fetched ref, read using git show')
    parser.add_argument('--bootstrap', action='store_true', help='One-time coordinator initialization')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = root_path(args.root)
    try:
        if args.track and not TRACK.fullmatch(args.track):
            raise ValueError('Invalid track ID')
        if args.dry_run:
            writes, warnings = plan(root, args)
        else:
            with lock(root, args.track or 'coordinator'):
                writes, warnings = plan(root, args)
                for path, content in writes.items():
                    atomic_write(path, content)
        for warning in warnings:
            print('WARN ' + warning)
        print(('Planned' if args.dry_run else 'Updated') + ' {} memory files; no scientific conclusions inferred'.format(len(writes)))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print('ERROR ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
