#!/usr/bin/env python3
"""Read-only AI entrypoint. No model import, fetch, file writes or task execution."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

from ai_memory_lib import (MEMORY, SHA, evidence_available, generated, git, handoff_provenance,
                           handoff_registry, is_ancestor, json_text, load, root_path,
                           track_views, validate_record)


def memory_only(paths):
    return all(p.startswith(('docs/ai/', 'tools/ai_memory', 'tests/test_ai_memory',
                              'tests/test_ai_handoff')) or
               p in {'AGENTS.md', 'SESSION_SUMMARY.md', 'tools/ai_handoff.py'} for p in paths)


def branch_name(root):
    try:
        return git(root, 'symbolic-ref', '--short', 'HEAD').strip()
    except ValueError:
        return None


def target_head(root, branch):
    for ref in ('refs/heads/' + branch, 'refs/remotes/origin/' + branch):
        try:
            return git(root, 'rev-parse', '--verify', ref + '^{commit}').strip()
        except ValueError:
            continue
    return None


def source_head(root, ref, branch):
    sha = git(root, 'rev-parse', '--verify', ref + '^{commit}').strip()
    symbolic = git(root, 'rev-parse', '--symbolic-full-name', ref).strip()
    allowed = {'refs/heads/' + branch, 'refs/remotes/origin/' + branch}
    if symbolic and symbolic not in allowed:
        raise ValueError('Requested source branch does not own the selected track')
    if not symbolic and not SHA.fullmatch(ref):
        raise ValueError('Use an owning branch or a full immutable commit')
    head = target_head(root, branch)
    if not symbolic and head and not is_ancestor(root, sha, head):
        raise ValueError('Immutable source is not on the selected track branch')
    return sha


def recover(root, track=None, role='codex', source=None):
    root = root.resolve()
    config = handoff_registry(root)
    if not config:
        raise ValueError('Persistent handoff is not installed; read docs/ai/START_HERE.md')
    actual = branch_name(root)
    owners = {route['branch']: key for key, route in config['tracks'].items()}
    inferred = owners.get(actual)
    if track is None:
        track = inferred
    if track not in config['tracks']:
        raise ValueError('Cannot infer track from branch; specify FE01 or FE02. Never infer from directory name.')
    if role == 'codex' and inferred and inferred != track:
        raise ValueError('Codex track/working branch mismatch; use the correct independent worktree')
    route = config['tracks'][track]
    directory = Path(route['directory'])
    warnings = []
    ref = None
    loaded_from = 'track_worktree'
    working_tree_source = False
    if source:
        ref = source_head(root, source, route['branch'])
        loaded_from = 'git_ref'
    elif actual != route['branch']:
        ref = target_head(root, route['branch'])
        loaded_from = 'git_ref'
    if actual == route['branch'] and source is None:
        working_tree_source = True
        record = validate_record(load(root / directory / 'record.json'))
        reader = lambda name: (root / directory / name).read_text(encoding='utf-8')
        view_head = git(root, 'rev-parse', 'HEAD').strip()
    else:
        present = False
        if ref:
            try:
                git(root, 'cat-file', '-e', ref + ':' + str(directory / 'record.json'))
                present = True
            except ValueError:
                pass
        if present:
            record = validate_record(json.loads(git(root, 'show', ref + ':' + str(directory / 'record.json'))))
            reader = lambda name: git(root, 'show', ref + ':' + str(directory / name))
            view_head = ref
        elif source:
            raise ValueError('Requested source has no track record; deployment is required')
        else:
            working_tree_source = True
            record = validate_record(load(root / directory / 'record.json'))
            reader = lambda name: (root / directory / name).read_text(encoding='utf-8')
            view_head = git(root, 'rev-parse', 'HEAD').strip()
            loaded_from = 'coordinator_snapshot'
            warnings.append('Target experiment branch has not adopted the entrypoint; using the pinned coordinator snapshot, not claiming deployment.')
    if record['track_id'] != track:
        raise ValueError('Directory/record track ID mismatch')
    h = handoff_provenance(root, record, config)
    for name, body in track_views(record).items():
        text = reader(name)
        if generated(text, track + ' ' + name[:-3], body) != text:
            raise ValueError('Stale generated view: ' + str(directory / name))
    state_dirty = False
    if working_tree_source:
        state_dirty = bool(git(root, 'diff', '--name-only', 'HEAD', '--', str(directory)).strip() or
                           git(root, 'ls-files', '--others', '--exclude-standard', '--', str(directory)).strip())
        if state_dirty:
            warnings.append('State documents include uncommitted changes; state_document_commit is null, not an invented committed version.')
    remote = load(root / MEMORY / 'audit/remote_refs.json')
    evidence_checks = [dict(evidence, available=evidence_available(root, evidence, remote))
                       for evidence in record['evidence_paths']]
    head = target_head(root, route['branch'])
    needs_review = False
    if head and head != record['commit_sha']:
        if not is_ancestor(root, record['commit_sha'], head):
            raise ValueError('Track branch diverged from the recorded scientific commit')
        changed = git(root, 'diff', '--name-only', record['commit_sha'], head).splitlines()
        if not memory_only(changed):
            needs_review = True
            warnings.append('New scientific/evidence changes since recorded commit: review status before implementation; retained facts are dated.')
    if actual == route['branch']:
        dirty = git(root, 'status', '--porcelain=v1', '-uno').splitlines()
        if dirty:
            warnings.append('Working tree has tracked uncommitted changes; preserve and inspect them before any implementation.')
    return dict(repository=h['repository'], role=role, track_id=track,
                actual_branch=actual, actual_head=git(root, 'rev-parse', 'HEAD').strip(),
                branch=record['branch'], scientific_commit=record['commit_sha'],
                state_document_commit=None if state_dirty else view_head,
                state_base_commit=view_head, state_document_dirty=state_dirty, loaded_from=loaded_from,
                default_action=h['default_action'], automatic_actions=['read', 'verify', 'report'],
                implementation_authorization='current user task only; NEXT_TASK grants none',
                role_scope='research review and AI-HANDOFF' if role == 'chatgpt' else 'verify and implement only the current authorized task',
                goal=h['goal'], current_state=record['current_state'], summary=record['summary'],
                last_verified_at=record['last_verified_at'], evidence_as_of=record['evidence_as_of'],
                verified_results=record['verified_results'], limitations=record['limitations'],
                pending_tasks=record['pending_tasks'], decisions=record['decisions'],
                next_task=h['next_task'], last_session=h['last_session'],
                documents=[str(directory / name) for name in ('STATUS.md', 'NEXT_TASK.md', 'DECISIONS.md')],
                evidence=evidence_checks, needs_status_review=needs_review, warnings=warnings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--track', choices=['FE01', 'FE02'])
    parser.add_argument('--role', choices=['codex', 'chatgpt'], default='codex')
    parser.add_argument('--source', help='Explicit owning branch or immutable full commit; read using git show')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        value = recover(root_path(args.root), args.track, args.role, args.source)
        if args.json:
            print(json_text(value), end='')
        else:
            for key in ('repository', 'role', 'track_id', 'actual_branch', 'actual_head',
                        'branch', 'scientific_commit', 'state_document_commit', 'loaded_from', 'default_action'):
                print('{}: {}'.format(key, value[key]))
            print('\n目标：' + value['goal'])
            print('\n已核验：' + value['summary'])
            print('\n下一步提案（未授权执行）：' + value['next_task']['objective'])
            print('\n未完成/恢复：\n' + '\n'.join('- ' + x for x in value['last_session']['unfinished'] + value['last_session']['resume_steps']))
            for message in value['warnings']:
                print('WARN ' + message)
            print('\n只读接手完成；没有写文件、执行训练、提交HPC作业或执行NEXT_TASK。')
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('ERROR ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
