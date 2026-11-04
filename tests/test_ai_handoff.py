"""Read-only recovery of real pinned facts in disposable, synthetic Git namespaces.

Uses actual FE01/FE02 records and evidence objects; never checks out or executes
scientific programs. The test repository's branch heads are synthetic fixtures.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from ai_handoff import recover
from ai_memory_lib import (MEMORY, generated, git, handoff_registry, json_text, load,
                           track_views, utcnow, validate_record)


class PersistentHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='persistent-handoff-test-')
        self.parent = Path(self.temp.name)
        self.repo = self.parent / 'coordinator'
        self.repo.mkdir()
        self.config = handoff_registry(ROOT)
        self.coordinator = self.config['coordinator_branches'][0]
        git(self.repo, 'init', '-b', self.coordinator)
        git(self.repo, 'config', 'user.name', 'Synthetic handoff fixture')
        git(self.repo, 'config', 'user.email', 'fixture@example.invalid')
        common = Path(git(ROOT, 'rev-parse', '--git-common-dir').strip())
        if not common.is_absolute():
            common = ROOT / common
        (self.repo / '.git/objects/info/alternates').write_text(str((common / 'objects').resolve()) + '\n')
        shutil.copytree(ROOT / 'docs/ai', self.repo / 'docs/ai', ignore=shutil.ignore_patterns('__pycache__'))
        (self.repo / 'tools').mkdir()
        for name in ('ai_memory_lib.py', 'ai_memory_update.py', 'ai_memory_check.py', 'ai_handoff.py'):
            shutil.copyfile(ROOT / 'tools' / name, self.repo / 'tools' / name)
        shutil.copyfile(ROOT / 'AGENTS.md', self.repo / 'AGENTS.md')
        (self.repo / '.gitignore').write_text('__pycache__/\n*.pyc\n')
        git(self.repo, 'add', '.')
        tree = git(self.repo, 'write-tree').strip()
        parents = {git(ROOT, 'rev-parse', 'HEAD').strip()}
        parents.update(load(self.repo / MEMORY / 'tracks' / t / 'record.json')['commit_sha'] for t in ('FE01', 'FE02'))
        args = ['commit-tree', tree]
        for sha in sorted(parents):
            args += ['-p', sha]
        self.initial = git(self.repo, *args, '-m', 'SYNTHETIC namespace with real pinned handoff documents').strip()
        git(self.repo, 'update-ref', 'HEAD', self.initial)
        self.worktrees = {}
        for track, route in self.config['tracks'].items():
            path = self.parent / track
            git(self.repo, 'worktree', 'add', '-b', route['branch'], str(path), self.initial)
            self.worktrees[track] = path

    def tearDown(self):
        self.temp.cleanup()

    def run_tool(self, root, name, *args, expect=0):
        result = subprocess.run([sys.executable, str(root / 'tools' / name), '--root', str(root), *args],
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, expect, result.stdout + result.stderr)
        return result

    def document_bytes(self, root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
                if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}

    def scenario(self, track, role):
        root = self.worktrees[track] if role == 'codex' else self.repo
        before = self.document_bytes(root)
        args = ['--role', role, '--json']
        if role == 'chatgpt':
            args += ['--track', track, '--source', self.config['tracks'][track]['branch']]
        value = json.loads(self.run_tool(root, 'ai_handoff.py', *args).stdout)
        self.assertEqual(value['track_id'], track)
        self.assertEqual(value['branch'], self.config['tracks'][track]['branch'])
        self.assertEqual(value['default_action'], 'read_verify_report')
        self.assertEqual(value['automatic_actions'], ['read', 'verify', 'report'])
        self.assertEqual(value['next_task']['execution_authorization'], 'not_granted_by_this_file')
        self.assertTrue(value['goal'] and value['verified_results'] and value['last_session']['unfinished'])
        self.assertTrue(value['last_session']['resume_steps'] and value['decisions'])
        for filename in ('STATUS.md', 'NEXT_TASK.md', 'DECISIONS.md'):
            self.assertIn(str(MEMORY / 'tracks' / track / filename), value['documents'])
        if track == 'FE01':
            self.assertEqual(value['scientific_commit'], '577a0cec43643597a61209209a770aeecf5fefa0')
            self.assertIn('前端', value['goal'])
            self.assertIn('TEAM', value['summary'])
            self.assertTrue(any('站点' in x for x in value['limitations']))
        else:
            self.assertEqual(value['scientific_commit'], '618661f99afb67f5c23c9f1a3fd713a0a156e7b5')
            self.assertIn('R0/A/B/C/M', value['goal'])
            self.assertIn('12轮', value['summary'])
            self.assertNotIn('5/12轮', value['summary'])
            self.assertTrue(any('sacct' in x for x in value['limitations']))
        self.assertEqual(before, self.document_bytes(root))
        return value

    def test_new_fe01_codex_handoff(self):
        self.scenario('FE01', 'codex')

    def test_new_fe02_codex_handoff(self):
        self.scenario('FE02', 'codex')

    def test_new_fe01_chatgpt_handoff(self):
        self.scenario('FE01', 'chatgpt')

    def test_new_fe02_chatgpt_handoff(self):
        self.scenario('FE02', 'chatgpt')

    def update_track(self, track):
        root = self.worktrees[track]
        directory = root / MEMORY / 'tracks' / track
        # Synthetic result is a test-only document, not a model/HPC outcome.
        result_path = directory / 'SYNTHETIC_RESULT.md'
        result_path.write_text('# SYNTHETIC result ' + track + '\n')
        git(root, 'add', str(result_path.relative_to(root)))
        git(root, 'commit', '-qm', 'SYNTHETIC result ' + track)
        sha = git(root, 'rev-parse', 'HEAD').strip()
        record = load(directory / 'record.json')
        record.update(commit_sha=sha, last_verified_at=utcnow(), summary='SYNTHETIC updated ' + track)
        record['handoff']['last_session'].update(result_commit=sha, result_branch=record['branch'],
                                                  scope='evidence_review', task_id='SYNTHETIC-' + track)
        candidate = self.parent / (track + '-candidate.json')
        candidate.write_text(json_text(record))
        self.run_tool(root, 'ai_memory_update.py', '--track', track, '--record', str(candidate))
        git(root, 'add', str(MEMORY / 'tracks' / track))
        self.run_tool(root, 'ai_memory_check.py', '--scope', track, '--base', sha, '--staged-only')
        git(root, 'commit', '-qm', 'SYNTHETIC status ' + track)
        return sha

    def test_concurrent_worktrees_never_overwrite_other_track_or_globals(self):
        before = {t: self.document_bytes(p) for t, p in self.worktrees.items()}
        coordinator_before = self.document_bytes(self.repo)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(self.update_track, t) for t in ('FE01', 'FE02')]
            results = [job.result() for job in jobs]
        self.assertEqual(coordinator_before, self.document_bytes(self.repo))
        for own, root in self.worktrees.items():
            other = 'FE02' if own == 'FE01' else 'FE01'
            for path, raw in before[own].items():
                if not path.startswith(str(MEMORY / 'tracks' / own) + '/'):
                    self.assertEqual((root / path).read_bytes(), raw, path)
            self.assertNotIn('SYNTHETIC updated ' + own, (root / MEMORY / 'tracks' / other / 'STATUS.md').read_text())
        self.run_tool(self.repo, 'ai_memory_update.py', '--coordinator',
                      '--source', 'FE01=' + self.config['tracks']['FE01']['branch'],
                      '--source', 'FE02=' + self.config['tracks']['FE02']['branch'])
        snapshot = load(self.repo / MEMORY / 'snapshot.json')
        for track, sha in zip(('FE01', 'FE02'), results):
            self.assertEqual(next(r for r in snapshot['tracks'] if r['track_id'] == track)['commit_sha'], sha)
            value = recover(self.repo, track)
            self.assertEqual(value['scientific_commit'], sha)
        self.assertEqual(git(self.repo, 'rev-parse', 'HEAD').strip(), self.initial)

    def test_wrong_track_and_coordinator_updates_are_rejected_without_writes(self):
        root = self.worktrees['FE01']
        before = self.document_bytes(root)
        self.run_tool(root, 'ai_handoff.py', '--track', 'FE02', expect=1)
        self.run_tool(root, 'ai_memory_update.py', '--track', 'FE02', expect=1)
        self.run_tool(root, 'ai_memory_update.py', '--coordinator', expect=1)
        self.assertEqual(before, self.document_bytes(root))
        with self.assertRaisesRegex(ValueError, 'does not own'):
            recover(self.repo, 'FE01', source=self.config['tracks']['FE02']['branch'])

    def test_fields_paths_commits_and_forged_authorization_are_rejected(self):
        original = load(self.worktrees['FE01'] / MEMORY / 'tracks/FE01/record.json')
        mutations = [lambda r: r['handoff'].pop('goal'),
                     lambda r: r['handoff']['next_task'].update(execution_authorization='run_sbatch_now'),
                     lambda r: r['handoff']['last_session'].update(result_commit='1234'),
                     lambda r: r['evidence_paths'][0].update(path='../outside'),
                     lambda r: r['handoff']['next_task'].update(task_id='FE02-WRONG')]
        for mutate in mutations:
            value = copy.deepcopy(original); mutate(value)
            with self.assertRaises(ValueError):
                validate_record(value)
        value = copy.deepcopy(original)
        value['handoff']['last_session']['result_commit'] = 'f' * 40
        candidate = self.parent / 'bad-candidate.json'; candidate.write_text(json_text(value))
        self.run_tool(self.worktrees['FE01'], 'ai_memory_update.py', '--track', 'FE01', '--record', str(candidate), expect=1)

    def test_next_task_text_is_never_executed_and_role_facts_agree(self):
        root = self.worktrees['FE02']; directory = root / MEMORY / 'tracks/FE02'
        record = load(directory / 'record.json')
        sentinel = self.parent / 'MUST_NOT_EXIST'
        record['handoff']['next_task']['read_steps'].append('sbatch job.sh; touch ' + str(sentinel))
        (directory / 'record.json').write_text(json_text(record))
        for name, body in track_views(record).items():
            p = directory / name;p.write_text(generated(p.read_text(), 'FE02 ' + name[:-3], body))
        before = self.document_bytes(root)
        a = recover(root, role='codex'); b = recover(root, role='chatgpt')
        self.assertFalse(sentinel.exists())
        self.assertEqual(a['goal'], b['goal']); self.assertEqual(a['verified_results'], b['verified_results'])
        self.assertNotEqual(a['role_scope'], b['role_scope'])
        self.assertTrue(a['state_document_dirty'])
        self.assertIsNone(a['state_document_commit'])
        self.assertEqual(before, self.document_bytes(root))

    def test_agents_entry_routes_and_ownership_directories_agree(self):
        agents = (ROOT / 'AGENTS.md').read_text()
        entry = (ROOT / 'docs/ai/START_HERE.md').read_text()
        self.assertIn('First read `docs/ai/START_HERE.md`', agents)
        self.assertIn('NEXT_TASK', agents)
        for track, route in self.config['tracks'].items():
            self.assertIn(route['branch'], entry)
            self.assertIn(route['directory'], entry)
        path = self.repo / MEMORY / 'registry.json'
        registry = load(path)
        registry['handoff']['tracks']['FE01']['directory'] = str(MEMORY / 'tracks/FE02')
        path.write_text(json_text(registry))
        with self.assertRaisesRegex(ValueError, 'ownership'):
            handoff_registry(self.repo)

    def test_stale_view_new_science_and_ambiguous_branch_are_reported(self):
        root = self.worktrees['FE02']; file = root / MEMORY / 'tracks/FE02/NEXT_TASK.md'
        original = file.read_text();file.write_text(original.replace('### 下一步提案', '### stale task'))
        with self.assertRaisesRegex(ValueError, 'Stale generated view'):
            recover(root)
        file.write_text(original)
        (root / 'SYNTHETIC_NEW_SCIENCE.txt').write_text('test only\n')
        git(root, 'add', 'SYNTHETIC_NEW_SCIENCE.txt');git(root, 'commit', '-qm', 'SYNTHETIC unrecorded change')
        self.assertTrue(recover(root)['needs_status_review'])
        with self.assertRaisesRegex(ValueError, 'Cannot infer'):
            recover(self.repo)

    def test_staged_scope_preserves_user_file_but_rejects_foreign_stage(self):
        root = self.worktrees['FE01']
        note = root / 'USER_UNTRACKED.txt';note.write_text('preserve me\n')
        self.run_tool(root, 'ai_memory_check.py', '--scope', 'FE01', '--base', 'HEAD', '--staged-only')
        self.assertEqual(note.read_text(), 'preserve me\n')
        other = root / MEMORY / 'tracks/FE02/STATUS.md';other.write_text(other.read_text() + '\nforeign edit\n')
        git(root, 'add', str(other.relative_to(root)))
        result = self.run_tool(root, 'ai_memory_check.py', '--scope', 'FE01', '--base', 'HEAD', '--staged-only', expect=1)
        self.assertIn('Outside track ownership', result.stdout)


if __name__ == '__main__':
    unittest.main()
