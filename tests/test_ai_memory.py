"""Memory workflow tests use disposable Git repos and synthetic evidence, never HPC."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from ai_memory_lib import (MEMORY, evidence_available, generated, git, json_text,
                           load, merge_risk, preserve_history, resource_errors,
                           track_views, utcnow, validate_record)
from ai_memory_check import check


def fixture_record(track, sha):
    return dict(track_id=track, branch='exp/' + track.lower(), commit_sha=sha,
                last_verified_at=utcnow(), evidence_as_of='2026-11-01',
                current_state='planned', completion_scope='SYNTHETIC fixture only',
                summary='SYNTHETIC initial ' + track,
                verified_results=[dict(text='SYNTHETIC source', evidence_ids=['report'])],
                pending_tasks=['SYNTHETIC next'], limitations=['Not scientific evidence'],
                evidence_paths=[dict(evidence_id='report', commit_sha=sha, path='evidence.md', kind='report')],
                decisions=[dict(decision_id='D-' + track, date='2026-11-01', problem='fixture',
                                options='fixture', decision='retain evidence', evidence_ids=['report'],
                                applies_to=track, superseded_by=None, history_status='source_recorded')],
                experiments=[dict(experiment_id='E-' + track, scientific_question='fixture',
                                  dataset_protocol='SYNTHETIC', result_nature='fixture',
                                  review_status='fixture', relationship='independent',
                                  evidence_ids=['report'], state='planned')])


class MemoryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ai-memory-test-')
        self.parent = Path(self.temp.name)
        self.root = self.parent / 'coordinator'
        self.root.mkdir()
        git(self.root, 'init', '-b', 'coordinator')
        git(self.root, 'config', 'user.name', 'Memory fixture')
        git(self.root, 'config', 'user.email', 'fixture@example.invalid')
        (self.root / 'tools').mkdir()
        for name in ('ai_memory_lib.py', 'ai_memory_check.py', 'ai_memory_update.py'):
            shutil.copyfile(TOOLS / name, self.root / 'tools' / name)
        (self.root / '.gitignore').write_text('__pycache__/\n')
        (self.root / 'evidence.md').write_text('# SYNTHETIC evidence\n')
        (self.root / 'gemini_models.py').write_text('# SYNTHETIC shared source\nvalue = 0\n')
        git(self.root, 'add', '.')
        git(self.root, 'commit', '-qm', 'Synthetic source fixture')
        sha = git(self.root, 'rev-parse', 'HEAD').strip()
        mem = self.root / MEMORY
        (mem / 'audit').mkdir(parents=True)
        (mem / 'registry.json').write_text(json_text({'track_ids': ['FE01', 'FE02']}))
        (mem / 'audit/remote_refs.json').write_text(json_text({'branches': [], 'remote_files': []}))
        for name in ('PROJECT_MEMORY.md', 'README.md', 'HISTORY_IMPORT_TEMPLATE.md', 'PARALLEL_WORKFLOW.md'):
            (mem / name).write_text('# SYNTHETIC fixture\n')
        for track in ('FE01', 'FE02'):
            directory = mem / 'tracks' / track
            directory.mkdir(parents=True)
            (directory / 'record.json').write_text(json_text(fixture_record(track, sha)))
        self.run_update(self.root, '--coordinator', '--bootstrap')
        git(self.root, 'add', '.')
        git(self.root, 'commit', '-qm', 'Install synthetic memory')
        self.base = git(self.root, 'rev-parse', 'HEAD').strip()

    def tearDown(self):
        self.temp.cleanup()

    def run_update(self, root, *args, expect=0):
        result = subprocess.run([sys.executable, str(root / 'tools/ai_memory_update.py'), '--root', str(root), *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, expect, result.stdout + result.stderr)
        return result

    def worktree(self, track):
        path = self.parent / track
        git(self.root, 'worktree', 'add', '-b', 'exp/' + track.lower(), str(path), self.base)
        return path

    def update_line(self, root, track, value):
        # Commit a fake result in a disposable repository; no production model is edited.
        (root / 'gemini_models.py').write_text('# SYNTHETIC ' + track + '\nvalue = ' + str(value) + '\n')
        git(root, 'add', 'gemini_models.py')
        git(root, 'commit', '-qm', 'SYNTHETIC result ' + track)
        source = git(root, 'rev-parse', 'HEAD').strip()
        file = root / MEMORY / 'tracks' / track / 'record.json'
        record = load(file)
        record.update(commit_sha=source, last_verified_at=utcnow(), current_state='in_progress',
                      summary='SYNTHETIC result ' + track)
        file.write_text(json_text(record))
        self.run_update(root, '--track', track)
        git(root, 'add', str(MEMORY / 'tracks' / track))
        git(root, 'commit', '-qm', 'SYNTHETIC track memory ' + track)
        return source

    def test_bootstrap_and_global_contradiction_detection(self):
        self.assertEqual(check(self.root)[0], [])
        snapshot = self.root / MEMORY / 'snapshot.json'
        data = load(snapshot)
        data['tracks'][0]['current_state'] = 'unknown'
        snapshot.write_text(json_text(data))
        self.assertTrue(any('Contradictory' in x for x in check(self.root)[0]))

    def test_parallel_two_worktrees_and_same_code_risk(self):
        before = (self.root / MEMORY / 'CURRENT_STATUS.md').read_bytes()
        a, b = self.worktree('FE01'), self.worktree('FE02')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(self.update_line, a, 'FE01', 1), pool.submit(self.update_line, b, 'FE02', 2)]
            source_shas = [job.result() for job in jobs]
        self.assertEqual((self.root / MEMORY / 'CURRENT_STATUS.md').read_bytes(), before)
        self.assertNotEqual((a / 'gemini_models.py').read_bytes(), (b / 'gemini_models.py').read_bytes())
        self.run_update(self.root, '--coordinator', '--source', 'FE01=exp/fe01', '--source', 'FE02=exp/fe02')
        snapshot = load(self.root / MEMORY / 'snapshot.json')
        self.assertEqual([r['commit_sha'] for r in snapshot['tracks']], source_shas)
        self.assertTrue(all(r['current_state'] == 'in_progress' for r in snapshot['tracks']))
        self.assertEqual({d['decision_id'] for r in snapshot['tracks'] for d in r['decisions']}, {'D-FE01', 'D-FE02'})
        self.assertIn('SYNTHETIC result FE01', (self.root / MEMORY / 'CHAT_HANDOFF.md').read_text())
        self.assertIn('SYNTHETIC result FE02', (self.root / MEMORY / 'CHAT_HANDOFF.md').read_text())
        self.assertEqual(check(self.root)[0], [])
        risk = merge_risk(self.root, 'exp/fe01', 'exp/fe02')
        self.assertIn('gemini_models.py', risk['shared_changed_files'])
        self.assertTrue(risk['requires_architecture_review'])
        # Review tool did not merge either branch or touch their files.
        self.assertEqual(git(self.root, 'rev-parse', 'HEAD').strip(), self.base)
        self.assertEqual(git(a, 'show', 'HEAD:gemini_models.py'), (a / 'gemini_models.py').read_text())

    def test_single_track_update_keeps_other_files_and_manual_notes(self):
        a = self.worktree('FE01')
        other = {p.relative_to(a): p.read_bytes() for p in (a / MEMORY / 'tracks/FE02').glob('*')}
        note = a / MEMORY / 'tracks/FE01/STATUS.md'
        note.write_text(note.read_text() + '\n人工补充保留\n')
        self.update_line(a, 'FE01', 1)
        self.assertIn('人工补充保留', note.read_text())
        self.assertTrue(all((a / p).read_bytes() == value for p, value in other.items()))

    def test_missing_remote_source_retains_verified_state(self):
        old = load(self.root / MEMORY / 'observations/FE02.json')
        result = self.run_update(self.root, '--coordinator', '--source', 'FE02=refs/remotes/unavailable')
        self.assertIn('previous verified state retained', result.stdout)
        self.assertEqual(load(self.root / MEMORY / 'observations/FE02.json'), old)
        self.assertEqual(load(self.root / MEMORY / 'snapshot.json')['sync']['FE02']['state'], 'source_unavailable_retained')

    def test_wrong_source_branch_cannot_import_an_inherited_other_track(self):
        self.worktree('FE02')
        old = load(self.root / MEMORY / 'observations/FE01.json')
        result = self.run_update(self.root, '--coordinator', '--source', 'FE01=exp/fe02')
        self.assertIn('does not own this track', result.stdout)
        self.assertEqual(load(self.root / MEMORY / 'observations/FE01.json'), old)

    def test_track_cannot_aggregate_or_write_on_another_branch(self):
        result = self.run_update(self.root, '--track', 'FE01', expect=1)
        self.assertIn('recorded branch', result.stderr)
        a = self.worktree('FE01')
        result = self.run_update(a, '--track', 'FE01', '--source', 'FE02=HEAD', expect=1)
        self.assertIn('require --coordinator', result.stderr)

    def test_historical_ids_and_older_records_cannot_disappear(self):
        old = load(self.root / MEMORY / 'tracks/FE01/record.json')
        new = copy.deepcopy(old)
        new['decisions'] = []
        with self.assertRaisesRegex(ValueError, 'historical IDs'):
            preserve_history(old, new)
        new = copy.deepcopy(old)
        new['last_verified_at'] = '2000-01-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'Older verification'):
            preserve_history(old, new)

    def test_schema_rejects_missing_fields_states_duplicate_ids(self):
        record = load(self.root / MEMORY / 'tracks/FE01/record.json')
        for mutate in (lambda r: r.pop('branch'), lambda r: r.update(current_state='completed'),
                       lambda r: r['experiments'].append(r['experiments'][0]),
                       lambda r: r.update(commit_sha='bad'), lambda r: r.update(evidence_paths=[])):
            bad = copy.deepcopy(record)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_record(bad)

    def test_invalid_evidence_path_and_commit_are_detected(self):
        record = load(self.root / MEMORY / 'tracks/FE01/record.json')
        remote = {'remote_files': []}
        with self.assertRaises(ValueError):
            evidence_available(self.root, {**record['evidence_paths'][0], 'path': 'missing.md'}, remote)
        with self.assertRaises(ValueError):
            evidence_available(self.root, {**record['evidence_paths'][0], 'commit_sha': 'f' * 40}, remote)

    def test_remote_snapshot_blob_tamper_is_detected(self):
        path = self.root / MEMORY / 'audit/remote.txt'
        raw = b'SYNTHETIC remote source\n'
        path.write_bytes(raw)
        evidence = dict(commit_sha='f'*40, path='report.md')
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        remote = {'remote_files': [dict(**evidence, blob_sha=blob, snapshot_path=str(path.relative_to(self.root)))]}
        self.assertEqual(evidence_available(self.root, evidence, remote), 'connector_snapshot')
        path.write_text('tampered\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evidence_available(self.root, evidence, remote)

    def test_generated_sections_do_not_erase_manual_text_or_accept_bad_markers(self):
        original = generated('', 'test', 'old') + '\nmanual\n'
        updated = generated(original, 'test', 'new')
        self.assertIn('manual', updated)
        self.assertNotIn('\nold\n', updated)
        with self.assertRaises(ValueError):
            generated('# manual file without markers', 'test', 'new')

    def test_resource_overlap_symlink_and_shared_readonly(self):
        shared = self.parent / 'shared'
        shared.mkdir()
        alias = self.parent / 'alias'
        alias.symlink_to(shared, target_is_directory=True)
        good = [dict(track_id='FE01', write_paths=[str(self.parent/'a')], readonly_paths=[str(shared)]),
                dict(track_id='FE02', write_paths=[str(self.parent/'b')], readonly_paths=[str(shared)])]
        self.assertEqual(resource_errors(good), [])
        bad = copy.deepcopy(good)
        bad[0]['write_paths'] = [str(alias)]
        self.assertTrue(resource_errors(bad))
        bad[0]['write_paths'] = ['$UNDEFINED_PROJECT_MEMORY_ROOT/cache']
        self.assertTrue(resource_errors(bad))

    def test_bootstrap_cannot_repeat_or_use_malicious_track(self):
        self.run_update(self.root, '--coordinator', '--bootstrap', expect=1)
        self.run_update(self.root, '--track', '../outside', expect=1)

    def test_dry_run_leaves_memory_bytes_unchanged(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in (self.root/MEMORY).rglob('*') if p.is_file()}
        self.run_update(self.root, '--coordinator', '--source', 'FE01=missing', '--dry-run')
        self.assertTrue(all((self.root/p).read_bytes() == raw for p,raw in before.items()))


if __name__ == '__main__':
    unittest.main()
