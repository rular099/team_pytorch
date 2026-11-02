import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workspace_sync', ROOT/'tools/workspace_sync.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)
POLICY = json.loads((ROOT/'docs/ai/sync/SYNC_POLICY.json').read_text())


class SyncTests(unittest.TestCase):
    def test_small_summary_and_png_are_allowed(self):
        self.assertEqual(sync.inspect_content('reports/summary.json', b'{"state":"unknown"}', POLICY), [])
        self.assertEqual(sync.inspect_content('reports/figure.png', b'\x89PNG\r\n\x1a\nfixture', POLICY), [])

    def test_size_archives_private_paths_and_invalid_images_are_rejected(self):
        self.assertIn('size_limit_exceeded', sync.inspect_content('table.csv', b'x'*(POLICY['max_file_bytes']+1), POLICY))
        self.assertIn('non_allowlisted_type', sync.inspect_content('weights.pth', b'fixture', POLICY))
        self.assertIn('non_allowlisted_type', sync.inspect_content('archive.zip', b'fixture', POLICY))
        self.assertIn('private_path', sync.inspect_content('.aws/credentials', b'fixture', POLICY))
        self.assertIn('invalid_binary_format', sync.inspect_content('fake.png', b'not an image', POLICY))

    def test_credentials_are_detected_without_echoing_values(self):
        token = 'ghp_' + 'A'*36
        errors = sync.inspect_content('notes.md', token.encode(), POLICY)
        self.assertTrue(errors)
        self.assertNotIn(token, str(errors))
        self.assertEqual(sync.inspect_content('env.example.sh', b'export GITHUB_TOKEN="${GITHUB_TOKEN}"\n', POLICY), [])

    def test_manifest_is_verified_only_against_real_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'repo/artifact.csv'; path.parent.mkdir()
            raw = b'fixture\n';path.write_bytes(raw)
            manifest = {'files':[{'workspace_relative_path':'repo/artifact.csv', 'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}
            self.assertTrue(sync.verify_artifacts(root, manifest)['complete'])
            path.write_bytes(b'damaged\n')
            self.assertFalse(sync.verify_artifacts(root, manifest)['complete'])
            path.unlink()
            self.assertEqual(sync.verify_artifacts(root, manifest)['results'][0]['state'], 'MISSING')

    def test_missing_external_hash_cannot_be_called_a_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=sync.verify_artifacts(Path(tmp), {'files':[], 'unavailable_external_dependencies':[{'logical_path':'chaosuan_res'}]})
            self.assertFalse(result['complete'])
            self.assertEqual(result['results'][0]['state'], 'NO_SOURCE_HASH_AVAILABLE')

    def test_paths_and_external_symlinks_are_rejected(self):
        for value in ('../escape', '/absolute', 'a/../../escape', 'C:\\path'):
            with self.assertRaises(ValueError):sync.safe_relative(value)
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as external:
            root=Path(tmp);(root/'link').symlink_to(Path(external), target_is_directory=True)
            raw=b'fixture';(Path(external)/'data.csv').write_bytes(raw)
            manifest={'files':[{'workspace_relative_path':'link/data.csv','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}
            self.assertFalse(sync.verify_artifacts(root,manifest)['complete'])

    def test_staged_check_does_not_add_untracked_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'summary.md').write_text('fixture\n');(root/'raw.npz').write_bytes(b'fixture')
            subprocess.run(['git','-C',str(root),'add','summary.md'],check=True)
            self.assertEqual(sync.check_staged(root,POLICY)['errors'],[])
            subprocess.run(['git','-C',str(root),'add','raw.npz'],check=True)
            self.assertTrue(sync.check_staged(root,POLICY)['errors'])


if __name__ == '__main__':unittest.main()
