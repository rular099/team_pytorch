"""Real shell bootstrap with synthetic module/Conda/Python, not HPC evidence."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ClusterEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        conda_root = self.root / 'conda'
        (conda_root / 'bin').mkdir(parents=True)
        python = conda_root / 'bin/python'
        python.write_text('#!/bin/bash\nprintf "%s\\n" "$1" >> "$FE02_TEST_TRACE"\n')
        python.chmod(0o755)
        conda_sh = self.root / 'conda.sh'
        # Legacy Conda and activation hooks may read PS1 / other optional vars.
        # The actual FE02 env.sh must survive this when its caller enabled -u.
        conda_sh.write_text('''
: "$PS1"
: "$FE02_TEST_UNSET_OPTIONAL"
conda() {
    [[ "$1" == activate && "$2" == zb ]] || return 81
    : "$PS1"
    : "$FE02_TEST_UNSET_OPTIONAL"
    [[ "$CONDA_CHANGEPS1" == false ]] || return 82
    [[ "${FE02_TEST_FAIL_CONDA:-0}" == 0 ]] || return 37
    export CONDA_PREFIX="$FE02_TEST_CONDA_ROOT"
}
''')
        values = dict(FE02_CODE_ROOT=str(ROOT), FE02_OUTPUT_ROOT=str(self.root / 'output'),
                      FE02_MODULE_LOADS='none', FE02_CONDA_SH=str(conda_sh), FE02_CONDA_ENV='zb')
        self.env_file = self.root / 'cluster.env'
        self.env_file.write_text('\n'.join('export ' + k + '=' +
            ('"${FE02_MODULE_LOADS:-none}"' if k == 'FE02_MODULE_LOADS' else shlex.quote(v))
            for k, v in values.items()))
        self.trace = self.root / 'python-trace'
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('FE02_')}
        self.env.update(FE02_ENV_FILE=str(self.env_file), FE02_TEST_CONDA_ROOT=str(conda_root),
                        FE02_TEST_TRACE=str(self.trace), SLURM_JOB_ID='12345')

    def bootstrap(self, prefix='', **overrides):
        script = '''
set -euo pipefail
unset PS1 FE02_TEST_UNSET_OPTIONAL
''' + prefix + '\nsource "$1"\n[[ "$-" == *u* ]]\necho BOOTSTRAP_PASS\n'
        return subprocess.run(['bash', '-c', script, 'unit-bootstrap', str(ROOT / 'scripts/fe02/env.sh')],
                              env={**self.env, **overrides}, capture_output=True, text=True)

    def test_inherited_nounset_and_missing_ps1_survive_conda_then_restore_strict_mode(self):
        result = self.bootstrap()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('BOOTSTRAP_PASS', result.stdout)
        self.assertEqual(self.trace.read_text().splitlines(), ['scripts/fe02/verify_source.py', '-c'])

    def test_module_initialization_can_read_optional_unset_variables(self):
        result = self.bootstrap('''
export FE02_MODULE_LOADS=unit/mpi
module() {
    : "$FE02_TEST_UNSET_OPTIONAL"
    echo "module:$1" >> "$FE02_TEST_TRACE"
    if [[ "$1" == purge ]]; then
        export LOADEDMODULES=''
    elif [[ "$1" == load ]]; then
        export LOADEDMODULES="$2"
    else
        return 83
    fi
}
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(),
                         ['module:purge', 'module:load', 'scripts/fe02/verify_source.py', '-c'])

    def test_activation_failure_stops_before_source_verification_or_torch(self):
        result = self.bootstrap(FE02_TEST_FAIL_CONDA='1')
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertNotIn('BOOTSTRAP_PASS', result.stdout)
        self.assertFalse(self.trace.exists())


if __name__ == '__main__':
    unittest.main()
