"""Isolated FE01-A01 policy, diagnostics and manually submitted experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = '20261007-fe01-a01-geometry-absolute-amplitude'
BASE_COMMIT = 'ba4fa7740d9fde5fde87a7b0ae0397037209baf4'
ON = 'physical_scale_keep_duration'
OFF = 'no_absolute_scale_keep_duration'
FAMILIES = ('team_original_scratch', 'phasenet_pretrained_frozen',
            'eqt_pretrained_frozen', 'diting_pretrained_frozen')
