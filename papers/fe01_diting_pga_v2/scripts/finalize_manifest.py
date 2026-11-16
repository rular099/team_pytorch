"""Add explicit method/config inputs and final manuscript source inventory to assets."""
import json
from common import *

e=Evidence();manifest=json.loads((PAPER/'assets/asset_manifest.json').read_text())
settings=json.loads((PAPER/'evidence/complete_model_settings.json').read_text())
for asset in manifest['assets']:
    if asset['asset_id'] in ['F1','T1'] and asset['panel_id']!='assembled':
        inputs=[]
        for family,row in settings.items():
            records=[v for v in e.records if v['full_commit']==row['source_commit'] and
                     (v['path']==row['source_path'] or (family=='DiTing' and v['path'].endswith('training/index3/model_interface_audit.json')))]
            for record in records:
                inputs.append(dict(repository=record['repository'],full_commit=record['full_commit'],path=record['path'],sha256=record['actual_sha256'],rows_before=None,role='resolved configuration and archived parameter audit'))
        asset['method_config_inputs']=inputs
    if asset.get('panel_id')=='F1_a':
        path=PAPER/'assets/source_data/F1_schematic.csv'
        asset['additional_source_data']=[dict(path=str(path.relative_to(PAPER)),sha256=sha(path.read_bytes()),kind='explicit synthetic schematic samples, not event observations')]
    asset['finalizer_script']='scripts/finalize_manifest.py'
    asset['finalizer_sha256']=sha(Path(__file__).read_bytes())
write_json(PAPER/'assets/asset_manifest.json',manifest)
print('PASS: schematic/config audit sources explicit in final asset manifest')
