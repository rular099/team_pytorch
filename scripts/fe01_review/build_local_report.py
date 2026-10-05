#!/usr/bin/env python3
"""Publish compact diagnostics from real sufficient statistics, without rereading raw predictions."""
import argparse
import itertools
import re
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
from fe01_review.provenance import read_json,write_json,new_output,sha256,seal,provenance
from fe01_review.requests import read_frame
from fe01_review.statistics import interval
from fe01_review.pack import validate_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','plan','inventory','test-log','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();source=Path(a.input);output=new_output(a.output)
    validate_manifest(source/'artifact_manifest.sha256')
    exclude={'event_fields.csv.gz','event_losses.csv.gz','artifact_manifest.sha256'}
    for f in source.iterdir():
        if f.name in exclude:continue
        if f.is_dir():shutil.copytree(f,output/f.name)
        else:shutil.copy2(f,output/f.name)
    for name in ('case_manifest.json','request_population_audit.json','verification_decisions.csv','random_draw_mapping.csv'):
        shutil.copy2(Path(a.plan)/name,output/name)
    for name in ('event_fields.csv.gz','event_losses.csv.gz'):
        table=read_frame(source/name);table=table[table.target_group.isin(['noninput','untriggered_noninput'])]
        table.to_csv(output/(name.replace('.csv','_primary.csv')),index=False)
    losses=read_frame(source/'event_losses.csv.gz');rows=[]
    ids=losses[['run_id','model_family','seed']].drop_duplicates()
    for seed in (42,43,44):
        for left,right in itertools.combinations(ids[ids.seed==seed].run_id,2):
            la,lb=losses[losses.run_id==left],losses[losses.run_id==right]
            for endpoint,times in [('one_second',[1]),('early',[1,3,5])]:
                for metric in ('abs_error','nll','crps','brier'):
                    rows.append(dict(left=left,right=right,seed=seed,endpoint=endpoint,metric=metric,
                        **interval(la[la.target_group=='noninput'],right=lb[lb.target_group=='noninput'],metric=metric,times=times)))
    pd.DataFrame(rows).to_csv(output/'paired_early_probability_event_ci.csv',index=False)
    inventory=read_json(Path(a.inventory)/'checkpoint_inventory.json')
    for r in inventory:
        for role in ('selected','init','last'):
            item=r[role]
            if 'path' in item:item['path']=Path(item['path']).name
            if isinstance(item.get('encoder_source'),str):item['encoder_source']=Path(item['encoder_source']).name
        if 'selected_csv' in r:r['selected_csv']['path']=Path(r['selected_csv']['path']).name
    write_json(output/'checkpoint_inventory.json',inventory)
    captions=output/'figures/CAPTIONS.md'
    captions.write_text('All physical PGA axes use log10(m/s²); errors/widths in dex, MSE in dex².\n\n'+captions.read_text().replace('event_fields.csv.gz','event_fields_primary.csv.gz (noninput/untriggered subset; complete other-role fields in full evidence)'))
    raw_log=Path(a.test_log).read_text()
    (output/'verification_pytest.txt').write_text(re.sub(r'/[^\s]*?/site-packages/','${SITE_PACKAGES}/',raw_log))
    write_json(output/'publication_provenance.json',provenance(source_analysis_manifest_sha256=sha256(source/'artifact_manifest.sha256'),
        full_event_fields_sha256=sha256(source/'event_fields.csv.gz'),full_event_losses_sha256=sha256(source/'event_losses.csv.gz'),
        compact_event_groups=['noninput','untriggered_noninput'],all_other_group_statistics='full local/HPC source package; aggregate metrics and intervals retained here',
        original_test_log_sha256=sha256(a.test_log),test_log_redaction='absolute site-packages prefix only',
        extra_bootstrap='noninput1s/early MAE,NLL,CRPS,Brier;5000 paired event draws from saved sums/counts, not new forward'))
    seal(output)


if __name__=='__main__':main()
