"""Small synthetic paired artifacts, not Japan measurements."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import numpy as np
from fe01.metrics import score_rows
from fe01.config import write_json
from fe02.analysis import compare


def fixtures(root):
    paths=[]
    for variant in ('R0','A','B','C','M'):
        path=Path(root)/variant;path.mkdir();paths.append(path)
        rows=[]
        for event in ('e1','e2'):
            for protocol in ('normal','random'):
                for time in (1,3,5,10,20,40,90):
                    for station in range(6):
                        truth=-2+station*.3
                        scores=score_rows(np.array([truth]),np.array([[1.]]),np.array([[truth+(.2 if variant=='R0' else .1)]]),np.array([[.3]]))
                        rows.append(dict(dataset_id='unit.h5',event_id=event,station_id=str(station),split='val',
                            seed=42,variant_id=variant,event_fusion='unit',model_family='diting_pretrained_frozen',
                            geometry_protocol=protocol,elapsed_time=time,target_role='untriggered_noninput',
                            input_ids='["input"]',input_count=1 if event=='e1' else 3,status='supported',
                            truth=truth,prediction=truth+(.2 if variant=='R0' else .1),latitude=35+station*.1,
                            longitude=139,event_latitude=35,event_longitude=139,
                            **{k:float(v[0]) for k,v in scores.items() if k!='prediction'}))
        pd.DataFrame(rows).to_csv(path/'predictions.csv.gz',index=False)
        pd.DataFrame(rows)[['dataset_id','event_id','geometry_protocol','elapsed_time','status']].drop_duplicates().to_csv(path/'support_status.csv',index=False)
        write_json(path/'provenance.json',dict(split='val',variant_id=variant,checkpoint_epoch=12,
            checkpoint_sha256=variant,split_manifest_sha256='unit',validation_population_sha256='unit',
            window_protocol='native_prefix_v2',stage='formal',encoder_checkpoint_sha256='unit',random_times_manifest_sha256=None))
    return paths


class AnalysisTests(unittest.TestCase):
    def test_variant_matrix_probability_and_single_multi_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=fixtures(tmp);output=Path(tmp)/'analysis';compare(paths,output,bootstrap_draws=30)
            headline=pd.read_csv(output/'headline_primary.csv')
            self.assertEqual(set(headline.variant_id),{'R0','A','B','C','M'})
            metrics=pd.read_csv(output/'metrics_time_geometry_count_target.csv')
            self.assertEqual(set(metrics.input_count_stratum),{'all_counts','single','multi'})
            self.assertEqual(set(metrics.geometry_protocol),{'normal','random'})
            paired=pd.read_csv(output/'paired_event_bootstrap.csv')
            delta=paired.loc[(paired.baseline=='R0')&(paired.competitor=='A'),'delta_mae']
            self.assertTrue((abs(delta+.1)<1e-8).all())

    def test_unpaired_encoder_or_population_and_missing_variant_refused(self):
        for mode in ('encoder','population','missing','test','numeric'):
            with tempfile.TemporaryDirectory() as tmp:
                paths=fixtures(tmp)
                if mode=='missing':paths.pop()
                elif mode in ('encoder','test'):
                    path=paths[-1]/'provenance.json';value=json.loads(path.read_text())
                    value['encoder_checkpoint_sha256' if mode=='encoder' else 'split']='changed' if mode=='encoder' else 'test'
                    write_json(path,value)
                else:
                    path=paths[-1]/'predictions.csv.gz';frame=pd.read_csv(path,dtype={'event_id':str,'station_id':str})
                    if mode=='population':frame.loc[0,'input_ids']='["wrong"]'
                    else:frame.loc[0,'status']='failure'
                    frame.to_csv(path,index=False)
                with self.assertRaises(ValueError):compare(paths,Path(tmp)/'analysis',bootstrap_draws=2)

if __name__=='__main__':unittest.main()
