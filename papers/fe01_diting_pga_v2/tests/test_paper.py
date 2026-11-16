"""Focused provenance, estimand and manuscript tests; no scientific model execution."""
import ast
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *

class PaperEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=Evidence();cls.m=cls.e.eid('E03');cls.p=cls.e.eid('E04')
        cls.tt=cls.e.eid('E10');cls.shadow=cls.e.eid('E17')

    def test_unique_row_failure_is_closed(self):
        one=self.m.iloc[:1]
        with self.assertRaises(ValueError):unique(pd.concat([one,one]),family=one.iloc[0].family)
        with self.assertRaises(ValueError):unique(one,family='absent')

    def test_epochs_and_active_model_path(self):
        inventory=self.e.eid('E31')
        self.assertEqual({r['family']:r['epoch'] for r in inventory if r['policy']=='selected'},EPOCHS)
        settings=json.loads((PAPER/'evidence/complete_model_settings.json').read_text())
        for f in FAMILIES:
            flags=settings[f]['active_flags']
            self.assertEqual(flags['station_context_mode'],'off')
            self.assertFalse(flags['pga_use_event_context']);self.assertTrue(flags['use_amplitude_info'])
            self.assertEqual(flags['pga_readout_layers'],4)

    def test_population_is_not_interchangeable(self):
        for g,n in [('all',194265),('noninput',139440)]:
            r=unique(self.m,family='DiTing',checkpoint_policy='selected',target_group=g,endpoint='primary_10cell')
            self.assertEqual(int(r.requests),n);self.assertEqual(int(r.events),1310)
        common=dict(family='DiTing',panel='fixed_never_input_targets',budget_dex=.2,gate='field_pass',scene='ideal_cutoff_replay',horizon=20)
        self.assertEqual(int(unique(self.tt,geometry_protocol='normal',**common).events),207)
        self.assertEqual(int(unique(self.tt,geometry_protocol='random',**common).events),539)
        r=unique(self.shadow,family='DiTing',geometry_protocol='normal',panel='fixed_never_input_targets',threshold_logpga=-1.2,p_alarm=.5,tau=3,scene='measured_rendered_map_replay')
        self.assertEqual((int(r.events),int(r.events_with_affected_targets),int(r.affected_targets)),(312,214,2336))

    def test_unfavorable_population_and_epoch_policy_survive(self):
        shared=dict(comparator='PhaseNet',checkpoint_policy='selected',endpoint='primary_10cell',metric='mae')
        self.assertGreater(unique(self.p,target_group='all',**shared).estimate,0)
        self.assertLess(unique(self.p,target_group='noninput',**shared).estimate,0)
        self.assertEqual(set(self.m.checkpoint_policy),{'selected','epoch12'})

    def test_unknown_bounds_and_numerical_arithmetic(self):
        self.assertTrue(((self.shadow.timely+self.shadow.failures+self.shadow.unknown)==self.shadow.affected_targets).all())
        self.assertTrue((self.shadow.pooled_failure_upper>=self.shadow.pooled_failure_lower).all())
        self.assertTrue(((self.tt.events+self.tt.ineligible_fewer_than_5)==self.tt.total_cohort_events).all())
        c=json.loads((PAPER/'qa/numerical_checks.json').read_text())
        self.assertEqual(c['status'],'PASS');self.assertTrue(all(r['passed'] for r in c['checks']))

    def test_claims_are_rows_not_blueprint_numbers(self):
        tr=json.loads((PAPER/'evidence/claim_trace.json').read_text())
        self.assertEqual({c['claim_id'] for c in tr['claims']},{'C01','C02','C03','C04','C05','C07','C08','C09','C10','C11','C12'})
        for c in tr['claims']:
            for src in c['inputs']:
                self.assertEqual(src['full_commit'],RESULT);self.assertEqual(len(src['sha256']),64);self.assertGreater(len(src['rows']),0)

    def test_materials_scope_and_length(self):
        manuscript=(PAPER/'manuscript.md').read_text();sup=(PAPER/'supplement.md').read_text()
        self.assertEqual(manuscript.splitlines()[0],'# 基于 DiTing 的早期 PGA 概率预测')
        self.assertEqual(re.findall(r'^## ([1-5]) ',manuscript,re.M),list('12345'))
        body=manuscript.split('## 1 引言')[1].split('## 数据与代码可用性')[0]
        abstract=manuscript.split('## 摘要')[1].split('关键词')[0]
        self.assertTrue(8000<=len(re.findall('[\u3400-\u9fff]',body))<=10000)
        self.assertTrue(400<=len(re.findall('[\u3400-\u9fff]',abstract))<=600)
        self.assertEqual(len(re.findall(r'^## S[1-8] ',sup,re.M)),8)
        for text in [manuscript,sup,(PAPER/'figure_captions.md').read_text()]:
            self.assertIsNone(re.search(r'ON\s*/\s*OFF|幅值消融|幅值开关|C06|FE02',text,re.I))
        for line in manuscript.splitlines():
            if line.startswith('#'):self.assertNotIn('：',line);self.assertNotIn(':',line)

    def test_main_assets_and_dpi(self):
        from PIL import Image
        manifest=json.loads((PAPER/'assets/asset_manifest.json').read_text())['assets']
        groups=[a for a in manifest if a['panel_id']=='assembled']
        self.assertEqual({a['asset_id'] for a in groups},{'F'+str(i) for i in range(1,8)})
        for a in manifest:
            if a['panel_id']=='assembled':continue
            self.assertTrue((PAPER/a['source_data_path']).is_file())
            self.assertEqual(sha((PAPER/a['source_data_path']).read_bytes()),a['source_data_sha256'])
            for s in a['sources']:self.assertEqual(len(s['sha256']),64)
        for i in range(1,8):
            for ext in ['svg','pdf','png']:self.assertTrue((PAPER/f'assets/figures/F{i}.{ext}').is_file())
            with Image.open(PAPER/f'assets/figures/F{i}.png') as im:self.assertGreaterEqual(min(im.info['dpi']),299.)

    def test_rounded_anchors_are_consistent(self):
        # Explicit QA only. Plot construction always reads original source rows.
        a=json.loads((PAPER/'evidence/blueprint_v2/04_CLAIM_REGISTER.json').read_text())['claims']
        a={v['claim_id']:v.get('rounded_QA_anchors',{}) for v in a}
        for f,value in a['C01'].items():
            r=unique(self.m,family=f,checkpoint_policy='selected',target_group='noninput',endpoint='primary_10cell')
            self.assertAlmostEqual(r.equal_cell_mae,value,delta=.00000051)
        r=unique(self.p,comparator='PhaseNet',checkpoint_policy='selected',target_group='noninput',endpoint='first_second_2cell',metric='mae')
        for field,key in [('estimate','delta_mae'),('ci_low','ci_low'),('ci_high','ci_high')]:self.assertAlmostEqual(r[field],a['C02'][key],delta=.00000051)
        for f in ['DiTing','PhaseNet']:
            r=unique(self.m,family=f,checkpoint_policy='selected',target_group='noninput',endpoint='primary_10cell')
            for col,suffix in [('equal_cell_covered95','coverage95'),('equal_cell_nll','nll')]:self.assertAlmostEqual(r[col],a['C03'][f+'_'+suffix],delta=.00000051)
        cases=[('C04',unique(self.p,comparator='PhaseNet',checkpoint_policy='selected',target_group='Kmulti_noninput',endpoint='primary_10cell',metric='mae'),'delta_mae'),
            ('C05',unique(self.e.eid('E08'),comparator='PhaseNet',checkpoint_policy='selected',target_group='noninput',metric='range_error'),'PhaseNet_noninput_delta_range'),
            ('C11',unique(self.e.eid('E05'),comparator='PhaseNet',checkpoint_policy='selected',factor='truth_logpga',stratum='y>0',metric='mae'),'delta_mae')]
        for cid,row,key in cases:
            for col,target in [('estimate',key),('ci_low','ci_low'),('ci_high','ci_high')]:self.assertAlmostEqual(row[col],a[cid][target],delta=.00000051)
        for scene,short in [('ideal_cutoff_replay','ideal'),('measured_compute_replay','numeric'),('measured_rendered_map_replay','rendered')]:
            for family,key in [('DiTing','D'),('PhaseNet','P')]:
                r=unique(self.tt,family=family,geometry_protocol='normal',panel='fixed_never_input_targets',budget_dex=.2,gate='field_pass',scene=scene,horizon=20)
                self.assertEqual(r.achieved,a['C07'][short+'_'+key])
        for f in ['DiTing','PhaseNet']:
            r=unique(self.e.eid('E13'),family=f,geometry_protocol='normal',panel='fixed_never_input_targets',budget_dex=.2,gate='field_pass',scene='ideal_cutoff_replay')
            self.assertEqual(r.post_confirmation_known_nonpassing_events,a['C08'][f+'_failed_after'])
            self.assertEqual(r.stable_confirmed,a['C08'][f+'_confirmed'])
            lat=unique(self.e.eid('E14'),family=f,scope='whole_map')
            self.assertAlmostEqual(lat.p95_seconds,a['C09']['whole_map_p95_'+('D' if f=='DiTing' else 'P')],delta=.00000051)
        r=unique(self.e.eid('E09'),comparator='PhaseNet',checkpoint_policy='selected',target_group='noninput')
        for col,key in [('loo_min','LOO_min'),('loo_max','LOO_max'),('event_win_rate','event_win_rate')]:self.assertAlmostEqual(r[col],a['C12'][key],delta=.00000051)

    def test_scripts_compile_and_do_not_import_scientific_models(self):
        forbidden={'torch','fe01','fe01_a01','fe01_a01_compare','gemini_models','train_light'}
        for path in (PAPER/'scripts').glob('*.py'):
            text=path.read_text();compile(text,str(path),'exec');tree=ast.parse(text)
            for n in ast.walk(tree):
                names=[x.name for x in n.names] if isinstance(n,ast.Import) else ([n.module] if isinstance(n,ast.ImportFrom) and n.module else [])
                self.assertFalse(any(name.split('.')[0] in forbidden for name in names),path.name)

if __name__=='__main__':unittest.main(verbosity=2)
