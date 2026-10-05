import copy
import numpy as np
import pandas as pd
import pytest
from fe01.metrics import score_rows,summarize
from fe01_review.diagnostics import fields,field_summary,fixed_panels,random_endpoint,station_recovery
from fe01_review.statistics import event_losses,interval,fixed_model_mean_loss
from fe01_review.comparison import assert_same_population,random_loss_table
from fe01_review.requests import population_audit,select_cases,verification_decisions


def frame(events=16):
    rows=[]
    for event in range(events):
        for protocol in ('normal','random'):
            for time in (1,3,5,10,20):
                for station in range(6):
                    rows.append(dict(dataset_id='a',event_id=str(event),station_id=str(station),geometry_protocol=protocol,elapsed_time=time,
                        truth=-2+station*.2,prediction=-2+station*.2+.1,input_count=1 if event%2 else 2,
                        target_role='observed_input' if station==0 and time>=10 else 'untriggered_noninput',
                        latitude=35+station*.001,longitude=139+station*.001,event_latitude=35,event_longitude=139,magnitude=3+event*.1,depth=10,
                        split='val',units='log10(m/s^2)',status='supported',input_ids='["0"]',input_coords='[[35,139,0]]',
                        requested_decision_sample=time*100,current_sample=time*100,cutout_exclusive=time*100+1,
                        history_start_sample=0,history_end_sample=time*100,latest_received_sample=time*100,
                        history_seconds=time+.01,valid_seconds=time+.01,physical_received_history_sha256='x',query_elevation=0.))
    f=pd.DataFrame(rows);scores=score_rows(f.truth.to_numpy(),np.ones((len(f),1)),f.prediction.to_numpy()[:,None],np.full((len(f),1),.2))
    for key,value in scores.items():f[key]=value
    return f


def test_mse_decomposition_constant_offset_and_residual_identity():
    f=frame();table=fields(f);g=table[table.status=='supported']
    np.testing.assert_allclose(g.field_mse,g.level_mse+g.shape_mse,atol=1e-14)
    np.testing.assert_allclose(g.shape_mse,0,atol=1e-14)
    np.testing.assert_allclose(g.level_mse,.01,atol=1e-14)
    np.testing.assert_allclose((f.prediction-.7)-(f.truth-.7),f.prediction-f.truth)
    assert g.pairwise_delta_mae.max()<1e-14


def test_group_recentering_and_minimum_range_undefined():
    f=frame();f.loc[f.station_id=='0','prediction']+=5
    table=fields(f);g=table[(table.elapsed_time==20)&(table.target_group=='noninput')]
    assert g.centered_mae.max()<1e-14
    f.truth=0;f.prediction=.1
    table=fields(f);g=table[table.status=='supported'];assert g.range_ratio.isna().all()
    assert g.equal_distance_sign_denominator.sum()==0
    assert g.equal_distance_sign_accuracy.isna().all()
    summary=field_summary(table);assert summary.range_ratio_field_denominator.sum()==0


def test_nan_failure_not_dropped_and_empty_field_explicit():
    f=frame();f.loc[0,'prediction']=np.nan
    with pytest.raises(ValueError):fields(f)
    f=frame().iloc[:3];table=fields(f)
    assert table.status.eq('N/A').all()
    assert summarize(f.assign(truth=0)).get('r2') is None


def test_primary_cell_equal_bootstrap_and_pair_denominator():
    f=frame();loss=event_losses(f);g=loss[loss.target_group=='noninput']
    result=interval(g,draws=101);assert result['events']==16;assert result['draws']==101
    assert result['estimate']==pytest.approx(.1);assert result['ci_low']==pytest.approx(.1)
    other=g.copy();other.abs_error_sum+=.03*other.targets
    paired=interval(g,draws=101,right=other);assert paired['estimate']==pytest.approx(.03)
    with pytest.raises(ValueError):interval(g,right=other.iloc[1:],draws=10)


def test_mean_seed_loss_is_not_ensemble():
    f=frame();f.prediction=f.truth+.2;a=event_losses(f)
    f.prediction=f.truth-.2;b=event_losses(f)
    mean=fixed_model_mean_loss([a,b]);g=mean[mean.target_group=='noninput']
    assert interval(g,draws=10)['estimate']==pytest.approx(.2)
    assert np.mean(np.abs((f.truth+.2+f.truth-.2)/2-f.truth))<1e-12


def test_pooled_rmse_differs_from_mean_cell_rmse():
    f=frame();f.prediction=f.truth+np.where(f.elapsed_time==1,.1,.5)
    pooled=summarize(f)['rmse'];cell=np.mean([summarize(g)['rmse'] for _,g in f.groupby(['geometry_protocol','elapsed_time'])])
    assert abs(pooled-cell)>.01


def test_fixed_panel_retains_population_and_migration():
    f=frame();f=f.drop(f[(f.event_id=='0')&(f.elapsed_time==1)&(f.station_id=='5')].index)
    scores,audit,migration=fixed_panels(f)
    excluded=audit[audit.panel=='fixed_targets_natural_inputs'].excluded_targets
    assert excluded.max()==1
    assert migration.query('from_role=="untriggered_noninput" and to_role=="observed_input"').event_station_trajectories.sum()>0


def test_duplicate_random_draws_keep_original_weight():
    f=frame(events=2);f=f[f.elapsed_time.isin([1,3])].copy()
    f.loc[f.elapsed_time==1,'prediction']=f.loc[f.elapsed_time==1,'truth']+.1
    f.loc[f.elapsed_time==3,'prediction']=f.loc[f.elapsed_time==3,'truth']+.4
    mapping=pd.DataFrame([dict(dataset_id='a',event_id=str(e),elapsed_time=t,draw=i) for e in range(2) for i,t in enumerate([1,1,3])])
    assert random_endpoint(f,mapping)==pytest.approx(.2)
    losses=random_loss_table(f,mapping,'noninput');assert interval(losses,draws=10)['estimate']==pytest.approx(.2)
    assert losses.targets.eq(3).all()


def test_exact_population_no_silent_intersection():
    f=frame();population_audit(f,production=False);assert_same_population(f,f.copy())
    with pytest.raises(ValueError):assert_same_population(f,f.iloc[1:])
    b=f.copy();b.loc[0,'input_ids']='["5"]'
    with pytest.raises(ValueError):assert_same_population(f,b)


def test_station_recovery_unique_event_count_and_recompute_bootstrap():
    f=frame();reference=dict(no_site_coefficients=[0]*6,site_reference_coefficients=[0]*6,train_site_effects={'0':.2})
    rows,summary,stability=station_recovery(f,reference,draws=101)
    assert rows.events.max()==16;assert summary.draws.min()==101
    assert 'elapsed_time' in stability and 'geometry_protocol' in stability
    assert rows.station_in_train_label_cohort.sum()>0
    assert summary.station_equal_recovery_mae.max()<1e-12


def test_metadata_case_and_probe_rules_ignore_predictions():
    f=frame();first=select_cases(f);f.prediction=1e20
    assert first==select_cases(f)
    probes=verification_decisions(f);assert 32<=len(probes)<=40
    assert probes.input_count.eq(1).any() and probes.input_count.gt(1).any()
