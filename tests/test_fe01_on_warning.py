import numpy as np
import pandas as pd
import pytest
from fe01_a01_compare.map_readiness import times_to_ready,frame_metrics,gate,evaluate
from fe01_a01_compare.shadow import classify,evaluate as shadow
from fe01_a01_compare.synthetic import fixture
from fe01_a01_compare.contracts import contract


def frames(passed):
    return [dict(elapsed_time=i+1,station_pass=p) for i,p in enumerate(passed)]


def test_ttum_onset_confirmation_and_retreat():
    r=times_to_ready(frames([True,False,True,True,True,False]+[False]*14))
    assert r['first_observed_pass']==1 and 'left_censored' in r['first_status']
    assert r['stable_onset_retrospective']==3 and r['stable_confirmation']==5
    assert r['degradations_after_first']==2


def test_ttum_missing_breaks_and_tail_censoring():
    r=times_to_ready(frames([False]*17+[True,None,True]))
    assert r['stable_confirmation'] is None and r['missing_frames']==1
    assert r['stable_status']=='tail_right_censored'
    assert times_to_ready(frames([False]*20))['stable_status']=='not_reached_within_20s'
    # Old 1/3/5 snapshots cannot certify a three-second streak.
    r=times_to_ready([dict(elapsed_time=t,station_pass=True) for t in (1,3,5)])
    assert r['stable_confirmation'] is None


def test_constant_prediction_field_gate_and_small_n():
    f,_=fixture();f=f[f.family.eq('DiTing')&f.elapsed_time.eq(20)].copy()
    f['prediction']=f.truth.mean()
    m=frame_metrics(f,set(f.station_id))
    assert gate(m,10)['station_pass'] is True
    assert gate(m,.25)['field_pass'] is False
    assert frame_metrics(f,set(f.station_id[:1]))['status']=='Unknown'
    f['truth']=1.;f['prediction']=1.
    assert frame_metrics(f,set(f.station_id))['range_ratio'] is None


def alarms(alert=5,delivered=None):
    return pd.DataFrame(dict(elapsed_time=list(range(1,21)),status=['supported']*20,
                             probability=[1. if t==alert else 0. for t in range(1,21)],
                             numeric_ready_seconds=delivered if delivered is not None else list(range(1,21))))


@pytest.mark.parametrize('cross,tau,status',[(5,0,'late_or_zero'),(8,3,'timely'),(7,3,'positive_but_short'),(4,0,'late_or_zero')])
def test_shadow_boundary(cross,tau,status):
    assert classify(True,cross,'exact_within_record',alarms(),.5,tau)['status']==status


def test_no_alarm_horizon_missing_crossing_and_no_input():
    f=alarms(alert=99)
    assert classify(True,30,'exact_within_record',f,.5,0)['status']=='unknown_prediction_horizon'
    assert classify(True,10,'left_censored',f,.5,0)['status']=='unknown_crossing'
    assert classify(True,10,'exact_within_record',f.drop(index=5),.5,0)['status']=='missing_export'
    f['status']='no_input'
    assert classify(True,10,'exact_within_record',f,.5,0)['status']=='no_input'


def test_nonmonotonic_and_delivered_order():
    f=alarms(alert=10)
    f.loc[f.elapsed_time.eq(3),'probability']=1
    f['numeric_ready_seconds']=f.elapsed_time+1
    f.loc[f.elapsed_time.eq(3),'numeric_ready_seconds']=19
    r=classify(True,15,'exact_within_record',f,.5,0,'measured_compute_replay')
    assert r['alert_time']==11


def test_always_alarm_control_exposes_false_alarms_and_empty_H():
    f,tr=fixture()
    f['mdn_mu']='[10.0]'
    o,s=shadow(f,tr,contract()['shadow'])
    nonaffected=o[~o.affected]
    assert nonaffected.status.eq('false_alarm').all()
    assert s.false_positive_rate.dropna().eq(1).all()
    tr['affected']=False
    _,empty=shadow(f,tr,contract()['shadow'])
    assert empty.failure_fraction_lower.isna().all()


def test_fixed_panel_keeps_queries_that_become_inputs():
    f,_=fixture()
    _,t=evaluate(f)
    allpanel=t[t.panel.eq('fixed_all_targets')]
    assert allpanel.targets.eq(6).all()
    never=t[t.panel.eq('fixed_never_input_targets')]
    assert never.targets.eq(4).all()


def test_missing_delivery_clock_is_not_a_true_negative_or_certain_late_alarm():
    f=alarms();f['numeric_ready_seconds']=np.nan
    r=classify(False,None,'not_exceeded',f,.5,0,'measured_compute_replay')
    assert r['status']=='unknown_negative' and not r['complete_observation']
    r=classify(True,12,'exact_within_record',f,.5,0,'measured_compute_replay')
    assert r['status']=='unknown_clock'
    f=alarms(alert=10);f.loc[f.elapsed_time.eq(3),'probability']=1.;f.loc[f.elapsed_time.eq(3),'numeric_ready_seconds']=np.nan
    assert classify(True,9,'exact_within_record',f,.5,0,'measured_compute_replay')['status']=='unknown_clock'


def test_all_no_input_has_unknown_probabilities_without_requiring_fake_mdn():
    from fe01_a01_compare.shadow import probabilities
    f=pd.DataFrame({'status':['no_input','unsupported_history']})
    assert np.isnan(probabilities(f,-1.2)).all()
