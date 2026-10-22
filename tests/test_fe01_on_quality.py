import numpy as np
import pytest
from fe01_a01_compare.quality import station_snr
from fe01_a01_compare.crossings import first_crossing
from fe01_a01_compare.clocks import audit_clock,fifo,jst_utc
from fe01_a01_compare.query_grid import make_grid,geometry_covariates,input_plan
from fe01_a01_compare.contracts import contract
from fe01.windows import build_window,CAPABILITIES,clock


def wave():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(3,1200))*.01
    x[:,600:700]*=10
    return x,np.ones(1200,bool)


def test_snr_missing_padding_is_not_noise():
    x,m=wave();m[:600]=False
    r=station_snr(x,m,600,700)
    assert r['status']=='Unknown'
    assert r['reason']=='insufficient_valid_noise_or_signal'


def test_snr_zero_noise_is_unknown_and_legal_zeros_not_masked():
    x,m=wave();x[:,:600]=0
    assert station_snr(x,m,600,700)['reason']=='zero_or_invalid_energy'
    x,m=wave();x[:,100]=0
    assert station_snr(x,m,600,700)['noise_coverage']==1


def test_snr_future_and_signal_availability():
    x,m=wave()
    assert station_snr(x,m,600,640)['status']=='Unknown'
    a=station_snr(x,m,600,700)
    x[:,700:]=np.inf
    b=station_snr(x,m,600,700)
    assert a['snr_db']==b['snr_db']
    assert station_snr(x,m,600,700,clipping=True)['status']=='Unknown'


def test_crossing_first_not_peak_and_gaps():
    x=np.zeros((3,100));x[0,20]=.2;x[0,60]=1
    m=np.ones(100,bool)
    y=np.log10(1+1e-10)
    assert first_crossing(x,m,0,100,100,-1,y)['crossing_sample']==20
    m[10]=False
    assert first_crossing(x,m,0,100,100,-1,y)['reason']=='gap_before_first_observed_crossing'
    m[:]=True;x[0,0]=.3
    assert first_crossing(x,m,0,100,100,-1,y)['status']=='left_censored'
    assert first_crossing(x,m,0,100,100,-1,2)['reason']=='peak_label_definition_mismatch'


def test_jst_utc_and_alignment():
    assert jst_utc('2020-01-01 09:00:00')==jst_utc('2020-01-01T00:00:00Z')
    r=audit_clock(['2020-01-01 09:00:00','2020-01-01 09:00:01'],[0,100],100,'2020-01-01 08:59:55',600)
    assert r['status']=='verified'
    assert r['first_p_minus_origin_seconds']==11
    assert audit_clock(['2020-01-01 09:00:00','2020-01-01 09:00:02'],[0,100],100)['status']=='unknown_clock'


def test_fifo_queue_does_not_assume_one_second_completion():
    r=fifo([1,2,3],[2,2,2])
    assert [x['numeric_ready_seconds'] for x in r]==[3,5,7]
    assert [x['queue_wait_seconds'] for x in r]==[0,1,2]


def test_grid_cap_units_fixed_and_degenerate():
    c=contract()['grid'];c=dict(c,max_points_per_event=100)
    coords=np.array([[30,130,.1],[30,134,.2],[35,130,.3],[35,134,.4]])
    a=make_grid(coords,c);b=make_grid(coords,c)
    assert a==b and len(a['points'])<=100
    assert a['spacing_km']>=10
    assert any(p[2] is None for p in a['points'])
    assert 'km' in a['third_coordinate']
    assert make_grid(coords[:1],c)['third_coordinate_status']=='station_supported_display_only'
    q=geometry_covariates(coords[:1],coords[:2])[0]
    assert q['nearest_input_distance_km']==0
    assert q['hull_status']=='degenerate'


@pytest.mark.parametrize('future',[np.nan,np.inf,1e8])
def test_causal_window_future_and_last_legal(future):
    x,m=wave();x=x[None];m=m[None]
    c=CAPABILITIES['phasenet_pretrained_frozen']
    a,am,info=build_window(x,m,600,1,c)
    changed=x.copy();changed[...,701:]=future
    b,bm,_=build_window(changed,m,600,1,c)
    assert np.array_equal(a,b) and np.array_equal(am,bm)
    changed=x.copy();changed[...,700]+=10
    b,_,_=build_window(changed,m,600,1,c)
    assert not np.array_equal(a,b)
    assert info['cutout_exclusive']==701
    assert build_window(x,m,600,40,c)[2]['status']=='unsupported'


def test_tiny_positive_noise_does_not_manufacture_high_snr():
    rng=np.random.default_rng(3)
    x=(rng.normal(size=(3,1200))*1e-22).astype(np.float32)
    x[:,600:701]=rng.normal(size=(3,101)).astype(np.float32)*.01
    r=station_snr(x,np.ones(1200,bool),600,701)
    assert r['status']=='Unknown' and r['reason']=='noise_energy_below_numeric_resolution'
    assert r['snr_db'] is None
