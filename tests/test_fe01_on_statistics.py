import json
import numpy as np
import pandas as pd
import pytest
from fe01_a01_compare.synthetic import fixture
from fe01_a01_compare.statistics import summarize,bootstrap_pair,paired,robustness
from fe01_a01_compare.io import csv_read,csv_write
from fe01_a01_compare.stratification import bins


def sample():
    f,_=fixture()
    a=f[f.family.eq('DiTing')].copy()
    # Fixture has a different family per row; physical request identity is common.
    for c,value in [('requested_decision_sample',600.),('current_sample',600),('cutout_exclusive',601),
                    ('history_start_sample',0),('history_end_sample',600),('latest_received_sample',600)]:
        a[c]=value
    b=a.copy()
    b['prediction']=b.prediction+.2
    return a,b


def test_paired_sign_and_transformed_rmse():
    a,b=sample()
    r=bootstrap_pair(a,b,times=(1,3,5),draws=30)
    by={x['metric']:x for x in r}
    assert by['mae']['estimate']==pytest.approx(-.2)
    assert by['mae']['ci_high']<0
    am=summarize(a,times=(1,3,5))
    bm=summarize(b,times=(1,3,5))
    assert by['rmse']['estimate']==pytest.approx(am['equal_cell_rmse']-bm['equal_cell_rmse'])
    assert by['rmse']['estimate']!=pytest.approx(-np.sqrt(abs(by['mse']['estimate'])))


def test_cell_equal_and_pooled_quantiles():
    a,b=sample()
    a=a[a.elapsed_time.isin([1,3])].copy()
    a.loc[a.elapsed_time.eq(1),'prediction']=a.loc[a.elapsed_time.eq(1),'truth']
    a.loc[a.elapsed_time.eq(3),'prediction']=a.loc[a.elapsed_time.eq(3),'truth']+1
    a=pd.concat([a,a[a.elapsed_time.eq(1)]]*2,ignore_index=True)
    m=summarize(a,(1,3))
    assert m['equal_cell_mae']==pytest.approx(.5)
    assert m['pooled_mae']==pytest.approx(1/3)
    assert m['pooled_p95_abs']==pytest.approx(1.)
    assert m['missing_cells']==2  # no random geometry


def test_population_mismatch_never_intersection():
    a,b=sample()
    with pytest.raises(ValueError,match='size'):
        paired(a,b.iloc[1:])
    b.loc[b.index[0],'truth']+=1
    with pytest.raises(ValueError,match='labels'):
        paired(a,b)


def test_low_variance_and_empty():
    a,_=sample()
    a['truth']=1.
    assert summarize(a)['pooled_r2'] is None
    assert summarize(a.iloc[:0])['status']=='NA_EMPTY'


def test_event_contribution_and_leave_one_out():
    a,b=sample()
    c=a.copy();d=b.copy()
    c['event_id']='e2';d['event_id']='e2'
    r,s=robustness(pd.concat([a,c]),pd.concat([b,d]))
    assert r.additive_primary_contribution.sum()==pytest.approx(s['primary_delta'])
    assert s['event_win_rate']==1
    assert s['loo_min']==pytest.approx(-.2)


def test_roundtrip_is_required(tmp_path):
    rng=np.random.default_rng(23)
    x=rng.normal(size=2000).astype(np.float32).astype(float)
    f=pd.DataFrame({'truth':x})
    path=tmp_path/'p.csv'
    csv_write(f,path)
    assert np.array_equal(csv_read(path).truth.to_numpy(),x)
    assert not np.array_equal(pd.read_csv(path).truth.to_numpy(),x)


def test_fixed_bin_boundaries_unknown():
    b=bins(pd.Series([0.,10.,20.,np.nan]),[0,10,20],['<0','0-10','10-20','20+'],right=False)
    assert list(b)==['0-10','10-20','20+','Unknown']
