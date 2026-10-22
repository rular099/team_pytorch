import numpy as np
import pandas as pd
import pytest
from fe01_a01_compare.interactions import contrast_arrays,interval


def test_joint_difference_of_differences_not_significance_comparison():
    rows=[]
    strata=[]
    for event in range(30):
        for time in (1.,3.):
            for k in ('1','2'):
                # DiTing effect grows by +.2 only at later time and K2.
                rows.append(dict(dataset_id='d',event_id=str(event),geometry_protocol='normal',elapsed_time=time,
                                 delta=.2 if time==3 and k=='2' else 0))
                strata.append(dict(time=str(time),input_count=k,snr='Unknown'))
    f=pd.DataFrame(rows);s=pd.DataFrame(strata)
    arr,support=contrast_arrays(f,s,'time','input_count','3.0','2')
    assert support['common_domains']==1
    r=interval(arr,draws=30)
    assert r['estimate']==pytest.approx(.2)
    assert r['ci_low']==pytest.approx(.2)


def test_no_snr_metadata_contrast_is_not_filled():
    f=pd.DataFrame([dict(dataset_id='d',event_id='e',geometry_protocol='normal',elapsed_time=1.,delta=0.)])
    s=pd.DataFrame([dict(time='1.0',input_count='1',snr='Unknown')])
    arr,support=contrast_arrays(f,s,'time','snr','3.0','[0,10)')
    assert arr is None and support['status']=='NOT_IDENTIFIABLE_NO_COMMON_SUPPORT'
