"""Paired fixed-model event bootstrap; cell and pooled estimands stay separate."""
import numpy as np
import pandas as pd
from .contracts import EVENT, KEYS, TIMES
from .io import require

ADDITIVE = ['mae', 'mse', 'bias', 'nll', 'crps', 'brier', 'covered68', 'covered95',
            'predictive_sigma', 'width95']
ENDPOINTS = {'primary_10cell': TIMES, 'first_second_2cell': (1,), 'early_6cell': (1, 3, 5)}


def paired(a, b):
    from fe01_review.requests import IDENTITY_KEYS
    a = a.sort_values(KEYS).reset_index(drop=True)
    b = b.sort_values(KEYS).reset_index(drop=True)
    require(len(a) == len(b) and not a.duplicated(KEYS).any() and not b.duplicated(KEYS).any(), 'Pair size/duplicate mismatch')
    require(a[IDENTITY_KEYS].equals(b[IDENTITY_KEYS]), 'Pair keys/labels/inputs/clocks mismatch; no intersection')
    return a, b


def groups(f):
    return {'all': np.ones(len(f), bool), 'noninput': f.target_role.ne('observed_input').to_numpy(),
            'triggered_noninput': f.target_role.eq('triggered_noninput').to_numpy(),
            'untriggered_noninput': f.target_role.eq('untriggered_noninput').to_numpy(),
            'K1_noninput': (f.input_count.eq(1) & f.target_role.ne('observed_input')).to_numpy(),
            'Kmulti_noninput': (f.input_count.ge(2) & f.target_role.ne('observed_input')).to_numpy()}


def values(f):
    error = f.prediction.to_numpy() - f.truth.to_numpy()
    result = dict(mae=np.abs(error), mse=error ** 2, bias=error)
    for c in ADDITIVE[3:]:
        result[c] = f[c].to_numpy(dtype=float)
    return pd.DataFrame(result, index=f.index)


def summarize(f, times=TIMES):
    f = f[f.elapsed_time.isin(times)]
    if not len(f):
        return dict(status='NA_EMPTY', events=0, requests=0, supported_cells=0, missing_cells=2 * len(times))
    from fe01.metrics import summarize as pooled
    v = values(f)
    cell = v.groupby([f.geometry_protocol, f.elapsed_time]).mean()
    means = cell.mean()
    p = pooled(f)
    result = dict(status='FULL_SUPPORT' if len(cell) == len(times) * 2 else 'SPARSE_AVAILABLE_CELLS_ONLY',
                  events=f[EVENT].drop_duplicates().shape[0], stations=f.station_id.nunique(),
                  requests=len(f), decisions=f[EVENT + ['geometry_protocol', 'elapsed_time']].drop_duplicates().shape[0],
                  supported_cells=len(cell), missing_cells=2 * len(times) - len(cell),
                  **{'equal_cell_' + k: float(x) for k, x in means.items()},
                  equal_cell_rmse=float(np.sqrt(means['mse'])))
    result.update({'pooled_' + k: v for k, v in p.items()})
    result['pooled_low_variance_reason'] = 'truth variance <=1e-12: R2/slope NA' if np.var(f.truth) <= 1e-12 else ''
    if np.var(f.truth) <= 1e-12:
        result['pooled_r2'] = result['pooled_prediction_on_truth_slope'] = None
    return result


def arrays(f, times):
    f = f[f.elapsed_time.isin(times)]
    cells = pd.MultiIndex.from_product([('normal', 'random'), times], names=['geometry_protocol', 'elapsed_time'])
    keys = EVENT + ['geometry_protocol', 'elapsed_time']
    table = pd.concat([f[keys], values(f)], axis=1).groupby(keys).sum()
    counts = f.groupby(keys).size().unstack(['geometry_protocol', 'elapsed_time']).reindex(columns=cells).fillna(0)
    n = counts.to_numpy()
    sums = np.stack([table[k].unstack(['geometry_protocol', 'elapsed_time']).reindex(index=counts.index, columns=cells).fillna(0).to_numpy()
                     for k in ADDITIVE], axis=-1)
    return counts.index, sums, n


def bootstrap_pair(a, b, times=TIMES, draws=2000, seed=20261007):
    """a minus b. Recompute each model's sqrt(cell-MSE mean) inside each draw."""
    ai, av, an = arrays(a, times)
    bi, bv, bn = arrays(b, times)
    require(ai.equals(bi) and np.array_equal(an, bn), 'Bootstrap support differs')
    supported = an.sum(0) > 0
    if not supported.any():
        return []
    av, bv, an = av[:, supported], bv[:, supported], an[:, supported]
    denominator = an.sum(0)
    am, bm = (x.sum(0) / denominator[:, None] for x in (av, bv))
    estimates = (am - bm).mean(0)
    est_rmse = np.sqrt(am[:, 1].mean()) - np.sqrt(bm[:, 1].mean())
    rng = np.random.default_rng(seed)
    samples = []
    # The same event indices are used for every model, time, geometry and metric.
    for start in range(0, draws, 50):
        idx = rng.integers(0, len(ai), size=(min(50, draws-start), len(ai)))
        den = an[idx].sum(1)
        good = (den > 0).all(1)
        aa = av[idx[good]].sum(1) / den[good, :, None]
        bb = bv[idx[good]].sum(1) / den[good, :, None]
        delta = (aa - bb).mean(1)
        rmse = np.sqrt(aa[:, :, 1].mean(1)) - np.sqrt(bb[:, :, 1].mean(1))
        samples.append(np.column_stack([delta, rmse]))
    sample = np.concatenate(samples)
    out = []
    for k, estimate, distribution in zip(ADDITIVE + ['rmse'], list(estimates) + [est_rmse], sample.T):
        out.append(dict(metric=k, estimate=float(estimate), ci_low=float(np.quantile(distribution, .025)) if len(distribution) else None,
                        ci_high=float(np.quantile(distribution, .975)) if len(distribution) else None,
                        events=len(ai), requests=int(an.sum()), supported_cells=int(supported.sum()),
                        missing_cells=int((~supported).sum()), effective_replicates=len(sample),
                        empty_cell_fraction=1-len(sample)/draws, draws=draws, rng_seed=seed,
                        descriptive_only=len(ai) < 20,
                        status='CONDITIONAL_UNSTABLE' if len(sample) < .95*draws else
                               'SPARSE_AVAILABLE_CELLS_ONLY' if not supported.all() else 'FULL_SUPPORT',
                        estimand='targets within cell; supported cells equal; DiTing minus comparator',
                        uncertainty='fixed model event clusters; single seed; exploratory uncorrected'))
    return out


def robustness(a, b, times=TIMES):
    idx, av, n = arrays(a, times)
    j, bv, other = arrays(b, times)
    require(idx.equals(j) and np.array_equal(n, other), 'Event support mismatch')
    supported = n.sum(0) > 0
    n, d = n[:, supported], (av-bv)[:, supported, 0]
    den = n.sum(0)
    total = d.sum(0)
    primary = float((total / den).mean())
    rows = []
    for i, key in enumerate(idx):
        good = n[i] > 0
        macro = float(np.mean(d[i, good] / n[i, good]))
        left = den - n[i]
        loo = float(np.mean((total-d[i]) / left)) if (left > 0).all() else None
        rows.append(dict(dataset_id=key[0], event_id=key[1], event_equal_available_cell_delta=macro,
                         event_win=macro < 0, event_tie=macro == 0,
                         additive_primary_contribution=float(np.mean(d[i] / den)),
                         leave_one_event_out_primary_delta=loo, available_cells=int(good.sum())))
    r = pd.DataFrame(rows)
    absolute = r.additive_primary_contribution.abs().sort_values(ascending=False)
    summary = dict(primary_delta=primary, event_win_rate=float(r.event_win.mean()),
                   event_macro_median_delta=float(r.event_equal_available_cell_delta.median()),
                   event_macro_mean_delta=float(r.event_equal_available_cell_delta.mean()),
                   loo_min=float(r.leave_one_event_out_primary_delta.min()), loo_max=float(r.leave_one_event_out_primary_delta.max()),
                   absolute_contribution_top1_share=float(absolute.iloc[0]/absolute.sum()) if absolute.sum() else None,
                   absolute_contribution_top10_share=float(absolute.head(10).sum()/absolute.sum()) if absolute.sum() else None,
                   concentration_definition='absolute shares; signed contributions retained; not causal attribution')
    return r, summary


def bootstrap_mae(a,b,draws=2000,times=TIMES):
    """Stratum MAE intervals with the same event draws, without unused metric tensors."""
    keys=EVENT+['geometry_protocol','elapsed_time']
    aa=a.assign(_delta=np.abs(a.prediction-a.truth)-np.abs(b.prediction-b.truth))
    aa=aa[aa.elapsed_time.isin(times)]
    if not len(aa):
        return []
    t=aa.groupby(keys)._delta.agg(['sum','count'])
    cells=pd.MultiIndex.from_product([('normal','random'),times],names=['geometry_protocol','elapsed_time'])
    num=t['sum'].unstack(['geometry_protocol','elapsed_time']).reindex(columns=cells).fillna(0).to_numpy()
    den=t['count'].unstack(['geometry_protocol','elapsed_time']).reindex(columns=cells).fillna(0).to_numpy()
    supported=den.sum(0)>0
    num,den=num[:,supported],den[:,supported]
    rng=np.random.default_rng(20261007)
    sample=[]
    for start in range(0,draws,100):
        idx=rng.integers(0,len(num),size=(min(100,draws-start),len(num)))
        n,d=num[idx].sum(1),den[idx].sum(1)
        valid=(d>0).all(1)
        sample.extend((n[valid]/d[valid]).mean(1).tolist())
    return [dict(metric='mae',estimate=float((num.sum(0)/den.sum(0)).mean()),
                 ci_low=float(np.quantile(sample,.025)) if sample else None,ci_high=float(np.quantile(sample,.975)) if sample else None,
                 events=len(num),requests=int(den.sum()),supported_cells=int(supported.sum()),missing_cells=int((~supported).sum()),
                 draws=draws,effective_replicates=len(sample),empty_cell_fraction=1-len(sample)/draws,rng_seed=20261007,
                 descriptive_only=len(num)<20,status='CONDITIONAL_UNSTABLE' if len(sample)<.95*draws else 'SPARSE' if not supported.all() else 'FULL_SUPPORT',
                 estimand='DiTing minus comparator; targets/cell then supported cells equal',uncertainty='fixed model event clusters; exploratory uncorrected')]
