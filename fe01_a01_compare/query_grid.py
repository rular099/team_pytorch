"""Static metric grid and shared input geometry; no predictions used."""
import json
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from scipy.interpolate import LinearNDInterpolator
from fe01.spatial import distance_km
from .io import fingerprint


def project(coords, center=None):
    c = np.asarray(coords, float)
    center = np.asarray(center if center is not None else c[:, :2].mean(0))
    # Declared local equirectangular metric projection; not an equal-area support map.
    x = (c[:, 1] - center[1]) * 111.195 * np.cos(np.deg2rad(center[0]))
    y = (c[:, 0] - center[0]) * 111.195
    return np.column_stack([x, y]), center


def geometry_covariates(query_coords, inputs):
    q, inp = np.asarray(query_coords), np.asarray(inputs)
    if not len(inp):
        return [dict(nearest_input_distance_km=np.nan, hull_status='no_input',
                     input_span_km=np.nan, azimuth_gap_degrees=np.nan) for _ in q]
    nearest = distance_km(q[:, 0, None], q[:, 1, None], inp[None, :, 0], inp[None, :, 1]).min(1)
    xy, origin = project(inp)
    qp, _ = project(q, origin)
    span = float(distance_km(inp[:, 0, None], inp[:, 1, None], inp[None, :, 0], inp[None, :, 1]).max())
    hull = None
    if len(inp) >= 3 and np.linalg.matrix_rank(xy-xy.mean(0)) == 2:
        try:
            hull = Delaunay(xy)
        except QhullError:
            pass
    result = []
    for i, point in enumerate(qp):
        angle = np.sort(np.arctan2(xy[:, 1]-point[1], xy[:, 0]-point[0]))
        gap = float(np.rad2deg(np.diff(np.r_[angle, angle[0]+2*np.pi]).max()))
        result.append(dict(nearest_input_distance_km=float(nearest[i]), input_span_km=span,
                           azimuth_gap_degrees=gap,
                           hull_status='degenerate' if hull is None else 'inside' if hull.find_simplex(point) >= 0 else 'outside'))
    return result


def make_grid(coords, config):
    coords = np.asarray(coords, float)
    xy, origin = project(coords)
    pad, base, cap = config['event_station_bounds_buffer_km'], config['base_spacing_km'], config['max_points_per_event']
    lo, hi = xy.min(0)-pad, xy.max(0)+pad
    multiplier = 1
    while True:
        spacing = base * multiplier
        xs = np.arange(np.floor(lo[0]/spacing), np.ceil(hi[0]/spacing)+1) * spacing
        ys = np.arange(np.floor(lo[1]/spacing), np.ceil(hi[1]/spacing)+1) * spacing
        if len(xs)*len(ys) <= cap:
            break
        multiplier += 1
    xx, yy = np.meshgrid(xs, ys)
    lat = origin[0]+yy.ravel()/111.195
    lon = origin[1]+xx.ravel()/(111.195*np.cos(np.deg2rad(origin[0])))
    height = np.full(len(lat), np.nan)
    status = 'station_supported_display_only'
    if len(coords) >= 3 and np.linalg.matrix_rank(xy-xy.mean(0)) == 2:
        try:
            height = np.asarray(LinearNDInterpolator(xy, coords[:, 2])(xx.ravel(), yy.ravel()))
            status = 'metadata_estimated_station_height_linear_inside_hull_km'
        except QhullError:
            pass
    points = [[float(a), float(b), float(c) if np.isfinite(c) else None] for a, b, c in zip(lat, lon, height)]
    spec = dict(protocol='grid_replay_v4', points=points, shape=[len(ys), len(xs)],
                spacing_km=spacing, projection='local equirectangular km', projection_origin=origin.tolist(),
                third_coordinate='station height km above datum; builder stationheight(m)/1000',
                third_coordinate_status=status, invalid_height_points=int((~np.isfinite(height)).sum()),
                truth='none; station truth retrospective only', bounds_rule='all static candidate stations plus20km',
                metadata_estimated=True, area_support='NA_no_common_independent_equal_area_support')
    spec['grid_sha256'] = fingerprint(spec)
    return spec


def input_plan(event, elapsed, geometry, capacity=25, random_counts=(1, 3, 5, 8, 12, 16)):
    from fe01.windows import clock
    from fe01.sampling import stable_rng
    current, cutout = clock(event['reference_sample'], elapsed, event['fs'])
    begin = int(np.floor(event['reference_sample']))-500
    lo, hi = max(0, begin), min(event['storage'].shape[-1], cutout)
    mask = event['storage'][:, lo:hi]
    if mask.ndim == 3:
        mask = mask.all(1)
    eligible = np.flatnonzero((event['picks'] > 0) & (event['picks'] <= current) & mask.any(-1))
    if geometry == 'random':
        rng = stable_rng(42, event['dataset_id'], event['event_id'], 0, 0, 'geometry')
        count = int(rng.choice(random_counts))
        priority = rng.permutation(len(event['ids']))
        available = set(eligible.tolist())
        chosen = np.asarray([i for i in priority if i in available][:count], dtype=int)
    else:
        chosen = eligible[np.argsort(event['picks'][eligible], kind='stable')][:capacity]
    # Keep original rule leaving a legal query outside random inputs. Uses only target identity, no label value.
    if geometry == 'random' and len(event['queries']) and set(event['queries']).issubset(set(chosen)):
        chosen = chosen[chosen != event['queries'][-1]]
    chosen = chosen[np.argsort(event['picks'][chosen], kind='stable')][:capacity]
    return chosen
