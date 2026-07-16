"""MOCRE-1 shared utilities: config, geometry, fuzzy memberships, anomaly normalizer."""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_name_from_argv(default="segment_sjc.json"):
    """First CLI arg, if given, is the config filename (with or without .json)."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        name = sys.argv[1]
        return name if name.endswith(".json") else name + ".json"
    return default


def load_config(name=None):
    name = name or config_name_from_argv()
    with open(os.path.join(ROOT, "config", name)) as f:
        return json.load(f)


def segment_paths(cfg):
    """Per-segment data/out dirs + slug, namespaced by segment_id. Callers build filenames as
    f"{something}_{slug}.ext" -- explicit, no magic."""
    slug = cfg["segment_id"].lower().replace("_", "-")
    cat_dir = os.path.join(ROOT, "data", "catalog", slug)
    gnss_dir = os.path.join(ROOT, "data", "gnss", slug)
    out_dir = os.path.join(ROOT, "out")
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(gnss_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    return {
        "slug": slug,
        "catalog_csv": os.path.join(cat_dir, "events.csv"),
        "gnss_dir": gnss_dir,
        "stations_json": os.path.join(gnss_dir, "stations.json"),
        "out_dir": out_dir,
    }


# ---------------------------------------------------------------- geometry
def local_xy_km(lat, lon, ref_lat, ref_lon):
    """Equirectangular projection to km around a reference point."""
    kx = 111.32 * math.cos(math.radians(ref_lat))
    ky = 110.57
    return (np.asarray(lon) - ref_lon) * kx, (np.asarray(lat) - ref_lat) * ky


def _load_polyline_trace(cfg):
    """Real multi-segment trace (USGS/CA GIS Quaternary Fault DB), if configured."""
    tr = cfg["fault_trace"].get("polyline_file")
    if not tr:
        return None
    with open(os.path.join(ROOT, "config", tr)) as f:
        d = json.load(f)
    return d["polylines"]  # list of [ [lon,lat], ... ] chains


def dist_to_trace_km(lat, lon, cfg):
    """Distance (km) and side (+1 NE / -1 SW) relative to the fault trace.

    Uses the real multi-segment mapped trace when config provides one
    (cfg['fault_trace']['polyline_file']); falls back to the 2-point approximation
    (cfg['fault_trace']['p1']/['p2']) otherwise. Side is relative to the mean strike."""
    polylines = _load_polyline_trace(cfg)
    lat_a, lon_a = np.atleast_1d(np.asarray(lat, dtype=float)), np.atleast_1d(np.asarray(lon, dtype=float))

    if polylines is None:
        p1, p2 = cfg["fault_trace"]["p1"], cfg["fault_trace"]["p2"]
        ref_lat, ref_lon = (p1["lat"] + p2["lat"]) / 2.0, (p1["lon"] + p2["lon"]) / 2.0
        x, y = local_xy_km(lat_a, lon_a, ref_lat, ref_lon)
        x1, y1 = local_xy_km(p1["lat"], p1["lon"], ref_lat, ref_lon)
        x2, y2 = local_xy_km(p2["lat"], p2["lon"], ref_lat, ref_lon)
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / L2, 0.0, 1.0)
        px, py = x1 + t * dx, y1 + t * dy
        dist = np.hypot(x - px, y - py)
        side = np.sign(dx * (y - y1) - dy * (x - x1))
        return (dist, side) if len(dist) > 1 else (dist[0], side[0])

    allpts = np.array([p for line in polylines for p in line])
    ref_lat, ref_lon = allpts[:, 1].mean(), allpts[:, 0].mean()
    x, y = local_xy_km(lat_a, lon_a, ref_lat, ref_lon)
    best_dist = np.full(len(x), np.inf)
    for line in polylines:
        lx, ly = local_xy_km([p[1] for p in line], [p[0] for p in line], ref_lat, ref_lon)
        for i in range(len(lx) - 1):
            x1, y1, x2, y2 = lx[i], ly[i], lx[i + 1], ly[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy + 1e-9
            t = np.clip(((x - x1) * dx + (y - y1) * dy) / L2, 0.0, 1.0)
            px, py = x1 + t * dx, y1 + t * dy
            d = np.hypot(x - px, y - py)
            best_dist = np.minimum(best_dist, d)
    az = math.radians(cfg["gnss"]["fault_azimuth_deg"])
    ux, uy = math.sin(az), math.cos(az)  # strike unit vector
    side = np.sign(-uy * x + ux * y)  # cross product sign vs strike, arbitrary but consistent
    return (best_dist, side) if len(best_dist) > 1 else (best_dist[0], side[0])


def fault_strike_azimuth_deg(cfg):
    """Mean strike (deg from north) of the real trace via PCA on all vertices."""
    polylines = _load_polyline_trace(cfg)
    if polylines is None:
        return cfg["gnss"]["fault_azimuth_deg"]
    allpts = np.array([p for line in polylines for p in line])
    ref_lat, ref_lon = allpts[:, 1].mean(), allpts[:, 0].mean()
    x, y = local_xy_km(allpts[:, 1], allpts[:, 0], ref_lat, ref_lon)
    xy = np.column_stack([x, y])
    xy -= xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy, full_matrices=False)
    dx, dy = Vt[0]
    if dy < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(dx, dy)) % 360.0


# ---------------------------------------------------------------- fuzzy
def get_membership(x, params):
    """Generic trimf (3 params) / trapmf (4 params), same contract as Avaltix engine_helper."""
    x = np.asarray(x, dtype=float)
    if len(params) == 3:
        a, b, c = params
        left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
        right = np.where(c > b, (c - x) / (c - b + 1e-12), 1.0)
        return np.clip(np.minimum(left, right), 0.0, 1.0)
    a, b, c, d = params
    left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
    right = np.where(d > c, (d - x) / (d - c + 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0.0, 1.0)


def membership_centroid(params):
    """Area centroid of a triangular or trapezoidal membership function.

    Several MOCRE training paths turn the extreme output terms into shoulder
    trapezoids.  Treating those four parameters as a triangle and dividing
    their sum by three can place the representative value outside the declared
    universe.  This helper keeps one mathematically correct implementation for
    every engine.
    """
    p = [float(v) for v in params]
    if len(p) == 3:
        a, b, c = p
        return (a + b + c) / 3.0
    if len(p) != 4:
        raise ValueError("membership params must contain 3 or 4 values")
    a, b, c, d = p
    if not a <= b <= c <= d:
        raise ValueError(f"invalid trapezoid geometry: {p}")
    parts = []
    if b > a:
        parts.append(((b - a) / 2.0, (a + 2.0 * b) / 3.0))
    if c > b:
        parts.append((c - b, (b + c) / 2.0))
    if d > c:
        parts.append(((d - c) / 2.0, (2.0 * c + d) / 3.0))
    area = sum(v[0] for v in parts)
    if area <= 0:
        return (a + d) / 2.0
    return sum(ar * x for ar, x in parts) / area


def causal_reindex(series, index, limit=None):
    """Reindex a time series without ever consulting a future observation.

    ``pandas.interpolate`` fills an historical gap from both endpoints and is
    therefore look-ahead leakage in a pseudo-prospective forecast.  A bounded
    forward fill is intentionally less smooth but has an explicit information
    cutoff: value at *t* can only come from an observation at or before *t*.
    """
    import pandas as pd

    s = pd.Series(series).sort_index()
    target = pd.DatetimeIndex(index)
    union = s.index.union(target).sort_values()
    return s.reindex(union).ffill(limit=limit).reindex(target)


# Universal 5-partition over anomaly space [-1, 1].
# Recalibrated 2026-07-03 (was a verbatim copy of Avaltix's ema200_sets, wrong domain): under
# a quiet-regime null z~N(0,1), a=tanh(z/tau) has median~0.32, q90~0.68, q99~0.86 (tau=2) — the
# original knots put "alto" onset at a=0.02, essentially firing on any positive noise. Widened
# "normal" to track the real bulk of the null distribution; alto/muy_alto now require genuinely
# elevated z (roughly >q75 / >q90), not "greater than zero". Derived from the tanh(z/tau)
# distribution, not fit to any one segment's outcome.
#
# RUSPINI-CORRECTED 2026-07-05 (Matt's audit of the fuzzy engine found the 3-jul partition was
# NOT a real Ruspini partition -- verified numerically: sum of memberships was != 1 across 76% of
# the universe, e.g. sum=0.50 at x=-0.60. Cause: adjacent terms' ascent/descent intervals were
# offset from each other instead of sharing the exact same interval (a real gap, not just an
# approximation). Fixed via the standard "shared-foot chain" construction (7 knots q0..q6, each
# interior triangle's ascent = its left neighbor's descent, verified sum=1.0 exactly, tolerance
# 1e-9, not just "close"). This shifts alto/muy_alto onset earlier (alto peak 0.55->0.35,
# muy_alto plateau start 0.80->0.70) -- a real, not cosmetic, change to every rule's firing
# strength across the whole pipeline. Re-verified CSEP on all 9 segments after this change (see
# fuzzycore-cre1-modelo-sismico.md memory for before/after numbers) -- no regression found.
ANOMALY_SETS = {
    "muy_bajo": [-1.0, -1.0, -0.70, -0.35],
    "bajo": [-0.70, -0.35, 0.0],
    "normal": [-0.35, 0.0, 0.35],
    "alto": [0.0, 0.35, 0.70],
    "muy_alto": [0.35, 0.70, 1.0, 1.0],
}


STATE01_SETS = {
    "muy_bajo": [0.0, 0.0, 0.175, 0.35],
    "bajo": [0.175, 0.35, 0.50],
    "normal": [0.35, 0.50, 0.65],
    "alto": [0.50, 0.65, 0.825],
    "muy_alto": [0.65, 0.825, 1.0, 1.0],
}


C_STATE_SETS = {
    # Raw recurrence loading is intentionally not clipped to [0,1]. The fuzzy gate starts rising
    # around 0.6 and is fully high around 1.2, while overdue states can continue up to the raw cap 3.
    "muy_bajo": [0.0, 0.0, 0.20, 0.45],
    "bajo": [0.20, 0.45, 0.60],
    "normal": [0.45, 0.60, 0.90],
    "alto": [0.60, 0.90, 1.20],
    "muy_alto": [0.90, 1.20, 3.0, 3.0],
}


INPUT_DOMAINS = {
    "A_state": [0.0, 1.0],
    "C_state": [0.0, 3.0],
    "K_state": [0.0, 1.0],
    "E_state": [0.0, 1.0],
    "F_state": [0.0, 1.0],
}


INPUT_SETS = {
    "A_state": STATE01_SETS,
    "C_state": C_STATE_SETS,
    "K_state": STATE01_SETS,
    "E_state": STATE01_SETS,
    "F_state": STATE01_SETS,
}


def sets_for_var(var_name):
    return INPUT_SETS.get(var_name, ANOMALY_SETS)


def domain_for_var(var_name):
    return INPUT_DOMAINS.get(var_name, [-1.0, 1.0])


def fuzzify_variable(var_name, value):
    sets = sets_for_var(var_name)
    return {name: get_membership(value, p) for name, p in sets.items()}


def fuzzify_anomaly(a):
    return {name: get_membership(a, p) for name, p in ANOMALY_SETS.items()}


# ---------------------------------------------------------------- normalizer
def robust_anomaly(series, baseline_days, tau, min_periods=None, return_raw_z=False):
    """Rolling robust z-score (median/MAD over trailing window) -> tanh(z/tau) in (-1,1).

    Uses only past data at each point (no lookahead). min_periods defaults to baseline_days//2
    (original behavior, correct for daily-dense channels like GNSS/seismicity/forward-filled
    monthly data). MUST be overridden lower for genuinely SPARSE channels: found real 2026-07-04
    that a_insar (InSAR, ~1 real sample per ~30 days, ~3% density) was silently 100% NaN end to
    end -- min_periods=182 (half of a 365-day window) can never be satisfied by a series with
    only ~8-12 real values total in its entire history, no matter how much real data exists.
    Verified with a synthetic reproduction before fixing: 0 non-NaN outputs from 10 real samples
    over 6 years at 30-day spacing. Silent, not an exception -- exactly why this needed a real
    audit, not just "the code runs".

    return_raw_z=True returns the untransformed z (median/MAD, no tau, no tanh) -- needed for
    real tau calibration: reconstructing z from an already-saturated tanh(z/tau) output via
    arctanh is NOT reliable once |tanh output| is close to 1 (float64 arctanh(0.999999) caps
    around +-7.4, an artifact of the clip, not the true z -- found this the hard way 2026-07-04
    auditing tau_gnss/tau_sse/tau_foreshock/tau_strain, several of which showed suspiciously
    identical "p99" values across unrelated channels/segments that turned out to be exactly this
    clipping ceiling, not real data). z itself does not depend on tau (tau only scales z right
    before the tanh), so this is safe to compute independently of whatever tau was previously
    guessed for that channel."""
    import pandas as pd

    if min_periods is None:
        min_periods = baseline_days // 2
    s = pd.Series(series)
    med = s.rolling(baseline_days, min_periods=min_periods).median().shift(1)
    mad = (s - med).abs().rolling(baseline_days, min_periods=min_periods).median().shift(1)
    # Forecast for day t is issued at its start: the observation at t is not
    # available yet. Shift the complete statistic, not only its baseline.
    z = ((s - med) / (1.4826 * mad + 1e-9)).shift(1)
    if return_raw_z:
        return z.to_numpy()
    return np.tanh(z.to_numpy() / tau)


def poisson_anomaly(counts, baseline_days, tau, obs_window=14, min_periods=None,
                    return_raw_z=False):
    """Count anomaly for event-RATE channels (a_seis), correct where robust_anomaly's median/MAD
    degenerates: on seismically sparse segments the base rate is ~0 most days, the MAD collapses,
    and tanh saturates (Cascadia a_seis was 98% saturated at |v|>0.95 -- 98.45% zero-days). This
    is the prerequisite for a global tessellation, where most of GEM's ~13.7k faults are sparse.

    Model: a trailing Poisson base rate lambda (events/day), regularized with a Jeffreys-style
    (+0.5 event) prior so it is NEVER exactly 0. Compare the observed count in the recent
    obs_window against its Poisson expectation mu=lambda*obs_window, standardized by the Poisson
    sd sqrt(mu). Unlike median/MAD, sqrt(mu) shrinks gracefully as the segment gets quieter
    instead of collapsing -- seeing 1 event after a long silence gives a finite, moderate z, not
    a saturated one. Causal (base uses only past data via shift(1))."""
    import pandas as pd

    if min_periods is None:
        min_periods = baseline_days // 2
    s = pd.Series(counts).fillna(0.0)
    base_events = s.rolling(baseline_days, min_periods=min_periods).sum().shift(1)
    base_days = s.rolling(baseline_days, min_periods=min_periods).count().shift(1)
    lam = (base_events + 0.5) / (base_days + 1.0)          # events/day, Jeffreys prior -> always >0
    obs = s.rolling(obs_window, min_periods=1).sum().shift(1)  # only observations before day t
    mu = lam * obs_window                                   # expected count in that window
    z = (obs - mu) / np.sqrt(mu + 1e-9)                     # Poisson standardization
    if return_raw_z:
        return z.to_numpy()
    return np.tanh(z.to_numpy() / tau)
