"""MOCRE-1 "InSAR" pass (4-jul-2026, continuation): real Sentinel-1 InSAR deformation as a
genuinely NEW variable -- NOT part of the 25-variable audited catalog. Source verified real
before writing this (same discipline as strain/wells/ocean/TEC): COMET-LiCS products, open, no
account, via JASMIN (gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products.public/).

ENGINEERING DECISION, stated plainly (same spirit as stress_graph.py's point-source-vs-Okada
choice): LiCSAR does NOT publish a ready-made cumulative displacement time series -- verified by
listing a real frame's directory structure myself: only raw pairwise unwrapped interferograms
(interferograms/{d1}_{d2}/*.geo.unw.tif) and a single STATIC long-term mean velocity map
(metadata/*.geo.vlos_eur.tif). Reconstructing a real absolute cumulative displacement history
would need a proper SBAS/small-baseline NETWORK INVERSION (Berardino et al. 2002) over thousands
of interferogram pairs per frame -- a legitimate but substantial standalone geodetic-processing
project (GB-scale downloads, redundant-network least-squares inversion), not a same-session
wire-in. Instead, following the exact same "rate, not absolute position" pattern this project
already uses for K_state (30-day rate vs 365-day trend from GNSS): use SHORT-BASELINE pairs
(preferring 12-day, the most complete network step per LiCSAR's own baseline distribution)
directly as short-term deformation RATE samples (displacement / pair span), fed into the same
robust_anomaly() pipeline as every other channel. This is honest and real -- it is NOT a
cumulative deformation reconstruction, and must not be presented as one.

Scope, stated plainly (same as TEC's 2-year window, wells' 2011-2024, etc.): real interferogram
archives run back to 2014-2019 depending on frame, but downloading+parsing the FULL archive for
4 frames is not a reasonable single-session backfill. BACKFILL_DAYS below bounds this to a real,
recent, honestly-documented window -- extend later if the channel proves useful.

Frame coverage confirmed real (downloaded+parsed an actual GeoTIFF per frame, not just directory
listings) for SJC-Anza+Mojave (shared frame), Parkfield, Cascadia -- Oklahoma has NO real LiCSAR
coverage (confirmed: no frame centroid near 36N/-97W across ~30 candidate tracks checked), and
Shumagin's frame was found by research but not independently re-verified with a real download in
this script-writing pass -- included with the same code path, flagged honestly if it fails.
"""
import os
import re
import sys
import time
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd
import tifffile

from common import ROOT, _load_polyline_trace, load_config, segment_paths

FRAME_TABLE = {
    "sjc-anza": ("173", "173D_05576_131313"),
    "saf-mojave": ("173", "173D_05576_131313"),
    "saf-parkfield": ("144", "144D_05298_131313"),
    "csz-central-or": ("137", "137A_04637_131313"),
    "aak-shumagin": ("153", "153A_03412_061209"),
    # ok-pawnee-prague: intentionally absent -- real, confirmed network gap, not a missing feature.
}
BASE = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products.public/{track}/{frame}"
LIST_CACHE = os.path.join(ROOT, "data", "insar", "_list_cache")
TIF_CACHE = os.path.join(ROOT, "data", "insar", "_tif_cache")
WAVELENGTH_M = 0.055465763  # Sentinel-1 C-band, standard published value
BACKFILL_DAYS = 425  # ~14 months -- real, bounded window, see module docstring
AOI_HALF_PX = 5  # ~1km box at 0.001 deg/px, small enough to stay near the fault centroid


def get(url, timeout=60, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def list_pairs(track, frame):
    os.makedirs(LIST_CACHE, exist_ok=True)
    cache_path = os.path.join(LIST_CACHE, f"{frame}.html")
    if not os.path.exists(cache_path):
        url = BASE.format(track=track, frame=frame) + "/interferograms/"
        html = get(url).decode(errors="replace")
        with open(cache_path, "w") as f:
            f.write(html)
    else:
        with open(cache_path) as f:
            html = f.read()
    pairs = re.findall(r'href="(\d{8})_(\d{8})/"', html)
    out = []
    for a, b in pairs:
        da, db = pd.Timestamp(a), pd.Timestamp(b)
        out.append((da, db, (db - da).days))
    return out


MAX_STACK_BASELINE = 48  # days -- widen beyond PREFERRED_BASELINES for stacking (see select_bins)


def select_bins(pairs, start, end, bin_days=30):
    """Group ALL short-to-moderate-baseline pairs (<=MAX_STACK_BASELINE) whose start date falls
    in each bin_days-wide window. Added after finding single-pair short-baseline rates are
    atmosphere-noise-dominated (real test on Parkfield: per-pair rates of several mm/day, ~1000x
    the real ~28mm/year creep rate, even after planar-ramp removal) -- averaging N independent
    interferograms sharing (most of) the same time window reduces uncorrelated atmospheric noise
    by ~sqrt(N), the standard "stacking" mitigation used in real InSAR practice when full SBAS
    network inversion isn't done. Returns {bin_start_date: [(d1,d2,gap), ...]}."""
    bins = {}
    for da, db, gap in pairs:
        if start <= da <= end and gap <= MAX_STACK_BASELINE:
            bin_key = start + pd.Timedelta(days=((da - start).days // bin_days) * bin_days)
            bins.setdefault(bin_key, []).append((da, db, gap))
    return bins


def fetch_tif(track, frame, d1, d2):
    cache_dir = os.path.join(TIF_CACHE, frame)
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"{d1:%Y%m%d}_{d2:%Y%m%d}.geo.unw.tif"
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        url = f"{BASE.format(track=track, frame=frame)}/interferograms/{d1:%Y%m%d}_{d2:%Y%m%d}/{fname}"
        try:
            data = get(url)
        except Exception as e:  # noqa: BLE001
            print(f"    {fname}: download failed ({e})")
            return None
        with open(path, "wb") as f:
            f.write(data)
    try:
        tif = tifffile.TiffFile(path)
        page = tif.pages[0]
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value
        arr = page.asarray()
    except Exception as e:  # noqa: BLE001
        print(f"    {fname}: parse failed ({e})")
        return None
    return arr, tie, scale


def trace_candidate_points(cfg):
    """Real fault-trace points to try for AOI placement, in priority order: trace midpoint first
    (the intended target), then along-trace points -- covers the real case (confirmed on
    Parkfield's actual data) where the exact midpoint sits in a genuinely low-coherence patch
    (agriculture/vegetation on the fault zone itself) while a nearby point on the same trace has
    good coverage within the same interferogram."""
    polylines = _load_polyline_trace(cfg)
    if polylines is not None:
        pts = [(lat, lon) for line in polylines for lon, lat in [line[0]] + [line[-1]]]
        n = len(pts)
        candidates = [pts[n // 2], pts[0], pts[-1], pts[n // 4], pts[3 * n // 4]]
    else:
        p1, p2 = cfg["fault_trace"]["p1"], cfg["fault_trace"]["p2"]
        mid = ((p1["lat"] + p2["lat"]) / 2, (p1["lon"] + p2["lon"]) / 2)
        candidates = [mid, (p1["lat"], p1["lon"]), (p2["lat"], p2["lon"])]
    seen, uniq = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def remove_planar_ramp(arr):
    """Fit and subtract a 2D plane (a + b*row + c*col) from the unwrapped phase over all real
    valid pixels, least-squares. Required correction (found on real data, replacing an earlier
    far-corner-reference attempt that made things WORSE): a single far reference pixel doesn't
    cancel a spatially-linear orbital/atmospheric ramp -- it maximizes exposure to it, since a
    linear ramp's contribution grows with distance from any one point. A full-image planar fit
    absorbs both the arbitrary per-interferogram unwrapping constant (the 'a' term) AND a
    first-order orbital/atmospheric trend (the 'b','c' terms) in one principled least-squares
    step -- standard practice for raw (non-atmosphere-corrected) InSAR products, not optional.
    Verified this is what was needed: pre-ramp-removal AOI rates were >1 m/year equivalent
    (impossible for the San Andreas); see module history in ingest_insar.py commit context."""
    # Nodata convention differs BY FRAME (found on real data, 4-jul): Parkfield/SJC/Mojave/Shumagin
    # use 0 for nodata, but Cascadia's frame uses NaN -- checking only `arr != 0` silently treated
    # every NaN pixel as "valid" (NaN != 0 is True in numpy), poisoning the lstsq fit and the
    # output median with NaN for that entire frame. valid_mask() below excludes both conventions
    # regardless of which one a given frame actually uses.
    mask = valid_mask(arr)
    rows, cols = np.nonzero(mask)
    if len(rows) < 1000:
        return None
    vals = arr[rows, cols]
    # subsample for speed -- a few thousand points is plenty to fit 3 plane parameters robustly
    if len(rows) > 20000:
        idx = np.random.default_rng(0).choice(len(rows), size=20000, replace=False)
        rows, cols, vals = rows[idx], cols[idx], vals[idx]
    A = np.column_stack([np.ones(len(rows)), rows, cols])
    coeffs, *_ = np.linalg.lstsq(A, vals, rcond=None)
    r_full, c_full = np.nonzero(mask)
    ramp = coeffs[0] + coeffs[1] * r_full + coeffs[2] * c_full
    corrected = arr.copy()
    corrected[r_full, c_full] = arr[r_full, c_full] - ramp
    return corrected


def valid_mask(arr):
    """True where a pixel carries real data -- excludes BOTH nodata conventions seen across real
    LiCSAR frames (0 in most, NaN in Cascadia's), see remove_planar_ramp() for how this was found."""
    return (arr != 0) & np.isfinite(arr)


def sample_aoi(arr, tie, scale, lat0, lon0, max_half_px=60):
    """Median unwrapped phase in a small window around (lat0, lon0). Real fault traces often sit
    on genuinely low-coherence ground (agriculture, dense vegetation, the fault gouge zone itself
    -- confirmed on real data: Parkfield's exact trace centroid has zero valid pixels in an
    11x11 window on real interferograms, not a parsing bug) -- expand the search window (up to
    max_half_px, ~12km at 0.001deg/px) around the intended point rather than silently requiring
    the literal centroid pixel, same honest-fallback spirit as strain's quality-flag filter."""
    px_scale_x, px_scale_y = scale[0], scale[1]
    tie_lon, tie_lat = tie[3], tie[4]
    col = int(round((lon0 - tie_lon) / px_scale_x))
    row = int(round((tie_lat - lat0) / px_scale_y))
    for half in (AOI_HALF_PX, 15, 30, max_half_px):
        r0, r1 = max(row - half, 0), min(row + half + 1, arr.shape[0])
        c0, c1 = max(col - half, 0), min(col + half + 1, arr.shape[1])
        win = arr[r0:r1, c0:c1]
        valid = win[valid_mask(win)]
        if valid.size >= 5:
            return float(np.median(valid))
    return None


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    slug = paths["slug"]
    print(f"segment: {cfg['segment_id']} ({slug})")

    out_dir = os.path.join(ROOT, "data", "insar", slug)
    os.makedirs(out_dir, exist_ok=True)
    agg_path = os.path.join(out_dir, "daily_aggregate.csv")

    if slug not in FRAME_TABLE:
        pd.DataFrame(columns=["los_rate_mm_day", "n_valid_px"]).to_csv(agg_path)
        print(f"{slug}: no real LiCSAR frame coverage (confirmed network gap, not a bug) -- "
              f"wrote empty file")
        return

    track, frame = FRAME_TABLE[slug]
    candidates = trace_candidate_points(cfg)
    print(f"frame: {frame} | trace candidate points (priority order): {candidates}")

    all_pairs = list_pairs(track, frame)
    end = pd.Timestamp(date.today())
    start = end - pd.Timedelta(days=BACKFILL_DAYS)
    bins = select_bins(all_pairs, start, end)
    n_pairs_total = sum(len(v) for v in bins.values())
    print(f"{len(bins)} ~30-day bins, {n_pairs_total} total pairs (<= {MAX_STACK_BASELINE}d "
          f"baseline) in [{start.date()}, {end.date()}] out of {len(all_pairs)} total pairs")

    # Lock in ONE AOI point for the whole series -- switching AOI point pair-to-pair would
    # introduce spurious jumps unrelated to real ground deformation. Real fault traces can have
    # genuine low-coherence patches (confirmed on Parkfield's own data: the exact trace midpoint
    # has zero valid pixels even in a ~12km search radius -- not a parsing bug), so try each
    # candidate in priority order against the first few real interferograms.
    first_bin_pairs = next(iter(sorted(bins.items())))[1] if bins else []
    lat0 = lon0 = None
    for d1, d2, gap in first_bin_pairs[:5]:
        result = fetch_tif(track, frame, d1, d2)
        if result is None:
            continue
        arr, tie, scale = result
        for clat, clon in candidates:
            if sample_aoi(arr, tie, scale, clat, clon) is not None:
                lat0, lon0 = clat, clon
                break
        if lat0 is not None:
            break
    if lat0 is None:
        pd.DataFrame(columns=["los_rate_mm_day", "n_valid_px"]).to_csv(agg_path)
        print(f"{slug}: none of the {len(candidates)} trace candidate points have valid "
              f"coverage in this frame -- real coherence gap, wrote empty file")
        return
    print(f"AOI locked to {lat0:.4f}, {lon0:.4f} (first candidate with real valid coverage)")

    rows = []
    for bin_start, pair_list in sorted(bins.items()):
        rates = []
        for d1, d2, gap in pair_list:
            result = fetch_tif(track, frame, d1, d2)
            if result is None:
                continue
            arr, tie, scale = result
            ramp_corrected = remove_planar_ramp(arr)
            if ramp_corrected is None:
                continue
            phase = sample_aoi(ramp_corrected, tie, scale, lat0, lon0)
            if phase is None:
                continue
            disp_mm = phase * WAVELENGTH_M * 1000.0 / (4 * np.pi)
            rates.append(disp_mm / gap)
        if not rates:
            continue
        stacked = float(np.median(rates))
        rows.append((bin_start + pd.Timedelta(days=15), stacked, len(rates)))
        if len(rows) % 5 == 0:
            print(f"  bin {bin_start.date()}: stacked_rate={stacked:.4f} mm/day from "
                  f"{len(rates)} pairs ({len(rows)}/{len(bins)})")

    if not rows:
        pd.DataFrame(columns=["los_rate_mm_day", "n_stacked_pairs"]).to_csv(agg_path)
        print(f"{slug}: no usable interferogram AOI samples -- wrote empty file (real gap: "
              f"either the frame doesn't actually cover the fault centroid, or all downloads "
              f"failed -- check log above)")
        return

    df = pd.DataFrame(rows, columns=["date", "los_rate_mm_day", "n_stacked_pairs"]).set_index("date")
    df = df.sort_index()
    df.to_csv(agg_path)
    print(f"wrote {len(df)} rate samples -> {agg_path}")
    print(f"real coverage window: {df.index.min()} -> {df.index.max()} (~14mo backfill of "
          f"short-baseline rate samples, NOT a cumulative displacement series -- see docstring)")
    print(df.tail(5))


if __name__ == "__main__":
    main()
