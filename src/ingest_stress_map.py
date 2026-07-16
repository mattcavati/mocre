"""MOCRE-1 "esfuerzo cortical" pass (4-jul-2026, continuation): World Stress Map as a per-segment
STATIC PRIOR -- NOT a time-varying a_X(t) anomaly channel like every other ingest_*.py in this
project. Verified this session (real download + parse of 100,842 rows) that WSM is fundamentally
a historical compilation of point measurements (focal mechanisms, borehole breakouts), each
anchored to its own past event/measurement date with no repeated observation at the same site --
there is no baseline-vs-anomaly structure to compute here, unlike strain/GNSS/wells/ocean-loading.
Implemented as a fixed feature computed ONCE per segment: circular-mean S_Hmax azimuth + dominant
tectonic regime + a consistency (circular concentration) score, from real GFZ Data Services data
(DOI 10.5880/WSM.2025.001, anonymous CSV download, no account) -- NOT wired into stress_graph.py's
already-validated Coulomb self_test() this session (see stress_graph.py comment at load_segments()
for why: that module's lobe_pattern()/mechanism_compat() physics are already self-tested against
real published decay/sign-pattern behavior, and properly resolving a real regional stress tensor
onto a receiver fault plane is a nontrivial derivation that deserves its own careful pass rather
than a same-session bolt-on risking a silent regression -- this ingests and computes the real prior
so that decision can be made deliberately, not blocked on data access).

S_Hmax azimuth is an AXIAL quantity (a stress orientation, not a direction: 10 deg and 190 deg are
the same axis) -- averaging it like a normal angle would be wrong (e.g. mean of 179 and 1 should be
~0/180, not ~90). Doubled-angle circular mean is standard practice for this exact reason (same fix
category as strain's symmetric bajo/alto, water's symmetric rule, ocean's symmetric rule -- this
project has hit "treating an axial/bidirectional quantity as if it had one preferred sign/direction"
as a bug pattern multiple times already; this is the SAME class of mistake, avoided up front here).
"""
import io
import os
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

from common import ROOT, load_config, segment_paths

WSM_URL = "https://datapub.gfz.de/download/10.5880.WSM.2025.001-Scbwez/WSM_Database_2025.csv"
WSM_CACHE = os.path.join(ROOT, "data", "stress_map", "_wsm_cache.csv")
GOOD_QUALITY = {"A", "B", "C"}  # WSM quality ranking A (best) .. E (worst) -- drop D/E, same
                                # spirit as strain's quality filter (keep 'g'/'i', drop 'b')
PAD_DEG = 2.0  # generous bbox padding, matches the real-coverage check already done this session


def fetch_wsm():
    os.makedirs(os.path.dirname(WSM_CACHE), exist_ok=True)
    if not os.path.exists(WSM_CACHE):
        with urllib.request.urlopen(WSM_URL, timeout=180) as r:
            data = r.read()
        with open(WSM_CACHE, "wb") as f:
            f.write(data)
    return pd.read_csv(WSM_CACHE, encoding="utf-8-sig", low_memory=False)


def circular_mean_axial(azimuths_deg):
    """Circular mean of an AXIAL quantity (period 180, not 360) -- double the angle, average the
    unit vectors, halve the result angle. Returns (mean_azimuth_deg in [0,180), R in [0,1] where
    R=1 is perfectly aligned and R=0 is uniformly scattered -- the standard circular concentration
    measure, NOT a linear stddev, since azimuth has no meaningful zero point)."""
    theta = np.radians(azimuths_deg * 2.0)
    c, s = np.mean(np.cos(theta)), np.mean(np.sin(theta))
    r = float(np.hypot(c, s))
    mean_az = (np.degrees(np.arctan2(s, c)) / 2.0) % 180.0
    return float(mean_az), r


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    print(f"segment: {cfg['segment_id']} ({paths['slug']})")
    b = cfg["catalog"]["bbox"]
    minlat, maxlat = b["minlat"] - PAD_DEG, b["maxlat"] + PAD_DEG
    minlon, maxlon = b["minlon"] - PAD_DEG, b["maxlon"] + PAD_DEG

    df = fetch_wsm()
    sub = df[(df["LAT"] >= minlat) & (df["LAT"] <= maxlat) &
             (df["LON"] >= minlon) & (df["LON"] <= maxlon)].copy()
    print(f"{len(sub)} WSM points in padded bbox ({minlat:.1f}-{maxlat:.1f}, "
          f"{minlon:.1f}-{maxlon:.1f})")

    sub_q = sub[sub["QUALITY"].isin(GOOD_QUALITY) & sub["AZI"].notna()]
    print(f"{len(sub_q)} after quality filter (A/B/C only, AZI present)")

    out_dir = os.path.join(ROOT, "data", "stress_map", paths["slug"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prior.json")

    if len(sub_q) < 5:
        prior = {"status": f"insufficient real WSM coverage ({len(sub_q)} quality A/B/C points) "
                            "-- honest network/data gap, not computed", "n_points": len(sub_q)}
    else:
        mean_az, r = circular_mean_axial(sub_q["AZI"].to_numpy())
        regime_counts = Counter(sub_q["REGIME"].dropna())
        dominant_regime = regime_counts.most_common(1)[0][0] if regime_counts else None
        prior = {
            "status": "ok",
            "shmax_azimuth_deg": round(mean_az, 1),
            "circular_consistency_r": round(r, 3),
            "dominant_regime": dominant_regime,
            "regime_counts": dict(regime_counts),
            "n_points": int(len(sub_q)),
            "n_points_total_any_quality": int(len(sub)),
            "quality_used": sorted(GOOD_QUALITY),
            "source": "World Stress Map 2025 (GFZ Data Services, DOI 10.5880/WSM.2025.001)",
            "note": "STATIC prior, not a time-varying channel -- see module docstring. "
                    "shmax_azimuth_deg is axial (mod 180), not a compass direction.",
        }
        print(f"S_Hmax azimuth: {mean_az:.1f} deg (axial, mod 180) | consistency R={r:.3f} "
              f"| dominant regime: {dominant_regime} ({regime_counts})")

    import json
    with open(out_path, "w") as f:
        json.dump(prior, f, indent=2)
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
