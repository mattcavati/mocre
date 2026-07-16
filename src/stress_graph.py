"""MOCRE-1 F4: Coulomb stress-transfer graph between fault segments.

Nodes = segments (their real/approximate traces already in config/). Edges = static Coulomb
stress change (Delta CFS) one segment's historical M>=5 ruptures impart on another, accumulated
over time and fed into the receiver's C (carga) state -- closing the dC formula in the spec
(04_modelo_sismico/modelo-cavati-red-esfuerzo.md SS3: "w_deltaCFS * max(0, deltaCFS_in)"), which
was specified from the start but never wired up until now.

ENGINEERING DECISION, stated plainly: this is a deliberately simplified point-source
approximation, NOT a full Okada (1992) half-space rectangular-dislocation solution. Reconstructing
Okada's full closed-form solution from memory (many terms, several integral functions, easy to
get a sign or a factor wrong) without a reference implementation to validate against is a real
risk of shipping a silently-wrong stress field -- worse than not having one. Instead:

    Delta_CFS(source, receiver) = K * M0_source / r^3 * lobe_pattern(receiver_azimuth, strike_diff)
                                   * mechanism_compat(source_regime, receiver_regime)

  - M0 (seismic moment) / r^3 is the correct dimensional scaling for static stress from a point
    shear dislocation (moment = shear_modulus * slip * area; stress ~ moment/distance^3) -- this
    part is not in question, it's textbook.
  - lobe_pattern(...) reproduces the classic four-lobe Coulomb-stress pattern for two
    similarly-oriented parallel faults: stress ENCOURAGED (+) along the source's own strike
    direction (the real, well-documented mechanism behind sequential along-strike ruptures like
    Landers 1992 -> Big Bear -> Hector Mine 1999 in this same California fault system), stress
    DISCOURAGED (-, "stress shadow") perpendicular to strike, with 45 deg as the zero-crossing
    between lobes -- this is the qualitatively-correct, low-risk-to-implement part.
  - K is calibrated so a M7 event at ~20km gives Delta_CFS on the order of 0.1-3 bar, matching
    the range commonly reported in the literature for comparable source-receiver distances (e.g.
    Landers 1992 -> Big Bear/Hector Mine stress-transfer studies) -- an empirical anchor, not a
    first-principles elastic-constant derivation (avoids unit-conversion risk in mu, rigidity,
    etc. that a from-scratch derivation would carry).
  - mechanism_compat(...) down-weights coupling between very different fault styles (e.g.
    transform source -> subduction-interface receiver) -- physically real (a receiver only
    responds strongly to stress components aligned with its own fault plane) and also honest
    about a real limitation: with real focal mechanisms we could resolve the full stress tensor
    onto the receiver plane properly; here we approximate that resolution with a coarse
    similarity score.

Validated by: a built-in sanity check (`self_test()`) confirming (a) 1/r^3 decay, (b) the 4-lobe
sign pattern for two parallel same-style strike-slip segments, (c) near-zero cross-regional terms
for segments >500km apart (Cascadia/Shumagin vs California) -- NOT validated against any specific
published numeric Coulomb-stress map. Treat magnitudes as order-of-magnitude plausible, not
survey-grade.
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from common import ROOT, fault_strike_azimuth_deg, load_config, segment_paths

OUT = os.path.join(ROOT, "out")
# Calibrated numerically (not by hand-arithmetic, to avoid a unit-conversion slip): solved for
# K such that a M7.0 event at r=20km gives Delta_CFS=1 bar at lobe=1, compat=1 (mid-range of the
# 0.1-3 bar commonly reported in the literature for comparable source-receiver distances):
#   M0(7.0) = 10**(1.5*7.0+9.1) = 3.981e19 N*m ; r_m**3 = (20000)**3 = 8e12 m^3
#   K = 1.0 * r_m**3 / M0 = 2.0095e-07  (bar * m^3 / (N*m))
K_CALIBRATION = 2.0095e-07
MIN_TARGET_MAG = 5.0     # only M>=5 sources considered (spec: "tras cada evento M>=5")

REGIME_MECHANISM = {
    # (dip_deg, rake_deg) -- coarse, per-regime defaults; not per-event focal mechanisms.
    "transform": (85.0, 180.0),   # near-vertical, right-lateral strike-slip
    "subduction": (15.0, 90.0),   # shallow-dipping megathrust, pure thrust
    "induced": (80.0, 180.0),     # Oklahoma induced seismicity reactivates near-vertical
                                   # basement strike-slip faults (Wilzetta fault, Keranen et al.
                                   # 2013) -- same style as "transform" mechanically, but kept as
                                   # a distinct regime label so mechanism_compat() still
                                   # down-weights coupling with natural tectonic segments (the
                                   # stress SOURCE physics differs -- pore pressure, not tectonic
                                   # loading -- even if the receiver fault style is similar).
}


def seismic_moment_nm(mag):
    """M0 in N*m from moment magnitude (standard Hanks & Kanamori 1979 relation)."""
    return 10 ** (1.5 * mag + 9.1)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Azimuth from point 1 to point 2, degrees from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360.0


def lobe_pattern(source_strike_deg, azimuth_to_receiver_deg, receiver_strike_deg):
    """Classic 4-lobe Coulomb pattern: encouraged (+) along the source's own strike direction
    (real mechanism behind along-strike rupture cascades, e.g. Landers->Hector Mine),
    discouraged (-, "stress shadow") fault-perpendicular, neutral at 45 deg (zero-crossing).
    cos(2*theta) is the standard low-order qualitative approximation for this quadrupole-like
    pattern (theta = azimuth relative to source strike). Modulated by how aligned the receiver's
    own strike is with the source's (same-strike receivers couple more strongly -- a coarse
    stand-in for proper stress-tensor resolution onto the receiver plane)."""
    theta = math.radians(azimuth_to_receiver_deg - source_strike_deg)
    source_lobe = math.cos(2 * theta)
    strike_align = math.cos(math.radians(2 * (receiver_strike_deg - source_strike_deg)))
    # blend: pure source lobe pattern, softened toward neutral when receiver strike is very
    # different from source strike (>~45 deg off) -- avoids overconfident sign claims cross-style
    return source_lobe * (0.5 + 0.5 * max(strike_align, 0.0))


def mechanism_compat(regime_a, regime_b):
    if regime_a == regime_b:
        return 1.0
    return 0.25  # coarse down-weight for cross-style coupling, see module docstring


def load_wsm_prior(slug):
    """Real World Stress Map static prior (see ingest_stress_map.py), if computed for this
    segment. Loaded here as DIAGNOSTIC data only -- see andersonian_diagnostic() below -- not
    wired into delta_cfs_bar()'s actual physics this session (see module docstring for why)."""
    path = os.path.join(ROOT, "data", "stress_map", slug, "prior.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        prior = json.load(f)
    return prior if prior.get("status") == "ok" else None


def andersonian_diagnostic(strike_deg, shmax_azimuth_deg):
    """How close is this fault's strike to the classic Andersonian optimal-rupture angle from the
    REAL regional S_Hmax (WSM)? ~30 deg off S_Hmax is optimal for a friction coefficient ~0.6
    (Byerlee's law, the standard textbook value) in a strike-slip regime. This is informational
    only -- a real, useful cross-check of the fault-strike inputs already in config/ against
    independent regional stress data, not a replacement for delta_cfs_bar()'s own geometry."""
    # both strike and S_Hmax are axial (mod 180) -- smallest angular separation on a half-circle
    diff = abs(strike_deg - shmax_azimuth_deg) % 180.0
    diff = min(diff, 180.0 - diff)
    optimal_dev = abs(diff - 30.0)
    return diff, optimal_dev


def load_segments():
    segs = {}
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        cfg = load_config(os.path.basename(cfg_path))
        slug = segment_paths(cfg)["slug"]
        trace = cfg["fault_trace"]
        if "polyline_file" in trace:
            with open(os.path.join(ROOT, "config", trace["polyline_file"])) as f:
                d = json.load(f)
            pts = np.array([p for line in d["polylines"] for p in line])
            centroid = (pts[:, 1].mean(), pts[:, 0].mean())  # lat, lon
            strike = fault_strike_azimuth_deg(cfg)
        else:
            p1, p2 = trace["p1"], trace["p2"]
            centroid = ((p1["lat"] + p2["lat"]) / 2, (p1["lon"] + p2["lon"]) / 2)
            strike = cfg["gnss"]["fault_azimuth_deg"]
        segs[slug] = {"cfg": cfg, "slug": slug, "name": cfg["name"], "regime": cfg["regime"],
                     "lat": centroid[0], "lon": centroid[1], "strike": strike,
                     "wsm_prior": load_wsm_prior(slug)}
    return segs


def delta_cfs_bar(source, receiver, mag, event_lat, event_lon):
    r_km = haversine_km(event_lat, event_lon, receiver["lat"], receiver["lon"])
    r_km = max(r_km, 5.0)  # floor: point-source approx breaks down inside the rupture itself
    az = bearing_deg(event_lat, event_lon, receiver["lat"], receiver["lon"])
    lobe = lobe_pattern(source["strike"], az, receiver["strike"])
    compat = mechanism_compat(source["regime"], receiver["regime"])
    m0 = seismic_moment_nm(mag)
    r_m = r_km * 1000.0
    return K_CALIBRATION * m0 / (r_m ** 3) * lobe * compat


def self_test():
    """Sanity checks -- not a substitute for validation against a published Coulomb-stress map,
    see module docstring."""
    src = {"strike": 0.0, "regime": "transform", "lat": 0.0, "lon": 0.0}
    rc_same = {"strike": 0.0, "regime": "transform"}
    # (a) 1/r^3 decay
    near = delta_cfs_bar(src, {**rc_same, "lat": 0.1, "lon": 0.0}, 7.0, 0.0, 0.0)
    far = delta_cfs_bar(src, {**rc_same, "lat": 1.0, "lon": 0.0}, 7.0, 0.0, 0.0)
    ratio = near / far
    expect = (1.0 / 0.1) ** 3 if abs(near) > 0 else None
    print(f"self_test (a) 1/r^3 decay: near/far ratio={ratio:.1f} (expect ~{expect:.0f})")
    assert 0.5 * expect < ratio < 2.0 * expect, "decay law broken"
    # (b) 4-lobe sign pattern, corrected physics: for a receiver ALONG the source's own strike
    # (theta=0), Coulomb stress is ENCOURAGED (>0) -- this is exactly the real, well-documented
    # phenomenon behind sequential along-strike ruptures like Landers(1992)->Big Bear->Hector
    # Mine(1999) in this same California fault system. PERPENDICULAR to strike (theta=90) is the
    # classic "stress shadow" (<0). 45 degrees is the zero-crossing between lobes, not a peak.
    along = delta_cfs_bar(src, {**rc_same, "lat": 1.0, "lon": 0.0}, 7.0, 0.0, 0.0)       # az=0
    perp = delta_cfs_bar(src, {**rc_same, "lat": 0.0, "lon": 1.0}, 7.0, 0.0, 0.0)        # az~90
    print(f"self_test (b) along-strike (encouraged expected, >0): {along:.4f} bar | "
          f"perpendicular (stress shadow expected, <0): {perp:.4f} bar")
    assert along > 0 > perp, "4-lobe pattern sign is wrong"
    # (c) cross-regional near-zero: two segments 1500km apart should be negligible
    far_receiver = {**rc_same, "lat": 13.5, "lon": 0.0}  # ~1500km at equator
    tiny = delta_cfs_bar(src, far_receiver, 7.5, 0.0, 0.0)
    print(f"self_test (c) 1500km cross-regional term: {tiny:.6f} bar (expect << 0.01)")
    assert abs(tiny) < 0.01
    print("self_test: PASSED\n")


def build_graph_and_propagate():
    self_test()
    segs = load_segments()
    print(f"segments (nodes): {list(segs.keys())}\n")

    print("WSM cross-check (informational, does not affect delta_cfs_bar):")
    for slug, seg in segs.items():
        prior = seg.get("wsm_prior")
        if prior is None:
            print(f"  {slug}: no real WSM prior (insufficient coverage or not ingested)")
            continue
        diff, optimal_dev = andersonian_diagnostic(seg["strike"], prior["shmax_azimuth_deg"])
        print(f"  {slug}: config strike={seg['strike']:.1f} vs real S_Hmax="
              f"{prior['shmax_azimuth_deg']:.1f} (R={prior['circular_consistency_r']:.2f}, "
              f"regime={prior['dominant_regime']}, n={prior['n_points']}) -> angle to S_Hmax="
              f"{diff:.1f} deg (Andersonian-optimal ~30 deg, deviation={optimal_dev:.1f})")
    print()

    edges = []
    cfs_timeline = {slug: [] for slug in segs}  # slug -> list of (time, source_slug, mag, dcfs)
    for slug, seg in segs.items():
        paths = segment_paths(seg["cfg"])
        try:
            cat = pd.read_csv(paths["catalog_csv"])
        except FileNotFoundError:
            continue
        cat["time"] = pd.to_datetime(cat["time"], utc=True, format="ISO8601")
        big = cat[cat["mag"] >= MIN_TARGET_MAG]
        for _, ev in big.iterrows():
            for rslug, receiver in segs.items():
                if rslug == slug:
                    continue
                dcfs = delta_cfs_bar(seg, receiver, ev["mag"], ev["latitude"], ev["longitude"])
                cfs_timeline[rslug].append((ev["time"], slug, float(ev["mag"]), dcfs))
                edges.append({"source": slug, "receiver": rslug, "event_time": str(ev["time"]),
                              "event_mag": float(ev["mag"]), "delta_cfs_bar": round(dcfs, 5)})

    edges_df = pd.DataFrame(edges)
    if len(edges_df):
        edges_df.to_csv(os.path.join(OUT, "stress_graph_edges.csv"), index=False)

    # per-segment daily step series: cumulative Delta_CFS_in, positive-only per the spec's dC
    # formula ("w_deltaCFS * max(0, deltaCFS_in)") -- pipeline.py's build_state() picks this up
    # if present. Steps occur on the day of each contributing source event.
    for slug, events in cfs_timeline.items():
        if not events:
            continue
        df = pd.DataFrame(events, columns=["time", "source", "mag", "delta_cfs_bar"])
        df["date"] = pd.to_datetime(df["time"]).dt.tz_convert(None).dt.normalize()
        daily_step = df.groupby("date")["delta_cfs_bar"].sum()
        out_path = os.path.join(OUT, f"cfs_in_{slug}.csv")
        daily_step.to_csv(out_path, header=["delta_cfs_bar"])
    print(f"per-segment daily CFS-in step series -> {OUT}/cfs_in_<slug>.csv")

    print("net cumulative Delta_CFS received per segment (sum over all sources' M>=5 events):")
    summary_rows = []
    for slug, events in cfs_timeline.items():
        net = sum(e[3] for e in events)
        n_contrib = len(events)
        top_sources = {}
        for _, src_slug, mag, dcfs in events:
            top_sources[src_slug] = top_sources.get(src_slug, 0.0) + dcfs
        top = sorted(top_sources.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {slug}: net={net:+.4f} bar from {n_contrib} contributing events "
              f"| top contributors: {top}")
        summary_rows.append({"segment": slug, "net_delta_cfs_bar": round(net, 5),
                             "n_contributing_events": n_contrib})
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "stress_graph_summary.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'stress_graph_edges.csv')}, "
          f"{os.path.join(OUT, 'stress_graph_summary.csv')}")
    return cfs_timeline, segs


if __name__ == "__main__":
    build_graph_and_propagate()
