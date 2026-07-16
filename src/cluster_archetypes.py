"""MOCRE-1 F2.5: unsupervised archetype discovery from pre-event state trajectories.

The idea from Vision 1 (genealogy doc SS1.3) that almost got lost: instead of imposing a
sismologist's taxonomy (subduction/intraplate/volcanic/...) by hand, let the data group
EARTHQUAKES by the SHAPE of their precursor-state trajectory in the window before they happened,
across regions -- not by which region they're in. If distinguishable clusters emerge, that's an
empirically discovered "regime," which may or may not line up with the geological taxonomy.

Honest scope for this first pass (2026-07-03, 3 segments, 9 target events total: SJC=7,
Mojave=0, Cascadia=2): 9 events is nowhere near enough to claim discovered clusters mean
anything statistically -- this is a sanity check that the pipeline runs correctly end-to-end,
not a result. What it delivers: (a) correct per-event trajectory extraction reusing pipeline.py
directly (no re-derivation, no drift risk), (b) a real distance matrix + hierarchical linkage
over actual events, (c) segment-level features too (works even for 0-event segments like
Mojave, which is itself archetype signal -- "locked and silent" IS a trajectory shape, just a
flat one). Re-run the moment more segments/events exist; nothing here needs to change, just more
data through the same path.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from common import ROOT, load_config, segment_paths
from pipeline import build_daily, build_state, load_catalog, load_gnss

OUT = os.path.join(ROOT, "out")
STATE_VARS = ["C_state", "A_state", "K_state"]
WINDOW_DAYS = 90  # trailing window before each event


def segment_level_features(df, summ):
    """Whole-window summary feature vector per segment -- works even with 0 target events, so
    Mojave (locked/silent) still participates instead of being excluded for lack of events."""
    feats = {}
    for v in STATE_VARS:
        s = df[v].dropna()
        feats[f"{v}_mean"] = s.mean()
        feats[f"{v}_std"] = s.std()
        feats[f"{v}_frac_hi"] = (s > (s.median() + 0.5 * s.std())).mean() if s.std() > 0 else 0.0
    feats["C_trend_per_decade"] = (df["C_state"].iloc[-1] - df["C_state"].iloc[0]) / \
                                   max((len(df) / 365.25) / 10, 0.1)
    feats["n_target_events"] = summ["n_target_events"]
    return feats


def decluster_for_archetype(ev, window_days=30):
    """Keep only the largest event within each window_days window, walking chronologically.

    Without this, mainshock+aftershock sequences on (near-)identical calendar days share
    (near-)identical trailing-window state features BY CONSTRUCTION (the window is anchored to
    the day, so events a day apart see almost the same 90-day history). That's not shared
    archetype, it's the same window counted N times -- it silently inflates segments with big
    aftershock sequences (Parkfield 2004, Shumagin 2020) as if they had many independent
    archetype samples. Found by inspecting distance-0.000 merges in the first 5-segment run,
    2026-07-03. Standard practice in seismology (this is literally what "declustering" means for
    an ETAS background rate too -- same principle, applied here to the archetype-feature step)."""
    if len(ev) == 0:
        return ev
    ev = ev.sort_values("time").reset_index(drop=True)
    keep = []
    last_kept_time = None
    pending = None
    for _, row in ev.iterrows():
        if last_kept_time is None or (row["time"] - last_kept_time).days > window_days:
            if pending is not None:
                keep.append(pending)
            pending = row
            last_kept_time = row["time"]
        else:
            if pending is None or row["mag"] > pending["mag"]:
                pending = row
    if pending is not None:
        keep.append(pending)
    return pd.DataFrame(keep).reset_index(drop=True)


def event_trajectory_features(daily, event_dates, slug):
    """For each target event, extract the WINDOW_DAYS trailing window of (C,A,K) and summarize
    its SHAPE (not just level) -- slope, peak timing, peak-to-onset ratio. This is the actual
    per-event archetype feature vector, distinct from the whole-segment averages above."""
    rows = []
    idx = daily.index
    for i, ed in enumerate(event_dates):
        pos = idx.get_indexer([ed], method="nearest")[0]
        start = max(0, pos - WINDOW_DAYS)
        win = daily.iloc[start:pos + 1]
        if len(win) < WINDOW_DAYS // 2:
            continue  # too close to series start, incomplete window
        f = {"segment": slug, "event_date": str(ed.date()), "event_idx": i}
        for v in STATE_VARS:
            s = win[v].dropna()
            if len(s) < 5:
                f[f"{v}_level"] = np.nan
                f[f"{v}_slope"] = np.nan
                f[f"{v}_peak_frac"] = np.nan
                continue
            f[f"{v}_level"] = s.iloc[-1]
            x = np.arange(len(s))
            f[f"{v}_slope"] = np.polyfit(x, s.to_numpy(), 1)[0] if len(s) > 2 else 0.0
            f[f"{v}_peak_frac"] = (s.to_numpy().argmax() / max(len(s) - 1, 1))  # 0=early,1=late peak
        rows.append(f)
    return rows


def gather_event_features():
    """Re-derives daily state + target-event list per segment by calling pipeline.py's own
    functions directly (single source of truth -- no re-implementation of the state math)."""
    all_events = []
    seg_summary = {}
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        cfg = load_config(os.path.basename(cfg_path))
        slug = segment_paths(cfg)["slug"]
        try:
            cat = load_catalog(cfg)
            stations, gnss = load_gnss(cfg)
        except FileNotFoundError:
            print(f"  {slug}: no ingested data, skipping")
            continue
        daily = build_daily(cfg, cat, stations, gnss)
        daily = build_state(cfg, daily)
        tgt = cfg["etas_lite"]["target_mag"]
        ev_raw = cat[(cat["mag"] >= tgt) & (cat["date"] >= daily.index[0])]
        ev = decluster_for_archetype(ev_raw)
        print(f"  {slug}: {len(ev_raw)} raw target events -> {len(ev)} after declustering "
              f"(30-day window, keep largest), {len(daily)} days of state")
        feats = event_trajectory_features(daily, ev["date"].tolist(), slug)
        all_events.extend(feats)
        seg_summary[slug] = (daily, {"n_target_events": len(ev_raw)})
    return all_events, seg_summary


def main():
    print("re-deriving state + events per segment (reuses pipeline.py, single source of truth):")
    event_feats, seg_summary = gather_event_features()

    print(f"\ntotal target-magnitude events with usable pre-event windows: {len(event_feats)}")

    # ---------------------------------------------------------------- segment-level (all segments)
    rows = []
    for slug, (df, summ) in seg_summary.items():
        f = segment_level_features(df, summ)
        f["segment"] = slug
        rows.append(f)
    seg_feat_df = pd.DataFrame(rows).set_index("segment")
    print("\nsegment-level feature table (includes 0-event segments, e.g. Mojave):")
    print(seg_feat_df.round(3).to_string())
    seg_feat_df.to_csv(os.path.join(OUT, "archetype_features_segment_level.csv"))

    Xs = seg_feat_df.drop(columns=["n_target_events"]).to_numpy()
    Xs = np.nan_to_num((Xs - np.nanmean(Xs, axis=0)) / (np.nanstd(Xs, axis=0) + 1e-9))
    if len(seg_feat_df) >= 2:
        Ds = squareform(pdist(Xs, metric="euclidean"))
        print("\nsegment-level pairwise distance (standardized):")
        print(pd.DataFrame(Ds, index=seg_feat_df.index, columns=seg_feat_df.index).round(2))

    # ---------------------------------------------------------------- event-level (the real ask)
    if len(event_feats) < 3:
        print(f"\nonly {len(event_feats)} usable per-event trajectories -- too few to cluster "
              "meaningfully. Reporting the raw feature table only, no clustering forced on it.")
        pd.DataFrame(event_feats).to_csv(os.path.join(OUT, "archetype_features_per_event.csv"),
                                          index=False)
        return

    ev_df = pd.DataFrame(event_feats)
    feat_cols = [c for c in ev_df.columns if c not in ("segment", "event_date", "event_idx")]
    Xe = ev_df[feat_cols].to_numpy()
    Xe = np.nan_to_num((Xe - np.nanmean(Xe, axis=0)) / (np.nanstd(Xe, axis=0) + 1e-9))
    Z = linkage(pdist(Xe, metric="euclidean"), method="average")
    print(f"\nper-event hierarchical linkage over {len(ev_df)} events "
          f"({', '.join(f'{s}={c}' for s, c in ev_df['segment'].value_counts().items())}):")
    labels = [f"{r.segment}/{r.event_date}" for r in ev_df.itertuples()]
    for i, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        na = labels[a] if a < len(labels) else f"cluster{a}"
        nb = labels[b] if b < len(labels) else f"cluster{b}"
        print(f"  merge {i}: {na} + {nb} at distance {row[2]:.3f}")
        labels.append(f"cluster{len(labels)}")
    ev_df.to_csv(os.path.join(OUT, "archetype_features_per_event.csv"), index=False)

    print(f"""
HONEST STATUS:
  - {len(ev_df)} declustered events total (counts by segment above) is a pipeline sanity check, not a
    statistically meaningful clustering result -- do not read cluster membership as a discovered
    archetype yet.
  - What IS real: the extraction is correct (reuses pipeline.py's own state-building code,
    trailing {WINDOW_DAYS}-day window, shape features -- level/slope/peak-timing -- not just
    mean level), and it's ready to re-run as-is the moment more segments/events exist. That's
    the actual deliverable of this pass: the pipe, not a premature conclusion through it.
  - Next real step: add segments with more target events (lower magnitude threshold trades
    event count for operational relevance -- a documented, deliberate choice, not free) before
    trusting cluster boundaries.
""")


if __name__ == "__main__":
    main()
