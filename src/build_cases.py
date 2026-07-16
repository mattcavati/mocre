"""MOCRE-1 fuzzy-inverse groundwork: build the GLOBAL case dataset. Each row = one day in one
segment, with that day's causal variable vector (the a_X anomaly channels + persistent states,
all computed from past-only data) and a label f: 1 if an M>=target event occurs in the next
HORIZON days, 0 otherwise. All segments stacked into one pool (Matt's decision: global, for
statistical power to discover a shared subset).

This is the honest, no-overfit-risk step: it only organizes data. No model, no rule discovery yet.
The subset-selection / membership-learning (Wang-Mendel + sparse feature selection, validated
out-of-sample) runs ON this dataset later.

Key discipline baked in for whoever models this next:
  - Features are causal (channel values at day t use only data up to t).
  - Label looks FORWARD (event in (t, t+HORIZON]) -- this is prediction, not fitting the present.
  - A `block` column (segment + 90-day bucket) is emitted so out-of-sample validation can split by
    TIME BLOCK, never by random day: consecutive days are near-identical (channels move slowly),
    so a random train/test split would leak and massively inflate apparent skill. Split by block.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from common import ROOT, load_config, segment_paths

HORIZONS = [30, 90]   # emit a label per horizon; the model picks which to predict
FEATURES = ["a_seis", "a_gnss", "a_swarm", "a_foreshock", "a_sse", "a_fluid", "a_water",
            "a_strain", "a_ocean", "a_insar", "a_tec", "a_noise",
            "A_state", "C_state", "K_state", "E_state", "F_state"]


def event_days(cfg, slug):
    """Days (normalized timestamps) of M>=target events for this segment, from the real catalog."""
    cat_path = os.path.join(ROOT, "data", "catalog", slug, "events.csv")
    if not os.path.exists(cat_path):
        return pd.DatetimeIndex([])
    cat = pd.read_csv(cat_path)
    tcol = "time" if "time" in cat.columns else cat.columns[0]
    mcol = "mag" if "mag" in cat.columns else "magnitude"
    cat[tcol] = pd.to_datetime(cat[tcol], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    tgt = cfg["etas_lite"]["target_mag"]
    return pd.DatetimeIndex(cat[cat[mcol] >= tgt][tcol].dropna().unique())


def main():
    rows = []
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        cfg = load_config(os.path.basename(cfg_path))
        slug = segment_paths(cfg)["slug"]
        series_path = os.path.join(ROOT, "out", f"series_{slug}.csv")
        if not os.path.exists(series_path):
            continue
        s = pd.read_csv(series_path, index_col=0, parse_dates=True)
        for f in FEATURES:
            if f not in s.columns:
                s[f] = 0.0
        s = s[FEATURES + (["lam_etas"] if "lam_etas" in s.columns else [])].copy()
        # drop the warm-up head where the baseline window hasn't filled (features all NaN/0)
        s = s.dropna(subset=["a_seis"])
        if len(s) < 400:
            continue

        evs = event_days(cfg, slug)
        idx = s.index
        # forward label per horizon: any target event in (t, t+H]
        ev_arr = evs.values.astype("datetime64[ns]")
        for H in HORIZONS:
            lab = np.zeros(len(idx), dtype=int)
            if len(ev_arr):
                for i, t in enumerate(idx.values):
                    hi = t + np.timedelta64(H, "D")
                    if np.any((ev_arr > t) & (ev_arr <= hi)):
                        lab[i] = 1
            s[f"f_{H}d"] = lab

        s["segment"] = slug
        s["date"] = idx
        # time-block id for leakage-safe CV: segment + 90-day bucket
        s["block"] = slug + "_" + ((idx - idx[0]).days // 90).astype(str)
        rows.append(s.reset_index(drop=True))
        pos30 = int(s["f_30d"].sum())
        print(f"  {slug:<18} {len(s):>5} días | eventos M>=target={len(evs):>3} | "
              f"f_30d=1 en {pos30} días ({pos30/len(s)*100:.1f}%)")

    if not rows:
        print("no series found -- run the pipeline first")
        return
    df = pd.concat(rows, ignore_index=True)
    out = os.path.join(ROOT, "out", "cases_global.csv")
    df.to_csv(out, index=False)
    print(f"\nGLOBAL pool: {len(df)} filas, {df['segment'].nunique()} segmentos, "
          f"{df['block'].nunique()} bloques temporales")
    for H in HORIZONS:
        p = df[f"f_{H}d"].mean()
        print(f"  f_{H}d: {int(df[f'f_{H}d'].sum())} positivos / {len(df)} ({p*100:.2f}%) "
              f"-- ratio 1:{int((1-p)/p) if p>0 else 0}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
