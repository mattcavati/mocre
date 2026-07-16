"""Dataset de episodios independientes para descubrir el estado crítico MOCRE-2."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from common import ROOT, load_config, segment_paths
from critical_state import derive_cluster_evidence, rule_firings
from pipeline import load_catalog

OUT = os.path.join(ROOT, "out")
CONTRACT = json.load(open(os.path.join(ROOT, "config", "critical_state_contract.json")))


def independent_events(cfg):
    cat = load_catalog(cfg)
    cat = cat[cat.mag >= cfg["etas_lite"]["target_mag"]].sort_values("time").copy()
    kept, last = [], None
    for _, event in cat.iterrows():
        when = pd.Timestamp(event["date"])
        if last is None or (when - last).days > CONTRACT["decluster_days"]:
            kept.append(event)
            last = when
    if not kept:
        return pd.DataFrame(columns=cat.columns)
    out = pd.DataFrame(kept).reset_index(drop=True)
    stable = (out.time.astype(str) + "|" + out.latitude.round(4).astype(str) + "|" +
              out.longitude.round(4).astype(str))
    if "id" not in out:
        out["id"] = stable
    else:
        out["id"] = out["id"].where(out["id"].notna() & out["id"].astype(str).ne(""), stable)
    return out


def event_splits(events):
    split = {}
    n = len(events)
    for i, row in events.iterrows():
        frac = (i + 1) / max(n, 1)
        block = "fit" if frac <= 0.60 else "cal" if frac <= 0.80 else "test"
        split[str(row.id)] = block
    return split


def valid_controls(index, event_dates, center, n, rng, used):
    exclusion = CONTRACT["control_exclusion_days"]
    candidates = index[(index >= center - pd.Timedelta(days=3 * 365)) &
                       (index <= center + pd.Timedelta(days=3 * 365))]
    if len(candidates):
        month_distance = np.minimum(np.abs(candidates.month - center.month),
                                    12 - np.abs(candidates.month - center.month))
        candidates = candidates[month_distance <= 1]
    ok = []
    for d in candidates:
        if d in used or d < index[180] or d > index[-2]:
            continue
        if all(abs((d - e).days) > exclusion for e in event_dates):
            ok.append(d)
    rng.shuffle(ok)
    chosen = ok[:n]
    used.update(chosen)
    return chosen


def row_at(features, date):
    pos = features.index.searchsorted(date, side="right") - 1
    if pos < 0:
        return None
    return features.iloc[int(pos)]


def main():
    rng = np.random.default_rng(170719)
    rows = []
    metadata = {"contract": CONTRACT, "segments": {}}
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        cfg = load_config(os.path.basename(cfg_path))
        slug = segment_paths(cfg)["slug"]
        path = os.path.join(OUT, f"series_{slug}.csv")
        if not os.path.exists(path):
            continue
        series = pd.read_csv(path, index_col=0, parse_dates=True)
        evidence = derive_cluster_evidence(series)
        firings = rule_firings(evidence).add_prefix("r_")
        features = pd.concat([evidence.add_prefix("e_"), firings], axis=1)
        events = independent_events(cfg)
        if events.empty:
            continue
        splits = event_splits(events)
        dates = [pd.Timestamp(x) for x in events.date]
        used_controls = set()
        seg_rows = 0
        for _, event in events.iterrows():
            eid = str(event.id)
            event_date = pd.Timestamp(event.date)
            block = splits[eid]
            valid_positive = 0
            for lead in CONTRACT["lead_days"]:
                anchor = event_date - pd.Timedelta(days=lead)
                values = row_at(features, anchor)
                if values is None:
                    continue
                row = values.to_dict()
                row.update({"segment": slug, "regime": cfg["regime"], "anchor": anchor,
                            "episode_id": f"event:{eid}", "event_id": eid,
                            "event_date": event_date, "lead_days": lead, "label": 1,
                            "target_mag": float(event.mag), "block": block})
                rows.append(row); seg_rows += 1; valid_positive += 1
            controls = valid_controls(features.index, dates, event_date,
                                      min(CONTRACT["controls_per_event"], valid_positive), rng,
                                      used_controls)
            for j, anchor in enumerate(controls):
                values = row_at(features, anchor)
                row = values.to_dict()
                row.update({"segment": slug, "regime": cfg["regime"], "anchor": anchor,
                            "episode_id": f"control:{eid}:{j}", "event_id": None,
                            "event_date": pd.NaT, "lead_days": np.nan, "label": 0,
                            "target_mag": np.nan, "block": block})
                rows.append(row); seg_rows += 1
        metadata["segments"][slug] = {"independent_events": len(events), "rows": seg_rows}
        print(f"{slug:<20} eventos independientes={len(events):3d} filas={seg_rows:4d}")
    data = pd.DataFrame(rows)
    # Cada terremoto pesa uno aunque tenga varios lead-times; controles asociados
    # pesan en total lo mismo para no recrear pseudorreplicación.
    counts = data.groupby("episode_id").episode_id.transform("size")
    data["sample_weight"] = 1.0 / counts
    path = os.path.join(OUT, "critical_episodes.csv.gz")
    data.to_csv(path, index=False, compression="gzip")
    metadata.update({"rows": len(data), "positive_rows": int(data.label.sum()),
                     "independent_positive_episodes": int(data[data.label == 1].episode_id.nunique())})
    json.dump(metadata, open(os.path.join(OUT, "critical_episodes_meta.json"), "w"), indent=2)
    print(f"-> {path}: {len(data)} filas, {metadata['independent_positive_episodes']} eventos")


if __name__ == "__main__":
    main()
