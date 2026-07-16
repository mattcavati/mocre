"""Emite el estado crítico fuzzy vivo, siempre en sombra hasta promoción completa."""
from __future__ import annotations

import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

from common import ROOT, load_config, segment_paths
from critical_state import RULES, infer_series

OUT = os.path.join(ROOT, "out")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(archive=True):
    model_path = os.path.join(OUT, "critical_state_model.json")
    model = json.load(open(model_path))
    segments = []
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        cfg = load_config(os.path.basename(cfg_path))
        slug = segment_paths(cfg)["slug"]
        path = os.path.join(OUT, f"series_{slug}.csv")
        if not os.path.exists(path):
            continue
        series = pd.read_csv(path, index_col=0, parse_dates=True)
        evidence, firings, state = infer_series(series, model)
        i = -1
        contributions = []
        for j, (name, _) in enumerate(RULES):
            contributions.append({"rule": name, "firing": round(float(firings.iloc[i, j]), 4),
                                  "weighted": round(float(firings.iloc[i, j] * model["weights"][j]), 4)})
        contributions.sort(key=lambda x: x["weighted"], reverse=True)
        cluster_values = {c: {suffix: round(float(evidence.iloc[i][f"{c}_{suffix}"]), 4)
                              for suffix in ("level", "quiet", "trend", "persistence", "coherence", "coverage")}
                          for c in ("regional", "seismic", "deformation", "mechanical", "hydro", "remote")}
        segments.append({"segment": slug, "segment_id": cfg["segment_id"],
                         "state_as_of": str(series.index[-1]),
                         "critical_state": round(float(state.iloc[i]), 5),
                         "threshold": round(float(model["threshold"]), 5),
                         "above_threshold": bool(state.iloc[i] >= model["threshold"]),
                         "clusters": cluster_values, "top_rules": contributions[:5]})
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "engine": "MOCRE-2 fuzzy hierarchical recurrent critical-state inference",
           "status": model["status"], "promoted": model["promoted"],
           "model_sha256": sha256(model_path), "promotion_checks": model["promotion_checks"],
           "segments": segments}
    target = os.path.join(OUT, "critical_shadow.json")
    with open(target, "w") as f:
        json.dump(out, f, indent=2)
    if archive:
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src", "global"))
        from forecast_ledger import archive_forecast
        archive_forecast(out, ledger_dir=os.path.join(OUT, "prospective-critical"))
    print(f"{len(segments)} estados críticos -> {target}; promoted={model['promoted']}")
    return out


if __name__ == "__main__":
    main()
