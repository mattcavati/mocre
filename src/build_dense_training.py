"""MOCRE-1 — "mapear los datos en tiempo real pasados de cada sismo para su entrenamiento" (Matt).
Análogo del enriched_data.csv del padre (candle t -> outcome t+1), pero DENSO y DIARIO: para cada
día t de la serie real de estado (series_*.csv, los 17 canales), calcula los 4 valores REALES del
sismo que le sigue dentro del horizonte, para que IA3 entrene cada salida contra su valor real.

Por día t (estado en t) -> targets (el "t+1" del padre, aquí "el sismo que viene"):
  occ_30d   : 1 si hay sismo M>=target en (t, t+30d], 0 si no          (ocurrencia)
  mag_next  : magnitud real de ese sismo (NaN si no hay)               (magnitud)
  days_next : días de t a ese sismo (NaN si no hay)                    (cuándo)
  radial_next: dist radial normalizada del epicentro al centro (NaN)   (dónde)
Salida: out/dense_training.csv (todos los segmentos, con block train/holdout y mag_scale/regime).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from common import ROOT, load_config, segment_paths
from pipeline import load_catalog

OUT = os.path.join(ROOT, "out")
CUTOFF = pd.Timestamp("2018-01-01")
HORIZON = 30
MAG_LO, MAG_HI = 3.5, 7.5
CHANNELS = ["a_seis", "a_swarm", "a_foreshock", "a_gnss", "a_sse", "a_fluid", "a_water", "a_strain",
            "a_ocean", "a_insar", "a_tec", "a_noise", "A_state", "C_state", "K_state", "E_state",
            "F_state"]


def load_events(cfg):
    path = segment_paths(cfg)["catalog_csv"]
    if not os.path.exists(path):
        return None
    cat = load_catalog(cfg)  # misma geometría/corredor que genera las entradas
    tm = cfg["etas_lite"]["target_mag"]
    cat = cat[cat["mag"] >= tm].sort_values("date").reset_index(drop=True)
    if "id" not in cat:
        cat["id"] = (cat["time"].astype(str) + "|" + cat["latitude"].round(4).astype(str) +
                     "|" + cat["longitude"].round(4).astype(str))
    return cat


def main():
    parts = []
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        cfg = load_config(os.path.basename(cfg_path))
        slug = segment_paths(cfg)["slug"]
        spath = os.path.join(OUT, f"series_{slug}.csv")
        if not os.path.exists(spath):
            continue
        series = pd.read_csv(spath, index_col=0, parse_dates=True)
        cat = load_events(cfg)
        if cat is None or len(cat) == 0:
            continue
        tm = cfg["etas_lite"]["target_mag"]
        mag_scale = float(np.clip(2 * (tm - MAG_LO) / (MAG_HI - MAG_LO) - 1, -1, 1))
        # centro + spread del segmento para el radial (mismo criterio que discover_rules)
        latc, lonc = cat["latitude"].median(), cat["longitude"].median()
        raw = np.sqrt((cat["latitude"] - latc) ** 2 + (cat["longitude"] - lonc) ** 2)
        p95 = raw.quantile(0.95) or 1e-9

        ev_dates = cat["date"].to_numpy()
        days = series.index.to_numpy()
        idx = np.searchsorted(ev_dates, days, side="right")   # primer evento estrictamente > t
        rows = []
        for i, t in enumerate(series.index):
            j = idx[i]
            occ = 0
            mag = dnext = radial = np.nan
            event_id = event_time = None
            if j < len(ev_dates):
                dd = (ev_dates[j] - np.datetime64(t)) / np.timedelta64(1, "D")
                if 0 < dd <= HORIZON:
                    occ = 1
                    e = cat.iloc[j]
                    mag = float(e["mag"]); dnext = float(dd)
                    event_id = str(e["id"]); event_time = str(e["time"])
                    radial = float(min(1.0, np.sqrt((e["latitude"]-latc)**2 + (e["longitude"]-lonc)**2) / p95))
            row = {c: float(series[c].iloc[i]) if c in series.columns and not pd.isna(series[c].iloc[i]) else 0.0
                   for c in CHANNELS}
            row.update({"segment": slug, "date": t, "mag_scale": mag_scale, "regime": cfg["regime"],
                        "occ_30d": occ, "mag_next": mag, "days_next": dnext, "radial_next": radial,
                        "event_id": event_id, "event_time": event_time,
                        "block": ("train" if t < CUTOFF - pd.Timedelta(days=HORIZON) else
                                  "purged" if t < CUTOFF else "holdout")})
            rows.append(row)
        parts.append(pd.DataFrame(rows))
        print(f"{slug:<18} {len(rows)} días | occ+ = {sum(r['occ_30d'] for r in rows)} "
              f"({100*np.mean([r['occ_30d'] for r in rows]):.1f}%)")

    dense = pd.concat(parts, ignore_index=True)
    out = os.path.join(OUT, "dense_training.csv")
    dense.to_csv(out, index=False)
    print(f"\n-> {out}  ({len(dense)} filas-día, {dense['occ_30d'].mean()*100:.1f}% positivas)")
    tr = dense[dense.block == "train"]; ho = dense[dense.block == "holdout"]
    print(f"   train={len(tr)} ({tr.occ_30d.mean()*100:.1f}% pos) | holdout={len(ho)} ({ho.occ_30d.mean()*100:.1f}% pos)")


if __name__ == "__main__":
    main()
