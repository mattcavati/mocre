"""MOCRE-1 fuzzy-inverse fix 4: link each POSITIVE case to the REAL outcome of the event that
made it positive -- magnitude, delta-t (days to the event), and epicenter coordinates. Matt's
exact point: "los valores de entradas a descubrir deben estar ligados a ocurrencias que sucedieron
con las salidas que queremos" (occurrence+confidence, coordinates, magnitude, when). Without this,
the fuzzy-inverse can only ever train the ocurrencia axis (which is all cases_global.csv carries)
-- magnitud/cuando/donde have no real target to learn from at all.

Reads out/cases_global.csv (built by the parallel session) + the real event catalogs, and adds:
  event_mag, event_lat, event_lon, event_delta_days   (NaN for negative cases -- no event to link)
Writes out/cases_enriched.csv (new file, does not touch cases_global.csv).
"""
import glob
import os

import numpy as np
import pandas as pd

from common import ROOT

OUT = os.path.join(ROOT, "out")
HORIZON = 30   # must match build_cases.py's f_30d horizon


def load_catalog_events(slug):
    path = os.path.join(ROOT, "data", "catalog", slug, "events.csv")
    if not os.path.exists(path):
        return None
    cat = pd.read_csv(path)
    cat["date"] = pd.to_datetime(cat["time"], utc=True, format="ISO8601").dt.tz_convert(None).dt.normalize()
    return cat[["date", "mag", "latitude", "longitude"]].sort_values("date")


def link_next_event(dates, cat, target_mag):
    """For each date in `dates`, find the FIRST event with mag>=target_mag strictly after it and
    within HORIZON days (matches f_30d's own definition of what made the case positive). Returns
    arrays (mag, lat, lon, delta_days), NaN where no such event (negative case, or no data)."""
    tgt = cat[cat["mag"] >= target_mag].sort_values("date")
    ev_dates = tgt["date"].to_numpy()
    n = len(dates)
    mag = np.full(n, np.nan); lat = np.full(n, np.nan)
    lon = np.full(n, np.nan); dd = np.full(n, np.nan)
    if len(ev_dates) == 0:
        return mag, lat, lon, dd
    idx = np.searchsorted(ev_dates, dates.to_numpy(), side="right")
    for i, j in enumerate(idx):
        if j >= len(ev_dates):
            continue
        delta = (ev_dates[j] - dates.to_numpy()[i]) / np.timedelta64(1, "D")
        if delta <= HORIZON:
            row = tgt.iloc[j]
            mag[i], lat[i], lon[i] = row["mag"], row["latitude"], row["longitude"]
            dd[i] = delta
    return mag, lat, lon, dd


def main():
    df = pd.read_csv(os.path.join(OUT, "cases_global.csv"), parse_dates=["date"])
    print(f"casos de entrada: {len(df)}")

    out_parts = []
    for seg, g in df.groupby("segment"):
        cat = load_catalog_events(seg)
        if cat is None:
            g = g.copy()
            for c in ["event_mag", "event_lat", "event_lon", "event_delta_days"]:
                g[c] = np.nan
            out_parts.append(g)
            continue
        # infer target_mag from the real events that ARE flagged positive in this segment's own
        # f_30d (avoids re-reading configs): the minimum mag among linked events at f_30d==1 rows
        # is unreliable if events.csv includes sub-threshold events, so instead re-derive target
        # from config directly (same source build_cases.py used).
        import json
        cfg_path = None
        for p in glob.glob(os.path.join(ROOT, "config", "segment_*.json")):
            c = json.load(open(p))
            if c.get("segment_id", "").lower().replace("_", "-") == seg:
                cfg_path = p
                break
        if cfg_path is None:
            g = g.copy()
            for c in ["event_mag", "event_lat", "event_lon", "event_delta_days"]:
                g[c] = np.nan
            out_parts.append(g)
            continue
        target_mag = json.load(open(cfg_path))["etas_lite"]["target_mag"]
        g = g.copy()
        mag, lat, lon, dd = link_next_event(g["date"], cat, target_mag)
        g["event_mag"], g["event_lat"], g["event_lon"], g["event_delta_days"] = mag, lat, lon, dd
        out_parts.append(g)

    out = pd.concat(out_parts, ignore_index=True)
    out_path = os.path.join(OUT, "cases_enriched.csv")
    out.to_csv(out_path, index=False)

    linked = out["event_mag"].notna().sum()
    pos = int(out["f_30d"].sum())
    print(f"casos con evento real ligado: {linked} de {pos} positivos "
          f"({100*linked/max(pos,1):.1f}% -- el resto son positivos de f_30d cuyo evento real "
          f"cae fuera de la ventana de {HORIZON}d exacta usada aqui, o el segmento no tiene "
          f"config/catalogo mapeado)")
    print(f"-> {out_path}")
    print("\nresumen real de magnitudes/deltas ligados:")
    lk = out.dropna(subset=["event_mag"])
    print(f"  magnitud: media={lk['event_mag'].mean():.2f} rango=[{lk['event_mag'].min():.1f},"
          f"{lk['event_mag'].max():.1f}]")
    print(f"  delta_dias: media={lk['event_delta_days'].mean():.1f} "
          f"rango=[{lk['event_delta_days'].min():.0f},{lk['event_delta_days'].max():.0f}]")


if __name__ == "__main__":
    main()
