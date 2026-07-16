"""MOCRE-GLOBAL — actualizador en TIEMPO REAL. Cada ejecución (systemd timer ~10 min):
1. Baja el feed USGS M>=4.5 de la última semana (cubre huecos si el timer falló).
2. Fusiona eventos nuevos en catalog_global.csv (dedupe espaciotemporal).
3. Recalcula el estado vivo de TODAS las celdas del grid a AHORA (rápido: 1 timestamp).
4. Re-emite globe/globe_data.json -> el globo lo recoge en su auto-refresh.
"""
import json
import os
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data", "global")
OUTG = os.path.join(ROOT, "out", "global")
sys.path.insert(0, os.path.dirname(__file__))
from build_global_dataset import cell_features  # noqa: E402

FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson"
CAT_COLUMNS = ["id", "time", "updated", "latitude", "longitude", "depth", "mag", "net"]


def fetch_feed():
    req = urllib.request.Request(FEED, headers={"User-Agent": "mocre-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        gj = json.load(r)
    rows = []
    for f in gj["features"]:
        p = f["properties"]; lon, lat, depth = f["geometry"]["coordinates"]
        if p.get("mag") is None:
            continue
        rows.append({"id": f.get("id") or p.get("code"),
                     "time": pd.Timestamp(p["time"], unit="ms"),
                     "updated": pd.Timestamp(p["updated"], unit="ms") if p.get("updated") else pd.NaT,
                     "latitude": lat, "longitude": lon, "depth": depth, "mag": p["mag"],
                     "net": p.get("net")})
    return pd.DataFrame(rows)


def _normalise(df):
    """Normaliza catálogos antiguos sin perder compatibilidad con sus columnas."""
    df = df.copy()
    for col in CAT_COLUMNS:
        if col not in df:
            df[col] = pd.NaT if col in {"time", "updated"} else np.nan
    for col in ("time", "updated"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(None)
    df["id"] = df["id"].astype("object")
    df["net"] = df["net"].astype("object")
    return df[CAT_COLUMNS]


def _distance_km(lat1, lon1, lat2, lon2):
    """Distancia de gran círculo; suficiente para reconocer revisiones de ComCat."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _atomic_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".catalog-", suffix=".csv", dir=os.path.dirname(path))
    try:
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def merge_catalog(new):
    path = os.path.join(DATA, "catalog_global.csv")
    cat = _normalise(pd.read_csv(path, dtype={"id": "string", "net": "string"}, low_memory=False))
    new = _normalise(new).sort_values(["time", "updated"])
    original_ids = set(cat.id.dropna().astype(str))
    inserted = revised = 0

    # Upsert por ID. Para catálogos históricos sin ID, se reconoce la misma
    # solución por proximidad temporal y espacial, sin agrupar enjambres reales.
    for _, row in new.iterrows():
        rid = str(row.id) if pd.notna(row.id) and str(row.id) else None
        exact = cat.index[cat.id.astype(str).eq(rid)] if rid else np.array([], dtype=int)
        if len(exact):
            idx = int(exact[-1])
            old_updated = cat.at[idx, "updated"]
            if pd.isna(old_updated) or pd.isna(row.updated) or row.updated >= old_updated:
                cat.loc[idx, CAT_COLUMNS] = row[CAT_COLUMNS].values
                revised += 1
            continue

        tdiff = (cat.time - row.time).abs().dt.total_seconds()
        candidates = cat.index[tdiff.le(5.0)]
        match = None
        if len(candidates):
            c = cat.loc[candidates]
            dist = _distance_km(c.latitude.to_numpy(), c.longitude.to_numpy(),
                                float(row.latitude), float(row.longitude))
            mdiff = np.abs(c.mag.to_numpy(dtype=float) - float(row.mag))
            ok = np.flatnonzero((dist <= 15.0) & (mdiff <= 0.5))
            if len(ok):
                match = int(candidates[int(ok[np.argmin(dist[ok])])])
        if match is not None:
            cat.loc[match, CAT_COLUMNS] = row[CAT_COLUMNS].values
            revised += 1
        else:
            cat.loc[len(cat), CAT_COLUMNS] = row[CAT_COLUMNS].values
            inserted += 1

    # El feed puede contener varias revisiones del mismo ID; la última gana.
    has_id = cat.id.notna() & cat.id.astype(str).ne("")
    identified = cat[has_id].sort_values(["id", "updated"]).drop_duplicates("id", keep="last")
    unidentified = cat[~has_id]
    both = pd.concat([identified, unidentified], ignore_index=True).sort_values("time").reset_index(drop=True)
    _atomic_csv(both, path)  # también persiste limpieza/revisiones si el saldo de filas es <= 0
    new_ids = len(set(both.id.dropna().astype(str)) - original_ids)
    print(f"feed: {len(new)} eventos; {inserted} altas, {revised} revisiones, "
          f"{new_ids} IDs nuevos (catálogo {len(both)})")
    return both


def recompute_live(cat):
    grid = json.load(open(os.path.join(OUTG, "grid.json")))
    la = np.floor(cat.latitude.to_numpy()).astype(int)
    lo = np.floor(cat.longitude.to_numpy()).astype(int)
    t_ns = cat.time.astype("int64").to_numpy()
    mags = cat.mag.to_numpy()
    reps = []
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            reps.append(pd.DataFrame({"la": la + dla, "lo": ((lo + dlo + 180) % 360) - 180,
                                      "idx": np.arange(len(cat))}))
    halo = pd.concat(reps, ignore_index=True)
    groups = {k: np.sort(v["idx"].to_numpy()) for k, v in halo.groupby(["la", "lo"])}
    now_ns = np.array([pd.Timestamp.utcnow().tz_localize(None).value])
    rows = []
    for g in grid:
        k = (g["la"], g["lo"])
        idx = groups.get(k)
        T = t_ns[idx] if idx is not None else np.array([], dtype="int64")
        M = mags[idx] if idx is not None else np.array([])
        f, _ = cell_features(T, M, now_ns)
        rows.append({nm: float(v[0]) for nm, v in f.items()} |
                    {"la": k[0], "lo": k[1], "slip": float(g.get("slip", 0.0))})
    pd.DataFrame(rows).to_csv(os.path.join(OUTG, "cells_live.csv"), index=False)
    print(f"estado vivo recalculado: {len(rows)} celdas")


def main():
    new = fetch_feed()
    cat = merge_catalog(new)
    recompute_live(cat)
    import emit_globe
    forecast = emit_globe.main()
    from forecast_ledger import archive_forecast
    snapshot = archive_forecast(forecast)
    print(f"snapshot prospectivo: {snapshot or 'ya existía para este intervalo'}")


if __name__ == "__main__":
    main()
