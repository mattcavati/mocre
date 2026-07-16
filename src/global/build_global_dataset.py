"""MOCRE-GLOBAL — grid mundial de micro-puntos + dataset de entrenamiento celda-mes + estado vivo.

1. GRID: celdas 1°x1°. Activa si >=3 eventos M>=4.5 propios desde 1980 O falla GEM la cruza.
   Cada celda activa = un micro-punto del globo (centro lat/lon).
2. EVENTOS por celda con HALO (vecindad Moore 3x3, ~150km): el estado de un punto lo definen los
   sismos a su alrededor, no solo dentro de su casilla exacta.
3. SNAPSHOTS mensuales 1990->hoy por celda (1980-90 = calentamiento de baselines): features de
   sismicidad + target = ¿hubo M>=5.0 en la vecindad en los 30 días siguientes?
   ESTO RESUELVE EL PROBLEMA DE N EFECTIVO del entrenamiento por segmentos (226 clusters): el
   planeta entero aporta decenas de miles de episodios M>=5 independientes.
4. ESTADO VIVO: las mismas features evaluadas AHORA para cada celda -> el globo en tiempo real.

Todo vectorizado por celda (searchsorted sobre todos los snapshots a la vez).
Salidas: out/global/cellmonths.csv.gz, out/global/cells_live.csv, out/global/grid.json.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data", "global")
OUTG = os.path.join(ROOT, "out", "global")
os.makedirs(OUTG, exist_ok=True)

TARGET_MAG = 5.0
MIN_OWN_EVENTS = 3
SNAP_START = "1990-01-01"
HORIZON = 30.0
D = 86400 * 10**9          # ns por día
T0 = pd.Timestamp("1980-01-01").value


def load_catalog():
    df = pd.read_csv(os.path.join(DATA, "catalog_global.csv"), parse_dates=["time"])
    df = df[(df.latitude.abs() <= 89) & df.mag.notna()]
    return df.sort_values("time").reset_index(drop=True)


def gem_cells():
    """celdas GEM + slip rate máximo (mm/año) por celda — proxy de carga tectónica (el ingrediente
    de strain de los modelos clase GEAR1, aquí desde la GAF-DB que ya tenemos)."""
    path = os.path.join(ROOT, "data", "gem", "gem_active_faults.geojson")
    slip = {}
    for f in json.load(open(path))["features"]:
        geom = f.get("geometry") or {}
        props = f.get("properties") or {}
        sr = props.get("net_slip_rate")
        try:
            # formato GEM: "(valor,min,max)" o número
            sr = float(str(sr).strip("() ").split(",")[0]) if sr not in (None, "") else 0.0
        except (ValueError, IndexError):
            sr = 0.0
        coords = geom.get("coordinates") or []
        lines = [coords] if geom.get("type") == "LineString" else \
                coords if geom.get("type") == "MultiLineString" else []
        for line in lines:
            for pt in line[::3]:
                lon, lat = pt[0], pt[1]
                if abs(lat) <= 89:
                    k = (int(np.floor(lat)), int(np.floor(lon)))
                    slip[k] = max(slip.get(k, 0.0), sr)
    return slip


def cell_features(T, M, ts_arr):
    """features vectorizadas para UNA celda en un array de instantes ts_arr (ns).
    T: tiempos de eventos (ns, ordenados), M: magnitudes."""
    i = np.searchsorted(T, ts_arr)                                  # eventos anteriores a cada ts
    yrs = np.maximum((ts_arr - T0) / D / 365.25, 1.0)
    out = {"rate_bg": i / yrs}
    for w, nm in ((7, "n7"), (30, "n30"), (90, "n90"), (365, "n365")):
        out[nm] = i - np.searchsorted(T, ts_arr - int(w * D))
    # días desde el último evento y Mmax histórico acumulado
    t_last = np.full(len(ts_arr), 99999.0)
    magmax = np.zeros(len(ts_arr))
    has = i > 0
    if len(T):
        cummax = np.maximum.accumulate(M)
        t_last[has] = (ts_arr[has] - T[i[has] - 1]) / D
        magmax[has] = cummax[i[has] - 1]
    out["t_last"] = t_last
    out["mag_max"] = magmax
    # Omori ponderado por productividad (proxy lam_etas), eventos del último año.
    # bucle solo sobre snapshots con actividad reciente (n365>0) — barato.
    omori = np.zeros(len(ts_arr))
    j0 = np.searchsorted(T, ts_arr - int(365 * D))
    act = np.where(out["n365"] > 0)[0]
    for s in act:
        dt = (ts_arr[s] - T[j0[s]:i[s]]) / D
        omori[s] = np.sum(10 ** (0.5 * (M[j0[s]:i[s]] - 4.5)) / (dt + 5.0) ** 1.1)
    out["omori"] = omori
    return out, i


def build():
    cat = load_catalog()
    print(f"catálogo: {len(cat)} eventos {cat.time.min().date()}..{cat.time.max().date()}", flush=True)
    la = np.floor(cat.latitude.to_numpy()).astype(int)
    lo = np.floor(cat.longitude.to_numpy()).astype(int)

    own = pd.DataFrame({"la": la, "lo": lo}).groupby(["la", "lo"]).size()
    seis_cells = set(own[own >= MIN_OWN_EVENTS].index)
    slip = gem_cells()
    active = sorted(seis_cells | set(slip))
    print(f"celdas activas: {len(active)} (sismicidad {len(seis_cells)}, resto GEM)", flush=True)

    # asignación evento->celdas con halo, vectorizada (9 réplicas + groupby)
    reps = []
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            reps.append(pd.DataFrame({
                "la": la + dla,
                "lo": ((lo + dlo + 180) % 360) - 180,
                "idx": np.arange(len(cat)),
            }))
    halo = pd.concat(reps, ignore_index=True)
    # CRÍTICO: ordenar los índices dentro de cada grupo — el groupby preserva orden de réplica
    # (9 concatenados), NO orden temporal; con T desordenado searchsorted da dt negativos y
    # features corruptas (lo delataron los RuntimeWarning de power en el primer run).
    groups = {k: np.sort(v["idx"].to_numpy()) for k, v in halo.groupby(["la", "lo"])}
    print("halo asignado", flush=True)

    t_ns = cat.time.astype("int64").to_numpy()
    mags = cat.mag.to_numpy()
    now = pd.Timestamp.utcnow().tz_localize(None)
    snaps = pd.date_range(SNAP_START, now, freq="MS")
    snap_ns = snaps.astype("int64").to_numpy()
    now_ns = np.array([now.value])

    parts, live_rows, grid = [], [], []
    for n_done, k in enumerate(active):
        idx = groups.get(k)
        if idx is not None:
            T = t_ns[idx]; M = mags[idx]          # idx ordenado -> T temporalmente ordenado
        else:
            T = np.array([], dtype="int64"); M = np.array([])
        srate = float(slip.get(k, 0.0))
        grid.append({"la": int(k[0]), "lo": int(k[1]), "n_hist": int(len(T)),
                     "n_own": int(own.get(k, 0)), "slip": srate})

        f, i = cell_features(T, M, snap_ns)
        # targets: ocurrencia (M>=5 en 30d) + VALOR/CUÁNDO del próximo M>=5 (hasta 365d vista,
        # para entrenar las salidas magnitud/delta-t del motor con eventos reales)
        j1 = np.searchsorted(T, snap_ns + int(HORIZON * D))
        tgt = np.zeros(len(snap_ns), dtype=int)
        mag_next = np.full(len(snap_ns), np.nan)
        days_next = np.full(len(snap_ns), np.nan)
        big = M >= TARGET_MAG
        big_idx = np.where(big)[0]
        big_T = T[big_idx]
        for s in range(len(snap_ns)):
            if j1[s] > i[s]:
                tgt[s] = 1 if np.max(M[i[s]:j1[s]]) >= TARGET_MAG else 0
            # próximo M>=5 estrictamente después del snapshot
            bpos = np.searchsorted(big_T, snap_ns[s], side="right")
            if bpos < len(big_T):
                dd = (big_T[bpos] - snap_ns[s]) / D
                if dd <= 365.0:
                    mag_next[s] = M[big_idx[bpos]]
                    days_next[s] = dd
        keep = ~((f["rate_bg"] == 0) & (f["mag_max"] == 0))    # celdas GEM sin historia: sin filas
        if keep.any():
            p = pd.DataFrame({nm: v[keep] for nm, v in f.items()})
            p["y"] = tgt[keep]; p["mag_next"] = mag_next[keep]; p["days_next"] = days_next[keep]
            p["slip"] = srate
            p["la"] = k[0]; p["lo"] = k[1]
            p["date"] = snaps[keep]
            parts.append(p)

        fl, _ = cell_features(T, M, now_ns)
        live_rows.append({nm: float(v[0]) for nm, v in fl.items()} |
                         {"la": k[0], "lo": k[1], "slip": srate})
        if (n_done + 1) % 5000 == 0:
            print(f"  {n_done+1}/{len(active)} celdas", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(OUTG, "cellmonths.csv.gz"), index=False, compression="gzip")
    pd.DataFrame(live_rows).to_csv(os.path.join(OUTG, "cells_live.csv"), index=False)
    json.dump(grid, open(os.path.join(OUTG, "grid.json"), "w"))
    print(f"celda-mes: {len(df)} filas ({df.y.mean()*100:.2f}% positivas)", flush=True)
    print(f"estado vivo: {len(live_rows)} celdas | grid: {len(grid)} micro-puntos", flush=True)


if __name__ == "__main__":
    build()
