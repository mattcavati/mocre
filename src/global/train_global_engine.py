"""MOCRE-GLOBAL — motor fuzzy global entrenado + calibración honesta.

Canales fuzzy [-1,1] derivados SOLO del catálogo (disponibles en todo el planeta):
  a_seis  : anomalía Poisson del conteo 30d vs tasa de fondo de la celda
  a_swarm : aceleración 30d vs el propio último año
  a_fore  : aceleración 7d vs 30d (foreshock proxy)
  A_state : nivel de actividad 90d vs fondo
  C_state : carga (intervalos medios transcurridos desde el último evento)
  mag_scale: escala tectónica local (Mmax histórico)
Score de ocurrencia = suma de (lift-1)*pertenencia (Wang-Mendel aditivo, el ranker honesto).
El problema de N efectivo del entrenamiento por segmentos (226 clusters) queda resuelto por
volumen: el planeta aporta decenas de miles de episodios M>=5 independientes.

CALIBRACIÓN 2D: bins (log-omori x score fuzzy) -> P(M>=5,30d) empírica en TRAIN (<2018);
verificación en HOLDOUT (>=2018): Brier, tabla de fiabilidad, lift por decil. Es λ=μ_ETAS·G en
forma calibrada: el eje omori es la persistencia (ETAS), el eje fuzzy la modula (G).
Salida: out/global/global_engine.json (taus+reglas+tabla calib) para emit_globe/update_realtime.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTG = os.path.join(ROOT, "out", "global")
CUTOFF = "2018-01-01"

SETS5 = {  # partición Ruspini compartida (la de common.py, duplicada aquí para autonomía del módulo)
    "muy_bajo": [-1.0, -1.0, -0.70, -0.35],
    "bajo": [-0.70, -0.35, 0.0],
    "normal": [-0.35, 0.0, 0.35],
    "alto": [0.0, 0.35, 0.70],
    "muy_alto": [0.35, 0.70, 1.0, 1.0],
}


def memb(x, p):
    x = np.asarray(x, float)
    if len(p) == 3:
        a, b, c = p
        left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
        right = np.where(c > b, (c - x) / (c - b + 1e-12), 1.0)
        return np.clip(np.minimum(left, right), 0, 1)
    a, b, c, d = p
    left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
    right = np.where(d > c, (d - x) / (d - c + 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


def derive_channels(df):
    """features crudas -> canales fuzzy [-1,1]."""
    out = pd.DataFrame(index=df.index)
    mu30 = df.rate_bg * 30 / 365.25
    out["a_seis"] = np.tanh(((df.n30 - mu30) / np.sqrt(mu30 + 0.5)) / 4.0)
    exp30 = df.n365 * 30 / 365.0
    out["a_swarm"] = np.tanh(((df.n30 - exp30) / np.sqrt(exp30 + 0.5)) / 4.0)
    exp7 = df.n30 * 7 / 30.0
    out["a_fore"] = np.tanh(((df.n7 - exp7) / np.sqrt(exp7 + 0.5)) / 4.0)
    mu90 = df.rate_bg * 90 / 365.25
    out["A_state"] = np.tanh((df.n90 / np.maximum(mu90, 0.25) - 1.0) / 2.0)
    x = df.t_last / 365.25 * df.rate_bg           # intervalos medios transcurridos
    out["C_state"] = np.tanh((x - 1.0) / 2.0)
    out["mag_scale"] = np.clip(2 * (df.mag_max - 4.5) / (8.0 - 4.5) - 1, -1, 1)
    return out


CH = ["a_seis", "a_swarm", "a_fore", "A_state", "C_state", "mag_scale"]


def learn_lifts(chan, y, min_support=500.0):
    """Wang-Mendel global: lift por (canal, término). Con el N global, estable."""
    base = y.mean()
    lifts = {}
    for v in CH:
        x = chan[v].to_numpy()
        for s, p in SETS5.items():
            m = memb(x, p)
            supp = m.sum()
            if supp < min_support:
                continue
            pr = float((m * y).sum() / supp)
            lifts[f"{v}|{s}"] = {"lift": pr / (base + 1e-12), "support": float(supp)}
    return lifts, float(base)


def fuzzy_score(chan, lifts):
    sc = np.zeros(len(chan))
    for key, d in lifts.items():
        v, s = key.split("|")
        sc += (d["lift"] - 1.0) * memb(chan[v].to_numpy(), SETS5[s])
    return sc


def calibrate_2d(omori, score, y, n_om=8, n_sc=6):
    """tabla 2D: bins de log-omori (bin 0 = omori==0) x cuantiles de score -> P empírica."""
    lo = np.log1p(omori)
    om_edges = [0.0] + list(np.quantile(lo[lo > 0], np.linspace(0, 1, n_om)[1:-1])) + [np.inf]
    sc_edges = list(np.quantile(score, np.linspace(0, 1, n_sc + 1)))
    sc_edges[0], sc_edges[-1] = -np.inf, np.inf
    table = np.zeros((len(om_edges) - 1, n_sc))
    counts = np.zeros_like(table)
    oi = np.clip(np.searchsorted(om_edges, lo, side="right") - 1, 0, len(om_edges) - 2)
    si = np.clip(np.searchsorted(sc_edges, score, side="right") - 1, 0, n_sc - 1)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            m = (oi == i) & (si == j)
            counts[i, j] = m.sum()
            table[i, j] = y[m].mean() if m.sum() >= 200 else np.nan
    # rellenar celdas raras con la marginal de su fila (omori manda) y luego la global
    for i in range(table.shape[0]):
        row = table[i]
        rm = np.nanmean(row)
        row[np.isnan(row)] = rm if not np.isnan(rm) else y.mean()
    return {"om_edges": [float(x) for x in om_edges[:-1]] + [1e18],
            "sc_edges": [(-1e18 if not np.isfinite(sc_edges[0]) else float(sc_edges[0]))] +
                        [float(x) for x in sc_edges[1:-1]] + [1e18],
            "table": table.tolist(), "counts": counts.tolist()}


def apply_calib(omori, score, calib):
    lo = np.log1p(omori)
    om_e = np.array(calib["om_edges"]); sc_e = np.array(calib["sc_edges"])
    t = np.array(calib["table"])
    oi = np.clip(np.searchsorted(om_e, lo, side="right") - 1, 0, t.shape[0] - 1)
    si = np.clip(np.searchsorted(sc_e, score, side="right") - 1, 0, t.shape[1] - 1)
    return t[oi, si]


def main():
    df = pd.read_csv(os.path.join(OUTG, "cellmonths.csv.gz"), parse_dates=["date"])
    print(f"celda-mes: {len(df)} filas, {df.y.mean()*100:.2f}% positivas")
    chan = derive_channels(df)
    y = df.y.to_numpy(float)
    tr = (df.date < CUTOFF).to_numpy()
    ho = ~tr
    # nº de episodios independientes (clusters 0->1 por celda) — el fix del N efectivo
    d2 = df[["la", "lo", "date", "y"]].sort_values(["la", "lo", "date"])
    trans = ((d2.y.to_numpy()[1:] == 1) & (d2.y.to_numpy()[:-1] == 0)).sum()
    print(f"episodios positivos independientes (0->1): ~{int(trans)} (vs 226 en los 9 segmentos)")

    lifts, base = learn_lifts(chan[tr], y[tr])
    print(f"\nreglas (lift por canal|término, train, base={base*100:.2f}%):")
    for k, v in sorted(lifts.items(), key=lambda kv: -kv[1]["lift"])[:8]:
        print(f"  {k:<22} lift={v['lift']:.2f} (soporte {v['support']:.0f})")

    sc_tr = fuzzy_score(chan[tr], lifts)
    sc_ho = fuzzy_score(chan[ho], lifts)
    calib = calibrate_2d(df.omori.to_numpy()[tr], sc_tr, y[tr])

    # ===== VERIFICACIÓN HONESTA EN HOLDOUT =====
    p_ho = apply_calib(df.omori.to_numpy()[ho], sc_ho, calib)
    yh = y[ho]
    brier = float(np.mean((p_ho - yh) ** 2)); brier_base = float(np.mean((yh.mean() - yh) ** 2))
    print(f"\n=== HOLDOUT (>= {CUTOFF}) ===")
    print(f"Brier calibrado={brier:.5f}  vs base-rate={brier_base:.5f}  "
          f"(mejora {100*(brier_base-brier)/brier_base:.1f}%)")
    # fiabilidad
    print("fiabilidad (P predicha vs frecuencia real):")
    for a, b in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1.01)]:
        m = (p_ho >= a) & (p_ho < b)
        if m.sum() > 300:
            print(f"  P[{a:.2f},{b:.2f}): n={int(m.sum()):7d}  pred={p_ho[m].mean()*100:5.1f}%  "
                  f"real={yh[m].mean()*100:5.1f}%")
    # discriminación por deciles de P
    q = np.quantile(p_ho, np.linspace(0, 1, 11))
    top = p_ho >= q[-2]
    print(f"decil superior de P: tasa real={yh[top].mean()*100:.1f}% vs base {yh.mean()*100:.1f}% "
          f"(lift {yh[top].mean()/yh.mean():.1f}x)")
    # ¿el fuzzy añade sobre omori solo? (misma tabla con score constante)
    calib_om = calibrate_2d(df.omori.to_numpy()[tr], np.zeros(tr.sum()), y[tr], n_sc=1)
    p_om = apply_calib(df.omori.to_numpy()[ho], np.zeros(ho.sum()), calib_om)
    print(f"Brier omori-solo={np.mean((p_om-yh)**2):.5f} vs omori+fuzzy={brier:.5f} "
          f"(delta fuzzy={np.mean((p_om-yh)**2)-brier:+.5f})")

    json.dump({"lifts": lifts, "base_rate": base, "calib": calib, "cutoff": CUTOFF,
               "target": "M>=5.0 en 30 dias, vecindad ~150km",
               "brier_holdout": brier, "brier_base": brier_base},
              open(os.path.join(OUTG, "global_engine.json"), "w"))
    print(f"\n-> out/global/global_engine.json")


if __name__ == "__main__":
    main()
