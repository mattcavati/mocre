"""MOCRE-GLOBAL — emite globe/globe_data.json: previsión de cada micro-punto AHORA con el motor
MAMDANI v2 (fuzzy-inverso con grupos + mu_ev medido + IA3 magnitud + calibración walk-forward-
validada). Añade la capa RADAR: sismos reales de los últimos 7 días.
Lo regenera update_realtime.py cada ~10 min.
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTG = os.path.join(ROOT, "out", "global")
GLOBE = os.path.join(ROOT, "globe")
os.makedirs(GLOBE, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))

from global_mamdani import (CH, SETS5, TERMS, derive_channels, memb,        # noqa: E402
                            mamdani_occurrence, calib_3d_apply)
from global_ia2_faithful import apply_v3, ia1_apply                          # noqa: E402
from forecast_ledger import model_hashes                                    # noqa: E402
from common import membership_centroid                                      # noqa: E402


def recent_quakes(days=7.0):
    cat = pd.read_csv(os.path.join(ROOT, "data", "global", "catalog_global.csv"),
                      usecols=["time", "latitude", "longitude", "mag"], parse_dates=["time"])
    now = pd.Timestamp.utcnow().tz_localize(None)
    r = cat[cat.time >= now - pd.Timedelta(days=days)]
    out = []
    for _, e in r.iterrows():
        hours = (now - e.time).total_seconds() / 3600
        out.append([round(float(e.latitude), 2), round(float(e.longitude), 2),
                    round(float(e.mag), 1), round(hours, 1)])
    return out


def mag_stats():
    cat = pd.read_csv(os.path.join(ROOT, "data", "global", "catalog_global.csv"),
                      usecols=["latitude", "longitude", "mag"])
    cat = cat[cat.mag >= 5.0]
    la = np.floor(cat.latitude.to_numpy()).astype(int)
    lo = np.floor(cat.longitude.to_numpy()).astype(int)
    df = pd.DataFrame({"la": la, "lo": lo, "mag": cat.mag.to_numpy()})
    stats = {}
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            g = df.copy(); g["la"] += dla; g["lo"] = ((g["lo"] + dlo + 180) % 360) - 180
            for k, v in g.groupby(["la", "lo"])["mag"]:
                stats.setdefault(k, []).extend(v.tolist())
    return {k: (float(np.mean(v)), float(np.max(v))) for k, v in stats.items()}


def main():
    live = pd.read_csv(os.path.join(OUTG, "cells_live.csv"))
    eng = json.load(open(os.path.join(OUTG, "global_engine_v2.json")))
    chan = derive_channels(live)
    MU = {}
    for c in CH:
        x = chan[c].to_numpy()
        for t in TERMS:
            MU[(c, t)] = memb(x, SETS5[t])
    # reconstruir reglas del JSON al formato del motor
    rules = [{"name": r["name"], "cols": [tuple(ct) for ct in r["cols"]], "chans": r["chans"],
              "lift_fit": r["lift"]} for r in eng["rules"]]
    mom, score = mamdani_occurrence(MU, rules, eng["mu_ev"], 0.0)
    om_l = live.omori.to_numpy(); rt_l = live.rate_bg.to_numpy()
    if "logistic" in eng:
        # calibrador CONTINUO (7-jul): gradúa la cima — sin el plateau de 43.9% del bin superior
        # de la tabla. Validado walk-forward (fiabilidad incl. bins altos) antes de desplegarse.
        lg = eng["logistic"]
        feature_values = {"log1p_omori": np.log1p(om_l),
                          "has_omori": (om_l > 0).astype(float),
                          "log1p_rate": np.log1p(rt_l), "score": score}
        X = np.column_stack([feature_values[name] for name in lg["features"]])
        z = X @ np.array(lg["coef"]) + lg["intercept"]
        p = 1.0 / (1.0 + np.exp(-z))
    else:
        p = calib_3d_apply(om_l, rt_l, score, eng["calib"])
    # Motor IA1->IA2->IA3 v3 conservado como señal experimental en sombra.
    # No se interpreta como alerta ni como probabilidad operativa.
    p3 = apply_v3(live)
    alert = (p3 >= 0.40).astype(int)
    # ENTRADAS por celda para el panel de verificación del globo (Matt 7-jul: "en el globo debe
    # constar todos los valores de entrada de cada punto — así se ve que las entradas se están
    # llevando en consideración"): los 7 canales fuzzy v2 + ia1 (score IA1 del motor fiel).
    eng3 = json.load(open(os.path.join(OUTG, "global_engine_v3.json")))
    ia1_m = {"om": [-np.inf] + eng3["ia1"]["om"] + [np.inf],
             "rt": [-np.inf] + eng3["ia1"]["rt"] + [np.inf],
             "t": np.array(eng3["ia1"]["t"]), "n": eng3["ia1"]["n"]}
    s_ia1 = ia1_apply(live.omori.to_numpy(), live.rate_bg.to_numpy(), ia1_m)

    # magnitud: motor IA3 (mag_sets entrenados) sobre mag_scale/a_tect
    mag_pairs = [("mag_scale", "muy_bajo", "moderada"), ("mag_scale", "bajo", "moderada"),
                 ("mag_scale", "normal", "fuerte"), ("mag_scale", "alto", "muy_fuerte"),
                 ("mag_scale", "muy_alto", "extrema"), ("a_tect", "muy_alto", "muy_fuerte")]
    strengths = {}
    for c, t, term in mag_pairs:
        strengths.setdefault(term, np.zeros(len(live), dtype=np.float32))
        np.maximum(strengths[term], MU[(c, t)], out=strengths[term])
    num = np.zeros(len(live)); den = np.zeros(len(live))
    for t, s in strengths.items():
        cc = membership_centroid(eng["mag_sets"][t])
        num += cc * s; den += s
    mag_engine = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    ms = mag_stats()
    cells = []
    for i, r in live.iterrows():
        k = (int(r.la), int(r.lo))
        mag_mean, mag_max = ms.get(k, (0.0, float(r.mag_max)))
        pi = float(p[i])
        lam = -np.log(max(1 - pi, 1e-9)) / 30.0
        mean_days = float(1 / lam) if lam > 0 else None
        me = mag_engine[i]
        if not eng.get("mag_skill_validated", False):
            me = mag_mean if mag_mean else np.nan
        cells.append([
            int(r.la), int(r.lo), round(pi, 5),
            round(float(me) if np.isfinite(me) else (mag_mean or float(r.mag_max)), 2),
            round(float(mag_max), 1) if mag_max else round(float(r.mag_max), 1),
            int(r.n365), int(min(r.t_last, 99999)),
            round(min(mean_days, 36500), 0) if mean_days else None,
            round(float(mom[i]), 3),
            int(alert[i]), round(float(p3[i]), 3),
            # ENTRADAS del motor (verificables por punto en el panel):
            round(float(chan["a_seis"].iloc[i]), 3), round(float(chan["a_swarm"].iloc[i]), 3),
            round(float(chan["a_fore"].iloc[i]), 3), round(float(chan["A_state"].iloc[i]), 3),
            round(float(chan["C_state"].iloc[i]), 3), round(float(chan["mag_scale"].iloc[i]), 3),
            round(float(chan["a_tect"].iloc[i]), 3), round(float(s_ia1[i]), 4),
            round(float(r.omori), 3), round(float(r.rate_bg), 3), int(r.n30),
        ])
    wf = eng.get("walkforward", [])
    ig_mocre = [w["MOCRE_mamdani"]["info_gain"] for w in wf if "MOCRE_mamdani" in w]
    issued = datetime.now(timezone.utc)
    cutoff = pd.to_datetime(pd.read_csv(os.path.join(ROOT, "data", "global", "catalog_global.csv"),
                                        usecols=["time"]).time, utc=True, errors="coerce").max()
    out = {
        "generated": issued.isoformat(timespec="seconds"),
        "valid_from": issued.isoformat(timespec="seconds"),
        "valid_to": (issued + timedelta(days=30)).isoformat(timespec="seconds"),
        "target": eng["target"],
        "engine": ("Omori+tasa (capa fuzzy en sombra)" if not eng.get("fuzzy_increment_selected")
                   else "Omori+tasa+fuzzy, mejora material walk-forward"),
        "deployed_model": eng.get("deployed_model", "legacy"),
        "fuzzy_increment_selected": eng.get("fuzzy_increment_selected", False),
        "magnitude_semantics": ("climatología histórica regional; IA3 no superó el baseline"
                                if not eng.get("mag_skill_validated", False)
                                else "salida IA3 validada"),
        "status": "investigación; no es predicción determinista ni alerta temprana",
        "time_semantics": ("poisson_mean_wait_days es 1/lambda derivado de p30; no es ventana, "
                           "fecha objetivo ni cuenta atrás"),
        "model_sha256": model_hashes(),
        "input_cutoff": cutoff.isoformat() if pd.notna(cutoff) else None,
        "n_cells": len(cells),
        "fields": ["la", "lo", "p30", "mag_typical", "mag_max", "n365", "t_last_d",
                   "poisson_mean_wait_days", "mom",
                   "alert", "p3",
                   "a_seis", "a_swarm", "a_fore", "A_state", "C_state", "mag_scale", "a_tect",
                   "ia1", "omori", "rate_bg", "n30"],
        "n_alerts": int(alert.sum()),
        "cells": cells,
        "recent": recent_quakes(7.0),
    }
    target = os.path.join(GLOBE, "globe_data.json")
    fd, tmp = tempfile.mkstemp(prefix=".globe-", suffix=".json", dir=GLOBE)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(out, f, allow_nan=False)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"{len(cells)} micro-puntos + {len(out['recent'])} sismos radar -> globe_data.json")
    for c in sorted(cells, key=lambda c: -c[2])[:5]:
        print(f"  top: ({c[0]},{c[1]}) P30d={c[2]*100:.1f}% M~{c[3]} mom={c[8]}")
    return out


if __name__ == "__main__":
    main()
