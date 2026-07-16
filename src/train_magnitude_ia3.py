"""MOCRE-1 — entrenamiento IA3 de la salida MAGNITUD, port fiel del auto-ajuste de Cícero
(ajusteprevision_new.php) aplicado a sismos reales en vez de precios.

"Mapear los datos en tiempo real pasados de cada sismo para su entrenamiento" (Matt): cada fila =
estado real (canales) en el momento previo a un sismo + su magnitud REAL observada. IA3 recorre esas
filas y mueve las particiones de salida hasta que el centroide predice la magnitud real. El skill es
el ERROR de predicción (MAE) sobre magnitudes reales, reducido por el entrenamiento -- NO AUC vs ETAS.
"""
import json
import os

import numpy as np
import pandas as pd

from common import ANOMALY_SETS, ROOT
from fuzzy_avx_engine import train_output_sets, predict_over, infer_value

CUTOFF = "2018-01-01"
MAG_LO, MAG_HI = 3.5, 7.5

# ENTRADA: mag_scale (escala tectónica del segmento) con SU partición = ANOMALY_SETS de 5 (todas las
# variables ya están normalizadas al mismo espacio [-1,1] vía tau; mag_scale también). Medido que el
# estado transitorio NO resuelve magnitud (|r|<=0.08) -> la magnitud la fija la tectónica, y IA3
# entrena dónde caen los términos de salida por clase tectónica desde las magnitudes reales.
INPUT_SETS = {"mag_scale": ANOMALY_SETS}

# 5 términos de salida (uno por clase tectónica), cada uno con SU regla -> IA3 los coloca en la escala
# Richter real. Inicialización deliberadamente separada y sin calibrar: que el entrenamiento demuestre
# que MUEVE los términos a su sitio (como el precio del padre parte de sets por defecto y se ajustan).
RULES = [
    ([("mag_scale", "muy_bajo")], "m1"),
    ([("mag_scale", "bajo")],     "m2"),
    ([("mag_scale", "normal")],   "m3"),
    ([("mag_scale", "alto")],     "m4"),
    ([("mag_scale", "muy_alto")], "m5"),
]
INIT_OUTPUT = {
    "m1": [3.0, 3.5, 4.0],
    "m2": [3.8, 4.3, 4.8],
    "m3": [4.5, 5.0, 5.5],
    "m4": [5.2, 5.8, 6.4],
    "m5": [6.0, 7.0, 8.0],
}


def load_rows():
    df = pd.read_csv(os.path.join(OUT_ENR := os.path.join(ROOT, "out", "cases_enriched.csv")),
                     parse_dates=["date"])
    seg_target = {}
    import glob
    for p in glob.glob(os.path.join(ROOT, "config", "segment_*.json")):
        c = json.load(open(p))
        seg_target[c.get("segment_id", "").lower().replace("_", "-")] = c["etas_lite"]["target_mag"]
    df["mag_scale"] = df["segment"].map(seg_target).apply(
        lambda t: float(np.clip(2 * (t - MAG_LO) / (MAG_HI - MAG_LO) - 1, -1, 1)) if pd.notna(t) else 0.0)
    ev = df.dropna(subset=["event_mag"]).copy()      # solo sismos reales (tienen magnitud real)
    tr = ev[ev["date"] < CUTOFF]
    ho = ev[ev["date"] >= CUTOFF]
    to_rows = lambda g: [{"mag_scale": r.mag_scale, "event_mag": r.event_mag} for r in g.itertuples()]
    return to_rows(tr), to_rows(ho)


def mae(pred, real):
    return float(np.mean(np.abs(pred - real))) if len(pred) else float("nan")


def main():
    tr_rows, ho_rows = load_rows()
    print(f"filas de sismos reales: train={len(tr_rows)}  holdout={len(ho_rows)}\n")

    # MAE ANTES de entrenar (sets iniciales sin calibrar) -- baseline del propio motor
    p0, r0 = predict_over(ho_rows, INPUT_SETS, RULES, INIT_OUTPUT, "event_mag")
    print(f"MAE holdout ANTES de IA3 (sets por defecto):   {mae(p0, r0):.3f}")

    # IA3: entrenar los sets de salida sobre train
    trained, err_hist = train_output_sets(tr_rows, INPUT_SETS, RULES, INIT_OUTPUT,
                                           target_key="event_mag", learning_rate=0.2,
                                           tolerance=0.02, passes=5)
    p1, r1 = predict_over(ho_rows, INPUT_SETS, RULES, trained, "event_mag")
    print(f"MAE holdout DESPUÉS de IA3 (sets entrenados):  {mae(p1, r1):.3f}")
    print(f"MAE train (media |error| durante ajuste):      {np.mean(err_hist):.3f}\n")

    print("=== términos de salida movidos por IA3 (centroide = magnitud predicha por clase) ===")
    for k in ["m1", "m2", "m3", "m4", "m5"]:
        a, b, c = trained[k]
        moved = "(sin datos que la disparen)" if trained[k] == INIT_OUTPUT[k] else ""
        print(f"  {k}: [{a:.2f},{b:.2f},{c:.2f}] centroide={sum(trained[k])/3:.2f} {moved}")

    # referencia honesta: MAE de la media global (lo que ya usaba discover_rules)
    all_real = np.array([r["event_mag"] for r in tr_rows])
    base = mae(np.full(len(r1), all_real.mean()), r1)
    print(f"\nreferencia MAE media-global: {base:.3f}  |  IA3 entrenado: {mae(p1, r1):.3f}")

    json.dump(trained, open(os.path.join(ROOT, "out", "magnitude_sets_ia3.json"), "w"), indent=2)
    print(f"-> sets entrenados en out/magnitude_sets_ia3.json (análogo de ajustes_entrenados.json)")


if __name__ == "__main__":
    main()
