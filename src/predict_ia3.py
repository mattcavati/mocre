"""MOCRE-1 — inferencia con el modelo IA3 de 4 salidas (out/ia3_model.json), para el mapa.
Dado el estado actual de un segmento, devuelve las salidas del motor de Cícero entrenado.

Honestidad incorporada (medido, no vendido): magnitud es la salida VALIDADA (MAE ~0.48 vs 0.78);
ocurrencia fuera de muestra casi NO discrimina globalmente (calibrada ~plana 11%, flag débil 1.4x
sobre base), robusta solo en Oklahoma (via CSEP, aparte); cuándo/dónde sin señal medida.
"""
import json
import os

import numpy as np

from common import ROOT, get_membership, membership_centroid, sets_for_var

_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = json.load(open(os.path.join(ROOT, "out", "ia3_model.json")))
    return _MODEL


def _strengths(state, rules):
    """fuerza por término = max sobre reglas (arity-1: la pertenencia de (var,set))."""
    strength = {}
    for (v, s, term) in rules:
        val = state.get(v)
        if val is None:
            continue
        mu = float(get_membership(np.array([float(val)]), sets_for_var(v)[s])[0])
        if mu > strength.get(term, 0.0):
            strength[term] = mu
    return strength


def _defuzz(strength, sets):
    num = den = 0.0
    for t, s in strength.items():
        if s > 0 and t in sets:
            c = membership_centroid(sets[t])
            num += c * s; den += s
    return (num / den) if den > 0 else None


def _calibrate_occ(raw, cal):
    if raw is None:
        return None
    for thr, rate in zip(cal["thr"], cal["rate"]):
        if raw <= thr:
            return rate
    return cal["rate"][-1]


def predict(state, mag_scale):
    """state: dict canal->valor [-1,1]. mag_scale: escala tectónica estática. Devuelve las 4 salidas
    del motor IA3 con sus notas de honestidad."""
    m = load_model()
    out = {}
    st = dict(state); st["mag_scale"] = mag_scale
    o = m["outputs"]

    # OCURRENCIA: score crudo -> calibrado (deciles holdout). Flag débil, no probabilidad graduada.
    occ_raw = _defuzz(_strengths(st, o["ocurrencia"]["rules"]), _dictsets(o["ocurrencia"]))
    occ_p = _calibrate_occ(occ_raw, m["occ_calibrator"]) if m.get("occ_skill_validated") else None
    out["ocurrencia"] = {
        "prob": None if occ_p is None else round(occ_p, 3),
        "nota": ("probabilidad no publicada: no mejora el Brier del baseline en holdout"
                 if not m.get("occ_skill_validated") else "calibrada antes del holdout"),
    }
    # MAGNITUD: la salida VALIDADA
    mag = _defuzz(_strengths(st, o["magnitud"]["rules"]), _dictsets(o["magnitud"]))
    mag_ok = o["magnitud"].get("skill_validated", False)
    out["magnitud"] = {
        "value": None if mag is None or not mag_ok else round(mag, 2),
        "nota": (f"IA3 no publicada: MAE {o['magnitud']['mae_ia3']:.2f} vs "
                 f"baseline condicionado {o['magnitud']['mae_baseline']:.2f}"
                 if not mag_ok else "salida superior al baseline condicionado en holdout"),
    }
    # CUÁNDO / DÓNDE: sin señal medida
    out["cuando"] = {"value": None, "nota": "sin señal de estado a 30d (IA3 converge a la media)"}
    dn = _defuzz(_strengths(st, o["donde"]["rules"]), _dictsets(o["donde"]))
    out["donde"] = {"radio": None if dn is None else round(dn, 3),
                    "nota": "sin señal validada (in-sample no generaliza)"}
    return out


def _dictsets(output_entry):
    return output_entry["trained_sets"]


if __name__ == "__main__":
    # demo con un estado de crisis sintético
    demo = {c: 0.0 for c in load_model()["channels"]}
    demo.update({"a_seis": 0.9, "F_state": 0.8, "C_state": 0.9})
    import json as _j
    print(_j.dumps(predict(demo, mag_scale=-0.5), indent=2, ensure_ascii=False))
