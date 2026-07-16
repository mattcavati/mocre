"""MOCRE-1 — port FIEL del motor fuzzy de Cícero Cavati (Avaltix IA2 + IA3), aplicado a nuestras
salidas sísmicas en vez de precios de activos. Matt: "eses motores deben ser utilizados para el
nuestro; buscan precios de activos, nosotros tenemos nuestras propias salidas".

Correspondencia 1:1 con el código real:
  - IA2  (fuzzy_logica_predicion.php): fuzzificar cada variable con SU partición -> reglas de
    producción (min=Y, max=agregación) -> recorte del consecuente por la fuerza -> agregación ->
    DEFUZZIFICACIÓN POR CENTROIDE -> un VALOR. Escala por clase de activo == aquí por RÉGIMEN.
  - IA3  (ajusteprevision_new.php): recorrer la serie real fila a fila; para cada fila predecir el
    valor del evento, comparar con el REAL, y ajustar las particiones de SALIDA por
        delta = error * learning_rate * fuerza_de_la_regla_disparada
    con los MISMOS constraints (geometría triangular válida, no colapso). Esto ENTRENA dónde se
    sitúan los términos de salida (moderada/fuerte/muy_fuerte en la escala Richter) para que el
    centroide caiga sobre la magnitud real. Es lo que hace que el sistema PREVEA un valor.

NO se compara contra ETAS por AUC (ese era mi error). El skill es el ERROR de predicción del valor
sobre sismos reales, reducido por el entrenamiento IA3 -- exactamente como el motor del padre mide
su acierto sobre el precio real de t+1.
"""
import json
import os

import numpy as np
import pandas as pd

from common import ROOT, get_membership, membership_centroid

OUT = os.path.join(ROOT, "out")

# --- régimen -> escala/ँnormalización de salida (análogo EXACTO de avx_get_volatility_params) ------
# el padre adapta tanh_macd/tanh_ema/pred_scale por clase de activo; aquí el régimen tectónico juega
# ese papel. pred_scale deja la magnitud tal cual (ya está en Richter absoluto); se deja el hook para
# cuando entrenemos cuándo/dónde, donde la escala por régimen sí importará.
REGIME_PARAMS = {
    "transform":  {"pred_scale": 1.0},
    "subduction": {"pred_scale": 1.0},
    "normal":     {"pred_scale": 1.0},
    "induced":    {"pred_scale": 1.0},
}


def trimf_centroid(params):
    """Centroide geométrico de triángulos y hombros trapezoidales."""
    return membership_centroid(params)


def fuzzify(value, sets):
    """mu por término para una variable con SU propia partición (como en IA2)."""
    return {name: float(get_membership(np.array([float(value)]), p)[0]) for name, p in sets.items()}


def infer_value(inputs, input_sets, rules, output_sets, pred_scale=1.0):
    """IA2: dado el estado (inputs), fuzzifica cada variable con su partición, evalúa las reglas
    (min=Y), agrega por término de salida (max=O), y DEFUZZIFICA POR CENTROIDE PONDERADO sobre los
    output_sets ENTRENADOS. Devuelve (valor, fuerzas_por_termino, label). Fiel a get_fuzzy_prediction.

    rules: lista de (antecedent, out_term) donde antecedent = [(var, term), ...] (min entre cláusulas).
    """
    mu = {v: fuzzify(inputs[v], input_sets[v]) for v in input_sets if v in inputs}
    # fuerza por término de salida = max sobre las reglas que apuntan a ese término
    strength = {term: 0.0 for term in output_sets}
    for antecedent, out_term in rules:
        if out_term not in strength:
            continue
        degs = []
        missing = False
        for v, s in antecedent:
            if v not in mu or s not in mu[v]:
                missing = True
                break
            degs.append(mu[v][s])
        if missing or not degs:
            continue
        r = min(degs)
        if r > strength[out_term]:
            strength[out_term] = r
    # defuzzificación por centroide ponderado (idéntico al padre: sum(centroide*fuerza)/sum(fuerza))
    num = den = 0.0
    for term, params in output_sets.items():
        s = strength[term]
        if s > 0:
            c = trimf_centroid(params)
            num += c * s
            den += s
    value = (num / den) if den > 0 else None
    if value is not None:
        value *= pred_scale
    label = max(strength, key=strength.get) if den > 0 else None
    return value, strength, label


def train_output_sets(rows, input_sets, rules, init_output_sets, target_key,
                      learning_rate=0.2, tolerance=0.02, pred_scale=1.0, passes=3):
    """IA3: ajuste supervisado en línea de las particiones de SALIDA contra el valor REAL.
    Port fiel del bucle de ajusteprevision_new.php:
      para cada fila -> predecir -> error = real - predicho -> si |error|>tol, para cada término
      DISPARADO mover su triángulo por error*lr*fuerza, con constraints de geometría triangular.
    `rows`: lista de dicts con los canales de entrada + rows[target_key] = valor real observado.
    Devuelve (output_sets_entrenados, historial_error)."""
    sets = {k: list(v) for k, v in init_output_sets.items()}   # copia mutable
    err_hist = []
    for _ in range(passes):
        for row in rows:
            target = row.get(target_key)
            if target is None or (isinstance(target, float) and np.isnan(target)):
                continue
            pred, strength, _ = infer_value(row, input_sets, rules, sets, pred_scale)
            if pred is None:
                continue
            error = float(target) - pred                       # real - predicho (como el padre)
            err_hist.append(abs(error))
            if abs(error) <= tolerance:
                continue
            for term, s in strength.items():
                if s <= 0:
                    continue
                delta = error * learning_rate * s              # proporcional a error, lr y fuerza
                sets[term][0] += delta
                sets[term][1] += delta
                sets[term][2] += delta
                # INTEGRITY CHECK: geometría triangular válida (inicio<pico<fin), como el padre
                mg = 1e-4
                if sets[term][0] >= sets[term][1]:
                    sets[term][0] = sets[term][1] - mg
                if sets[term][2] <= sets[term][1]:
                    sets[term][2] = sets[term][1] + mg
    return sets, err_hist


def predict_over(rows, input_sets, rules, output_sets, target_key, pred_scale=1.0):
    """Evalúa el motor (ya entrenado) sobre filas y devuelve (preds, reales) para las que tienen
    valor real -- para medir el ERROR de predicción (el skill real, no AUC)."""
    preds, reals = [], []
    for row in rows:
        t = row.get(target_key)
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        p, _, _ = infer_value(row, input_sets, rules, output_sets, pred_scale)
        if p is None:
            continue
        preds.append(p)
        reals.append(float(t))
    return np.array(preds), np.array(reals)
