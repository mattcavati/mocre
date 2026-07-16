"""MOCRE-1 — sistema fuzzy de 4 SALIDAS entrenado por IA3 (port del motor de Cícero), sobre la serie
DENSA diaria (dense_training.csv). Un motor por salida (ocurrencia, magnitud, cuándo, dónde), cada
uno: reglas input->término (derivadas de los datos reales, Wang-Mendel) + IA3 que entrena dónde caen
los términos de salida contra el valor REAL. Skill = ERROR de predicción del valor, NO AUC vs ETAS.

Optimización fiel: las fuerzas de regla por día son constantes durante IA3 (el estado de entrada no
cambia; solo se mueven las particiones de salida) -> se precomputan una vez; IA3 solo reajusta los
centroides de los términos. Idéntico en resultado a recomputar infer_value cada fila como el padre.
"""
import json
import os

import numpy as np
import pandas as pd

from common import ROOT, get_membership, membership_centroid, sets_for_var

OUT = os.path.join(ROOT, "out")
CHANNELS = ["a_seis", "a_swarm", "a_foreshock", "a_gnss", "a_sse", "a_fluid", "a_water", "a_strain",
            "a_ocean", "a_insar", "a_tec", "a_noise", "A_state", "C_state", "K_state", "E_state",
            "F_state"]


def mu_matrix(df, var):
    """mu[fila][term] para una variable, como matriz (n, 5)."""
    x = df[var].fillna(0.0).to_numpy(dtype=float)
    return {s: get_membership(x, p) for s, p in sets_for_var(var).items()}


def extract_rules(tr, cands, target, out_terms, min_support=40.0, min_dev=0.0):
    """Wang-Mendel: para cada (var, partición), media del target ponderada por pertenencia sobre
    TRAIN; la regla dispara el TÉRMINO de salida cuyo centroide inicial esté más cerca de esa media.
    Devuelve lista de (var, term_set_name, out_term). Además arity-2 desde learned_rules si es occ."""
    y = tr[target].to_numpy(dtype=float)
    mask = ~np.isnan(y)
    rules = []
    term_c = {k: membership_centroid(v) for k, v in out_terms.items()}
    for v in cands:
        if v not in tr.columns:
            continue
        mm = mu_matrix(tr, v)
        for s, m in mm.items():
            mw = m * mask
            supp = mw.sum()
            if supp < min_support:
                continue
            mean_t = float((mw * np.nan_to_num(y)).sum() / supp)
            out_term = min(term_c, key=lambda k: abs(term_c[k] - mean_t))
            rules.append((v, s, out_term))
    return rules


def precompute_strengths(df, rules):
    """Para cada fila, fuerza por término de salida = max sobre reglas que apuntan a ese término
    (min dentro de la regla; aquí reglas arity-1 => la propia pertenencia). Devuelve dict term->array."""
    n = len(df)
    mus = {}
    for v in set(r[0] for r in rules):
        mus[v] = mu_matrix(df, v)
    terms = set(r[2] for r in rules)
    strength = {t: np.zeros(n) for t in terms}
    for (v, s, out_term) in rules:
        np.maximum(strength[out_term], mus[v][s], out=strength[out_term])
    return strength


def ia3_train(strength_tr, y_tr, out_terms, lr=0.2, tol=0.02, passes=5):
    """IA3: ajusta los centroides de los términos de salida por error*lr*fuerza (port fiel).
    Trabaja sobre triángulos [a,b,c]; mueve los 3 puntos, mantiene geometría válida."""
    sets = {k: list(v) for k, v in out_terms.items()}
    order = np.arange(len(y_tr))
    valid = ~np.isnan(y_tr)
    for _ in range(passes):
        for i in order:
            if not valid[i]:
                continue
            num = den = 0.0
            s_i = {t: strength[i] for t, strength in strength_tr.items()}
            for t, s in s_i.items():
                if s > 0:
                    c = membership_centroid(sets[t])
                    num += c * s; den += s
            if den == 0:
                continue
            pred = num / den
            err = y_tr[i] - pred
            if abs(err) <= tol:
                continue
            for t, s in s_i.items():
                if s <= 0:
                    continue
                d = err * lr * s
                sets[t][0] += d; sets[t][1] += d; sets[t][2] += d
                if sets[t][0] >= sets[t][1]: sets[t][0] = sets[t][1] - 1e-4
                if sets[t][2] <= sets[t][1]: sets[t][2] = sets[t][1] + 1e-4
    return sets


def complete_output_partition(sets, universe):
    """Project trained output terms onto a complete Ruspini chain over the declared universe."""
    lo, hi = map(float, universe)
    names = list(sets)
    centers = np.array([membership_centroid(sets[name]) for name in names], dtype=float)
    centers = np.clip(np.sort(centers), lo, hi)
    if len(centers) > 1:
        min_step = max((hi - lo) * 1e-4, 1e-6)
        for i in range(1, len(centers)):
            if centers[i] <= centers[i - 1]:
                centers[i] = min(hi, centers[i - 1] + min_step)
        for i in range(len(centers) - 2, -1, -1):
            if centers[i] >= centers[i + 1]:
                centers[i] = max(lo, centers[i + 1] - min_step)
    out = {}
    for i, name in enumerate(names):
        if i == 0:
            out[name] = [lo, lo, float(centers[i]), float(centers[i + 1])]
        elif i == len(names) - 1:
            out[name] = [float(centers[i - 1]), float(centers[i]), hi, hi]
        else:
            out[name] = [float(centers[i - 1]), float(centers[i]), float(centers[i + 1])]
    return out


def predict(strength, sets):
    n = len(next(iter(strength.values())))
    num = np.zeros(n); den = np.zeros(n)
    for t, s in strength.items():
        c = membership_centroid(sets[t])
        num += c * s; den += s
    out = np.full(n, np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def evaluate(name, tr, ho, target, cands, universe, train_on_pos=False, is_occ=False, lr=0.2):
    lo, hi = universe
    out_terms = {f"t{i+1}": [lo + (hi-lo)*i/5, lo + (hi-lo)*(i+0.5)/5, lo + (hi-lo)*(i+1)/5]
                 for i in range(5)}
    tr_use = tr[tr["occ_30d"] == 1] if train_on_pos else tr
    if train_on_pos and "event_id" in tr_use:
        tr_use = tr_use.dropna(subset=["event_id"]).drop_duplicates("event_id")
    rules = extract_rules(tr_use, cands, target, out_terms)
    if not rules:
        print(f"  [{name}] sin reglas"); return None
    st_tr = precompute_strengths(tr_use, rules)
    y_tr = tr_use[target].to_numpy(dtype=float)
    # baseline: MAE con sets SIN entrenar
    pred0 = predict(st_tr, out_terms)
    trained_raw = ia3_train(st_tr, y_tr, out_terms, lr=lr)
    trained = complete_output_partition(trained_raw, universe)
    # evaluar en holdout
    ho_use = ho[ho["occ_30d"] == 1] if train_on_pos else ho
    if train_on_pos and "event_id" in ho_use:
        ho_use = ho_use.dropna(subset=["event_id"]).drop_duplicates("event_id")
    st_ho = precompute_strengths(ho_use, rules)
    y_ho = ho_use[target].to_numpy(dtype=float)
    p0 = predict(st_ho, out_terms); p1 = predict(st_ho, trained)
    p_tr = predict(st_tr, trained)
    m = ~np.isnan(y_ho) & ~np.isnan(p1)
    mae0 = float(np.mean(np.abs(p0[m] - y_ho[m]))); mae1 = float(np.mean(np.abs(p1[m] - y_ho[m])))
    # Baseline justo para salidas condicionadas: magnitud por escala objetivo y
    # cuándo/dónde por segmento. Nunca usa la media del propio holdout.
    group_col = "mag_scale" if target == "mag_next" else "segment"
    if train_on_pos and group_col in tr_use and group_col in ho_use:
        medians = tr_use.groupby(group_col)[target].median()
        base_pred = ho_use[group_col].map(medians).fillna(np.nanmedian(y_tr)).to_numpy(float)
    else:
        base_pred = np.full(len(ho_use), np.nanmean(y_tr))
    base = float(np.mean(np.abs(base_pred[m] - y_ho[m])))
    line = f"  [{name:<10}] reglas={len(rules):3d} | MAE holdout: sin-entrenar={mae0:.3f} -> IA3={mae1:.3f} | baseline-condicionado={base:.3f}"
    if is_occ:
        brier = float(np.mean((p1[m] - y_ho[m])**2))
        line += f" | Brier={brier:.4f}"
    print(line)
    return {"name": name, "rules": rules, "n_rules": len(rules), "universe": list(universe),
            "mae_untrained": mae0, "mae_ia3": mae1, "mae_baseline": base, "trained_sets": trained,
            "p_tr": p_tr, "y_tr": y_tr, "p_ho": p1, "y_ho": y_ho,
            "rules_raw": rules}


def main():
    dense = pd.read_csv(os.path.join(OUT, "dense_training.csv"), parse_dates=["date"])
    tr = dense[dense.block == "train"].copy(); ho = dense[dense.block == "holdout"].copy()
    cal_start = pd.Timestamp("2015-01-01")
    fit = tr[tr.date < cal_start].copy()
    cal = tr[tr.date >= cal_start].copy()
    if len(fit) == 0 or len(cal) == 0:
        raise RuntimeError("se requieren bloques cronológicos separados fit/calibración")
    print(f"dense fit={len(fit)} calibración={len(cal)} purga={(dense.block == 'purged').sum()} "
          f"holdout={len(ho)}\n")
    print("skill = ERROR de predicción del valor real (NO AUC vs ETAS):\n")
    results = {}
    # OCURRENCIA: valor = probabilidad de sismo en 30d (target 0/1), universo [0,1], todo el estado
    results["ocurrencia"] = evaluate("ocurrencia", fit, ho, "occ_30d", CHANNELS + ["mag_scale"],
                                     (0.0, 1.0), train_on_pos=False, is_occ=True, lr=0.2)
    # MAGNITUD: valor = Richter real, universo [3,8], keyed SOLO en mag_scale (tectónico) — medido
    # que el estado transitorio no la resuelve (|r|<=0.08), incluir canales solo mete ruido en IA3.
    results["magnitud"] = evaluate("magnitud", fit, ho, "mag_next", ["mag_scale"],
                                   (3.0, 8.0), train_on_pos=True, lr=0.05)
    # CUÁNDO: valor = días al sismo, universo [0,30] — solo sismos
    results["cuando"] = evaluate("cuando", fit, ho, "days_next", CHANNELS,
                                 (0.0, 30.0), train_on_pos=True, lr=0.05)
    # DÓNDE: valor = radio normalizado, universo [0,1] — solo sismos
    results["donde"] = evaluate("donde", fit, ho, "radial_next", CHANNELS,
                                (0.0, 1.0), train_on_pos=True, lr=0.05)

    # Calibración en un bloque anterior al holdout. El holdout se toca una sola
    # vez para evaluación y nunca se convierte en tabla de producción.
    from sklearn.isotonic import IsotonicRegression
    oc = results["ocurrencia"]
    st_cal = precompute_strengths(cal, oc["rules_raw"])
    p_cal_fit = predict(st_cal, oc["trained_sets"])
    y_cal_fit = cal["occ_30d"].to_numpy(float)
    mtr = ~np.isnan(p_cal_fit) & ~np.isnan(y_cal_fit)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_fit[mtr], y_cal_fit[mtr])
    mho = ~np.isnan(oc["p_ho"]) & ~np.isnan(oc["y_ho"])
    p_raw, y = oc["p_ho"][mho], oc["y_ho"][mho]
    p_cal = iso.predict(p_raw)
    brier_raw = float(np.mean((p_raw - y)**2)); brier_cal = float(np.mean((p_cal - y)**2))
    base_rate = float(y_cal_fit[mtr].mean())
    brier_base = float(np.mean((base_rate - y)**2))
    print(f"\n=== CALIBRACIÓN ocurrencia (fit->calibración->holdout) ===")
    print(f"  Brier: crudo={brier_raw:.4f} -> calibrado={brier_cal:.4f} | base-rate constante={brier_base:.4f}")
    # tabla de calibración por bins (fiabilidad)
    print("  fiabilidad (prob predicha calibrada vs tasa real):")
    edges = [0, .05, .1, .2, .4, 1.01]
    for a, b in zip(edges[:-1], edges[1:]):
        mm = (p_cal >= a) & (p_cal < b)
        if mm.sum() > 20:
            print(f"    pred[{a:.2f},{b:.2f}): n={int(mm.sum()):5d}  tasa_real={y[mm].mean()*100:5.1f}%  pred_media={p_cal[mm].mean()*100:5.1f}%")

    cal_thr = [float(x) for x in iso.X_thresholds_]
    cal_rate = [float(x) for x in iso.y_thresholds_]
    skill_occ = brier_cal + 1e-4 < brier_base
    print(f"  calibrador aprendido antes de 2018; skill confirmada={skill_occ}")

    # --- PERSISTIR el modelo completo para inferencia (mapa) ---
    model = {"outputs": {}, "occ_calibrator": {"thr": cal_thr, "rate": cal_rate},
             "occ_brier_holdout": brier_cal, "occ_brier_base": brier_base,
             "occ_skill_validated": skill_occ,
             "splits": {"fit_end": str(cal_start.date()), "holdout_start": "2018-01-01",
                        "purge_days": HORIZON if 'HORIZON' in globals() else 30},
             "channels": CHANNELS, "mag_lo": 3.5, "mag_hi": 7.5}
    for k, r in results.items():
        if not r:
            continue
        model["outputs"][k] = {"rules": r["rules"], "trained_sets": r["trained_sets"],
                               "universe": r["universe"], "mae_ia3": r["mae_ia3"],
                               "mae_baseline": r["mae_baseline"],
                               "skill_validated": r["mae_ia3"] < r["mae_baseline"]}
    model["outputs"]["ocurrencia"]["brier_cal"] = brier_cal
    model["outputs"]["ocurrencia"]["skill_validated"] = skill_occ
    json.dump(model, open(os.path.join(OUT, "ia3_model.json"), "w"), indent=2)
    print(f"\n-> modelo IA3 completo (reglas+sets+calibrador) en out/ia3_model.json")


if __name__ == "__main__":
    main()
