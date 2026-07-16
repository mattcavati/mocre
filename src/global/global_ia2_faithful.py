"""MOCRE-GLOBAL v3 — port FIEL de IA1→IA2→IA3 de Avaltix (fuzzy_logica_predicion.php +
ajusteprevision_new.php), sin sustituciones estadísticas. Correspondencia exacta:

  IA1 (score1new)      -> núcleo estadístico de tasa (omori×tasa_fondo -> score [0,1]); IA2 lo
                          REFINA como variable de entrada — igual que IA2 Avaltix refina IA1.
  vol_params por clase -> normalización adaptativa por CLASE DE ACTIVIDAD de la celda (cuartiles
                          de tasa de fondo = forex/crypto/commodity/stock de la sismicidad): tau
                          por (clase, canal) para que cada canal OCUPE su universo en cada clase.
  sets por variable    -> partición Ruspini de 5 POR VARIABLE, nudos en los cuantiles reales de
                          esa variable (cada variable su universo de discurso, como rsi_sets vs
                          score1new_sets vs ema200_sets en el PHP).
  reglas de producción -> fuzzy-inverso (aridad 1+2) con gate de generalización de dos épocas.
  salida 7 términos    -> universo de OCURRENCIA [0,1] con 7 términos graduados (nulo..extremo),
                          como price_change_sets (7: baja_grande..sube_grande).
  IA3                  -> entrena las POSICIONES de los 7 términos contra el resultado real
                          (delta = error×lr×fuerza, constraints de orden/geometría) — IA3 es el
                          calibrador por realimentación; NO tabla estadística.
  defuzzificación      -> centroide ponderado de centroides (la fórmula EXACTA de
                          get_fuzzy_prediction: sum(centroide_i×fuerza_i)/sum(fuerza_i)).

Contrato de infalibilidad operativa: la salida correcta = calibración perfecta (diagonal de
fiabilidad). Toda desviación de la diagonal se trata como error de implementación. Validación
walk-forward 5 épocas + comparación directa con el v2 (tabla 3D) y los núcleos estadísticos.
"""
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTG = os.path.join(ROOT, "out", "global")

CH_RAW = ["a_seis", "a_swarm", "a_fore", "A_state", "C_state", "mag_scale", "a_tect"]
CH_ALL = ["ia1"] + CH_RAW          # IA1 entra al fuzzy como variable (= score1new en IA2)
N_OUT = 7
OUT_NAMES = ["nulo", "muy_bajo", "bajo", "medio", "alto", "muy_alto", "extremo"]


# ============================ universo por variable (nudos en cuantiles) ========================
def ruspini_from_knots(k):
    """5 términos Ruspini de una cadena de 5 nudos interiores + extremos del rango."""
    q0, q1, q2, q3, q4 = k
    return {
        "muy_bajo": [q0 - 1e-6, q0 - 1e-6, q0, q1],
        "bajo": [q0, q1, q2],
        "normal": [q1, q2, q3],
        "alto": [q2, q3, q4],
        "muy_alto": [q3, q4, q4 + 1e-6, q4 + 1e-6],
    }


def ruspini_chain(knots, names=None):
    """cadena Ruspini genérica de n nudos -> n términos (extremos trapezoidales, centro triangular)."""
    n = len(knots)
    names = names or [f"t{i}" for i in range(n)]
    sets = {}
    for i in range(n):
        if i == 0:
            sets[names[i]] = [knots[0] - 1e-6, knots[0] - 1e-6, knots[0], knots[1]]
        elif i == n - 1:
            sets[names[i]] = [knots[-2], knots[-1], knots[-1] + 1e-6, knots[-1] + 1e-6]
        else:
            sets[names[i]] = [knots[i - 1], knots[i], knots[i + 1]]
    return sets


def fit_var_sets(x, n_terms=5):
    """nudos = cuantiles reales de la variable -> sus sets ocupan SU universo de discurso.
    n_terms=5 para variables físicas (spec); IA1 admite más términos.
    NOTA DE DISEÑO (auditoría + revalidación 6-jul): en variables con MASA PUNTUAL (a_tect
    bimodal, a_fore ~80% ceros) los cuantiles colapsan y salen términos ultra-estrechos. Se probó
    "arreglarlo" deduplicando nudos -> la revalidación walk-forward lo RECHAZÓ: la cola extrema
    de IA3 (57-58% real, el nivel de ALERTA en producción) colapsó, porque esos términos estrechos
    NO son defecto sino DETECTORES CATEGÓRICOS precisos (un término clavado en -1.0 exacto dispara
    exactamente para "sin falla GEM", donde vive el 67% del planeta). Se mantiene la construcción
    original con separación mínima; los términos-sliver intermedios quedan inertes sin dañar."""
    if n_terms == 5:
        qs = np.quantile(x, [0.02, 0.25, 0.50, 0.75, 0.98])
        names = None
    else:
        qs = np.quantile(x, np.linspace(0.02, 0.98, n_terms))
        names = [f"n{i}" for i in range(n_terms)]
    qs = list(qs)
    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = qs[i - 1] + 1e-4
    if n_terms == 5:
        return ruspini_from_knots(qs)
    return ruspini_chain(qs, names)


def memb(x, p):
    x = np.asarray(x, dtype=np.float32)
    if len(p) == 3:
        a, b, c = p
        left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
        right = np.where(c > b, (c - x) / (c - b + 1e-12), 1.0)
        return np.clip(np.minimum(left, right), 0, 1).astype(np.float32)
    a, b, c, d = p
    left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
    right = np.where(d > c, (d - x) / (d - c + 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1).astype(np.float32)


# ============================ normalización adaptativa por clase (vol_params) ===================
def activity_class(rate_bg):
    """clase de actividad de la celda = el 'asset class' de la sismicidad (cuartiles de tasa)."""
    edges = [0.5, 2.0, 8.0]                                  # eventos/año: rala/moderada/activa/muy
    c = np.zeros(len(rate_bg), dtype=int)
    for i, e in enumerate(edges):
        c += (rate_bg > e).astype(int)
    return c


def derive_channels_classed(df, taus=None, fit_mask=None):
    """canales con z crudo y tau POR (clase, canal) — el análogo exacto de vol_params: cada clase
    ocupa su universo. Si taus=None, se ajustan sobre fit_mask (p85 de |z| -> tanh 0.7)."""
    z = {}
    mu30 = df.rate_bg * 30 / 365.25
    z["a_seis"] = ((df.n30 - mu30) / np.sqrt(mu30 + 0.5)).to_numpy()
    exp30 = df.n365 * 30 / 365.0
    z["a_swarm"] = ((df.n30 - exp30) / np.sqrt(exp30 + 0.5)).to_numpy()
    exp7 = df.n30 * 7 / 30.0
    z["a_fore"] = ((df.n7 - exp7) / np.sqrt(exp7 + 0.5)).to_numpy()
    mu90 = df.rate_bg * 90 / 365.25
    z["A_state"] = (df.n90 / np.maximum(mu90, 0.25) - 1.0).to_numpy()
    z["C_state"] = (df.t_last / 365.25 * df.rate_bg - 1.0).to_numpy()
    z["mag_scale"] = (2 * (df.mag_max - 4.5) / (8.0 - 4.5) - 1).to_numpy()   # ya acotada
    z["a_tect"] = (2 * np.log1p(df.slip) / np.log1p(50.0) - 1).to_numpy()    # ya acotada
    cls = activity_class(df.rate_bg.to_numpy())
    if taus is None:
        taus = {}
        for c in ["a_seis", "a_swarm", "a_fore", "A_state", "C_state"]:
            for k in range(4):
                m = (cls == k) & fit_mask
                p85 = np.percentile(np.abs(z[c][m]), 85) if m.sum() > 500 else 1.0
                taus[f"{c}|{k}"] = float(max(p85, 1e-3) / np.arctanh(0.7))
    out = pd.DataFrame(index=df.index)
    for c in CH_RAW:
        if c in ("mag_scale", "a_tect"):
            out[c] = np.clip(z[c], -1, 1)
        else:
            tau_arr = np.ones(len(df))
            for k in range(4):
                tau_arr[cls == k] = taus[f"{c}|{k}"]
            out[c] = np.tanh(z[c] / tau_arr)
    return out.astype(np.float32), taus, cls


# ============================ IA1: núcleo estadístico como score ================================
def ia1_fit(omori, rate, y, n=12):
    """IA1 = mapa (log-omori, log-tasa) -> tasa empírica, como score continuo [0,1] (el score1new
    de nuestro dominio). Ajustado SOLO en fit; IA2 lo refina."""
    lo = np.log1p(omori); lr = np.log1p(rate)
    om_e = [-np.inf] + list(np.quantile(lo, np.linspace(0, 1, n + 1))[1:-1]) + [np.inf]
    rt_e = [-np.inf] + list(np.quantile(lr, np.linspace(0, 1, n + 1))[1:-1]) + [np.inf]
    oi = np.clip(np.searchsorted(om_e, lo, side="right") - 1, 0, n - 1)
    ri = np.clip(np.searchsorted(rt_e, lr, side="right") - 1, 0, n - 1)
    flat = oi * n + ri
    cnt = np.bincount(flat, minlength=n * n)
    ys = np.bincount(flat, weights=y, minlength=n * n)
    tab = np.where(cnt >= 200, ys / np.maximum(cnt, 1), np.nan)
    gm = np.nanmean(tab)
    tab = np.nan_to_num(tab, nan=gm)
    return {"om": om_e, "rt": rt_e, "t": tab.reshape(n, n), "n": n}


def ia1_apply(omori, rate, m):
    lo = np.log1p(omori); lr = np.log1p(rate)
    n = m["n"]
    oi = np.clip(np.searchsorted(m["om"], lo, side="right") - 1, 0, n - 1)
    ri = np.clip(np.searchsorted(m["rt"], lr, side="right") - 1, 0, n - 1)
    return m["t"][oi, ri].astype(np.float32)


# ============================ fuzzy-inverso (reglas) con gate de dos épocas =====================
def extract_rules(MU, terms_by_var, y, fit_m, val_m, retain=0.5, min_gain_group=1.15):
    base_f, base_v = y[fit_m].mean(), y[val_m].mean()
    rules, lift1 = [], {}

    def ev(cols, chans):
        m = MU[cols[0]]
        for c in cols[1:]:
            m = np.minimum(m, MU[c])
        sf = float(m[fit_m].sum())
        if sf < 1000:
            return None
        lf = float((m[fit_m] * y[fit_m]).sum() / sf / (base_f + 1e-12))
        sv = float(m[val_m].sum())
        if sv < 300:
            return None
        lv = float((m[val_m] * y[val_m]).sum() / sv / (base_v + 1e-12))
        exf, exv = lf - 1, lv - 1
        ret = exv / exf if abs(exf) > 1e-6 else 0.0
        return {"cols": cols, "chans": chans, "lift": lf,
                "ok": (exf * exv > 0) and (ret >= retain), "ret": ret}

    for v in CH_ALL:
        for t in terms_by_var[v]:
            r = ev([(v, t)], [v])
            if r is None:
                continue
            lift1[(v, t)] = r["lift"]
            if r["ok"]:
                rules.append(r)
    for va, vb in combinations(CH_ALL, 2):
        for ta in terms_by_var[va]:
            for tb in terms_by_var[vb]:
                r = ev([(va, ta), (vb, tb)], [va, vb])
                if r is None:
                    continue
                marginal = [lift1.get((va, ta), 1.0), lift1.get((vb, tb), 1.0)]
                bm = max(marginal) if r["lift"] > 1 else min(marginal)
                gain = r["lift"] / (bm + 1e-12)
                genuine = gain >= min_gain_group if r["lift"] > 1 else gain <= 1 / min_gain_group
                if r["ok"] and genuine:
                    rules.append(r)
    return rules


# ============================ IA2: fuerzas por término de salida ================================
def rule_output_term(lift, base):
    """mapear el lift de la regla al término de salida graduado (7): la tasa que la regla implica
    (lift×base) posicionada en el rango de tasas -> índice 0..6. Reglas protectoras -> términos
    bajos, reglas fuertes -> altos. (= asignación consecuente en IA2, aquí derivada de datos)."""
    implied = np.clip(lift * base, 0, 1)
    # escala log entre base/5 y base*8 -> 7 niveles
    lo, hi = np.log(base / 5 + 1e-9), np.log(base * 8 + 1e-9)
    t = (np.log(implied + 1e-9) - lo) / (hi - lo + 1e-12)
    return int(np.clip(round(t * (N_OUT - 1)), 0, N_OUT - 1))


def ia2_strengths(MU, rules, base):
    """fuerza por término de salida = max sobre reglas asignadas a ese término (OR de IA2)."""
    n = len(next(iter(MU.values())))
    S = [np.zeros(n, dtype=np.float32) for _ in range(N_OUT)]
    for r in rules:
        m = MU[r["cols"][0]]
        for c in r["cols"][1:]:
            m = np.minimum(m, MU[c])
        k = rule_output_term(r["lift"], base)
        np.maximum(S[k], m, out=S[k])
    return S


# ============================ IA3: entrenar posiciones de los 7 términos ========================
def ia3_train_output(S, y, base, lr=0.25, iters=25):
    """posiciones iniciales log-espaciadas alrededor de la base; IA3 batch mueve cada término por
    delta = error×lr×fuerza (constraints de orden, como el PHP). Universo [0,1] (probabilidad)."""
    init = np.clip(np.exp(np.linspace(np.log(base / 5 + 1e-9), np.log(min(base * 8, 0.9)),
                                       N_OUT)), 1e-4, 0.95)
    cent = init.copy()
    den = np.zeros(len(y), dtype=np.float64)
    for s in S:
        den += s
    fired = den > 1e-6
    for _ in range(iters):
        num = np.zeros(len(y), dtype=np.float64)
        for k, s in enumerate(S):
            num += cent[k] * s
        pred = np.where(fired, num / np.maximum(den, 1e-9), base)
        err = y - pred
        for k, s in enumerate(S):
            sm = s[fired]
            tot = sm.sum()
            if tot < 100:
                continue
            cent[k] += lr * float((err[fired] * sm).sum() / tot)
        cent = np.clip(cent, 1e-4, 0.97)
        cent = np.maximum.accumulate(cent)                   # orden nulo<=...<=extremo (constraint)
    return cent


def ia2_predict(S, cent, base):
    den = np.zeros(len(S[0]), dtype=np.float64)
    num = np.zeros(len(S[0]), dtype=np.float64)
    for k, s in enumerate(S):
        num += cent[k] * s
        den += s
    return np.where(den > 1e-6, num / np.maximum(den, 1e-9), base).astype(np.float32)


# ============================ métricas ==========================================================
def metrics(p, y, p0):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    brier = float(np.mean((p - y) ** 2))
    ig = float(np.mean(y * np.log(p / p0) + (1 - y) * np.log((1 - p) / (1 - p0))))
    top = p >= np.quantile(p, 0.9)
    lift = float(y[top].mean() / y.mean()) if top.sum() else np.nan
    return brier, ig, lift


def reliability(p, y, label):
    print(f"  fiabilidad {label} (LA DIAGONAL = el contrato):")
    for a, b in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1.01)]:
        m = (p >= a) & (p < b)
        if m.sum() > 300:
            print(f"    pred[{a:.2f},{b:.2f}): n={int(m.sum()):7d}  pred={p[m].mean()*100:5.1f}%  "
                  f"real={y[m].mean()*100:5.1f}%  desvío={abs(p[m].mean()-y[m].mean())*100:+.1f}pt")


# ============================ walk-forward ======================================================
def run_fold(df, origin, years_test=4.0):
    t_or = pd.Timestamp(origin)
    tr = (df.date < t_or).to_numpy()
    te = ((df.date >= t_or) & (df.date < t_or + pd.Timedelta(days=365.25 * years_test))).to_numpy()
    y = df.y.to_numpy(float)
    dates_tr = df.date[tr]
    mid = dates_tr.quantile(0.55)
    fit_m = tr & (df.date < mid).to_numpy()
    val_m = tr & (df.date >= mid).to_numpy()

    # normalización por clase ajustada en fit (vol_params)
    chan, taus, cls = derive_channels_classed(df, taus=None, fit_mask=fit_m)
    # IA1 en fit; entra como variable
    ia1 = ia1_fit(df.omori.to_numpy()[fit_m], df.rate_bg.to_numpy()[fit_m], y[fit_m])
    s_ia1 = ia1_apply(df.omori.to_numpy(), df.rate_bg.to_numpy(), ia1)
    chan_all = chan.copy()
    chan_all["ia1"] = s_ia1
    # sets POR VARIABLE (nudos en cuantiles del fit — cada variable su universo)
    var_sets, terms_by_var, MU = {}, {}, {}
    for v in CH_ALL:
        x = chan_all[v].to_numpy()
        var_sets[v] = fit_var_sets(x[fit_m], n_terms=11 if v == "ia1" else 5)
        terms_by_var[v] = list(var_sets[v])
        for t, prm in var_sets[v].items():
            MU[(v, t)] = memb(x, prm)
    rules = extract_rules(MU, terms_by_var, y, fit_m, val_m)
    base = y[tr].mean()
    S = ia2_strengths(MU, rules, base)
    cent = ia3_train_output([s[tr] for s in S], y[tr], base)
    p_te = ia2_predict([s[te] for s in S], cent, base)
    b, ig, lift = metrics(p_te, y[te], base)
    return {"origin": origin, "rules": len(rules),
            "groups": sum(1 for r in rules if len(r["cols"]) == 2),
            "brier": b, "ig": ig, "lift": lift,
            "_p": p_te, "_y": y[te], "_cent": cent, "_taus": taus, "_ia1": ia1,
            "_var_sets": var_sets, "_rules": rules, "_base": base}


def main():
    df = pd.read_csv(os.path.join(OUTG, "cellmonths.csv.gz"), parse_dates=["date"])
    print(f"dataset: {len(df)} celda-mes\n")
    print("===== IA1→IA2→IA3 FIEL · WALK-FORWARD 5 ÉPOCAS =====")
    print("(comparar con v2-tabla3D: lift 4.37/4.41/4.45/4.33/4.56; B3: 4.19/4.20/4.25/4.12/4.32)\n")
    results, final = [], None
    for origin in ["2006-01-01", "2010-01-01", "2014-01-01", "2018-01-01", "2022-01-01"]:
        r = run_fold(df, origin)
        results.append({k: v for k, v in r.items() if not k.startswith("_")})
        final = r
        print(f"origen {origin}: {r['rules']} reglas ({r['groups']} grupos) | "
              f"Brier={r['brier']:.5f}  IG={r['ig']:+.4f}  lift@top10%={r['lift']:.2f}")
    print()
    reliability(final["_p"], final["_y"], f"época {final['origin']} (IA3 como calibrador)")
    print("\n  centroides IA3 de los 7 términos de salida (entrenados, no fijados):")
    print("  " + "  ".join(f"{n}={c:.4f}" for n, c in zip(OUT_NAMES, final["_cent"])))

    # persistir el motor fiel (época 2022)
    eng = {
        "version": "v3-ia2-fiel",
        "out_names": OUT_NAMES,
        "out_centroids": [float(c) for c in final["_cent"]],
        "taus": final["_taus"],
        "ia1": {"om": list(map(float, final["_ia1"]["om"][1:-1])),
                "rt": list(map(float, final["_ia1"]["rt"][1:-1])),
                "t": final["_ia1"]["t"].tolist(), "n": final["_ia1"]["n"]},
        "var_sets": {v: {t: [float(x) for x in prm] for t, prm in s.items()}
                     for v, s in final["_var_sets"].items()},
        "rules": [{"cols": [[c, t] for c, t in r["cols"]], "lift": round(r["lift"], 3)}
                  for r in final["_rules"]],
        "base": float(final["_base"]),
        "walkforward": results,
    }
    json.dump(eng, open(os.path.join(OUTG, "global_engine_v3.json"), "w"))
    print(f"\n-> out/global/global_engine_v3.json")


if __name__ == "__main__":
    main()


# ============================ aplicador del motor fiel persistido (para emit/globo) =============
def apply_v3(live_df):
    """P del motor fiel IA1→IA2→IA3 sobre el estado vivo (celdas ahora). Usa el engine persistido."""
    eng = json.load(open(os.path.join(OUTG, "global_engine_v3.json")))
    taus = eng["taus"]
    chan, _, _ = derive_channels_classed(live_df, taus=taus)
    ia1 = {"om": [-np.inf] + eng["ia1"]["om"] + [np.inf],
           "rt": [-np.inf] + eng["ia1"]["rt"] + [np.inf],
           "t": np.array(eng["ia1"]["t"]), "n": eng["ia1"]["n"]}
    chan = chan.copy()
    chan["ia1"] = ia1_apply(live_df.omori.to_numpy(), live_df.rate_bg.to_numpy(), ia1)
    MU = {}
    for v, sets in eng["var_sets"].items():
        x = chan[v].to_numpy()
        for t, prm in sets.items():
            MU[(v, t)] = memb(x, prm)
    rules = [{"cols": [tuple(ct) for ct in r["cols"]], "lift": r["lift"]} for r in eng["rules"]]
    base = eng["base"]
    n = len(live_df)
    S = [np.zeros(n, dtype=np.float32) for _ in range(N_OUT)]
    for r in rules:
        m = MU[r["cols"][0]]
        for c in r["cols"][1:]:
            m = np.minimum(m, MU[c])
        k = rule_output_term(r["lift"], base)
        np.maximum(S[k], m, out=S[k])
    cent = np.array(eng["out_centroids"])
    return ia2_predict(S, cent, base)
