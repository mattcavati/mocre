"""MOCRE-GLOBAL v2 — el motor de Cícero COMPLETO a escala planetaria. Sin atajos estadísticos:

1. UNIVERSOS DE DISCURSO: cada canal en [-1,1], partición Ruspini de 5 (centros triangulares,
   extremos trapezoidales) — la spec literal.
2. FUZZY-INVERSO con GRUPOS (aridad 1 y 2): reglas extraídas de celda-meses solapados. El gate
   entre dos épocas reduce selección oportunista, pero las filas no son episodios independientes
   (halos espaciales y horizontes temporales se solapan), por lo que no valida por sí solo reglas.
3. INFLUENCIA POR VARIABLE (mu_ev) MEDIDA: retención de lift por canal entre épocas — "definir
   cual la influencia de cada variable separadamente", derivada de datos, inspeccionable.
4. INFERENCIA MAMDANI REAL (IA2): fuzzificación → reglas de producción (t-norma min × mu_ev ×
   peso) → implicación por recorte → agregación max → defuzzificación MOM sobre rampas Ruspini
   {no,sí} en [-1,1] → VALOR de ocurrencia. El valor se ancla a probabilidad con calibración
   monótona (frecuencias reales) — el motor produce el valor, la calibración lo aterriza.
5. IA3 BATCH para la salida MAGNITUD (universo Richter, particiones entrenadas por error×fuerza
   contra la magnitud real del próximo evento) — el mecanismo de ajusteprevision_new.php en modo
   batch (mismo principio delta = error×lr×fuerza, agregado por término).
6. VALIDACIÓN WALK-FORWARD multi-época: orígenes 2006/2010/2014/2018/2022 — entrenar SOLO con el
   pasado de cada origen, predecir los 4 años siguientes. "Predecir como si estuviéramos en el
   pasado", literal.
7. ESCALERA DE BENCHMARKS (los mecanismos núcleo de los sistemas operacionales existentes):
   B0 climatología · B1 sismicidad suavizada (núcleo GEAR1) · B2 omori (núcleo ETAS/OEF) ·
   B3 omori×tasa · MOCRE = Mamdani×omori (λ=μ·G calibrado). Métrica CSEP: ganancia de
   información por celda-mes + Brier + lift del decil superior.
"""
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import membership_centroid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTG = os.path.join(ROOT, "out", "global")

SETS5 = {
    "muy_bajo": [-1.0, -1.0, -0.70, -0.35],
    "bajo": [-0.70, -0.35, 0.0],
    "normal": [-0.35, 0.0, 0.35],
    "alto": [0.0, 0.35, 0.70],
    "muy_alto": [0.35, 0.70, 1.0, 1.0],
}
TERMS = list(SETS5)
CH = ["a_seis", "a_swarm", "a_fore", "A_state", "C_state", "mag_scale", "a_tect"]


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


def derive_channels(df):
    out = pd.DataFrame(index=df.index)
    mu30 = df.rate_bg * 30 / 365.25
    out["a_seis"] = np.tanh(((df.n30 - mu30) / np.sqrt(mu30 + 0.5)) / 4.0)
    exp30 = df.n365 * 30 / 365.0
    out["a_swarm"] = np.tanh(((df.n30 - exp30) / np.sqrt(exp30 + 0.5)) / 4.0)
    exp7 = df.n30 * 7 / 30.0
    out["a_fore"] = np.tanh(((df.n7 - exp7) / np.sqrt(exp7 + 0.5)) / 4.0)
    mu90 = df.rate_bg * 90 / 365.25
    out["A_state"] = np.tanh((df.n90 / np.maximum(mu90, 0.25) - 1.0) / 2.0)
    x = df.t_last / 365.25 * df.rate_bg
    out["C_state"] = np.tanh((x - 1.0) / 2.0)
    out["mag_scale"] = np.clip(2 * (df.mag_max - 4.5) / (8.0 - 4.5) - 1, -1, 1)
    # carga tectónica estática (GEM slip rate mm/año, escala log; 0 si no hay falla GEM)
    out["a_tect"] = np.clip(2 * np.log1p(df.slip) / np.log1p(50.0) - 1, -1, 1)
    return out.astype(np.float32)


# ============================ FUZZY-INVERSO con gate de generalización ==========================
def _lift(mu_col, y, base):
    supp = float(mu_col.sum())
    if supp < 1000:
        return None, supp
    return float((mu_col * y).sum() / supp / (base + 1e-12)), supp


def extract_rules(MU, y, fit_mask, val_mask, min_gain_group=1.15, retain=0.5):
    """Reglas aridad-1 y aridad-2 con doble época: admitida solo si el exceso de lift retiene
    >= retain en la época de validación. Devuelve (rules, mu_ev, tabla)."""
    base_f = y[fit_mask].mean()
    base_v = y[val_mask].mean()
    rules, rows = [], []
    retention_by_ch = {c: [] for c in CH}

    def eval_combo(cols, name, chans):
        m = MU[cols[0]]
        for c in cols[1:]:
            m = np.minimum(m, MU[c])
        lf, sf = _lift(m[fit_mask], y[fit_mask], base_f)
        if lf is None:
            return None
        lv, sv = _lift(m[val_mask], y[val_mask], base_v)
        if lv is None:
            return None
        exc_f, exc_v = lf - 1.0, lv - 1.0
        ret = (exc_v / exc_f) if abs(exc_f) > 1e-6 else 0.0
        ok = (exc_f * exc_v > 0) and (ret >= retain)         # misma dirección y retiene
        return {"name": name, "chans": chans, "cols": cols, "lift_fit": lf, "lift_val": lv,
                "retention": ret, "support": sf, "ok": ok}

    # aridad-1
    lift1 = {}
    for c in CH:
        for t in TERMS:
            r = eval_combo([(c, t)], f"{c}={t}", [c])
            if r is None:
                continue
            lift1[(c, t)] = r["lift_fit"]
            retention_by_ch[c].append(max(0.0, min(1.0, r["retention"])))
            rows.append(r)
            if r["ok"]:
                rules.append(r)
    # aridad-2 (grupos): solo si el lift conjunto SUPERA al mejor marginal en ambas épocas
    for ca, cb in combinations(CH, 2):
        for ta in TERMS:
            for tb in TERMS:
                r = eval_combo([(ca, ta), (cb, tb)], f"{ca}={ta} & {cb}={tb}", [ca, cb])
                if r is None:
                    continue
                marg = [lift1.get((ca, ta), 1.0), lift1.get((cb, tb), 1.0)]
                best_marg = max(marg) if r["lift_fit"] > 1 else min(marg)
                gain = r["lift_fit"] / (best_marg + 1e-12)
                genuine = (gain >= min_gain_group) if r["lift_fit"] > 1 else (gain <= 1 / min_gain_group)
                r["gain_vs_parts"] = gain
                rows.append(r)
                if r["ok"] and genuine:
                    rules.append(r)
    # mu_ev por canal = retención media de sus reglas (influencia MEDIDA)
    mu_ev = {c: float(np.clip(np.mean(v) if v else 0.3, 0.2, 1.0)) for c, v in retention_by_ch.items()}
    return rules, mu_ev, pd.DataFrame(rows)


# ============================ INFERENCIA MAMDANI (IA2) ==========================================
def mamdani_occurrence(MU, rules, mu_ev, base):
    """Vectorizado sobre todas las filas: para cada regla, fuerza = min(pertenencias) × mu_ev(min
    de sus canales) × peso; consecuente 'sí' si lift>1 (peso=exceso normalizado), 'no' si lift<1.
    Agregación max por lado; MOM sobre rampas Ruspini => valor = ws si ws>wn, -wn si wn>ws.
    Devuelve (valor_mom, score_fino=ws-wn)."""
    n = len(next(iter(MU.values())))
    ws = np.zeros(n, dtype=np.float32)
    wn = np.zeros(n, dtype=np.float32)
    max_exc = max((abs(r["lift_fit"] - 1.0) for r in rules), default=1.0)
    for r in rules:
        m = MU[r["cols"][0]]
        for c in r["cols"][1:]:
            m = np.minimum(m, MU[c])
        ev = min(mu_ev[c] for c in r["chans"])
        w = float(np.clip(abs(r["lift_fit"] - 1.0) / max_exc, 0, 1))
        f = m * (ev * w)
        if r["lift_fit"] > 1.0:
            np.maximum(ws, f, out=ws)
        else:
            np.maximum(wn, f, out=wn)
    mom = np.where(ws > wn, ws, -wn).astype(np.float32)
    mom[np.isclose(ws, wn)] = 0.0
    return mom, (ws - wn)


# ============================ IA3 batch (salida magnitud) =======================================
def ia3_batch(strength_cols, y_real, init_sets, lr=0.3, iters=12):
    """IA3 en modo batch: mismo principio delta = error × lr × fuerza, agregado por término.
    strength_cols: dict término -> fuerza por fila. y_real: objetivo real (NaN se ignora)."""
    sets = {k: list(v) for k, v in init_sets.items()}
    ok = ~np.isnan(y_real)
    for _ in range(iters):
        num = np.zeros(len(y_real)); den = np.zeros(len(y_real))
        for t, s in strength_cols.items():
            c = membership_centroid(sets[t])
            num += c * s; den += s
        pred = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        err = y_real - pred
        m = ok & ~np.isnan(pred)
        for t, s in strength_cols.items():
            sm = s[m]
            tot = sm.sum()
            if tot < 1:
                continue
            d = float(lr * (err[m] * sm).sum() / tot)
            for j in range(3):
                sets[t][j] += d
            if sets[t][0] >= sets[t][1]:
                sets[t][0] = sets[t][1] - 1e-3
            if sets[t][2] <= sets[t][1]:
                sets[t][2] = sets[t][1] + 1e-3
    return sets


# ============================ calibración monótona por bins =====================================
def calib_2d_fit(omori, score, y, n_om=8, n_sc=8):
    lo = np.log1p(omori)
    pos = lo[lo > 0]
    om_edges = [0.0] + list(np.quantile(pos, np.linspace(0, 1, n_om)[1:-1])) + [np.inf]
    sc_edges = [-np.inf] + list(np.quantile(score, np.linspace(0, 1, n_sc + 1))[1:-1]) + [np.inf]
    oi = np.clip(np.searchsorted(om_edges, lo, side="right") - 1, 0, len(om_edges) - 2)
    si = np.clip(np.searchsorted(sc_edges, score, side="right") - 1, 0, n_sc - 1)
    table = np.zeros((len(om_edges) - 1, n_sc))
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            m = (oi == i) & (si == j)
            table[i, j] = y[m].mean() if m.sum() >= 300 else np.nan
        row = table[i]
        rm = np.nanmean(row)
        row[np.isnan(row)] = rm if not np.isnan(rm) else y.mean()
        table[i] = np.maximum.accumulate(row)               # monótono en el eje fuzzy
    return {"om": [float(v) if np.isfinite(v) else 1e18 for v in om_edges],
            "sc": [float(v) if np.isfinite(v) else (1e18 if v > 0 else -1e18) for v in sc_edges],
            "t": table.tolist()}


def calib_2d_apply(omori, score, cal):
    lo = np.log1p(omori)
    t = np.array(cal["t"])
    oi = np.clip(np.searchsorted(np.array(cal["om"]), lo, side="right") - 1, 0, t.shape[0] - 1)
    si = np.clip(np.searchsorted(np.array(cal["sc"]), score, side="right") - 1, 0, t.shape[1] - 1)
    return t[oi, si]


def calib_3d_fit(omori, rate, score, y, n_om=7, n_rt=5, n_sc=4):
    """λ=μ·G completo: tasa de fondo (μ largo plazo) × omori (persistencia) × score fuzzy (G).
    Superconjunto de información de B3 -> no puede ser peor que B3 salvo ruido de bins; si el
    fuzzy aporta sobre el núcleo estadístico, aquí se ve."""
    lo = np.log1p(omori); lr = np.log1p(rate)
    om_e = [0.0] + list(np.quantile(lo[lo > 0], np.linspace(0, 1, n_om)[1:-1])) + [np.inf]
    rt_e = [-np.inf] + list(np.quantile(lr, np.linspace(0, 1, n_rt + 1))[1:-1]) + [np.inf]
    sc_e = [-np.inf] + list(np.quantile(score, np.linspace(0, 1, n_sc + 1))[1:-1]) + [np.inf]
    oi = np.clip(np.searchsorted(om_e, lo, side="right") - 1, 0, n_om - 1)
    ri = np.clip(np.searchsorted(rt_e, lr, side="right") - 1, 0, n_rt - 1)
    si = np.clip(np.searchsorted(sc_e, score, side="right") - 1, 0, n_sc - 1)
    flat = (oi * n_rt + ri) * n_sc + si
    tab = np.full(n_om * n_rt * n_sc, np.nan)
    cnt = np.bincount(flat, minlength=len(tab))
    ysum = np.bincount(flat, weights=y, minlength=len(tab))
    ok = cnt >= 250
    tab[ok] = ysum[ok] / cnt[ok]
    # relleno jerárquico: celda vacía <- media de su (omori,rate) marginal <- global
    t3 = tab.reshape(n_om, n_rt, n_sc)
    for i in range(n_om):
        for j in range(n_rt):
            row = t3[i, j]
            rm = np.nanmean(row)
            row[np.isnan(row)] = rm if not np.isnan(rm) else np.nan
            if np.isnan(row).all():
                t3[i, j] = np.nanmean(t3[i]) if not np.isnan(np.nanmean(t3[i])) else y.mean()
            else:
                t3[i, j] = np.maximum.accumulate(row)       # monótono en el eje fuzzy
    t3[np.isnan(t3)] = y.mean()
    def _san(edges):
        return [(-1e18 if not np.isfinite(v) and v < 0 else 1e18 if not np.isfinite(v) else float(v))
                for v in edges]
    return {"om": _san(om_e), "rt": _san(rt_e), "sc": _san(sc_e), "t": t3.tolist()}


def calib_3d_apply(omori, rate, score, cal):
    lo = np.log1p(omori); lr = np.log1p(rate)
    t = np.array(cal["t"])
    oi = np.clip(np.searchsorted(np.array(cal["om"]), lo, side="right") - 1, 0, t.shape[0] - 1)
    ri = np.clip(np.searchsorted(np.array(cal["rt"]), lr, side="right") - 1, 0, t.shape[1] - 1)
    si = np.clip(np.searchsorted(np.array(cal["sc"]), score, side="right") - 1, 0, t.shape[2] - 1)
    return t[oi, ri, si]


def calib_1d_fit(x, y, n=10):
    edges = [-np.inf] + list(np.quantile(x, np.linspace(0, 1, n + 1))[1:-1]) + [np.inf]
    xi = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)
    tab = np.array([y[xi == i].mean() if (xi == i).sum() >= 300 else np.nan for i in range(n)])
    gm = np.nanmean(tab)
    tab[np.isnan(tab)] = gm
    tab = np.maximum.accumulate(tab)
    return {"e": [float(v) if np.isfinite(v) else (1e18 if v > 0 else -1e18) for v in edges],
            "t": tab.tolist()}


def calib_1d_apply(x, cal):
    t = np.array(cal["t"])
    xi = np.clip(np.searchsorted(np.array(cal["e"]), x, side="right") - 1, 0, len(t) - 1)
    return t[xi]


# ============================ métricas =========================================================
def metrics(p, y, p0):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    brier = float(np.mean((p - y) ** 2))
    ig = float(np.mean(y * np.log(p / p0) + (1 - y) * np.log((1 - p) / (1 - p0))))
    thr = np.quantile(p, 0.9)
    top = p >= thr
    lift = float(y[top].mean() / y.mean()) if top.sum() else np.nan
    return brier, ig, lift


# ============================ pipeline por origen (walk-forward) ================================
def run_fold(df, chan, MU, origin, years_test=4.0):
    t_or = pd.Timestamp(origin)
    tr = (df.date < t_or).to_numpy()
    te = ((df.date >= t_or) & (df.date < t_or + pd.Timedelta(days=365.25 * years_test))).to_numpy()
    if te.sum() < 10000:
        return None
    y = df.y.to_numpy(float)
    # dos épocas DENTRO de train para el gate de generalización
    dates_tr = df.date[tr]
    mid = dates_tr.quantile(0.55)
    fit_m = tr & (df.date < mid).to_numpy()
    val_m = tr & (df.date >= mid).to_numpy()

    rules, mu_ev, _tab = extract_rules(MU, y, fit_m, val_m)
    mom, score = mamdani_occurrence(MU, rules, mu_ev, y[tr].mean())
    om = df.omori.to_numpy()

    p0 = y[tr].mean()
    out = {"origin": origin, "n_test": int(te.sum()), "base_test": float(y[te].mean()),
           "n_rules": len(rules), "n_groups": sum(1 for r in rules if len(r["cols"]) == 2)}
    # escalera
    ladder = {}
    cal = calib_1d_fit(df.rate_bg.to_numpy()[tr], y[tr])
    ladder["B1_suavizada"] = calib_1d_apply(df.rate_bg.to_numpy()[te], cal)
    cal = calib_1d_fit(om[tr], y[tr])
    ladder["B2_omori"] = calib_1d_apply(om[te], cal)
    rate = df.rate_bg.to_numpy()
    cal = calib_2d_fit(om[tr], rate[tr], y[tr])
    ladder["B3_omori_x_tasa"] = calib_2d_apply(om[te], rate[te], cal)
    cal_m = calib_3d_fit(om[tr], rate[tr], score[tr], y[tr])
    ladder["MOCRE_lam_muG"] = calib_3d_apply(om[te], rate[te], score[te], cal_m)
    # CALIBRADOR CONTINUO (7-jul, Matt: "el máximo siempre da 43.9%"): la tabla por bins tiene
    # plateau en el bin superior (toda celda caliente recibe la media del bin). La logística
    # continua sobre las MISMAS señales gradúa la cima sin plateau. Indicador has_omori maneja la
    # masa puntual en 0 (la lección del artefacto logístico del 5-jul: era degeneración de masa
    # puntual + regularización, no la logística en sí).
    from sklearn.linear_model import LogisticRegression
    def feats_fuzzy(m):
        return np.column_stack([np.log1p(om[m]), (om[m] > 0).astype(float),
                                np.log1p(rate[m]), score[m]])
    def feats_base(m):
        return np.column_stack([np.log1p(om[m]), (om[m] > 0).astype(float), np.log1p(rate[m])])
    lr_b = LogisticRegression(C=1e4, max_iter=3000).fit(feats_base(tr), y[tr])
    lr_c = LogisticRegression(C=1e4, max_iter=3000).fit(feats_fuzzy(tr), y[tr])
    ladder["B3_continuo"] = lr_b.predict_proba(feats_base(te))[:, 1]
    ladder["MOCRE_continuo"] = lr_c.predict_proba(feats_fuzzy(te))[:, 1]

    for name, p in ladder.items():
        b, ig, lift = metrics(p, y[te], p0)
        out[name] = {"brier": round(b, 5), "info_gain": round(ig, 4), "lift_top10": round(lift, 2)}
    b0 = metrics(np.full(te.sum(), p0), y[te], p0)
    out["B0_climatologia"] = {"brier": round(b0[0], 5), "info_gain": 0.0, "lift_top10": 1.0}
    out["fuzzy_increment"] = {
        "brier_delta_base_minus_fuzzy": float(
            np.mean((ladder["B3_continuo"] - y[te]) ** 2) -
            np.mean((ladder["MOCRE_continuo"] - y[te]) ** 2)),
        "minimum_material_delta": 1e-4,
    }
    extras = {"lr_coef": lr_c.coef_[0].tolist(), "lr_intercept": float(lr_c.intercept_[0]),
              "base_coef": lr_b.coef_[0].tolist(), "base_intercept": float(lr_b.intercept_[0]),
              "p_cont_te": ladder["MOCRE_continuo"], "y_te": y[te]}
    return out, rules, mu_ev, cal_m, mom, score, extras


def main():
    df = pd.read_csv(os.path.join(OUTG, "cellmonths.csv.gz"), parse_dates=["date"])
    print(f"dataset: {len(df)} celda-mes | targets: y={df.y.mean()*100:.1f}%, "
          f"mag_next={df.mag_next.notna().mean()*100:.0f}%, slip>0={(df.slip > 0).mean()*100:.0f}%")
    chan = derive_channels(df)
    MU = {}
    for c in CH:
        x = chan[c].to_numpy()
        for t in TERMS:
            MU[(c, t)] = memb(x, SETS5[t])
    print("pertenencias precomputadas (35 columnas)")

    # ===== WALK-FORWARD =====
    print("\n===== VALIDACIÓN WALK-FORWARD (entrenar en el pasado, predecir el 'futuro') =====")
    results = []
    final = None
    for origin in ["2006-01-01", "2010-01-01", "2014-01-01", "2018-01-01", "2022-01-01"]:
        r = run_fold(df, chan, MU, origin)
        if r is None:
            continue
        out = r[0]
        results.append(out)
        final = r                                            # el último origen = motor más entrenado
        print(f"\norigen {origin} (test 4 años, {out['n_test']} filas, base {out['base_test']*100:.1f}%, "
              f"{out['n_rules']} reglas de las cuales {out['n_groups']} grupos):")
        for k in ["B1_suavizada", "B2_omori", "B3_omori_x_tasa", "B3_continuo",
                  "MOCRE_lam_muG", "MOCRE_continuo"]:
            m = out[k]
            print(f"  {k:<16} Brier={m['brier']:.5f}  IG={m['info_gain']:+.4f}  "
                  f"lift@top10%={m['lift_top10']:.2f}")

    # ===== IA3 magnitud (entrenado <2018, evaluado >=2018) =====
    print("\n===== SALIDA MAGNITUD (IA3 batch, universo Richter) =====")
    tr = (df.date < "2018-01-01").to_numpy()
    te = ~tr
    y_mag = df.mag_next.to_numpy(float)
    init = {"moderada": [4.5, 5.0, 5.6], "fuerte": [5.2, 5.9, 6.6],
            "muy_fuerte": [6.2, 6.9, 7.6], "extrema": [7.2, 8.0, 8.8]}
    # reglas magnitud: mag_scale y a_tect (la tectónica manda; medido que el estado no la resuelve)
    strengths = {}
    pairs = [("mag_scale", "muy_bajo", "moderada"), ("mag_scale", "bajo", "moderada"),
             ("mag_scale", "normal", "fuerte"), ("mag_scale", "alto", "muy_fuerte"),
             ("mag_scale", "muy_alto", "extrema"), ("a_tect", "muy_alto", "muy_fuerte")]
    for c, t, term in pairs:
        strengths.setdefault(term, np.zeros(len(df), dtype=np.float32))
        np.maximum(strengths[term], MU[(c, t)], out=strengths[term])
    tr_str = {k: v[tr] for k, v in strengths.items()}
    trained_mag = ia3_batch(tr_str, y_mag[tr], init)
    te_str = {k: v[te] for k, v in strengths.items()}
    num = np.zeros(te.sum()); den = np.zeros(te.sum())
    for t, s in te_str.items():
        cc = membership_centroid(trained_mag[t])
        num += cc * s; den += s
    pred = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    okm = ~np.isnan(y_mag[te]) & ~np.isnan(pred)
    mae_engine = float(np.mean(np.abs(pred[okm] - y_mag[te][okm])))
    mae_base = float(np.mean(np.abs(np.nanmean(y_mag[tr]) - y_mag[te][okm])))
    print(f"MAE magnitud holdout: Mamdani-IA3={mae_engine:.3f}  vs media-global={mae_base:.3f}")
    for t, v in trained_mag.items():
        print(f"  {t:<11} {v} centroide={membership_centroid(v):.2f}")

    # ===== fiabilidad del calibrador CONTINUO en la última época (incl. bins ALTOS, el plateau) ==
    _, _, _, _, _, _, extras = final
    pc, yc = extras["p_cont_te"], extras["y_te"]
    print("\n=== FIABILIDAD del calibrador CONTINUO (época 2022, la cima ya graduada) ===")
    for a, b in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .35), (.35, .5), (.5, 1.01)]:
        m = (pc >= a) & (pc < b)
        if m.sum() > 200:
            print(f"  P[{a:.2f},{b:.2f}): n={int(m.sum()):7d}  pred={pc[m].mean()*100:5.1f}%  "
                  f"real={yc[m].mean()*100:5.1f}%")
    print(f"  P máx continuo: {pc.max()*100:.1f}%  |  valores distintos en top-1%: "
          f"{len(np.unique(np.round(pc[pc >= np.quantile(pc, 0.99)], 4)))}")

    # Persistir el motor final. El fold 2022 es evaluación exploratoria, no confirmación prospectiva.
    out_fold, rules, mu_ev, cal_m, mom, score, extras = final
    fuzzy_deltas = [r["fuzzy_increment"]["brier_delta_base_minus_fuzzy"] for r in results]
    fuzzy_selected = bool(np.mean(fuzzy_deltas) >= 1e-4 and fuzzy_deltas[-1] >= 1e-4)
    primary_logistic = ({"coef": extras["lr_coef"], "intercept": extras["lr_intercept"],
                         "features": ["log1p_omori", "has_omori", "log1p_rate", "score"]}
                        if fuzzy_selected else
                        {"coef": extras["base_coef"], "intercept": extras["base_intercept"],
                         "features": ["log1p_omori", "has_omori", "log1p_rate"]})
    eng = {
        "version": "v2-mamdani",
        "channels": CH, "sets5": SETS5,
        "mu_ev": mu_ev,
        "rules": [{"name": r["name"], "cols": [[c, t] for c, t in r["cols"]],
                   "chans": r["chans"], "lift": round(r["lift_fit"], 3),
                   "retention": round(r["retention"], 3)} for r in rules],
        "calib": cal_m,
        "deployed_model": "omori_rate_plus_fuzzy" if fuzzy_selected else "omori_rate_baseline",
        "fuzzy_increment_selected": fuzzy_selected,
        "fuzzy_increment_brier_deltas": fuzzy_deltas,
        "selection_threshold_brier": 1e-4,
        "logistic": primary_logistic,
        "shadow_fuzzy_logistic": {"coef": extras["lr_coef"], "intercept": extras["lr_intercept"],
                                  "features": ["log1p_omori", "has_omori", "log1p_rate", "score"]},
        "mag_sets": trained_mag,
        "mag_skill_validated": bool(mae_engine < mae_base),
        "mag_holdout": {"mae_ia3": mae_engine, "mae_constant_baseline": mae_base},
        "walkforward": results,
        "target": "M>=5.0 en 30 dias, vecindad ~150km",
    }
    json.dump(eng, open(os.path.join(OUTG, "global_engine_v2.json"), "w"))
    print(f"\nmu_ev (influencia por canal, MEDIDA como retención de lift entre épocas):")
    for c, v in sorted(mu_ev.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<10} {v:.2f}")
    print(f"\n-> out/global/global_engine_v2.json ({len(rules)} reglas)")


if __name__ == "__main__":
    main()
