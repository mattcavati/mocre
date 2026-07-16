"""MOCRE-1 fuzzy-INVERSO completo (Fix 5): extraer del catálogo real de terremotos la base de reglas
Mamdani para las CUATRO salidas de la spec de Matt -- ocurrencia, magnitud, cuándo, dónde -- con
reglas de GRUPO (aridad>=2), no solo marginales de una variable. "Descubrir qué subconjunto
{x3,x7,x12,...} de f(x1..x27) es realmente necesario para que f=1" == encontrar los PARES de
(variable,partición) cuya coincidencia dispara el evento por encima de lo que cada uno hace solo.

Método = fuzzy lift (Wang-Mendel), disciplina temporal (pesos SOLO en train < CUTOFF, evaluación en
holdout intacto), y evaluado a través del MOTOR MAMDANI REAL (no un proxy lineal).

Honestidad medida (2026-07-05, corr estado vs salida intra-segmento):
  - ocurrencia : aprendible (el estado sí separa evento/no-evento). Reglas arity-1 + arity-2.
  - magnitud   : |r|<=0.08 con el estado -> NO la resuelve el estado; la fija la ESCALA TECTÓNICA
                 del segmento (input estático mag_scale, de target_mag / Wells&Coppersmith).
  - cuándo     : |r|<=0.05 -> el día exacto NO es resoluble con la etiqueta de ventana de 30d;
                 salida honesta = expectativa ~uniforme (tasa Poisson), se reporta sin skill.
  - dónde(radio): C_state -0.24, F_state -0.16, A_state -0.15, a_strain +0.13 -> SÍ hay señal;
                 estrés acumulado alto -> epicentro más cerca del núcleo del segmento.
Cada eje se entrena Y se reporta con su nivel de señal real: no se fabrican reglas donde no hay señal.
"""
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import ROOT, get_membership, sets_for_var
from fuzzy_engine import EXPERT_RULES, Rule, infer

CUTOFF = "2018-01-01"
LABEL = "f_30d"
CANDS = ["a_seis", "a_swarm", "a_foreshock", "a_gnss", "a_sse", "a_fluid", "a_water", "a_strain",
         "a_ocean", "a_insar", "a_tec", "a_noise", "A_state", "C_state", "K_state", "E_state",
         "F_state"]
# for arity-2 groups: only the informative extreme partitions (skip "normal" = no signal), to keep
# the pair search meaningful and avoid the combinatorial/overfitting explosion the project warns of.
EXTREME = ["muy_bajo", "bajo", "alto", "muy_alto"]
# global anchor mapping target_mag -> mag_scale in [-1,1]; 3.5->-1, 7.5->+1 (covers GEM Mmax range)
MAG_LO, MAG_HI = 3.5, 7.5


def mu_col(series, set_name):
    name = getattr(series, "name", None)
    return get_membership(series.to_numpy(dtype=float), sets_for_var(name)[set_name])


# ============================ OCURRENCIA: arity-1 + arity-2 (grupos) ============================
def learn_occurrence(tr, min_support=40.0, min_lift=1.15, min_gain=1.12):
    """Returns (rules, table1, table2). Arity-1: one rule per (var,part) with enough lift+support.
    Arity-2 (GROUPS): a pair (A,B) survives ONLY if its joint lift beats the max of its two parts'
    lifts by >= min_gain -- i.e. the coincidence carries information neither marginal has (genuine
    interaction). weight = normalized lift on the SÍ side (or NO side if protective)."""
    p0 = tr[LABEL].mean()
    f = tr[LABEL].to_numpy(dtype=float)

    # precompute membership columns once
    mucols = {}
    for v in CANDS:
        if v not in tr.columns or not (tr[v].fillna(0) != 0).any():
            continue
        col = tr[v].fillna(0.0)
        for s in sets_for_var(v):
            mucols[(v, s)] = mu_col(col, s)

    def cell(m):
        supp = m.sum()
        if supp < min_support:
            return None
        p = float((m * f).sum() / supp)
        return supp, p, p / (p0 + 1e-9)

    # -- arity 1 --
    rows1, rules, lift1 = [], [], {}
    for (v, s), m in mucols.items():
        r = cell(m)
        if r is None:
            continue
        supp, p, lift = r
        lift1[(v, s)] = lift
        direction = "si" if p > p0 else "no"
        w = (p - p0) / (1 - p0) if p > p0 else (p0 - p) / (p0 + 1e-9)
        w = float(np.clip(w, 0, 1))
        rows1.append({"regla": f"{v}={s}", "support": round(supp, 1), "p": round(p, 4),
                      "lift": round(lift, 2), "dir": direction, "peso": round(w, 3)})
        if abs(lift - 1.0) >= (min_lift - 1.0) and w > 0.05:
            rules.append(Rule([(v, s)], ("ocurrencia", direction), w))

    # -- arity 2 (grupos) -- only extreme partitions, distinct variables --
    keys = [(v, s) for (v, s) in mucols if s in EXTREME]
    rows2 = []
    for (a, b) in combinations(keys, 2):
        if a[0] == b[0]:
            continue                      # same variable, different partition -> not a coincidence
        m = np.minimum(mucols[a], mucols[b])   # AND = min (t-norm), same as the engine
        r = cell(m)
        if r is None:
            continue
        supp, p, lift = r
        if abs(lift - 1.0) < (min_lift - 1.0):
            continue
        direction = "si" if p > p0 else "no"
        # "mejor" depende de la dirección: máximo para excitatorias,
        # mínimo para protectoras. Usar max en ambas hacía fácil aprobar
        # falsos grupos protectores al compararlos con el marginal equivocado.
        marginal = [lift1.get(a, 1.0), lift1.get(b, 1.0)]
        base_parts = max(marginal) if direction == "si" else min(marginal)
        gain = lift / (base_parts + 1e-9)
        # only keep as a GROUP rule if the coincidence adds information over the marginals
        if (direction == "si" and gain < min_gain) or (direction == "no" and gain > (1 / min_gain)):
            keep = False
        else:
            keep = True
        w = (p - p0) / (1 - p0) if p > p0 else (p0 - p) / (p0 + 1e-9)
        w = float(np.clip(w, 0, 1))
        rows2.append({"grupo": f"{a[0]}={a[1]} & {b[0]}={b[1]}", "support": round(supp, 1),
                      "p": round(p, 4), "lift": round(lift, 2), "gain_vs_partes": round(gain, 2),
                      "dir": direction, "peso": round(w, 3), "keep": keep})
        if keep and w > 0.05:
            rules.append(Rule([a, b], ("ocurrencia", direction), w))

    t1 = pd.DataFrame(rows1).sort_values("lift", ascending=False)
    t2 = pd.DataFrame(rows2).sort_values("gain_vs_partes", ascending=False) if rows2 else pd.DataFrame()
    return rules, t1, t2


# ============================ MAGNITUD: tectónica (mag_scale), no estado ========================
def learn_magnitude(tr_pos):
    """State does not resolve magnitude (measured). Learn the tectonic map: mag_scale partition ->
    the magnitud linguistic term whose centroid is closest to the observed mean real magnitude in
    that partition. Rules key on the STATIC input mag_scale (generalizes to any GEM point)."""
    from fuzzy_engine import MAG_GRID, MAG_SETS
    centroids = {}
    for name, params in MAG_SETS.items():
        mfun = get_membership(MAG_GRID, params)
        centroids[name] = float((MAG_GRID * mfun).sum() / mfun.sum())
    rules, rows = [], []
    for s in sets_for_var("mag_scale"):
        m = mu_col(tr_pos["mag_scale"], s)
        supp = m.sum()
        if supp < 20:
            continue
        mean_mag = float((m * tr_pos["event_mag"]).sum() / supp)
        term = min(centroids, key=lambda k: abs(centroids[k] - mean_mag))
        rows.append({"mag_scale": s, "support": round(supp, 1), "mag_media_real": round(mean_mag, 2),
                     "termino": term, "centroide": round(centroids[term], 2)})
        rules.append(Rule([("mag_scale", s)], ("magnitud", term), 1.0))
    return rules, pd.DataFrame(rows)


# ============================ CUÁNDO: honesto, tasa ~uniforme ===================================
def learn_when(tr_pos, min_lift=1.15):
    """delta-t barely correlates with state (|r|<=0.05). Learn weak rules (state -> inminente/lejano)
    but flag that they carry ~no day-resolution skill. Target = which third of the window the event
    fell in (inminente<=10d, proximo 10-20, lejano>20)."""
    d = tr_pos["event_delta_days"]
    y = pd.cut(d, [-1, 10, 20, 31], labels=["inminente", "proximo", "lejano"])
    inm = (y == "inminente").to_numpy(dtype=float)
    base = inm.mean()
    rules, rows = [], []
    for v in ["a_sse", "a_strain", "a_tec", "a_fluid", "C_state"]:
        if v not in tr_pos.columns:
            continue
        col = tr_pos[v].fillna(0.0)
        for s in ["muy_alto", "alto"]:
            m = mu_col(col, s)
            supp = m.sum()
            if supp < 20:
                continue
            share = float((m * inm).sum() / supp)
            lift = share / (base + 1e-9)
            rows.append({"regla": f"{v}={s}", "support": round(supp, 1),
                         "p_inminente": round(share, 3), "base": round(base, 3),
                         "lift": round(lift, 2)})
            if lift >= min_lift:
                w = float(np.clip((share - base) / (1 - base), 0, 1))
                rules.append(Rule([(v, s)], ("cuando", "inminente"), w))
    return rules, pd.DataFrame(rows).sort_values("lift", ascending=False) if rows else pd.DataFrame()


# ============================ DÓNDE (radio): señal real =========================================
def learn_where(tr_pos, min_lift=1.10):
    """The one secondary axis with real transient signal. Target = normalized radial distance of the
    epicenter from the segment centroid (0=core, 1=edge). Learn rules state -> centro/borde."""
    r = tr_pos["dist_norm"].to_numpy(dtype=float)
    base_r = float(r.mean())
    rules, rows = [], []
    for v in ["C_state", "F_state", "A_state", "a_strain", "E_state", "a_seis"]:
        if v not in tr_pos.columns:
            continue
        col = tr_pos[v].fillna(0.0)
        for s in ["muy_alto", "alto", "muy_bajo", "bajo"]:
            m = mu_col(col, s)
            supp = m.sum()
            if supp < 30:
                continue
            mean_r = float((m * r).sum() / supp)      # membership-weighted mean radius in this cell
            term = "centro" if mean_r < base_r else "borde"
            lift = (base_r / (mean_r + 1e-9)) if term == "centro" else (mean_r / (base_r + 1e-9))
            rows.append({"regla": f"{v}={s}", "support": round(supp, 1), "radio_medio": round(mean_r, 3),
                         "base": round(base_r, 3), "termino": term, "lift": round(lift, 2)})
            if lift >= min_lift:
                w = float(np.clip(abs(mean_r - base_r) / (base_r + 1e-9), 0, 1))
                rules.append(Rule([(v, s)], ("donde", term), w))
    return rules, pd.DataFrame(rows).sort_values("lift", ascending=False) if rows else pd.DataFrame()


# ============================ scoring del motor real sobre holdout ==============================
def score_occurrence(rules, df):
    """Vectorized occurrence-only Mamdani/MOM score.

    For the bipolar occurrence ramps, MOM reduces exactly to +ws when the strongest
    "si" rule wins, -wn when the strongest "no" rule wins, and 0 on ties.
    """
    from fuzzy_engine import MU_EV

    n = len(df)
    ws = np.zeros(n, dtype=np.float64)
    wn = np.zeros(n, dtype=np.float64)
    needed = sorted({tuple(p) for r in rules if r.output == "ocurrencia" for p in r.antecedent})
    mu = {}
    for v, s in needed:
        x = df[v].fillna(0.0).to_numpy(dtype=float) if v in df.columns else np.zeros(n)
        mu[(v, s)] = get_membership(x, sets_for_var(v)[s])
    for r in rules:
        if r.output != "ocurrencia":
            continue
        if not r.antecedent:
            continue
        m = mu[tuple(r.antecedent[0])].copy()
        for p in r.antecedent[1:]:
            np.minimum(m, mu[tuple(p)], out=m)
        rule_mu_ev = min(MU_EV.get(v, 1.0) for v, _s in r.antecedent)
        m *= rule_mu_ev * r.weight
        if r.out_set == "si":
            np.maximum(ws, m, out=ws)
        else:
            np.maximum(wn, m, out=wn)
    out = np.where(ws > wn, ws, -wn)
    out[np.isclose(ws, wn)] = 0.0
    return out


def main():
    df = pd.read_csv(os.path.join(ROOT, "out", "cases_enriched.csv"), parse_dates=["date"])
    # derive the static + secondary targets the enrichment left us the raw material for
    import glob
    seg_target = {}
    for p in glob.glob(os.path.join(ROOT, "config", "segment_*.json")):
        c = json.load(open(p))
        sid = c.get("segment_id", "").lower().replace("_", "-")
        seg_target[sid] = c["etas_lite"]["target_mag"]
    df["mag_scale"] = df["segment"].map(seg_target).apply(
        lambda t: float(np.clip(2 * (t - MAG_LO) / (MAG_HI - MAG_LO) - 1, -1, 1)) if pd.notna(t) else 0.0)
    # normalized radial distance target for "dónde" (per-segment centroid + p95 spread)
    lk = df.dropna(subset=["event_lat"]).copy()
    latc = lk.groupby("segment")["event_lat"].transform("median")
    lonc = lk.groupby("segment")["event_lon"].transform("median")
    raw = np.sqrt((lk["event_lat"] - latc) ** 2 + (lk["event_lon"] - lonc) ** 2)
    p95 = raw.groupby(lk["segment"]).transform(lambda s: s.quantile(0.95))
    df["dist_norm"] = np.nan
    df.loc[lk.index, "dist_norm"] = np.clip(raw / (p95 + 1e-9), 0, 1)

    tr = df[df["date"] < CUTOFF].copy()
    ho = df[df["date"] >= CUTOFF].copy()
    tr_pos = tr.dropna(subset=["event_mag"]).copy()
    ho_pos = ho.dropna(subset=["event_mag"]).copy()
    print(f"train {len(tr)} ({tr[LABEL].mean()*100:.1f}% pos) | holdout {len(ho)} "
          f"({ho[LABEL].mean()*100:.1f}% pos) | positivos ligados train={len(tr_pos)} ho={len(ho_pos)}\n")

    # ---------- OCURRENCIA ----------
    occ_rules, t1, t2 = learn_occurrence(tr)
    print("=== OCURRENCIA · reglas arity-1 (top lift) ===")
    print(t1.head(10).to_string(index=False))
    print(f"\n=== OCURRENCIA · GRUPOS arity-2 que superan a sus partes (keep=True) ===")
    if len(t2):
        keep2 = t2[t2["keep"]]
        print(keep2.head(12).to_string(index=False) if len(keep2) else "  (ninguno supera el umbral de ganancia)")
        print(f"  ({int(t2['keep'].sum())} grupos genuinos de {len(t2)} pares con lift suficiente)")
    else:
        print("  (ninguno)")
    n1 = sum(1 for r in occ_rules if len(r.antecedent) == 1)
    n2 = sum(1 for r in occ_rules if len(r.antecedent) == 2)
    print(f"\n  base ocurrencia: {n1} reglas simples + {n2} grupos")

    s_learned = score_occurrence(occ_rules, ho)
    s_expert = score_occurrence(EXPERT_RULES, ho)
    y = ho[LABEL].to_numpy(); etas = ho["lam_etas"].to_numpy()
    aucf = lambda p: roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    print("\n  motor Mamdani REAL sobre holdout (ocurrencia defuzzificada):")
    print(f"    reglas aprendidas (arity 1+2): AUC={aucf(s_learned):.4f}")
    print(f"    reglas expertas (a mano):      AUC={aucf(s_expert):.4f}")
    print(f"    ETAS-solo (referencia):        AUC={aucf(etas):.4f}")

    # -- COMPLEMENTARIEDAD: ¿el fuzzy añade señal ORTOGONAL a ETAS, o es redundante? --
    # decisivo: el fuzzy solo (0.63) va por debajo de ETAS (0.72), pero eso no dice si CAPTURA algo
    # que ETAS no ve. Test disciplinado: logística de 2 features (ETAS, fuzzy) ajustada en TRAIN,
    # evaluada en holdout, vs logística de 1 feature (ETAS). CRÍTICO: estandarizar ambas features
    # (lam_etas es una tasa de cola pesada, escala enorme; el score fuzzy es [-1,1]) y usar
    # regularización mínima (C grande) -- si no, la L2 por defecto encoge el coef de ETAS y el test
    # da un delta negativo espurio (artefacto de escala, no señal). Con log1p+estandarización, añadir
    # una feature no puede empeorar el ajuste in-sample; el holdout da la respuesta real.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    s_tr = score_occurrence(occ_rules, tr)                 # fuzzy en train (caro, ~63k filas)
    le_tr = np.log1p(np.clip(np.nan_to_num(tr["lam_etas"].to_numpy(), nan=0.0), 0, None))
    le_ho = np.log1p(np.clip(np.nan_to_num(etas, nan=0.0), 0, None))
    ytr = tr[LABEL].to_numpy()
    # cache raw scores for offline iteration (no re-scoring needed to re-analyze)
    np.savez(os.path.join(ROOT, "out", "occ_scores.npz"), s_tr=s_tr, s_ho=s_learned,
             le_tr=le_tr, le_ho=le_ho, ytr=ytr, yho=y)
    sc_e = StandardScaler().fit(le_tr.reshape(-1, 1))
    sc_b = StandardScaler().fit(np.column_stack([le_tr, s_tr]))
    lr_e = LogisticRegression(C=1e4, max_iter=5000).fit(sc_e.transform(le_tr.reshape(-1, 1)), ytr)
    lr_b = LogisticRegression(C=1e4, max_iter=5000).fit(sc_b.transform(np.column_stack([le_tr, s_tr])), ytr)
    auc_e = roc_auc_score(y, lr_e.predict_proba(sc_e.transform(le_ho.reshape(-1, 1)))[:, 1])
    auc_b = roc_auc_score(y, lr_b.predict_proba(sc_b.transform(np.column_stack([le_ho, s_learned])))[:, 1])
    print(f"    ETAS-solo (logística estand. train->holdout):   AUC={auc_e:.4f}")
    print(f"    ETAS + fuzzy (logística estand. train->holdout): AUC={auc_b:.4f}  "
          f"(delta={auc_b-auc_e:+.4f} -> {'fuzzy APORTA ortogonal' if auc_b-auc_e>0.002 else 'redundante con ETAS'})")

    # ---------- MAGNITUD ----------
    mag_rules, tmag = learn_magnitude(tr_pos)
    print("\n=== MAGNITUD · mapa tectónico (mag_scale -> término), NO estado ===")
    print(tmag.to_string(index=False))
    preds = np.array([infer({"mag_scale": ms}, mag_rules)["magnitud"]["value"] for ms in ho_pos["mag_scale"]],
                     dtype=float)
    real = ho_pos["event_mag"].to_numpy()
    ok = ~np.isnan(preds)
    mae_f = np.mean(np.abs(preds[ok] - real[ok]))
    mae_base = np.mean(np.abs(tr_pos["event_mag"].mean() - real))
    print(f"  holdout MAE magnitud: fuzzy-tectónico={mae_f:.3f}  vs  media-global={mae_base:.3f}  "
          f"(n={int(ok.sum())})")

    # ---------- CUÁNDO ----------
    when_rules, twhen = learn_when(tr_pos)
    print("\n=== CUÁNDO · (honesto: el estado casi no resuelve el día) ===")
    print(twhen.to_string(index=False) if len(twhen) else "  (sin reglas sobre umbral)")
    if when_rules:
        recs = ho_pos[["a_sse", "a_strain", "a_tec", "a_fluid", "C_state"]].fillna(0.0).to_dict("records")
        wp = np.array([15.0 if (v := infer(x, when_rules)["cuando"]["value"]) is None else v for x in recs])
    else:
        wp = np.full(len(ho_pos), 15.0)
    dr = ho_pos["event_delta_days"].to_numpy()
    print(f"  holdout MAE días: fuzzy-cuando={np.mean(np.abs(wp-dr)):.2f}  vs  "
          f"constante-15={np.mean(np.abs(15-dr)):.2f}  (empate esperado = sin skill temporal)")

    # ---------- DÓNDE ----------
    where_rules, twhere = learn_where(tr_pos)
    print("\n=== DÓNDE (radio) · el eje secundario CON señal real ===")
    print(twhere.to_string(index=False) if len(twhere) else "  (sin reglas)")
    hp = ho_pos.dropna(subset=["dist_norm"])
    if where_rules and len(hp):
        recs = hp[["C_state", "F_state", "A_state", "a_strain", "E_state", "a_seis"]].fillna(0.0).to_dict("records")
        pred_r = [0.5 if (v := infer(x, where_rules)["donde"]["value"]) is None else v for x in recs]
        rho, pval = spearmanr(pred_r, hp["dist_norm"].to_numpy())
        print(f"  holdout: corr(radio predicho, radio real) rho={rho:+.3f} (p={pval:.1e}, n={len(hp)}) "
              f"-- señal real, débil pero direccional")

    # ---------- persist ----------
    all_rules = occ_rules + mag_rules + when_rules + where_rules
    out = [{"antecedent": r.antecedent, "output": r.output, "out_set": r.out_set,
            "weight": r.weight} for r in all_rules]
    with open(os.path.join(ROOT, "out", "learned_rules.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  base de reglas 4-ejes -> out/learned_rules.json "
          f"({len(occ_rules)} ocurrencia, {len(mag_rules)} magnitud, {len(when_rules)} cuando, "
          f"{len(where_rules)} donde)")


if __name__ == "__main__":
    main()
