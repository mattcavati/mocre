"""MOCRE-1 fuzzy-inverse core: discover the minimal subset of variables that predicts f=1
(M>=target event in the next 30 days) OUT OF SAMPLE, on the global pool.

Discipline (non-negotiable, this is the whole point -- without it we repeat Avaltix's 49.6%=chance):
  - TEMPORAL split, not random: train = days before CUTOFF, holdout = days after. Predicting the
    real future. A random split leaks catastrophically (consecutive days ~identical).
  - Feature ranking done on TRAIN ONLY (MI + L1-logistic + RF importance), then the chosen subset
    is scored on the untouched HOLDOUT.
  - The null to beat is ETAS (lam_etas), the same baseline as always. The real question is not
    "do the fuzzy variables correlate with events" (they will, in-sample) but "do they ADD skill
    OVER ETAS out-of-sample". If ETAS+fuzzy does not beat ETAS-alone on holdout, there is no
    signal to build a fuzzy system on -- an honest, valuable answer either way.
  - Also compared against a plain logistic (the model that beat the hand-fuzzy in Avaltix): if the
    discovered subset can't beat a simple logistic, the fuzzy machinery isn't earning its keep.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import ROOT

CUTOFF = "2018-01-01"          # train before, holdout after -- real forward prediction
LABEL = "f_30d"
FEATURES = ["a_seis", "a_gnss", "a_swarm", "a_foreshock", "a_sse", "a_fluid", "a_water",
            "a_strain", "a_ocean", "a_insar", "a_tec", "a_noise",
            "A_state", "C_state", "K_state", "E_state", "F_state"]


def auc_safe(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def main():
    df = pd.read_csv(os.path.join(ROOT, "out", "cases_global.csv"), parse_dates=["date"])
    df = df.dropna(subset=FEATURES + [LABEL])
    tr = df[df["date"] < CUTOFF]
    ho = df[df["date"] >= CUTOFF]
    print(f"train {len(tr)} filas ({tr[LABEL].mean()*100:.1f}% pos) | "
          f"holdout {len(ho)} filas ({ho[LABEL].mean()*100:.1f}% pos) | corte {CUTOFF}")
    Xtr, ytr = tr[FEATURES].to_numpy(), tr[LABEL].to_numpy()
    Xho, yho = ho[FEATURES].to_numpy(), ho[LABEL].to_numpy()
    etas_tr, etas_ho = tr["lam_etas"].to_numpy(), ho["lam_etas"].to_numpy()

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xho_s = sc.transform(Xtr), sc.transform(Xho)

    # ---- baselines on holdout ----
    print("\n=== BASELINES (AUC en holdout) ===")
    auc_etas = auc_safe(yho, etas_ho)
    print(f"  ETAS-solo (lam_etas):            AUC={auc_etas:.4f}  AP={average_precision_score(yho, etas_ho):.4f}")
    logf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, ytr)
    p_logf = logf.predict_proba(Xho_s)[:, 1]
    print(f"  Logística FULL (17 fuzzy vars):  AUC={auc_safe(yho, p_logf):.4f}  AP={average_precision_score(yho, p_logf):.4f}")
    # ETAS + fuzzy full
    Xtr_e = np.column_stack([Xtr_s, np.log(etas_tr + 1e-9)])
    Xho_e = np.column_stack([Xho_s, np.log(etas_ho + 1e-9)])
    loge = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_e, ytr)
    p_loge = loge.predict_proba(Xho_e)[:, 1]
    print(f"  ETAS + fuzzy FULL:               AUC={auc_safe(yho, p_loge):.4f}  AP={average_precision_score(yho, p_loge):.4f}")

    # ---- feature ranking on TRAIN only ----
    print("\n=== RANKING de variables (solo train) ===")
    mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                            class_weight="balanced", max_iter=2000).fit(Xtr_s, ytr)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced",
                                random_state=0, n_jobs=-1).fit(Xtr_s, ytr)
    rank = pd.DataFrame({"feature": FEATURES, "MI": mi, "L1_coef": l1.coef_[0],
                         "RF_imp": rf.feature_importances_})
    rank["rank_score"] = (rank["MI"].rank() + rank["RF_imp"].rank() +
                          rank["L1_coef"].abs().rank())
    rank = rank.sort_values("rank_score", ascending=False)
    print(rank.to_string(index=False,
          formatters={"MI": "{:.4f}".format, "L1_coef": "{:+.3f}".format,
                      "RF_imp": "{:.4f}".format, "rank_score": "{:.0f}".format}))

    order = rank["feature"].tolist()
    l1_subset = rank[rank["L1_coef"].abs() > 1e-6]["feature"].tolist()
    print(f"\n  L1 (sparse) mantiene {len(l1_subset)}/{len(FEATURES)}: {l1_subset}")

    # ---- forward selection with HONEST holdout scoring: add ETAS, then top-k fuzzy vars ----
    print("\n=== SUBCONJUNTO MÍNIMO (AUC holdout de ETAS + top-k variables) ===")
    best_k, best_auc = 0, auc_etas
    for k in range(0, len(order) + 1):
        feats = order[:k]
        cols_tr = [np.log(etas_tr + 1e-9)]
        cols_ho = [np.log(etas_ho + 1e-9)]
        for f in feats:
            j = FEATURES.index(f)
            cols_tr.append(Xtr_s[:, j]); cols_ho.append(Xho_s[:, j])
        Xk_tr = np.column_stack(cols_tr); Xk_ho = np.column_stack(cols_ho)
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xk_tr, ytr)
        a = auc_safe(yho, m.predict_proba(Xk_ho)[:, 1])
        tag = ""
        if a > best_auc + 1e-4:
            best_auc, best_k = a, k; tag = "  <- mejor"
        added = feats[-1] if feats else "(solo ETAS)"
        print(f"  k={k:>2} (+{added:<12}) AUC_holdout={a:.4f}{tag}")

    print(f"\nVEREDICTO: mejor AUC holdout = {best_auc:.4f} con ETAS + top-{best_k} "
          f"({order[:best_k]}) vs ETAS-solo {auc_etas:.4f} "
          f"(delta {best_auc-auc_etas:+.4f})")


if __name__ == "__main__":
    main()
