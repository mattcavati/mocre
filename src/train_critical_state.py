"""Entrena consecuentes fuzzy y ejecuta validaciones/falsificaciones de MOCRE-2."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, roc_auc_score

from common import ROOT
from critical_state import RULES

OUT = os.path.join(ROOT, "out")
CONTRACT = json.load(open(os.path.join(ROOT, "config", "critical_state_contract.json")))
RCOLS = [f"r_{name}" for name, _ in RULES]


def fit_model(df):
    X = df[RCOLS].to_numpy(float); y = df.label.to_numpy(float)
    sw = df.sample_weight.to_numpy(float)
    def loss(theta):
        z = np.clip(theta[0] + X @ theta[1:], -25, 25)
        p = 1 / (1 + np.exp(-z))
        ce = -(y * np.log(p + 1e-12) + (1-y) * np.log(1-p + 1e-12))
        return float(np.average(ce, weights=sw) + 0.02 * np.sum(theta[1:] ** 2))
    result = minimize(loss, np.r_[-2.0, np.ones(len(RCOLS)) * 0.3], method="L-BFGS-B",
                      bounds=[(-8, 2)] + [(0, 6)] * len(RCOLS))
    return {"intercept": float(result.x[0]), "weights": result.x[1:].tolist(),
            "converged": bool(result.success)}


def predict(model, df):
    z = model["intercept"] + df[RCOLS].to_numpy(float) @ np.asarray(model["weights"])
    return 1 / (1 + np.exp(-np.clip(z, -25, 25)))


def metrics(df, p):
    y = df.label.to_numpy(int); w = df.sample_weight.to_numpy(float)
    if len(np.unique(y)) < 2:
        return {"n": len(y), "events": int(df[df.label == 1].episode_id.nunique()), "auc": None}
    return {"n": len(y), "events": int(df[df.label == 1].episode_id.nunique()),
            "auc": float(roc_auc_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w))}


def leave_region_out(data):
    rows = []
    for segment in sorted(data.segment.unique()):
        train = data[data.segment != segment]
        test = data[data.segment == segment]
        if train.label.nunique() < 2 or test.label.nunique() < 2:
            continue
        model = fit_model(train)
        m = metrics(test, predict(model, test))
        rows.append({"segment": segment, **m})
    return rows


def time_shift_falsification(model, data):
    """Desplaza las features positivas un año dentro de cada segmento.

    Se usa la fila de episodio/control temporalmente más próxima al ancla
    desplazada. Si la señal es específica del preevento, el AUC debe colapsar.
    """
    actual, shifted = [], []
    pos = data[(data.block == "test") & (data.label == 1)].copy()
    for _, row in pos.iterrows():
        seg = data[data.segment == row.segment].copy()
        target = pd.Timestamp(row.anchor) + pd.Timedelta(days=365)
        delta = (pd.to_datetime(seg.anchor) - target).abs()
        if len(delta) and delta.min().days <= 120:
            actual.append(row)
            shifted.append(seg.loc[delta.idxmin()])
    if len(actual) < 3:
        return {"pairs": len(actual), "auc_actual_vs_shifted": None}
    a = pd.DataFrame(actual); s = pd.DataFrame(shifted)
    scores = np.r_[predict(model, a), predict(model, s)]
    labels = np.r_[np.ones(len(a)), np.zeros(len(s))]
    return {"pairs": len(a), "auc_actual_vs_shifted": float(roc_auc_score(labels, scores))}


def main():
    data = pd.read_csv(os.path.join(OUT, "critical_episodes.csv.gz"), parse_dates=["anchor", "event_date"])
    fit = data[data.block == "fit"]; cal = data[data.block == "cal"]; test = data[data.block == "test"]
    model = fit_model(fit)
    # Umbral fijado con controles de calibración, nunca con terremotos de test.
    p_cal = predict(model, cal)
    cal_controls = p_cal[cal.label.to_numpy() == 0]
    threshold = float(np.quantile(cal_controls, 0.95)) if len(cal_controls) else 0.8
    p_test = predict(model, test)
    test_metrics = metrics(test, p_test)
    event_scores = pd.DataFrame({"episode": test.episode_id, "label": test.label, "p": p_test}).groupby(
        ["episode", "label"]).p.max().reset_index()
    positives = event_scores[event_scores.label == 1].p
    controls = event_scores[event_scores.label == 0].p
    sensitivity = float((positives >= threshold).mean()) if len(positives) else 0.0
    false_fraction = float((controls >= threshold).mean()) if len(controls) else 1.0
    region = leave_region_out(data[data.block != "test"])
    region_aucs = [r["auc"] for r in region if r.get("auc") is not None]
    shifted = time_shift_falsification(model, data)
    crit = CONTRACT["promotion"]
    checks = {
        "enough_test_events": test_metrics.get("events", 0) >= crit["min_test_events"],
        "episode_auc": (test_metrics.get("auc") or 0) >= crit["min_episode_auc"],
        "leave_region_auc": (float(np.median(region_aucs)) if region_aucs else 0) >= crit["min_leave_region_median_auc"],
        "false_alarm_fraction": false_fraction <= crit["max_false_alarm_fraction"],
        "event_sensitivity": sensitivity >= crit["min_event_sensitivity"],
        "time_shift_collapses": shifted.get("auc_actual_vs_shifted") is not None and
                                shifted["auc_actual_vs_shifted"] <= crit["max_time_shift_auc"],
    }
    report = {"version": CONTRACT["version"], "rule_names": [r[0] for r in RULES],
              "intercept": model["intercept"], "weights": model["weights"],
              "threshold": threshold, "fit": metrics(fit, predict(model, fit)),
              "cal": metrics(cal, p_cal), "test": test_metrics,
              "test_event_sensitivity": sensitivity, "test_control_false_fraction": false_fraction,
              "leave_region_out": region, "leave_region_median_auc": float(np.median(region_aucs)) if region_aucs else None,
              "time_shift": shifted, "promotion_checks": checks,
              "ia3_outputs": {
                  "occurrence": "shadow_state_only" if not all(checks.values()) else "eligible_for_calibration",
                  "magnitude": "not_trained_until_occurrence_promotes",
                  "when": "not_trained_until_occurrence_promotes",
                  "where": "not_trained_until_spatial_state_promotes"
              },
              "promoted": bool(all(checks.values())),
              "status": "candidate" if all(checks.values()) else "shadow_research"}
    json.dump(report, open(os.path.join(OUT, "critical_state_model.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
