"""Comparación CSEP por evento de MOCRE frente a ETAS.

Implementa la forma de Rhoades usada por pyCSEP: a cada terremoto le
corresponde ``log(lambda_A/lambda_B)`` y se corrige la diferencia del número
total esperado. Las antiguas contribuciones mensuales no eran el T/W-test de
comparación de pronósticos y daban una falsa muestra de meses vacíos.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT, load_config, segment_paths

OUT = os.path.join(ROOT, "out")


def load_events(cfg):
    from pipeline import load_catalog
    cat = load_catalog(cfg)
    return cat[cat["mag"] >= cfg["etas_lite"]["target_mag"]]


def event_log_ratios(series, events, model_a="lam_mocre", model_b="lam_etas"):
    """Log-ratios en los bins observados, uno por evento (no uno por mes)."""
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    ratios = []
    for d in pd.to_datetime(events["date"]):
        if d < series.index[0] or d > series.index[-1]:
            continue
        pos = series.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]
        a = float(series.iloc[pos][model_a]); b = float(series.iloc[pos][model_b])
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            ratios.append(np.log(a / b))
    return np.asarray(ratios, dtype=float)


def n_test(expected, observed, alpha=0.05):
    """Prueba N de Poisson, con colas inclusivas y p bilateral conservador."""
    lower = float(stats.poisson.cdf(observed, expected))
    upper = float(stats.poisson.sf(observed - 1, expected))
    p_two = min(1.0, 2.0 * min(lower, upper))
    return {"expected": round(float(expected), 4), "observed": int(observed),
            "p_two_sided": round(p_two, 6), "passes_95pct": bool(p_two >= alpha)}


def comparison_test(series, events):
    """T/W oficiales y N-tests para dos pronósticos discretos Poisson."""
    d = event_log_ratios(series, events)
    n = len(d)
    expected_a = float(series["lam_mocre"].sum())
    expected_b = float(series["lam_etas"].sum())
    if n < 2:
        return {"status": "insufficient_events", "n_events": n,
                "n_test_mocre": n_test(expected_a, n), "n_test_etas": n_test(expected_b, n)}

    count_correction = (expected_a - expected_b) / n
    centered = d - count_correction
    t_stat, t_p = stats.ttest_1samp(d, popmean=count_correction)
    try:
        w_stat, w_p = stats.wilcoxon(centered, alternative="two-sided", method="auto")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    igpe = float(centered.mean())
    return {
        "status": "ok", "n_events": n,
        "mean_event_log_ratio": round(float(d.mean()), 6),
        "count_correction_per_event": round(float(count_correction), 6),
        "information_gain_per_event_nats": round(igpe, 6),
        "total_loglikelihood_difference": round(float(centered.sum()), 6),
        "t_test": {"statistic": round(float(t_stat), 6), "p_value": round(float(t_p), 6)},
        "w_test": {"statistic": None if np.isnan(w_stat) else round(float(w_stat), 6),
                   "p_value": None if np.isnan(w_p) else round(float(w_p), 6)},
        "n_test_mocre": n_test(expected_a, n),
        "n_test_etas": n_test(expected_b, n),
        "low_power": bool(n < 20),
    }


def run_segment(cfg_path):
    cfg = load_config(os.path.basename(cfg_path))
    slug = segment_paths(cfg)["slug"]
    series_path = os.path.join(OUT, f"series_{slug}.csv")
    params_path = os.path.join(OUT, f"etas_params_{slug}.json")
    if not (os.path.exists(series_path) and os.path.exists(params_path)):
        return None
    series = pd.read_csv(series_path, index_col=0, parse_dates=True)
    # Compatibilidad con artefactos antiguos; el pipeline nuevo ya almacena este producto crudo.
    series["lam_mocre"] = series["lam_etas"] * series["G"]
    train_end = pd.Timestamp(json.load(open(params_path))["train_end"])
    cons_path = os.path.join(OUT, f"consequents_trained_{slug}.json")
    if os.path.exists(cons_path):
        train_end = max(train_end, pd.Timestamp(json.load(open(cons_path))["train_end"]))
    holdout = series[series.index >= train_end]
    events = load_events(cfg)
    events = events[(pd.to_datetime(events["date"]) >= holdout.index[0]) &
                    (pd.to_datetime(events["date"]) <= holdout.index[-1])]
    result = comparison_test(holdout, events)
    result.update({"segment": slug,
                   "holdout_window": [str(holdout.index[0].date()), str(holdout.index[-1].date())],
                   "prospective": False,
                   "confirmatory": False,
                   "interpretation": "exploratorio: el holdout fue reutilizado durante el desarrollo"})
    return result


def main():
    results = []
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        result = run_segment(cfg_path)
        if result:
            results.append(result)
            print(json.dumps(result, indent=2))
    path = os.path.join(OUT, "csep_paired_test_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {path}\nLos p-valores son exploratorios; la confirmación comienza con el ledger prospectivo.")


if __name__ == "__main__":
    main()
