"""MOCRE-1 F3: walk-forward training of G's consequents.

Trains the 9 free consequents (2 anchors stay fixed at 0.0) to maximize the training-window
point-process log-likelihood, with a monotonicity constraint enforced as an ORDERED CHAIN across
the seismicity-anomaly spectrum: c(muy_bajo) <= c(bajo) <= c(normal)=0 <= c(alto) <= c(muy_alto).

CORRECTED 2026-07-03: the original version used independent softplus per rule, forcing EVERY
consequent >=0 -- which is a stronger, WRONG constraint (it made G structurally unable to ever
predict below the ETAS baseline, a real bug found via csep_test.py's T/W-test disagreement; see
pipeline.py's RULE_NAMES comment for the full story). R4 ("more anomaly never lowers gain") means
monotonic across the anomaly spectrum, not "all consequents non-negative" -- those are different
constraints, and the low-anomaly rules (seis_bajo/seis_muy_bajo) legitimately need negative
consequents to express suppressed hazard during genuinely quiet periods.

Training window: same cutoff as the ETAS MLE fit (train_end), so the fuzzy refiner and the
statistical backbone never see the holdout window during fitting -- avoids double-dipping.
Holdout is evaluated with the SAME code path pipeline.py uses (gain_from_weights), not re-derived.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import ROOT, load_config, segment_paths
from pipeline import (DEFAULT_CONSEQUENTS, RULE_NAMES, TRAINABLE_RULES, build_daily, build_state,
                       etas_fitted, gain_from_weights, load_catalog, load_gnss, loglik,
                       rule_weights)

OUT = os.path.join(ROOT, "out")


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)  # numerically stable


# Generalized 2026-07-03 after the SAME bug (rule silently trained back to non-negative) hit
# TWICE more the same session for new "_bajo"-style rules (fluid_bajo, enjambre_bajo,
# foreshock_calmo) that unpack() didn't know about -- it only special-cased the original
# seismicity chain by hardcoded name. Any rule meant to express a "low anomaly -> lower gain"
# response MUST be listed here or it silently gets the generic positive-only softplus treatment
# and F3 training will flip a carefully-chosen negative default back to positive, recreating the
# one-sided-G bug via the training path instead of the default path. Two structures:
#   NEGATIVE_CHAINS: (low, very_low) pairs where very_low <= low <= 0 (ordered, like seis/fluid)
#   NEGATIVE_SINGLES: single rules with only one negative-direction level, no "very" tier
NEGATIVE_CHAINS = [("seis_bajo", "seis_muy_bajo"), ("fluid_bajo", "fluid_muy_bajo")]
NEGATIVE_SINGLES = ["enjambre_bajo", "foreshock_calmo", "agua_bajo", "strain_bajo", "ocean_bajo",
                    "insar_bajo", "tec_bajo", "noise_bajo"]
POSITIVE_CHAINS = [("seis_alto", "seis_muy_alto")]  # (level1, level2) where level2 >= level1 >= 0


def unpack(theta, cfg):
    """Unconstrained theta -> consequents respecting monotonic ordering for every rule listed in
    NEGATIVE_CHAINS/NEGATIVE_SINGLES/POSITIVE_CHAINS; every other rule is independently
    non-negative (coincidence/state indicators -- no "lower than what" for them to violate)."""
    lo, hi = cfg["gain"]["log10_min"], cfg["gain"]["log10_max"]
    idx = {r: i for i, r in enumerate(TRAINABLE_RULES)}
    vals = {}
    for lo_name, vlo_name in NEGATIVE_CHAINS:
        lo_val = -softplus(theta[idx[lo_name]])
        vlo_val = np.maximum(lo_val - softplus(theta[idx[vlo_name]]), lo)
        vals[lo_name] = float(np.maximum(lo_val, lo))
        vals[vlo_name] = float(vlo_val)
    for name in NEGATIVE_SINGLES:
        vals[name] = float(np.maximum(-softplus(theta[idx[name]]), lo))
    for a_name, b_name in POSITIVE_CHAINS:
        a_val = softplus(theta[idx[a_name]])
        b_val = np.minimum(a_val + softplus(theta[idx[b_name]]), hi)
        vals[a_name] = float(np.minimum(a_val, hi))
        vals[b_name] = float(b_val)
    for r in TRAINABLE_RULES:
        if r not in vals:
            vals[r] = float(min(softplus(theta[idx[r]]), hi))
    return vals


def neg_loglik_train(theta, weights, cfg, lam_etas, ev_idx, train_mask, prior, reg_strength):
    cons = unpack(theta, cfg)
    G = gain_from_weights(weights, cons, cfg)
    lam = lam_etas * G
    lam_t = lam[train_mask]
    idx_t = [i for i in ev_idx if train_mask[i]]
    # remap event indices into the train-masked array
    remap = np.cumsum(train_mask) - 1
    idx_local = [remap[i] for i in idx_t]
    nll = -loglik(lam_t, np.array(idx_local, dtype=int))
    # ridge penalty toward the expert prior -- with only a handful of training events, the
    # likelihood alone cannot constrain 7 free parameters (see report "degrees_of_freedom_note").
    # This is a weakly-informative Bayesian prior, not a fudge: pulls the fit back from the
    # gain ceiling that pure MLE hits with n_train=3.
    penalty = reg_strength * sum((unpack(theta, cfg)[r] - prior[r]) ** 2 for r in TRAINABLE_RULES)
    return nll + penalty


def main():
    cfg = load_config()
    slug = segment_paths(cfg)["slug"]
    cat = load_catalog(cfg)
    stations, gnss = load_gnss(cfg)
    daily = build_daily(cfg, cat, stations, gnss)
    daily = build_state(cfg, daily)

    with open(os.path.join(OUT, f"etas_params_{slug}.json")) as f:
        etas_params = json.load(f)
    lam_etas, _ = etas_fitted(cfg, cat, daily.index, etas_params)
    weights = rule_weights(daily)

    tgt = cfg["etas_lite"]["target_mag"]
    ev = cat[(cat["mag"] >= tgt) & (cat["date"] >= daily.index[0])]
    ev_idx = daily.index.get_indexer(ev["date"], method="nearest")

    train_end = pd.Timestamp(etas_params["train_end"])
    train_mask = np.asarray(daily.index < train_end)
    n_train_events = int(sum(1 for i in ev_idx if train_mask[i]))
    print(f"training window: {daily.index[0].date()} .. {train_end.date()} "
          f"({train_mask.sum()} days, {n_train_events} target events)")
    DEFAULT = DEFAULT_CONSEQUENTS
    dof_note = None
    underpowered = n_train_events < 2 * len(TRAINABLE_RULES)
    if underpowered:
        dof_note = (f"n_train_events={n_train_events} vs {len(TRAINABLE_RULES)} free consequents: "
                    "likelihood alone cannot constrain this fit. Using a ridge prior toward the "
                    "hand-picked defaults (reg_strength below) -- this is damage control, not a "
                    "real fix. Real fix is more training events (more segments in F2.5, or a "
                    "lower target magnitude with a matching ETAS refit).")
        print("WARNING:", dof_note)

    reg_strength = 2.0  # weakly-informative; see dof_note
    x0 = np.zeros(len(TRAINABLE_RULES))  # softplus(0)=log(2)=0.69 -> mild positive start
    if underpowered:
        # No se convierte una optimización subdeterminada en motor operativo.
        # Se conserva el prior experto y se marca explícitamente como no entrenado.
        res = SimpleNamespace(success=False)
        trained_cons = dict(DEFAULT)
    else:
        res = minimize(neg_loglik_train, x0,
                       args=(weights, cfg, lam_etas, ev_idx, train_mask, DEFAULT, reg_strength),
                       method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-4, "fatol": 1e-4})
        trained_cons = {k: round(float(v), 4) for k, v in unpack(res.x, cfg).items()}

    # ---------------------------------------------------------------- scoreboard: train vs holdout
    def score(cons, mask):
        G = gain_from_weights(weights, cons, cfg)
        lam = lam_etas * G
        idx_m = [i for i in ev_idx if mask[i]]
        remap = np.cumsum(mask) - 1
        idx_local = np.array([remap[i] for i in idx_m], dtype=int)
        return loglik(lam[mask], idx_local), loglik(lam_etas[mask], idx_local), len(idx_local)

    hold_mask = ~train_mask

    ll_train_hand, ll_train_etas, n_train = score(DEFAULT, train_mask)
    ll_train_fit, _, _ = score(trained_cons, train_mask)
    ll_hold_hand, ll_hold_etas, n_hold = score(DEFAULT, hold_mask)
    ll_hold_fit, _, _ = score(trained_cons, hold_mask)

    report = {
        "train_end": str(train_end.date()),
        "n_train_events": n_train, "n_holdout_events": n_hold,
        "trained_consequents": trained_cons,
        "hand_picked_consequents": DEFAULT,
        "converged": bool(res.success), "optimization_attempted": not underpowered,
        "train_window": {
            "loglik_etas_only": round(ll_train_etas, 2),
            "loglik_hand_picked": round(ll_train_hand, 2),
            "loglik_trained": round(ll_train_fit, 2),
            "info_gain_hand_nats": round((ll_train_hand - ll_train_etas) / max(n_train, 1), 4),
            "info_gain_trained_nats": round((ll_train_fit - ll_train_etas) / max(n_train, 1), 4),
        },
        "holdout_window": {
            "loglik_etas_only": round(ll_hold_etas, 2),
            "loglik_hand_picked": round(ll_hold_hand, 2),
            "loglik_trained": round(ll_hold_fit, 2),
            "info_gain_hand_nats": round((ll_hold_hand - ll_hold_etas) / max(n_hold, 1), 4),
            "info_gain_trained_nats": round((ll_hold_fit - ll_hold_etas) / max(n_hold, 1), 4),
        },
        "reg_strength": reg_strength,
        "degrees_of_freedom_note": dof_note,
        "caveat": f"n_holdout_events={n_hold}, n_train_events={n_train} -- both small, "
                  "log-lik deltas have high variance. Trained-vs-hand-picked comparison on the "
                  "SAME holdout is the fairer read than either number in isolation. Even WITH "
                  "regularization, do not treat this as a validated model -- treat it as a "
                  "diagnostic that F3 needs more events before it says anything trustworthy.",
    }
    deploy_trained = bool(res.success and not underpowered and ll_train_fit > ll_train_hand)
    deployed = trained_cons if deploy_trained else DEFAULT
    with open(os.path.join(OUT, f"consequents_trained_{slug}.json"), "w") as f:
        json.dump({"train_end": str(train_end.date()), "consequents": deployed,
                   "model_source": "trained" if deploy_trained else "expert_default_unvalidated",
                   "skill_validated": False}, f, indent=2)
    with open(os.path.join(OUT, f"f3_training_report_{slug}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
