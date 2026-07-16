"""MOCRE-1 F5b: rolling-origin (expanding window) walk-forward evaluation.

csep_test.py already gives a real paired significance test, but only on ONE train/holdout split
per segment -- e.g. Parkfield's ETAS params are frozen at 2010 and then applied unchanged across
16 years of holdout (2010-2026). That's a single, large-variance replication. Standard
forecast-verification practice (and closer to how CSEP actually operates over years of
submissions) is ROLLING-ORIGIN evaluation: refit the statistical backbone periodically as more
data becomes available, and treat each subsequent period as an independent out-of-sample test.
This multiplies statistical power using the SAME underlying data already ingested -- no new
segments or ingestion needed, which is why it's the highest-leverage next step available.

Design:
  - State (C,A,K) and G's consequents are NOT re-fit per epoch -- they represent the model
    design itself (already walk-forward trained once in F3), not something that needs constant
    recalibration. What DOES go stale and needs refitting is ETAS's own background/triggering
    parameters (mu, K, c, p, alpha) -- exactly the piece a real forecaster would periodically
    update as the catalog grows.
  - Epoch boundaries: EVENT-COUNT-based, not fixed calendar width. First attempt used fixed
    4-year calendar bins and it badly underused the data -- target events cluster in time
    (e.g. Parkfield: 16 of 27 events fall in a single 4-year window around the 2004 sequence,
    leaving most other 4-year bins empty), so a uniform-calendar-time assumption is simply wrong
    for these point processes. Fixed instead: sort events chronologically, group into epochs of
    EVENTS_PER_EPOCH consecutive events -- this guarantees every epoch carries comparable
    statistical power and adapts to each segment's actual event tempo instead of assuming one
    that doesn't hold.
  - For epoch [e_start, e_end): ETAS is refit on ALL data strictly before e_start (expanding
    window, not sliding -- more data always available to later epochs, matching how a real
    operational system would accumulate history), then lambda_ETAS and lambda_MOCRE (raw, not
    renormalized -- see csep_test.py's fix) are evaluated only within the epoch, scored with the
    same Rhoades T-test/W-test machinery per epoch.
  - Epoch p-values combined via Fisher's method (chi2 = -2*sum(ln(p_i)), df=2*n_epochs) for an
    overall verdict, alongside a simple count of "how many epochs individually favored MOCRE-1
    at p<0.05" (a Fisher combination can be significant even if no single epoch is, and vice
    versa -- report both, don't cherry-pick whichever looks better).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT, load_config, segment_paths
from csep_test import comparison_test, load_events
from etas_fit import fit as etas_fit_fn
from pipeline import (build_daily, build_state, etas_fitted, gain_from_weights, load_catalog,
                       load_gnss, rule_weights)

OUT = os.path.join(ROOT, "out")
EVENTS_PER_EPOCH = 5
MIN_HISTORY_YEARS = 5


def run_segment(cfg_path):
    cfg = load_config(os.path.basename(cfg_path))
    slug = segment_paths(cfg)["slug"]
    try:
        cat = load_catalog(cfg)
        stations, gnss = load_gnss(cfg)
    except FileNotFoundError:
        return None
    daily = build_daily(cfg, cat, stations, gnss)
    daily = build_state(cfg, daily)
    weights = rule_weights(daily)

    trained_path = os.path.join(OUT, f"consequents_trained_{slug}.json")
    consequent_train_end = None
    if os.path.exists(trained_path):
        with open(trained_path) as f:
            trained = json.load(f)
            consequents = trained["consequents"]
            consequent_train_end = pd.Timestamp(trained["train_end"])
        cons_kind = "trained"
    else:
        from pipeline import DEFAULT_CONSEQUENTS
        consequents, cons_kind = DEFAULT_CONSEQUENTS, "hand-picked"
    G = gain_from_weights(weights, consequents, cfg)

    events_all = load_events(cfg).sort_values("time").reset_index(drop=True)
    t0 = daily.index[0]
    tN = daily.index[-1]
    first_origin = t0 + pd.DateOffset(years=MIN_HISTORY_YEARS)
    if consequent_train_end is not None:
        first_origin = max(first_origin, consequent_train_end)
    ev_after_origin = events_all[pd.to_datetime(events_all["date"]) >= first_origin].reset_index(drop=True)

    epochs = []
    e_start = first_origin
    i = 0
    while i < len(ev_after_origin):
        chunk = ev_after_origin.iloc[i:i + EVENTS_PER_EPOCH]
        e_end = pd.Timestamp(chunk["date"].max()) + pd.Timedelta(days=1)
        epochs.append((e_start, min(e_end, tN + pd.Timedelta(days=1))))
        e_start = e_end
        i += EVENTS_PER_EPOCH

    print(f"\n{slug}: {len(epochs)} rolling epochs (~{EVENTS_PER_EPOCH} events each, "
          f"event-count-based boundaries), consequents={cons_kind}")
    epoch_results = []
    for e_start, e_end in epochs:
        params = etas_fit_fn(cfg, str(e_start.date()))
        lam_etas, _ = etas_fitted(cfg, cat, daily.index, params)
        lam_mocre_raw = lam_etas * G  # no renormalization -- see csep_test.py's fix

        mask = (daily.index >= e_start) & (daily.index < e_end)
        ep_series = pd.DataFrame({"lam_etas": lam_etas[mask], "lam_mocre": lam_mocre_raw[mask]},
                                 index=daily.index[mask])
        ep_events = events_all[(pd.to_datetime(events_all["date"]) >= e_start) &
                               (pd.to_datetime(events_all["date"]) < e_end)]
        n_ev = len(ep_events)
        if n_ev < 3 or len(ep_series) < 60:
            epoch_results.append({"epoch": [str(e_start.date()), str(e_end.date())],
                                  "n_events": n_ev, "status": "skipped (too few events)"})
            continue
        comp = comparison_test(ep_series, ep_events)
        if comp["status"] != "ok":
            epoch_results.append({"epoch": [str(e_start.date()), str(e_end.date())],
                                  "n_events": n_ev, "status": "skipped (insufficient events)"})
            continue
        t_stat = comp["t_test"]["statistic"]
        t_p = comp["t_test"]["p_value"]
        mean_x = comp["information_gain_per_event_nats"]
        epoch_results.append({
            "epoch": [str(e_start.date()), str(e_end.date())], "n_events": n_ev,
            "status": "ok", "mean_X_per_bin": round(mean_x, 5),
            "t_statistic": round(float(t_stat), 4), "p_value": round(float(t_p), 5),
            "favors_mocre": bool(mean_x > 0),
        })
        print(f"  {e_start.date()}..{e_end.date()}: n_ev={n_ev} mean_X={mean_x:+.4f} "
              f"t_p={t_p:.4f} {'MOCRE+' if mean_x>0 else 'MOCRE-'}")

    valid = [e for e in epoch_results if e["status"] == "ok"]
    if len(valid) < 2:
        return {"segment": slug, "status": f"only {len(valid)} usable epochs, too few to combine",
               "epochs": epoch_results}

    ps = np.array([max(e["p_value"], 1e-10) for e in valid])
    fisher_chi2 = -2 * np.sum(np.log(ps))
    fisher_p = 1 - stats.chi2.cdf(fisher_chi2, df=2 * len(ps))
    n_favor = sum(1 for e in valid if e["favors_mocre"])
    n_favor_sig = sum(1 for e in valid if e["favors_mocre"] and e["p_value"] < 0.05)
    n_against_sig = sum(1 for e in valid if not e["favors_mocre"] and e["p_value"] < 0.05)

    result = {
        "segment": slug, "status": "ok", "consequents": cons_kind, "n_epochs_total": len(epochs),
        "n_epochs_usable": len(valid),
        "n_epochs_favoring_mocre": n_favor, "n_epochs_favoring_etas": len(valid) - n_favor,
        "n_epochs_significant_favoring_mocre_p05": n_favor_sig,
        "n_epochs_significant_favoring_etas_p05": n_against_sig,
        "fisher_combined_chi2": round(float(fisher_chi2), 3),
        "fisher_combined_p_value": round(float(fisher_p), 5),
        "confirmatory": False,
        "interpretation": "exploratorio; épocas dependientes y diseño adaptado retrospectivamente",
        "epochs": epoch_results,
    }
    print(f"  -> Fisher combined p={fisher_p:.5f} over {len(valid)} epochs "
          f"({n_favor} favor MOCRE, {len(valid)-n_favor} favor ETAS)")
    return result


def main():
    all_results = []
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        r = run_segment(cfg_path)
        if r:
            all_results.append(r)
    with open(os.path.join(OUT, "rolling_csep_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved -> {os.path.join(OUT, 'rolling_csep_results.json')}")

    print("\n=== SUMMARY ===")
    for r in all_results:
        if r.get("status") == "ok":
            print(f"{r['segment']}: {r['n_epochs_usable']} epochs, Fisher combined p="
                  f"{r['fisher_combined_p_value']:.4f}, {r['n_epochs_favoring_mocre']}/"
                  f"{r['n_epochs_usable']} favor MOCRE-1")
        else:
            print(f"{r['segment']}: {r.get('status')}")

    print("""
HONEST STATUS:
  - This multiplies test replications using EXISTING data (no new ingestion) -- ETAS is
    refit at each epoch origin using only strictly-prior data (expanding window), G's
    consequents are held fixed (they represent the trained model design, not something
    re-fit per epoch).
  - A segment showing epochs split between favoring MOCRE-1 and favoring ETAS is a REAL,
    honest result (model performance isn't uniform across time) -- do not average that away
    into a single misleading headline number.
  - Still not a true CSEP submission (forecasts registered before evaluation windows close);
    this is the best achievable rigor from historical data alone.
""")


if __name__ == "__main__":
    main()
