"""MOCRE-1 F1: temporal ETAS maximum-likelihood fit on the segment catalog.

Conditional intensity for events M>=Mc (days):
    lambda(t) = mu + sum_{ti<t} K * 10^(alpha*(Mi-Mc)) / (t - ti + c)^p

Fit (mu, K, c, p, alpha) by MLE (Nelder-Mead on log-params). Target-magnitude rate is
obtained by Gutenberg-Richter scaling: lambda_tgt = lambda_Mc * 10^(-b*(Mtgt-Mc)).
Params are fitted on a training window only (walk-forward hygiene); saved to out/.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import ROOT, dist_to_trace_km, load_config, segment_paths

MC = 3.0  # completeness magnitude for the fit


def load_times(cfg, train_end=None):
    df = pd.read_csv(segment_paths(cfg)["catalog_csv"])
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    dist, _ = dist_to_trace_km(df["latitude"].to_numpy(), df["longitude"].to_numpy(), cfg)
    df = df[(dist <= cfg["corridor_half_width_km"]) & (df["mag"] >= MC)].copy()
    df = df.sort_values("time").reset_index(drop=True)
    if train_end is not None:
        df = df[df["time"] < pd.Timestamp(train_end, tz="UTC")]
    t0 = df["time"].iloc[0]
    t_days = (df["time"] - t0).dt.total_seconds().to_numpy() / 86400.0
    return t_days, df["mag"].to_numpy(), t0


def neg_loglik(theta, t, m, T):
    mu, K, c, p, alpha = np.exp(theta)
    w = 10.0 ** (alpha * (m - MC))
    # intensity at each event time (strictly causal)
    dt = t[:, None] - t[None, :]
    mask = dt > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        kern = np.where(mask, 1.0 / np.power(dt + c, p, where=mask), 0.0)
    lam = mu + K * (kern * w[None, :]).sum(axis=1)
    if np.any(lam <= 0) or not np.isfinite(lam).all():
        return 1e12
    # integral of the intensity over [0, T]
    if abs(p - 1.0) < 1e-6:
        integ_i = np.log(T - t + c) - np.log(c)
    else:
        integ_i = ((T - t + c) ** (1 - p) - c ** (1 - p)) / (1 - p)
    integral = mu * T + K * (w * integ_i).sum()
    ll = np.log(lam).sum() - integral
    return -ll


def fit(cfg, train_end):
    t, m, t0 = load_times(cfg, train_end)
    T = t[-1] + 1.0
    n = len(t)
    x0 = np.log([n / T * 0.5, 0.02, 0.02, 1.1, 0.9])  # mu, K, c, p, alpha
    res = minimize(neg_loglik, x0, args=(t, m, T), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-4})
    mu, K, c, p, alpha = np.exp(res.x)
    branching = None
    if p > 1:  # expected offspring per average event (rough diagnostic)
        w_mean = float(np.mean(10.0 ** (alpha * (m - MC))))
        branching = K * w_mean * (c ** (1 - p)) / (p - 1)
    out = {
        "Mc": MC, "n_events_fit": int(n), "train_end": str(train_end),
        "catalog_t0": str(t0), "T_days": float(T),
        "mu": float(mu), "K": float(K), "c": float(c), "p": float(p), "alpha": float(alpha),
        "neg_loglik": float(res.fun), "converged": bool(res.success),
        "branching_ratio_approx": None if branching is None else float(branching),
    }
    return out


def main():
    cfg = load_config()
    slug = segment_paths(cfg)["slug"]
    train_end = cfg.get("etas_lite", {}).get("train_end", "2010-01-01")
    params = fit(cfg, train_end)
    path = os.path.join(ROOT, "out", f"etas_params_{slug}.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(json.dumps(params, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
