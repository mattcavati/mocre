"""MOCRE-1 F0 vertical slice: series -> state -> ETAS-lite -> fuzzy gain -> lambda(t) + scoreboard.

lambda_seg(t) = mu_ETAS(t) * G(state, coincidences; regime)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (ANOMALY_SETS, ROOT, causal_reindex, dist_to_trace_km, fuzzify_anomaly,
                    load_config, poisson_anomaly, robust_anomaly, segment_paths)

OUT = os.path.join(ROOT, "out")


# ---------------------------------------------------------------- inputs
def load_catalog(cfg):
    paths = segment_paths(cfg)
    df = pd.read_csv(paths["catalog_csv"])
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    dist, _ = dist_to_trace_km(df["latitude"].to_numpy(), df["longitude"].to_numpy(), cfg)
    df = df[dist <= cfg["corridor_half_width_km"]].copy()
    df["date"] = df["time"].dt.tz_convert(None).dt.normalize()
    return df.sort_values("time").reset_index(drop=True)


def load_gnss(cfg):
    paths = segment_paths(cfg)
    # GNSS optional (added 2026-07-05 for Vía A global scaling): a segment can have no GNSS
    # ingested (remote region, or ingestion skipped) -- the deformation channels (a_gnss/a_sse/
    # K_state, all derived from D_mm) then stay 0, same honest "channel absent" the model already
    # handles for strain/fluids/etc. Not an error.
    if not os.path.exists(paths["stations_json"]):
        return [], {}
    with open(paths["stations_json"]) as f:
        stations = json.load(f)
    az = np.radians(cfg["gnss"]["fault_azimuth_deg"])
    ux, uy = np.sin(az), np.cos(az)  # fault-parallel unit vector (E, N), NW-positive
    series = {}
    for s in stations:
        path = os.path.join(paths["gnss_dir"], f"{s['sta']}.tenv3")
        cols = ["site", "YYMMMDD", "decyr", "MJD", "week", "d", "reflon",
                "e0", "east", "n0", "north", "u0", "up"]
        df = pd.read_csv(path, sep=r"\s+", usecols=range(13), names=cols, skiprows=1)
        t = pd.to_datetime(df["MJD"] + 2400000.5 - 2440587.5, unit="D").dt.normalize()
        par_mm = (df["east"] * ux + df["north"] * uy) * 1000.0
        series[s["sta"]] = pd.Series(par_mm.to_numpy(), index=t)
    return stations, series


# ---------------------------------------------------------------- daily series
def build_daily(cfg, cat, stations, gnss, as_of=None):
    m2 = cat[cat["mag"] >= cfg["catalog"]["min_magnitude"]]
    counts = m2.groupby("date").size()
    if gnss:
        start = max(counts.index.min(), min(s.index.min() for s in gnss.values()))
    else:
        start = counts.index.min()          # no GNSS -> catalog span drives the window
    # El estado existe también en días sin terremotos; terminar en el último
    # evento congelaba silenciosamente segmentos enteros durante años.
    end = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.utcnow().tz_localize(None).normalize()
    end = max(end, counts.index.max())
    days = pd.date_range(start, end, freq="D")
    daily = pd.DataFrame(index=days)
    daily["n_seis"] = counts.reindex(days).fillna(0.0)

    # Differential fault-parallel motion: mean(SW/Pacific side) - mean(NE side). Absent GNSS ->
    # D_mm=0 (deformation channels stay 0), same for a one-sided network with no pair to difference.
    sides = {s["sta"]: s["side"] for s in stations}
    sw = [causal_reindex(gnss[k], days, limit=30) for k, v in sides.items() if v < 0 and k in gnss]
    ne = [causal_reindex(gnss[k], days, limit=30) for k, v in sides.items() if v > 0 and k in gnss]
    if sw and ne:
        D = pd.concat(sw, axis=1).mean(axis=1) - pd.concat(ne, axis=1).mean(axis=1)
        daily["has_gnss_data"] = D.notna().astype(float)
    else:
        D = pd.Series(0.0, index=days)
        daily["has_gnss_data"] = 0.0
    daily["D_mm"] = D

    # Loading rate: trailing 365-day slope (mm/yr), robust to gaps.
    w = 365
    slope = np.full(len(days), np.nan)
    Dv = D.to_numpy()
    x = np.arange(len(days), dtype=float)
    for i in range(w, len(days)):
        seg = Dv[i - w:i]
        msk = ~np.isnan(seg)
        if msk.sum() > w * 0.6:
            slope[i] = np.polyfit(x[i - w:i][msk], seg[msk], 1)[0] * 365.25
    daily["v_load_mm_yr"] = slope
    return daily


def add_regional_precursors(cfg, daily):
    """Features regionales causales de preparación, fuera del corredor estrecho.

    El target continúa limitado al corredor de falla, pero la preparación puede
    extenderse cientos de kilómetros. Todas las ventanas terminan en t-1.
    """
    path = segment_paths(cfg)["catalog_csv"]
    raw = pd.read_csv(path)
    raw["time"] = pd.to_datetime(raw["time"], utc=True, format="ISO8601")
    raw["date"] = raw.time.dt.tz_convert(None).dt.normalize()
    mc = cfg["catalog"]["min_magnitude"]
    raw = raw[raw.mag >= mc].copy()
    if raw.empty:
        for name in ("a_regional_rate", "a_regional_spread", "a_bvalue_drop",
                     "a_fault_concentration", "a_mag_accel"):
            daily[name] = 0.0
        daily["has_regional_catalog_data"] = 0.0
        return daily
    dist, _ = dist_to_trace_km(raw.latitude.to_numpy(), raw.longitude.to_numpy(), cfg)
    raw["dist_fault"] = dist
    raw["cell"] = (raw.latitude.mul(4).round().astype(str) + "|" +
                   raw.longitude.mul(4).round().astype(str))
    g = raw.groupby("date").agg(n=("mag", "size"), mag_sum=("mag", "sum"),
                                 radial_sum=("dist_fault", "sum"), cells=("cell", "nunique"))
    idx = daily.index
    n = g.n.reindex(idx).fillna(0.0)
    mag_sum = g.mag_sum.reindex(idx).fillna(0.0)
    radial_sum = g.radial_sum.reindex(idx).fillna(0.0)
    cell_days = g.cells.reindex(idx).fillna(0.0)
    # Todas las agregaciones conocidas al inicio de t.
    n30 = n.rolling(30, min_periods=1).sum().shift(1)
    n90 = n.rolling(90, min_periods=1).sum().shift(1)
    n730 = n.rolling(730, min_periods=90).sum().shift(1)
    mag30 = mag_sum.rolling(30, min_periods=1).sum().shift(1) / n30.replace(0, np.nan)
    mag730 = mag_sum.rolling(730, min_periods=90).sum().shift(1) / n730.replace(0, np.nan)
    radial90 = radial_sum.rolling(90, min_periods=1).sum().shift(1) / n90.replace(0, np.nan)
    spread30 = cell_days.rolling(30, min_periods=1).sum().shift(1) / np.sqrt(n30 + 1.0)
    # Gutenberg-Richter MLE a partir de magnitud media; una caída positiva
    # implica relativamente más eventos grandes en la ventana reciente.
    b_recent = np.log10(np.e) / np.maximum(mag30 - mc + 0.05, 0.05)
    b_base = np.log10(np.e) / np.maximum(mag730 - mc + 0.05, 0.05)
    b_drop = (b_base - b_recent).clip(-2, 2)
    st = cfg["state"]
    daily["a_regional_rate"] = poisson_anomaly(n.to_numpy(), st["baseline_window_days"],
                                                 st["tau_seismicity"], obs_window=30)
    daily["a_regional_spread"] = robust_anomaly(spread30.to_numpy(), 730, 2.5, min_periods=90)
    daily["a_bvalue_drop"] = np.tanh(b_drop.fillna(0).to_numpy() / 0.35)
    daily["a_fault_concentration"] = robust_anomaly((-radial90).to_numpy(), 730, 2.5,
                                                      min_periods=90)
    daily["a_mag_accel"] = robust_anomaly((mag30 - mag730).to_numpy(), 730, 2.5,
                                           min_periods=90)
    daily["has_regional_catalog_data"] = (n730.notna() & n730.gt(0)).astype(float)
    return daily


# ---------------------------------------------------------------- state S=(C,A) + anomalies
def build_state(cfg, daily):
    daily = add_regional_precursors(cfg, daily)
    st = cfg["state"]
    # seismicity-rate anomaly. Poisson count anomaly (not median/MAD) since 2026-07-04: robust_
    # anomaly degenerated on seismically sparse segments (Cascadia 98.5% zero-days -> a_seis was
    # 98% saturated at |v|>0.95, a dead binary channel), which would poison a global tessellation
    # where most of GEM's ~13.7k faults are sparse. poisson_anomaly's sqrt(mu) shrinks gracefully
    # instead of collapsing -- verified: Cascadia a_seis saturation 98%->~0, dense segments
    # (Oklahoma/Shumagin) keep only ~1% saturation from REAL extreme swarms/aftershocks. Same
    # ~14-day observation window as before (obs_window), same tau_seismicity=4.0 (the Poisson z
    # scale landed close to the old median/MAD scale, verified before switching).
    daily["a_seis"] = poisson_anomaly(daily["n_seis"].to_numpy(), st["baseline_window_days"],
                                      st["tau_seismicity"], obs_window=14)
    # deformation anomaly: residual of D after removing trailing-year trend
    resid = daily["D_mm"] - daily["D_mm"].rolling(365, min_periods=200).mean().shift(1)
    daily["a_gnss"] = robust_anomaly(resid.to_numpy(), st["baseline_window_days"], st["tau_gnss"])

    # A (agitacion): EWMA of high-anomaly membership
    mu = fuzzify_anomaly(daily["a_seis"].to_numpy())
    drive = np.nan_to_num(np.maximum(mu["alto"], mu["muy_alto"]))
    A = np.zeros(len(daily))
    lam = st["lambda_A_daily"]
    for i in range(1, len(A)):
        A[i] = lam * A[i - 1] + (1 - lam) * drive[i]
    daily["A_state"] = A

    # Enjambre (swarm) and foreshock/acceleration indicators, added 2026-07-03 ("todas las
    # variables" pass). Both are algorithmic features on the catalog we already have -- no new
    # data source. Distinct from A (which just flags "elevated rate", full stop): these
    # distinguish HOW the rate is elevated, which is exactly the mechanism difference the
    # catalog draws between the two archetypes.
    #
    # a_swarm: elevated 30-day rate that is NOT decaying (an Omori-decaying rate after a
    # mainshock has a strongly negative slope; a swarm's defining trait per the catalog is
    # "sostenido... sin decaimiento tipo Omori"). Ratio of 30d to 180d trailing rate (catches
    # "elevated"), gated by the 14-day slope of the 30d rate not being strongly negative
    # (catches "not decaying").
    # Ventanas conocidas al inicio del día: excluyen el recuento del propio día.
    n = daily["n_seis"].shift(1).fillna(0.0).to_numpy()
    r30 = pd.Series(n).rolling(30, min_periods=15).mean().to_numpy()
    r180 = pd.Series(n).rolling(180, min_periods=90).mean().to_numpy()
    # rate-ratio gate (added 2026-07-04, same prerequisite as a_seis's Poisson fix): a ratio of
    # two rates is statistically meaningless when both windows are near-empty. CRUCIAL detail: the
    # gate must be applied as NaN BEFORE robust_anomaly, not to its output -- otherwise the median/
    # MAD baseline is computed over the thousands of low-rate ~0 days and collapses, so the few
    # active days that pass the gate saturate against a degenerate baseline (Cascadia a_foreshock
    # stayed 63% saturated when the gate was applied only to the output). Setting the ratio to NaN
    # on low-base days makes robust_anomaly build its baseline from the active days only.
    enough_base_sw = (r180 * 180.0) >= 5.0
    with np.errstate(divide="ignore", invalid="ignore"):
        swarm_ratio = np.log((r30 + 0.01) / (r180 + 0.01))
    swarm_ratio = np.where(np.nan_to_num(enough_base_sw), swarm_ratio, np.nan)
    slope14 = pd.Series(r30).diff(14).to_numpy() / 14.0
    a_swarm_raw = robust_anomaly(swarm_ratio, st["baseline_window_days"], st.get("tau_swarm", 2.5))
    not_decaying = slope14 >= -0.01  # near-flat or rising 30d rate, not a sharp Omori decline
    daily["a_swarm"] = np.where(np.nan_to_num(not_decaying), np.nan_to_num(a_swarm_raw), 0.0)

    # a_foreshock: short-window (7d) rate accelerating relative to the medium-window (30d)
    # baseline -- the "cascada de nucleacion" signature. Used directly as a rule input (no
    # persistent EWMA state of its own, unlike A/K/F): it's inherently a short-horizon feature,
    # smoothing it into a slow state would blur exactly the short-term signal it's meant to catch.
    r7 = pd.Series(n).rolling(7, min_periods=4).mean().to_numpy()
    # same NaN-before-anomaly gate as a_swarm: a 7d/30d ratio is noise-over-noise when the 30d
    # window is near-empty (Cascadia a_foreshock was 76% saturated, 63% even with an output-only
    # gate). Require >=3 expected events in the 30d window; low-base days become NaN so they don't
    # poison the baseline, and drop out to 0 in the output.
    enough_base_fo = (r30 * 30.0) >= 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fore_ratio = np.log((r7 + 0.01) / (r30 + 0.01))
    fore_ratio = np.where(np.nan_to_num(enough_base_fo), fore_ratio, np.nan)
    a_fore_raw = robust_anomaly(fore_ratio, st["baseline_window_days"], st.get("tau_foreshock", 2.0))
    daily["a_foreshock"] = np.nan_to_num(a_fore_raw)

    # C (carga): slow loading integral anchored at recurrence fraction.
    # NOT clipped to [0,1] here -- a segment can be genuinely "overdue" (C0>1, e.g. Mojave:
    # 169/150=1.13 already on day one) and clipping at the raw-state level saturates it for the
    # entire window, destroying resolution exactly where it matters most (found by inspecting
    # the multi-segment comparison table, 2026-07-03). Only clipped at a generous numerical
    # ceiling; the [0,1]-bounded *gate* used by fuzzy rules is computed separately, downstream
    # (see C_hi in fuzzy_gain/rule_weights) -- raw "how overdue" and "does this count as high
    # for rule-firing purposes" are different questions and should not share one clipped number.
    C0 = st["years_since_last_major"] / st["T_rec_years"]
    v_norm = (daily["v_load_mm_yr"] / st["v_ref_mm_yr"]).clip(0, 3).fillna(1.0)
    dC = v_norm / (st["T_rec_years"] * 365.25)

    # F4: external Delta_CFS from other segments' M>=5 ruptures (stress_graph.py), added to dC
    # per the spec formula (04_modelo_sismico SS3): "w_deltaCFS * max(0, deltaCFS_in)". Positive
    # only -- receiving positive stress consumes part of the recurrence cycle; per the spec's own
    # documented design choice, negative (relieving) transfers do NOT reduce C here. Conversion
    # bar -> "fraction of a recurrence cycle": a typical earthquake stress drop is ~1-10 MPa
    # (10-100 bar) and resets a fault's OWN C to ~0 -- so 1 bar of received external stress is
    # approximated as consuming 1/STRESS_DROP_BAR of a full cycle. STRESS_DROP_BAR=30 (3 MPa) is
    # a representative mid-range value, not segment-specific -- flagged as an approximation.
    STRESS_DROP_BAR = 30.0
    slug = segment_paths(cfg)["slug"]
    cfs_path = os.path.join(OUT, f"cfs_in_{slug}.csv")
    dC_cfs = pd.Series(0.0, index=daily.index)
    if os.path.exists(cfs_path):
        steps = pd.read_csv(cfs_path, index_col=0, parse_dates=True)["delta_cfs_bar"]
        steps = steps.reindex(daily.index, fill_value=0.0).shift(1).fillna(0.0).clip(lower=0.0)
        dC_cfs = steps / STRESS_DROP_BAR
        daily["dC_from_stress_graph"] = dC_cfs  # kept for inspection/plots
    daily["C_state"] = np.clip(C0 + dC.cumsum() + dC_cfs.cumsum(), 0, 3.0)

    # K (acoplamiento / SSE transient): a Slow Slip Event shows as a *temporary* acceleration
    # of fault-parallel motion -- a short-window rate spike that reverts, not a permanent trend
    # change (that's C's job). Detector: 30-day rolling rate of D_mm minus the 365-day trailing
    # rate (v_load_mm_yr), robust-normalized against its own recent variability. Positive-only
    # (SSEs accelerate loading in this geometry; a negative spike is instrument/offset noise,
    # not physically an SSE signature) -- reuses D_mm, no new data ingestion.
    #
    # Gated to zero for regime="induced" (real bug found+fixed 2026-07-04 via CSEP diagnosis on
    # Oklahoma): K is built from the SAME GNSS D_mm series as C, and the config's own v_ref_comment
    # already documents that GNSS-based loading is expected to be flat/uninformative for induced
    # segments (near-zero natural tectonic strain -- the real driver there is fluid pressure, F's
    # job). That caveat was applied to C_state but never to K_state, which stayed fully active and
    # mistook GNSS processing noise (or local well-related deformation, not fault-coupling SSE) for
    # genuine transient signal. Quantitatively confirmed before this fix: K_state correlated -0.34
    # with the paired-test per-bin miss X (F_state correlated ~0, essentially nothing) and the worst
    # CSEP quartile for Oklahoma had K_mean=0.35 (2-3x every other quartile) while F_mean=0.0 exactly
    # -- K was inflating G on quiet months with no real hazard, not F. Same architectural lesson as
    # F's own regime gate: a channel built for one physical mechanism must not run unconditionally
    # in a regime the project already knows it doesn't apply to.
    if cfg.get("regime") != "induced":
        w30 = 30
        Dv = daily["D_mm"].to_numpy()
        x = np.arange(len(daily), dtype=float)
        v30 = np.full(len(daily), np.nan)
        for i in range(w30, len(daily)):
            seg = Dv[i - w30:i]
            msk = ~np.isnan(seg)
            if msk.sum() > w30 * 0.6:
                v30[i] = np.polyfit(x[i - w30:i][msk], seg[msk], 1)[0] * 365.25
        v_excess = v30 - daily["v_load_mm_yr"].to_numpy()  # short-term rate minus long-term trend
        a_sse_raw = robust_anomaly(v_excess, st["baseline_window_days"], st.get("tau_sse", 3.0))
        a_sse = np.where(np.nan_to_num(v_excess) > 0, a_sse_raw, 0.0)  # positive-only, see above
        daily["a_sse"] = a_sse
        mu_k = fuzzify_anomaly(a_sse)
        drive_k = np.nan_to_num(np.maximum(mu_k["alto"], mu_k["muy_alto"]))
        K = np.zeros(len(daily))
        lam_k = st.get("lambda_K_daily", 0.90)  # faster decay than A: SSEs are transient by definition
        for i in range(1, len(K)):
            K[i] = lam_k * K[i - 1] + (1 - lam_k) * drive_k[i]
        daily["K_state"] = K
    else:
        daily["a_sse"] = np.zeros(len(daily))
        daily["K_state"] = np.zeros(len(daily))

    # E (enjambre): EWMA of high a_swarm membership, same pattern as A/K. Slower decay than K
    # (SSE, days-weeks) since swarms persist weeks-months by definition.
    mu_sw = fuzzify_anomaly(daily["a_swarm"].to_numpy())
    drive_e = np.nan_to_num(np.maximum(mu_sw["alto"], mu_sw["muy_alto"]))
    E = np.zeros(len(daily))
    lam_e = st.get("lambda_E_daily", 0.97)
    for i in range(1, len(E)):
        E[i] = lam_e * E[i - 1] + (1 - lam_e) * drive_e[i]
    daily["E_state"] = E

    # F (facilitacion / presion de fluidos): the channel that was a placeholder since F0,
    # activated 2026-07-03 after the catalog audit found real, open, monthly injection data
    # (Oklahoma Corporation Commission UIC wells -- mu_ev=0.90, the single highest-confidence
    # precursor in the whole 25-variable catalog). Only populated for segments with a real
    # wells/ dataset (regime="induced" so far); all other segments get F=0 identically, which is
    # honest -- there is no fluid-injection mechanism to represent for natural tectonic segments,
    # not a missing feature.
    paths = segment_paths(cfg)
    wells_path = os.path.join(ROOT, "data", "wells", paths["slug"], "monthly_aggregate.csv")
    F = np.zeros(len(daily))
    daily["a_fluid"] = np.zeros(len(daily))
    daily["has_fluid_data"] = np.zeros(len(daily))
    if os.path.exists(wells_path):
        monthly = pd.read_csv(wells_path, index_col=0, parse_dates=True)
        # El libro OCC es anual y no hay snapshots que prueben disponibilidad
        # mes a mes. Se usa available_at; artefactos antiguos reciben el mismo
        # supuesto conservador (15-feb del año siguiente).
        if "available_at" in monthly:
            available = pd.to_datetime(monthly["available_at"])
        else:
            available = pd.DatetimeIndex([pd.Timestamp(year=d.year + 1, month=2, day=15)
                                          for d in monthly.index])
        released = pd.Series(monthly["total_vol_bbls"].to_numpy(), index=available)
        # Al liberarse un libro anual, agregamos sus meses como observaciones
        # conocidas en esa fecha; no retroinyectamos información al año previo.
        released = released.groupby(level=0).last()
        vol_daily = causal_reindex(released, daily.index)
        daily["has_fluid_data"] = vol_daily.notna().astype(float)
        a_fluid = robust_anomaly(vol_daily.to_numpy(), st["baseline_window_days"],
                                 st.get("tau_fluid", 3.0))
        daily["a_fluid"] = a_fluid
        mu_f = fuzzify_anomaly(a_fluid)
        drive_f = np.nan_to_num(np.maximum(mu_f["alto"], mu_f["muy_alto"]))
        lam_f = st.get("lambda_F_daily", 0.93)  # weeks-months decay, pore-pressure diffusion timescale
        for i in range(1, len(F)):
            F[i] = lam_f * F[i - 1] + (1 - lam_f) * drive_f[i]
    daily["F_state"] = F

    # Nivel de aguas subterraneas (groundwater level), mu_ev=0.35 per the catalog -- "mecanismo
    # real (respuesta poroelastica a cambios de esfuerzo) pero debil e inconsistente como
    # precursor en solitario", and confounded with drought/pumping effects the catalog itself
    # doesn't try to separate out. Deliberately NOT given its own persistent EWMA state (unlike
    # C/A/K/F/E) -- treated as an instantaneous z-score input feeding low-weight coincidence
    # rules directly, matching its low confidence rating; a whole new dynamical state variable
    # would overstate how much this channel is trusted. Symmetric bajo/alto from the START this
    # time (both directions are physically meaningful -- deep water table OR shallow water table
    # can each indicate a poroelastic stress change, direction depends on the local mechanism --
    # and this project has now hit the one-sided-rule bug three times in one session for skipping
    # this the first time a variable gets added).
    water_path = os.path.join(ROOT, "data", "groundwater", paths["slug"], "daily_aggregate.csv")
    daily["a_water"] = np.zeros(len(daily))
    daily["has_water_data"] = np.zeros(len(daily))
    if os.path.exists(water_path):
        wdf = pd.read_csv(water_path, index_col=0, parse_dates=True)
        if len(wdf) > 30:
            z_daily = causal_reindex(wdf["mean_z_depth"], daily.index, limit=60)
            daily["has_water_data"] = z_daily.notna().astype(float)
            daily["a_water"] = robust_anomaly(z_daily.to_numpy(), st["baseline_window_days"],
                                              st.get("tau_water", 3.0))

    # Borehole strain (inclinometros/extensometros, mu_ev 0.60->0.65-0.70 per the 3-jul audit --
    # real PBO/EarthScope Gladwin Tensor Strainmeter network, Level-2 processed areal strain
    # (dilatation), see ingest_strain.py. Same treatment as a_water: no persistent EWMA state of
    # its own (an instantaneous z-score feeding rules directly) -- but higher catalog confidence
    # than water, so its consequent magnitudes sit in the SSE tier, not water's near-zero tier
    # (see DEFAULT_CONSEQUENTS). Symmetric bajo/alto from the start (same lesson as everywhere
    # else: dilatation and contraction are BOTH physically real precursor candidates -- dilatancy-
    # diffusion models predict expansion, poroelastic loading can predict contraction -- and the
    # catalog does not establish which direction should dominate, so asserting one is stronger
    # would fabricate precision). Only present at all for segments with real PB network coverage
    # (SJC-Anza, Parkfield, Cascadia, Oklahoma) -- SAF-Mojave/Shumagin get a_strain=0 identically,
    # a true network gap, not a missing feature.
    strain_path = os.path.join(ROOT, "data", "strain", paths["slug"], "daily_aggregate.csv")
    daily["a_strain"] = np.zeros(len(daily))
    daily["has_strain_data"] = np.zeros(len(daily))
    if os.path.exists(strain_path):
        sdf = pd.read_csv(strain_path, index_col=0, parse_dates=True)
        if len(sdf) > 30:
            z_daily = causal_reindex(sdf["mean_z_areal"], daily.index, limit=14)
            daily["has_strain_data"] = z_daily.notna().astype(float)
            daily["a_strain"] = robust_anomaly(z_daily.to_numpy(), st["baseline_window_days"],
                                               st.get("tau_strain", 3.0))

    # Non-tidal ocean loading (Up-component GNSS displacement, ECCO2 model via EOST/ITES, see
    # ingest_ocean_load.py) -- a genuinely NEW variable added 2026-07-04, NOT part of the audited
    # 25-variable catalog. Same instantaneous-z treatment as water/strain (no dedicated EWMA
    # state). Confidence tier deliberately kept LOW (near water's, see DEFAULT_CONSEQUENTS) --
    # unlike strain/fluids this mechanism has not been through this project's own audit process,
    # only a today's-session literature/data-access check. Stale past 2024-12-31 (EOST's real
    # data currency, same order of staleness already accepted for Oklahoma's wells data) -- the
    # anomaly for any date after that is computed against a frozen tail, an honest limitation, not
    # hidden. Symmetric bajo/alto from the start (same repeated lesson).
    ocean_path = os.path.join(ROOT, "data", "ocean_load", paths["slug"], "daily_aggregate.csv")
    daily["a_ocean"] = np.zeros(len(daily))
    daily["has_ocean_data"] = np.zeros(len(daily))
    if os.path.exists(ocean_path):
        odf = pd.read_csv(ocean_path, index_col=0, parse_dates=True)
        if len(odf) > 30:
            z_daily = causal_reindex(odf["mean_z_up"], daily.index, limit=14)
            daily["has_ocean_data"] = z_daily.notna().astype(float)
            daily["a_ocean"] = robust_anomaly(z_daily.to_numpy(), st["baseline_window_days"],
                                              st.get("tau_ocean", 3.0))

    # InSAR (Sentinel-1 COMET-LiCS, short-baseline stacked LOS rate, see ingest_insar.py) --
    # LOWEST confidence tier of any channel in this model, deliberately. Real, hard-won finding
    # from this session: single short-baseline interferogram-pair rate estimates are atmosphere-
    # noise-dominated (tested on real Parkfield data -- per-pair rates implied >1 m/year LOS
    # motion, impossible for the San Andreas). Two corrections applied and verified real
    # (planar-ramp removal for the per-interferogram unwrapping-constant + orbital/atmospheric
    # trend, then stacking ~20 independent pairs per ~30-day bin) bring the residual noise down
    # to ~sub-mm/day, still ~10x the real expected creep signal (~0.08mm/day) -- a genuine, known
    # limitation of this approach vs a proper SBAS network inversion (not attempted this session,
    # see ingest_insar.py docstring for why: GB-scale downloads + real network-inversion code, a
    # separate project). Included anyway because the existing robust_anomaly() z-score is relative
    # to this channel's OWN recent baseline, not an absolute physical threshold -- a stationary
    # noise floor can still let genuine large outliers surface, at reduced sensitivity. Not part
    # of the audited 25-variable catalog. Only present for segments with real LiCSAR frame
    # coverage (SJC-Anza, Parkfield, Mojave, Cascadia, Shumagin) -- Oklahoma gets a_insar=0
    # identically, a confirmed real network gap (no LiCSAR frame near 36N/-97W), not a bug.
    insar_path = os.path.join(ROOT, "data", "insar", paths["slug"], "daily_aggregate.csv")
    daily["a_insar"] = np.zeros(len(daily))
    daily["has_insar_data"] = np.zeros(len(daily))
    if os.path.exists(insar_path):
        idf = pd.read_csv(insar_path, index_col=0, parse_dates=True)
        if len(idf) > 5:
            # Real bug found+fixed 2026-07-04 (caught by Matt asking "are all universes of
            # discourse ready", not by routine testing): with interpolate(limit=20) and the
            # default robust_anomaly() min_periods=182, this channel was SILENTLY 100% NaN in
            # every segment -- ~30-day-spaced stacked bins can't satisfy either the 20-day
            # interpolation gap limit or a 182-real-sample density requirement. Verified with a
            # synthetic reproduction (see common.py's robust_anomaly docstring) before fixing.
            # limit=35 bridges the real ~30-day bin spacing; min_periods=5 accepts this channel's
            # real, much sparser sample density instead of assuming daily-dense coverage.
            z_daily = causal_reindex(idf["los_rate_mm_day"], daily.index, limit=35)
            daily["has_insar_data"] = z_daily.notna().astype(float)
            daily["a_insar"] = robust_anomaly(z_daily.to_numpy(), st["baseline_window_days"],
                                              st.get("tau_insar", 3.0), min_periods=5)

    # TEC ionosferico (JPL GIM, see ingest_tec.py) -- WAS on the original 25-variable audited
    # catalog (3-jul) but marked blocked (CDDIS/NASA Earthdata account required); reopened this
    # session after finding the open JPL sideshow mirror. Data itself is clean and real (daily
    # global grid, verified current to yesterday) -- the uncertainty here is NOT data quality like
    # InSAR, it's that ionospheric earthquake precursors are one of the most scientifically
    # disputed precursor mechanisms in the literature (many studies report no reproducible
    # correlation) -- confidence tier set at water's level (0.35-ish) for that reason: real
    # measurement, contested mechanism, same epistemic posture as a confounded-but-real signal.
    # Global 2.5x5 degree grid -- segments closer together than that share nearly the same cell,
    # an honest resolution limit, not a bug. Symmetric bajo/alto from the start (same lesson).
    tec_path = os.path.join(ROOT, "data", "tec", paths["slug"], "daily_aggregate.csv")
    daily["a_tec"] = np.zeros(len(daily))
    daily["has_tec_data"] = np.zeros(len(daily))
    if os.path.exists(tec_path):
        tdf = pd.read_csv(tec_path, index_col=0, parse_dates=True)
        if len(tdf) > 30:
            z_daily = causal_reindex(tdf["mean_vtec"], daily.index, limit=14)
            daily["has_tec_data"] = z_daily.notna().astype(float)
            daily["a_tec"] = robust_anomaly(z_daily.to_numpy(), st["baseline_window_days"],
                                            st.get("tau_tec", 3.0))

    # Ruido sísmico ambiental (EarthScope MUSTANG sample_rms, see ingest_noise.py) -- catalog
    # variable mu_ev=0.35, un-blocked 4-jul (the 3-jul "MUSTANG doesn't cover CI" was a bad-metric-
    # name error, not a real gap). LOW confidence tier (water), for a mechanism-proxy reason, not
    # data quality: this is the noise LEVEL (log10 daily RMS), a proxy -- the real precursor is
    # dv/v (velocity change from ambient-noise cross-correlation), a fase-2 pipeline not built.
    # NO coincidence rule with seismicity: sample_rms spikes on earthquake days, so a coincid_seis
    # rule would be spurious self-confirmation (same reason InSAR has none). Symmetric bajo/alto.
    noise_path = os.path.join(ROOT, "data", "noise", paths["slug"], "daily_aggregate.csv")
    daily["a_noise"] = np.zeros(len(daily))
    daily["has_noise_data"] = np.zeros(len(daily))
    if os.path.exists(noise_path):
        ndf = pd.read_csv(noise_path, index_col=0, parse_dates=True)
        if len(ndf) > 30:
            z_daily = causal_reindex(ndf["log_rms"], daily.index, limit=14)
            daily["has_noise_data"] = z_daily.notna().astype(float)
            daily["a_noise"] = robust_anomaly(z_daily.to_numpy(), st["baseline_window_days"],
                                              st.get("tau_noise", 3.0))
    return daily


# ---------------------------------------------------------------- ETAS (F1: fitted MLE params)
def etas_fitted(cfg, cat, days, params):
    """Daily intensity for M>=target from MLE-fitted temporal ETAS at Mc, GR-scaled.

    Params are frozen at train_end; intensity at t uses all past events (causal)."""
    e = cfg["etas_lite"]
    mc = params["Mc"]
    t0 = days[0]
    tdays = (days - t0).days.to_numpy(dtype=float)  # intensidad al inicio del día
    par = cat[cat["mag"] >= mc]
    pt = (par["time"].dt.tz_convert(None) - t0).dt.total_seconds().to_numpy() / 86400.0
    w = 10.0 ** (params["alpha"] * (par["mag"].to_numpy() - mc))
    mu, K, c, p = params["mu"], params["K"], params["c"], params["p"]
    lam = np.full(len(tdays), mu)
    chunk = 500
    for i in range(0, len(tdays), chunk):
        dt = tdays[i:i + chunk][:, None] - pt[None, :]
        contrib = np.where(dt > 0, w[None, :] / np.power(np.abs(dt) + c, p), 0.0)
        lam[i:i + chunk] += K * contrib.sum(axis=1)
    gr = 10.0 ** (-e["b_value"] * (e["target_mag"] - mc))
    meta = dict(mu_bg=mu * gr, K0=K, n_obs=float("nan"),
                n_bg_expected=mu * gr * len(tdays))
    return lam * gr, meta


# ---------------------------------------------------------------- ETAS-lite baseline (fallback)
def etas_lite(cfg, cat, days):
    e = cfg["etas_lite"]
    tgt, b = e["target_mag"], e["b_value"]
    t0 = days[0]
    tdays = (days - t0).days.to_numpy(dtype=float)

    # crude background: M>=ref declustered by removing 30d/30km windows after M>=5
    ref = cat[cat["mag"] >= e["decluster_ref_mag"]].copy()
    big = cat[cat["mag"] >= 5.0]
    keep = np.ones(len(ref), dtype=bool)
    for _, bg in big.iterrows():
        dt = (ref["time"] - bg["time"]).dt.total_seconds() / 86400.0
        dx, _ = dist_to_trace_km(ref["latitude"].to_numpy(), ref["longitude"].to_numpy(), cfg)
        near = (np.hypot((ref["latitude"] - bg["latitude"]) * 110.6,
                         (ref["longitude"] - bg["longitude"]) * 92.6) < 30.0)
        keep &= ~(((dt > 0) & (dt < 30)) & near.to_numpy())
    span_days = (cat["time"].max() - cat["time"].min()).days
    mu_ref = keep.sum() / span_days
    mu_bg = mu_ref * 10 ** (-b * (tgt - e["decluster_ref_mag"]))  # GR scaling

    # Omori-triggered part from M>=trigger_min_mag parents
    par = cat[cat["mag"] >= e["trigger_min_mag"]]
    pt = (par["time"].dt.tz_convert(None) - t0).dt.total_seconds().to_numpy() / 86400.0
    pm = par["mag"].to_numpy()
    c, p, alpha = e["omori_c_days"], e["omori_p"], e["alpha"]
    weights = 10 ** (alpha * (pm - e["trigger_min_mag"]))

    lam_trig_unit = np.zeros(len(tdays))
    chunk = 500
    for i in range(0, len(tdays), chunk):
        T = tdays[i:i + chunk][:, None]
        dt = T - pt[None, :]
        contrib = np.zeros_like(dt, dtype=float)
        past = dt > 0
        contrib[past] = np.broadcast_to(weights[None, :], dt.shape)[past] / np.power(dt[past] + c, p)
        lam_trig_unit[i:i + chunk] = contrib.sum(axis=1)

    # calibrate K0 so total expected M>=tgt equals observed count in the window
    n_obs = int(((cat["mag"] >= tgt) & (cat["date"] >= days[0])).sum())
    total_trig_unit = lam_trig_unit.sum()
    n_bg = mu_bg * len(tdays)
    K0 = max(n_obs - n_bg, 0.05 * max(n_obs, 1)) / max(total_trig_unit, 1e-9)
    lam = mu_bg + K0 * lam_trig_unit
    return lam, dict(mu_bg=mu_bg, K0=K0, n_obs=n_obs, n_bg_expected=n_bg)


# ---------------------------------------------------------------- fuzzy gain G
# Rule names, in the fixed order used everywhere (pipeline default consequents + F3 training).
# BUG FOUND AND FIXED 2026-07-03: every original consequent was >=0 (by construction, softplus
# in train_consequents.py forced non-negativity), so the weighted average num/den -- a convex
# combination of nonnegative numbers -- could NEVER be negative. G was structurally one-sided:
# it could only ever elevate the rate above ETAS baseline, never suppress it below. Confirmed
# empirically: G was >=1.0 on 100% of holdout days for every segment, 0% at exactly 1.0. This
# silently inflated the info_gain numbers reported throughout F0-F4 (the global budget
# renormalization was, in retrospect, band-aiding this asymmetry, not a neutral convenience) and
# was only surfaced by csep_test.py's T-test/W-test DISAGREEING -- Wilcoxon (rank/sign-based)
# flagged a systematic effect the parametric T-test's variance masked, which is exactly the kind
# of discrepancy worth chasing rather than picking whichever test looks better.
# FIX: added seis_bajo/seis_muy_bajo (mirror of seis_alto/seis_muy_alto, negative consequents) --
# the low-anomaly end of the partition was simply missing a rule, not a design contradiction.
# Monotonicity (R4: "more anomaly never lowers gain") is preserved as an ORDERED CHAIN across the
# full spectrum: c(muy_bajo) <= c(bajo) <= c(normal)=0 <= c(alto) <= c(muy_alto) -- not "all
# consequents >=0", which was the actual bug. See train_consequents.py's `unpack()` for how this
# chain is now enforced during F3 training.
RULE_NAMES = ["anchor_no_evidence", "seis_normal", "seis_bajo", "seis_muy_bajo",
              "seis_alto", "seis_muy_alto", "coincid_seis_gnss", "seis_x_carga",
              "agitacion_x_carga", "sse_solo", "sse_x_seis",
              "fluid_bajo", "fluid_muy_bajo", "fluid_solo", "fluid_x_seis",
              "enjambre_bajo", "enjambre_solo", "foreshock_calmo", "foreshock_agudo",
              "agua_bajo", "agua_alto",
              "strain_bajo", "strain_alto", "coincid_seis_strain",
              "ocean_bajo", "ocean_alto", "coincid_seis_ocean",
              "insar_bajo", "insar_alto",
              "tec_bajo", "tec_alto",
              "noise_bajo", "noise_alto"]
TRAINABLE_RULES = RULE_NAMES[2:]  # exclude the two anchors fixed at 0.0
DEFAULT_CONSEQUENTS = {
    "seis_bajo": -0.30, "seis_muy_bajo": -0.60,
    "seis_alto": 0.30, "seis_muy_alto": 0.60, "coincid_seis_gnss": 1.00,
    "seis_x_carga": 0.80, "agitacion_x_carga": 0.50, "sse_solo": 0.40, "sse_x_seis": 0.85,
    # fluid_* consequents set higher than their sse_* analogues: mu_ev=0.90 (fluid pressure) is
    # the single highest-confidence causal mechanism in the whole 25-variable catalog, vs 0.60
    # for SSE -- the consequent hierarchy should reflect that, not just mirror SSE's numbers.
    # fluid_bajo/fluid_muy_bajo added 2026-07-03 after the Oklahoma run showed G systematically
    # inflated (+34% vs ETAS in holdout) -- fluid_solo/fluid_x_seis were the SAME one-sided-rule
    # bug already fixed once for the seismicity ladder (seis_bajo/seis_muy_bajo), just
    # reintroduced for a new variable. Oklahoma's real post-2016 regulatory-driven injection
    # decline is exactly the period this was missing a "low injection -> lower gain" response.
    "fluid_bajo": -0.35, "fluid_muy_bajo": -0.70,
    "fluid_solo": 0.60, "fluid_x_seis": 1.10,
    # enjambre_solo/foreshock_agudo: FIRST version (2026-07-03) was deliberately left one-sided
    # on the reasoning that "no swarm/no acceleration" is mere absence of evidence, not a
    # distinct negative-hazard state -- WRONG, disproven within the hour by the same segments
    # that caught the fluid-pressure version of this bug: Parkfield's G ratio hit 1.8x, Oklahoma
    # 1.36x, both flipped from "not distinguishable" to "significantly worse" on CSEP re-test.
    # The reasoning wasn't really about scientific justification for a negative-hazard claim --
    # it's structural: these features fire on a continuous ratio that sits below baseline just
    # as often as above (swarm_ratio, fore_ratio both oscillate around their own baseline), and
    # any one-sided rule with real firing frequency shifts the weighted average up regardless of
    # what the "story" for the missing negative side would be. Fixed the same way, third time:
    # symmetric mirror rules, consequents smaller in magnitude than the positive side (weaker
    # literature backing for "quiescence signals reduced hazard" than for fluid/seismicity-level,
    # so the asymmetric MAGNITUDE is a real, considered choice; asymmetric SIGN COVERAGE, which
    # is what caused the bug twice, is not).
    "enjambre_solo": 0.45, "enjambre_bajo": -0.20,
    "foreshock_agudo": 0.55, "foreshock_calmo": -0.25,
    # agua_bajo/agua_alto: mu_ev=0.35, weakest-confidence active channel by design -- both
    # directions given equal, small magnitude (deep or shallow water table are each only weakly
    # informative per the catalog, and the catalog doesn't establish which direction should
    # matter more, so asserting one is stronger than the other would be fabricated precision).
    "agua_bajo": -0.15, "agua_alto": 0.15,
    # strain_bajo/strain_alto: mu_ev 0.60->0.65-0.70, meaningfully higher-confidence than water
    # (0.35) -- real geodetic strain measurement vs a confounded (drought/pumping) proxy -- so
    # magnitude sits near the sse_solo tier (0.40) rather than water's near-zero tier, still
    # symmetric since direction isn't established. coincid_seis_strain mirrors coincid_seis_gnss
    # (both are cross-cluster B<->A confirmations of the same architectural pattern: coincidence
    # across independently-measured clusters suppresses false alarms quadratically per the spec) --
    # set slightly below coincid_seis_gnss (1.00) since the GNSS coincidence rule has been running
    # and validated across all 6 segments since F0/F1, this one is new and unproven.
    "strain_bajo": -0.35, "strain_alto": 0.35, "coincid_seis_strain": 0.90,
    # ocean_bajo/ocean_alto: NEW variable (4-jul-2026), not in the audited 25-variable catalog --
    # deliberately kept at water's near-zero tier (0.35), the WEAKEST active channel, not strain's
    # (0.60-0.70). Reasoning: strain earned its higher tier through this project's own catalog
    # audit (mu_ev sourced, cross-checked against real endpoints); ocean loading only has today's
    # data-access verification plus published literature Matt/this session found, not the same
    # audit rigor -- treat it as unproven until it goes through that process, same epistemic
    # posture as the catalog's own unaudited LST placeholder (mu_ev=0.20). coincid_seis_ocean set
    # below coincid_seis_strain (0.90) for the same reason -- newest, least track record of the
    # three B-cluster coincidence rules.
    "ocean_bajo": -0.15, "ocean_alto": 0.15, "coincid_seis_ocean": 0.70,
    # insar_bajo/insar_alto: LOWEST-confidence active channel in the whole model, smaller in
    # magnitude than even water's near-zero tier (0.35->0.15) -- real, documented reason: the
    # stacked short-baseline LOS rate still carries ~10x the real tectonic-rate signal in
    # atmosphere-driven noise even after ramp removal + 20-pair stacking (see pipeline.py's
    # build_state comment and ingest_insar.py). No coincid_seis_insar rule -- deliberately not
    # trusted enough yet to serve as a cross-cluster confirmation signal.
    "insar_bajo": -0.08, "insar_alto": 0.08,
    # tec_bajo/tec_alto: real data, contested mechanism (see build_state comment) -- water's tier.
    "tec_bajo": -0.15, "tec_alto": 0.15,
    # noise_bajo/noise_alto: ambient noise LEVEL proxy (not dv/v), water tier -- see build_state.
    "noise_bajo": -0.15, "noise_alto": 0.15,
}


def rule_weights(daily):
    """Firing strength w_r(t) for each rule, independent of consequents (features only)."""
    mu_s = fuzzify_anomaly(daily["a_seis"].to_numpy())
    mu_g = fuzzify_anomaly(daily["a_gnss"].to_numpy())
    C = daily["C_state"].to_numpy()
    A = daily["A_state"].to_numpy()
    K = daily["K_state"].to_numpy()
    F = daily["F_state"].to_numpy() if "F_state" in daily else np.zeros(len(C))
    hi_s = np.nan_to_num(np.maximum(mu_s["alto"], mu_s["muy_alto"]))
    hi_g = np.nan_to_num(np.maximum(mu_g["alto"], mu_g["muy_alto"]))
    C_hi = np.clip((C - 0.6) / 0.3, 0, 1)
    mu_f = fuzzify_anomaly(daily["a_fluid"].to_numpy()) if "a_fluid" in daily else \
        {k: np.zeros(len(C)) for k in ["bajo", "muy_bajo"]}
    return {
        "anchor_no_evidence": np.full(len(C), 0.5),
        "seis_normal": np.nan_to_num(mu_s["normal"]),
        "seis_bajo": np.nan_to_num(mu_s["bajo"]),
        "seis_muy_bajo": np.nan_to_num(mu_s["muy_bajo"]),
        "seis_alto": np.nan_to_num(mu_s["alto"]),
        "seis_muy_alto": np.nan_to_num(mu_s["muy_alto"]),
        "coincid_seis_gnss": np.minimum(hi_s, hi_g),
        "seis_x_carga": np.minimum(hi_s, C_hi),
        "agitacion_x_carga": np.minimum(A, C_hi),
        "sse_solo": K,
        "sse_x_seis": np.minimum(K, hi_s),
        "fluid_bajo": np.nan_to_num(mu_f["bajo"]),
        "fluid_muy_bajo": np.nan_to_num(mu_f["muy_bajo"]),
        "fluid_solo": F,
        "fluid_x_seis": np.minimum(F, hi_s),
        "enjambre_bajo": _sided_weight(daily, "a_swarm", "bajo", C),
        "enjambre_solo": daily["E_state"].to_numpy() if "E_state" in daily else np.zeros(len(C)),
        "foreshock_calmo": _sided_weight(daily, "a_foreshock", "bajo", C),
        "foreshock_agudo": _foreshock_weight(daily, C),
        "agua_bajo": _sided_weight(daily, "a_water", "bajo", C),
        "agua_alto": _sided_weight(daily, "a_water", "alto", C),
        "strain_bajo": _sided_weight(daily, "a_strain", "bajo", C),
        "strain_alto": _sided_weight(daily, "a_strain", "alto", C),
        "coincid_seis_strain": np.minimum(hi_s, _hi_membership(daily, "a_strain", C)),
        "ocean_bajo": _sided_weight(daily, "a_ocean", "bajo", C),
        "ocean_alto": _sided_weight(daily, "a_ocean", "alto", C),
        "coincid_seis_ocean": np.minimum(hi_s, _hi_membership(daily, "a_ocean", C)),
        "insar_bajo": _sided_weight(daily, "a_insar", "bajo", C),
        "insar_alto": _sided_weight(daily, "a_insar", "alto", C),
        "tec_bajo": _sided_weight(daily, "a_tec", "bajo", C),
        "tec_alto": _sided_weight(daily, "a_tec", "alto", C),
        "noise_bajo": _sided_weight(daily, "a_noise", "bajo", C),
        "noise_alto": _sided_weight(daily, "a_noise", "alto", C),
    }


def _hi_membership(daily, col, C):
    """max(alto, muy_alto) membership of an anomaly column, 0 if absent -- used for cross-cluster
    coincidence rules (e.g. coincid_seis_strain), same contract as hi_s/hi_g in rule_weights."""
    if col not in daily:
        return np.zeros(len(C))
    mu = fuzzify_anomaly(daily[col].to_numpy())
    return np.nan_to_num(np.maximum(mu["alto"], mu["muy_alto"]))


def _sided_weight(daily, col, side, C):
    """side='bajo' -> max(bajo,muy_bajo) membership of the named anomaly column, or zeros if
    the column isn't present in this segment's daily frame."""
    if col not in daily:
        return np.zeros(len(C))
    mu = fuzzify_anomaly(daily[col].to_numpy())
    if side == "bajo":
        return np.nan_to_num(np.maximum(mu["bajo"], mu["muy_bajo"]))
    return np.nan_to_num(np.maximum(mu["alto"], mu["muy_alto"]))


def _foreshock_weight(daily, C):
    if "a_foreshock" not in daily:
        return np.zeros(len(C))
    mu_fs = fuzzify_anomaly(daily["a_foreshock"].to_numpy())
    return np.nan_to_num(np.maximum(mu_fs["alto"], mu_fs["muy_alto"]))


def gain_from_weights(weights, consequents, cfg):
    """Sugeno-0 in log10 space given precomputed rule weights + a consequent dict.

    OR of low-arity clauses; min() only for cross-cluster coincidence (already baked into
    `rule_weights`). consequents dict may omit the 2 fixed anchors (always 0.0), AND may be a
    stale cached consequents_trained_*.json missing rules added after it was saved (this project
    keeps adding rule variables faster than every segment gets retrained) -- fall back to the
    current hand-picked default for any missing rule rather than crashing (KeyError) or silently
    zeroing it (0.0 would mean "never fires", a bigger behavior change than "use the untrained
    default"). Retraining stale files is still the right eventual fix; this just stops one
    missing key from taking down a whole segment run in the meantime."""
    c = {**DEFAULT_CONSEQUENTS, "anchor_no_evidence": 0.0, "seis_normal": 0.0, **consequents}
    num = sum(weights[r] * c[r] for r in RULE_NAMES)
    den = sum(weights[r] for r in RULE_NAMES) + 1e-9
    log10G = np.clip(num / den, cfg["gain"]["log10_min"], cfg["gain"]["log10_max"])
    return 10 ** log10G


def fuzzy_gain(cfg, daily, consequents=None):
    """Convenience wrapper: default hand-picked consequents unless a trained set is given."""
    return gain_from_weights(rule_weights(daily), consequents or DEFAULT_CONSEQUENTS, cfg)


# ---------------------------------------------------------------- scoreboard
def loglik(lam_daily, event_idx):
    """Point-process log-likelihood on the daily grid: sum log(lam) at events - integral."""
    lam = np.clip(lam_daily, 1e-9, None)
    return float(np.log(lam[event_idx]).sum() - lam.sum())


def active_channels(cfg, paths):
    """Which physical channels have REAL data wired in for this segment, computed from the same
    file-existence + regime checks build_state() uses -- not a hardcoded list. Fixed 2026-07-04:
    the list was static since F0 and silently never grew to mention fluid/water/strain/ocean as
    each was added over 3-jul/4-jul (cosmetic bug, did not affect any actual computation -- G/
    lambda_mocre never read this list, only the reported summary JSON did)."""
    active, missing = ["seismicity(A-cluster)", "deformation(B-cluster)"], []
    (active if cfg.get("regime") != "induced" else missing).append("SSE-coupling(K, from GNSS)")
    (active if os.path.exists(os.path.join(ROOT, "data", "wells", paths["slug"],
                                            "monthly_aggregate.csv")) else missing
     ).append("fluid-pressure(F, injection wells)")
    (active if os.path.exists(os.path.join(ROOT, "data", "groundwater", paths["slug"],
                                            "daily_aggregate.csv")) else missing
     ).append("groundwater-level(water)")
    (active if os.path.exists(os.path.join(ROOT, "data", "strain", paths["slug"],
                                            "daily_aggregate.csv")) else missing
     ).append("borehole-strain(PBO/EarthScope)")
    (active if os.path.exists(os.path.join(ROOT, "data", "ocean_load", paths["slug"],
                                            "daily_aggregate.csv")) else missing
     ).append("ocean-loading(ECCO2, unaudited)")
    (active if os.path.exists(os.path.join(ROOT, "data", "insar", paths["slug"],
                                            "daily_aggregate.csv")) else missing
     ).append("InSAR(COMET-LiCS, lowest-confidence, noise-limited)")
    (active if os.path.exists(os.path.join(ROOT, "data", "tec", paths["slug"],
                                            "daily_aggregate.csv")) else missing
     ).append("TEC-ionosphere(JPL GIM, contested mechanism)")
    (active if os.path.exists(os.path.join(ROOT, "data", "noise", paths["slug"],
                                            "daily_aggregate.csv")) else missing
     ).append("ambient-noise(MUSTANG sample_rms, level-proxy not dv/v)")
    active += ["swarm/foreshock(E, algorithmic)"]
    missing += ["static-stress-field(World Stress Map, static prior only, see data/stress_map/)",
                "geochem(D)", "Vp/Vs(no published time series exists, dv/v pivot not built)"]
    return active, missing


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    slug = paths["slug"]
    print(f"segment: {cfg['segment_id']} ({slug})")
    cat = load_catalog(cfg)
    stations, gnss = load_gnss(cfg)
    daily = build_daily(cfg, cat, stations, gnss)
    daily = build_state(cfg, daily)

    etas_params_path = os.path.join(OUT, f"etas_params_{slug}.json")
    if os.path.exists(etas_params_path):
        with open(etas_params_path) as f:
            etas_params = json.load(f)
        lam_etas, meta = etas_fitted(cfg, cat, daily.index, etas_params)
        etas_kind = f"ETAS MLE (train<{etas_params['train_end']})"
    else:
        lam_etas, meta = etas_lite(cfg, cat, daily.index)
        etas_params, etas_kind = None, "ETAS-lite"

    trained_path = os.path.join(OUT, f"consequents_trained_{slug}.json")
    if os.path.exists(trained_path):
        with open(trained_path) as f:
            trained = json.load(f)
        consequents, cons_kind = trained["consequents"], f"trained (F3, train<{trained['train_end']})"
    else:
        consequents, cons_kind = None, "hand-picked (F0/F2 default)"
    G = fuzzy_gain(cfg, daily, consequents)
    lam_mocre = lam_etas * G
    # No se renormaliza con el futuro de toda la serie. La intensidad almacenada
    # es exactamente la que se evalúa: ETAS causal multiplicado por G(t).

    tgt = cfg["etas_lite"]["target_mag"]
    ev = cat[(cat["mag"] >= tgt) & (cat["date"] >= daily.index[0])]
    ev_idx = daily.index.get_indexer(ev["date"], method="nearest")

    ll_e, ll_m = loglik(lam_etas, ev_idx), loglik(lam_mocre, ev_idx)
    ig_per_eq = (ll_m - ll_e) / max(len(ev_idx), 1)

    # holdout scoreboard: events after the ETAS training cutoff only
    holdout = {}
    if etas_params is not None:
        cut = pd.Timestamp(etas_params["train_end"])
        hmask = daily.index >= cut
        h_ev = ev[ev["date"] >= cut]
        h_idx_local = daily.index[hmask].get_indexer(h_ev["date"], method="nearest")
        ll_eh = loglik(lam_etas[hmask], h_idx_local)
        ll_mh = loglik(lam_mocre[hmask], h_idx_local)
        holdout = {
            "window": [str(cut.date()), str(daily.index[-1].date())],
            "n_events": int(len(h_idx_local)),
            "loglik_etas": round(ll_eh, 2),
            "loglik_mocre": round(ll_mh, 2),
            "info_gain_per_eq_nats": round((ll_mh - ll_eh) / max(len(h_idx_local), 1), 4),
        }

    daily["lam_etas"] = lam_etas
    daily["lam_mocre"] = lam_mocre
    daily["G"] = G
    daily.to_csv(os.path.join(OUT, f"series_{slug}.csv"))

    channels_active, channels_missing = active_channels(cfg, paths)
    summary = {
        "segment": cfg["segment_id"],
        "window": [str(daily.index[0].date()), str(daily.index[-1].date())],
        "stations": [s["sta"] for s in stations],
        "target_mag": tgt,
        "n_target_events": int(len(ev_idx)),
        "baseline": etas_kind,
        "consequents": cons_kind,
        "etas_meta": {k: round(float(v), 6) for k, v in meta.items()},
        "loglik_etas": round(ll_e, 2),
        "loglik_mocre": round(ll_m, 2),
        "info_gain_per_eq_nats": round(ig_per_eq, 4),
        "holdout": holdout,
        "channels_active": channels_active,
        "channels_missing": channels_missing,
        "note": "F0+F1+F2(K): expert consequents, in-sample+holdout. Walk-forward training = F3. "
                "Real Anza-section trace + sourced recurrence as of 2026-07-03 (see config comments).",
    }
    with open(os.path.join(OUT, f"summary_{slug}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ------------------------------------------------------------ plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    t = daily.index
    axes[0].plot(t, daily["a_seis"], lw=0.6, color="#c44", label="a_seis (anomalía sismicidad)")
    axes[0].plot(t, daily["a_gnss"], lw=0.6, color="#37c", alpha=0.8, label="a_gnss (anomalía deformación)")
    axes[0].axhline(0, color="k", lw=0.3)
    axes[0].set_ylabel("anomalía [-1,1]")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(t, daily["A_state"], color="#c44", label="A (agitación)")
    axes[1].plot(t, daily["C_state"], color="#583", label="C (carga)")
    axes[1].plot(t, daily["K_state"], color="#c80", lw=0.8, label="K (acoplamiento/SSE)")
    if daily["F_state"].abs().sum() > 0:
        axes[1].plot(t, daily["F_state"], color="#0a9", lw=0.9, label="F (presión fluidos)")
    axes[1].plot(t, np.log10(daily["G"]), color="#a5a", lw=0.8, label="log10 G (ganancia)")
    axes[1].set_ylabel("estado / log10 G")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].semilogy(t, daily["lam_etas"], color="#888", lw=0.8, label="λ ETAS-lite")
    axes[2].semilogy(t, daily["lam_mocre"], color="#c22", lw=0.8, label="λ MOCRE-1")
    for _, e in ev.iterrows():
        axes[2].axvline(e["date"], color="k", lw=0.8, alpha=0.6)
        axes[2].annotate(f"M{e['mag']:.1f}", (e["date"], axes[2].get_ylim()[1]),
                         fontsize=7, rotation=90, va="top")
    axes[2].set_ylabel(f"λ(t) M≥{tgt} / día")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))

    fig.suptitle(f"MOCRE-1 F0 vertical slice — {cfg['name']}", fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(OUT, f"lambda_{slug}.png")
    fig.savefig(out_png, dpi=130)
    print(f"plot -> {out_png}")


if __name__ == "__main__":
    main()
