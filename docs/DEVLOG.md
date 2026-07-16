# MOCRE-1 — Development log

> **This is the unabridged, chronological engineering journal of MOCRE-1** — every phase, every
> bug found and fixed, and every negative result, kept verbatim on purpose. For an overview of
> what MOCRE is and how to run it, start with the [README](../README.md). The radical-honesty
> tone below is deliberate: this is what building the model actually looked like.

Hybrid statistical-fuzzy earthquake forecasting model. `lambda_seg(t) = mu_ETAS(t) * G(state,
coincidences; regime)` — an ETAS backbone times a thin fuzzy gain refiner, inheriting the Avaltix
IA1→IA2 production pattern.

## Status — F0 vertical slice (San Jacinto, Southern California) ✅ 2026-07-02

End-to-end on real open data, first scoreboard:

| | |
|---|---|
| Catalog | USGS ComCat, 28,827 events M≥2, 1990→2026, SJC corridor (25 km half-width) |
| GNSS | 113 NGL stations (IGS20/NA frame) within 40 km of trace, both sides |
| Targets | 13 events M≥4.5 in corridor since 1995 (GNSS era) |
| log-lik ETAS-lite | −104.37 |
| log-lik MOCRE-1 | −99.89 |
| **Info gain** | **+0.344 nats/event (≈×1.41 likelihood/event) — in-sample, expert consequents** |

Active channels: seismicity (cluster A) + deformation (cluster B). Missing: C/D/E (stress,
geochem, remote) — coverage honestly partial, see `out/summary_sjc.json`.

## Layout

```
config/segment_sjc.json   segment definition + all model params
src/common.py             geometry, fuzzy memberships (Avaltix-compatible), tanh anomaly normalizer
src/ingest_catalog.py     ComCat download (recursive time-split, 20k limit safe)
src/ingest_gnss.py        NGL station selection + tenv3 download
src/pipeline.py           series -> state (C,A) -> ETAS-lite -> fuzzy G -> lambda(t) + scoreboard + plot
data/                     downloaded data (not committed)
out/                      lambda_sjc.png, series_sjc.csv, summary_sjc.json
```

Run: `cd src && python3 ingest_catalog.py && python3 ingest_gnss.py && python3 pipeline.py`

## Known F0 limitations (fix queue)

1. ~~ETAS-lite is crude~~ — **fixed by F1** (MLE fit, see below).
2. ~~a_seis saturates~~ — **partially fixed**: root cause was `ANOMALY_SETS` knots copied
   verbatim from Avaltix's `ema200_sets` (a different variable's calibration), not tau itself —
   even tau=12 left ~28% of days at hi_s>0.3. Bumped tau_seismicity 2.0→4.0 as a sane default
   (holdout info gain unchanged within noise: 0.225 vs 0.230, n=6 — expected, not tuned-to-metric).
   Real fix (recalibrate the shared partition against actual tanh(z/tau) quantiles, or let τ_v
   be a free parameter) belongs to F3 walk-forward training, not manual retuning now.
3. **In-sample full-window scoreboard remains** (0.340 nats/event) — kept as the "everything
   available" reference number. **Holdout-only scoreboard now exists** (see F1 section) and is
   the one that matters.
4. Only 2 of 4 state vars active (C from GNSS loading, A from seismicity). F (fluids) likely
   unavailable for SJC (no public well-pressure feed); K (SSE) is buildable from existing GNSS
   series (transient detector) without new data — planned for F2.
5. Single segment by design until F1-F4 close (see spec "Decisión de secuencia" 2026-07-02).
   F4 = SCEC CFM segment graph + Coulomb transfer edges.

## Status — F1 ETAS MLE fit ✅ 2026-07-02

`src/etas_fit.py`: temporal ETAS (mu,K,c,p,alpha) fit by Nelder-Mead MLE on Mc=3.0 events,
trained on data before 2010-01-01, frozen and applied causally forward. Wired into
`pipeline.py` as the default backbone (falls back to `etas_lite` heuristic if
`out/etas_params_sjc.json` is missing).

Holdout scoreboard (2010-01-01 → today, 6 events M≥4.5, **params never saw this window**):

| | |
|---|---|
| log-lik ETAS (fitted) | −43.51 |
| log-lik MOCRE-1 | −42.13 |
| **Info gain, holdout** | **+0.230 nats/event — real out-of-sample signal, not in-sample fitting artifact** |

G's consequents are still hand-picked (not yet walk-forward trained — that's F3). This holdout
number is the first honest checkpoint: the fuzzy refiner beats ETAS-alone on data its rules
never touched, even before any training.

## Status — geometry + recurrence sourced, K (SSE) live ✅ 2026-07-03

1. **Real fault trace** replaces the hand-picked 2-point line: `config/sjc_anza_trace.json`,
   428 vertices / 64 polyline chains, CA GIS `GeoscientificInformation/Fault_Lines/MapServer/3`
   (NAME="San Jacinto fault zone, Anza section", includes the Clark/Hog Lake strand where the
   paleoseismic recurrence data comes from). `dist_to_trace_km` now does nearest-segment-of-any-
   polyline distance; strike recomputed by PCA = 309° (was a 305° hand estimate — close).
2. **Recurrence sourced, not guessed**: Hog Lake paleoseismic trenching (Clark fault, Anza
   section) — 3 surface ruptures M~7.0-7.5 c. 1210/1530/~1795-1800, recurrence ~250-254 yr.
   `T_rec_years` 250→254, `years_since_last_major` 226→227, cited in config. Not re-verified
   against the primary paper — literature-grade, not re-derived.
3. **ANOMALY_SETS recalibrated**: the original partition was copied verbatim from Avaltix's
   `ema200_sets` (wrong domain) — "normal" covered only ±0.05, so "alto" fired on almost any
   positive noise. Rederived from the actual quantiles of tanh(z/tau) under a quiet-regime null
   (not fit to this dataset's outcome): normal widened to ±0.35, alto/muy_alto now require
   genuinely elevated z.
4. **K (acoplamiento/SSE) is live**: 30-day rolling rate of `D_mm` minus the 365-day trend,
   positive-only (transient acceleration, not deceleration — see code comment for the physical
   reasoning), robust-normalized, EWMA-decayed faster than A (λ_K=0.90 vs λ_A=0.95, SSEs are
   shorter-lived by definition). Wired into `fuzzy_gain` as its own low-arity clause + a K×A
   coincidence clause. Built entirely from GNSS data already downloaded — no new ingestion.

**Honest caveat on the updated scoreboard**: the corrected geometry is *tighter* to the real
Anza section (which is a known "seismic gap" in the literature — quieter than neighboring SJC
segments), so `n_target_events` dropped 13→7 (holdout 6→4). Info gain moved to +0.74 nats/event
in-sample / +0.76 holdout — **do not read this as "the model got much better"**: with n=4
holdout events, log-likelihood differences have huge variance. The geometry/recurrence/partition
fixes are real corrections (verified against sourced literature and derived math, not tuned to
the metric); the headline number swinging this much on n=4 is exactly why F3's proper
train/holdout discipline over more events (not just more segments) matters before trusting any
single info-gain figure.

## Status — F3 walk-forward consequent training ✅ 2026-07-03 (with an honest failure caught mid-way)

`src/train_consequents.py`: optimizes the 7 free consequents of $G$ (2 anchors stay fixed at
0.0) via Nelder-Mead, softplus-reparametrized so no trained value can go negative (literal R4:
"more anomaly never lowers gain"). Trained strictly on `train<2010-01-01` (same cutoff as the
ETAS fit — no double-dipping), scored on the untouched holdout.

**First run, no regularization: degenerate.** With only 3 training events for 7 free
parameters, MLE alone pushed 4 of 7 consequents straight to the gain ceiling (1.7) — textbook
overfitting to a handful of points, not learned signal. Holdout info gain looked spectacular
(+1.76 nats) and was worthless. This is exactly the failure mode F3's discipline exists to
catch, and it caught it immediately.

**Fix applied: ridge penalty toward the hand-picked defaults** (weakly-informative Bayesian
prior, reg_strength=2.0) — explicitly labeled in the code and report as *damage control, not a
real fix*. With the prior, trained consequents stay close to the expert values with small,
directionally sensible nudges (e.g. `seis_muy_alto` 0.60→0.84, `seis_x_carga` 0.80→1.03).
Holdout info gain: **+1.02 nats/event vs +0.76 hand-picked, n=4 holdout events** — a modest,
plausible improvement, not a validated one.

**The real conclusion of F3 is not the number — it's the diagnosis: 7 training events (3
train + 4 holdout) cannot support 7 free parameters, regularized or not.** Trust this
comparison as "the trained consequents are at least as good as hand-picked and didn't need to
cheat," not as "the model is calibrated." Genuine fix is more events: F2.5 (multi-segment) adds
data volume the honest way; a secondary, faster option is refitting ETAS + consequents at a
lower target magnitude (e.g. M≥4.0) for more training events on the same single segment, at the
cost of a less operationally interesting output threshold.

## Status — multi-segment (F2.5 groundwork) ✅ 2026-07-03

Pipeline is now segment-parameterized (`python3 script.py segment_X.json`, defaults to SJC for
backward compat). Two new segments online end-to-end (ingest → state → ETAS → gain → plot):

| Segment | Regime | Trace source | Stations | M≥4.5 events (corridor) | ETAS fit |
|---|---|---|---|---|---|
| `segment_sjc.json` (Anza) | transform | real CA GIS trace | 113 | 7 | converged, n_bg=12.1 |
| `segment_mojave.json` | transform | real CA GIS trace | 88 | **0** | converged, n_bg=3.4 |
| `segment_cascadia.json` | subduction | **approximate 2-pt coastline** (no CA-GIS equivalent found for the offshore megathrust — documented in config, revisit with a real slab model e.g. Slab2) | 48 | 2 | **did not converge**, degenerate (n=8 training events, K≈0) |

**Real archetype differentiation already visible in raw state summaries** (`out/multisegment_comparison.csv`), before any formal clustering:

| Segmento | C̄ | C_final | A activo (>0.3) | K activo (>0.3) |
|---|---|---|---|---|
| SJC (Anza) | 0.95 | 1.00 | 12% | 17% |
| Mojave | 1.00 | 1.00 | 15% | 18% |
| Cascadia | 0.68 | 0.69 | **26%** | 17% |

Mojave sits at maximum load with zero release (the "silent/locked" archetype from the genealogy
doc §1.2) while Cascadia never saturates (slower nominal recurrence) but has the highest
agitation-active fraction (frequent smaller sequences + real SSE signal — Cascadia is the
textbook ETS region, and K is picking something up there, though not yet validated against a
real tremor catalog).

**Bug found by inspecting this table, not fixed yet**: Mojave's $C_0 = 169/150 = 1.13$ already
exceeds 1 on day one (segment is "overdue" relative to its own naive recurrence fraction), so
`clip(C, 0, 1)` saturates it for the *entire* 30-year window — exactly the segment where $C$'s
resolution matters most. Needs a real fix before F2.5 clustering trusts this feature (options:
don't hard-clip C, carry an explicit "overdue multiplier" past 1.0, or renormalize by a longer
window than T_rec). Filed as the next concrete task, ahead of writing the clustering script
itself.

**Cascadia's ETAS non-convergence is itself informative, not just noise**: only 8 training
events in a 60km corridor over 20 years reflects the real, much lower shallow-crustal seismicity
rate near the locked megathrust compared to a California strike-slip corridor — expected given
the physics, but it means no lambda-based scoring is trustworthy for this segment yet. The state
trajectories (C, A, K) are still valid and usable for archetype comparison; the ETAS/gain
scoring side needs either a longer catalog window, a wider corridor, or accepting Cascadia stays
state-only until enough events accumulate.

## Status — C-saturation bug fixed + F2.5 clustering pipeline live ✅ 2026-07-03

**C-saturation fixed**: `C_state` is no longer clipped to [0,1] at the raw-state level (was
losing all resolution for any "overdue" segment, e.g. Mojave sat at exactly 1.0 for 30 years).
Raw C now ranges freely (ceiling 3.0 for numerical sanity only); the [0,1]-bounded *gate* used
by fuzzy rules (`C_hi`) is computed separately downstream and was never affected. Confirmed:
Mojave's C now moves 1.127→1.280 over the window instead of being pinned at 1.0. No regression
in SJC/Cascadia scores (re-ran all three, numbers unchanged within rounding).

**`src/cluster_archetypes.py`**: the actual F2.5 deliverable, not a stub. Re-derives daily state
by calling `pipeline.py`'s own functions directly per segment (single source of truth, zero
re-implementation drift), then:
1. **Segment-level features** (works even for 0-event segments like Mojave -- "locked and
   silent for 30 years" is itself a valid trajectory shape, not a data gap to exclude).
2. **Per-event trajectory features**: 90-day trailing window before each target-magnitude event,
   summarized by *shape* (level, slope, peak-timing), not just mean level -- this is what the
   genealogy doc's "cluster pre-event trajectories" idea actually meant.
3. Standardized pairwise distance + hierarchical linkage (scipy), printed in full (no forced
   cluster count imposed -- that would be fabricating structure that 9 events can't support).

**First real run** (3 segments, 9 target events: SJC=7, Cascadia=2, Mojave=0): segment-level
distances already separate physically sensibly -- **Cascadia (subduction) is the most distinct**
(dist 6.15 from Mojave, 5.46 from SJC) while **Mojave and SJC (both California transform) sit
closer to each other** (dist 4.73). This is exactly what the geology would predict, and it fell
out of the standardized-feature distance calculation with no thumb on the scale -- a genuine
sanity-check pass, not a tuned result (nothing in this script was adjusted to produce this
outcome). **Still: 9 events is a pipeline correctness check, not a statistical finding** -- don't
read cluster membership as validated archetypes yet, see the script's own printed caveat.

## Status — 2 more segments + declustering fix ✅ 2026-07-03 (continued)

**Two new segments, chosen for volume, not just variety:**

| Segment | Regime | Trace | Stations | M≥target events | Why this one |
|---|---|---|---|---|---|
| `segment_parkfield.json` | transform | real CA-GIS | 47 | 27 (target M≥3.5) | Most heavily instrumented fault section in the world; site of the 1985-1993 Parkfield Earthquake Prediction Experiment -- a documented FAILED point-prediction attempt, directly on-theme for this project's R1-R4 design requirements. Real 2004-09-28 M6.0 + aftershock sequence clearly visible in the state series. |
| `segment_shumagin.json` | subduction | approximate 2-pt (Alaska Peninsula coast proxy, same honesty caveat as Cascadia) | 6 (sparse) | 23 (target M≥5.5) | Real, large, well-documented 2020-07-22 M7.8 + 2020-10-19 M7.6 doublet (broke part of a gap unruptured since 1917) -- unlike Cascadia, gives genuine large target events plus real ETS/SSE activity in the same corridor. train_end set to 2018 specifically to keep the doublet in the HOLDOUT window. |

**F3 retrained with real volume, and it delivered an honest negative result — worth reporting
as-is, not smoothed over.** Parkfield: 21 training events (finally >= 2×7 free parameters, no
degrees-of-freedom warning) -- but the trained consequents scored **worse on holdout than
hand-picked** (+0.358 vs +0.424 nats/event). This is walk-forward discipline doing exactly its
job: a fit that looks fine in-sample doesn't automatically generalize, and reporting that
honestly is the point of having a holdout at all. Shumagin (7 train / 16 holdout, still
regularized) did generalize better (+0.663 vs +0.482 nats/event) -- mixed, not a clean win
either way, which is the realistic state of a system this early.

**Declustering bug found and fixed in `cluster_archetypes.py`**: the first 5-segment clustering
run showed several events merging at distance exactly 0.000 -- turned out to be same-day
aftershocks of the 2004 Parkfield and 2020 Shumagin sequences, whose 90-day trailing windows are
near-identical *by construction* (the window is anchored to the calendar day). Without
declustering, big aftershock sequences silently inflate their segment's event count with
non-independent, near-duplicate feature vectors -- exactly the same principle as ETAS
background-rate declustering, applied here to the archetype-feature step.
`decluster_for_archetype()` keeps only the largest event per 30-day window. Result: 59 raw
events -> **37 genuinely independent declustered events** (Parkfield 27->18, Shumagin 23->10,
others unaffected). Segment-level distances unchanged (they don't depend on event declustering);
per-event linkage is now clean, no more spurious 0-distance merges.

## Status — F4 Coulomb stress-transfer graph ✅ 2026-07-03

`src/stress_graph.py`: nodes = the 5 segments (real/approximate traces already in `config/`),
edges = static Coulomb stress change (Delta_CFS) each segment's historical M>=5 ruptures impart
on every other segment, closing the `dC` term the spec always specified
(`w_deltaCFS * max(0, deltaCFS_in)`) but never wired up until now.

**Engineering decision, stated plainly (see the module's own docstring for the full
reasoning)**: this is deliberately a simplified point-source approximation, NOT a full Okada
(1992) half-space rectangular-dislocation solution. Reconstructing Okada's closed-form solution
from memory without a reference implementation to validate against carries real risk of a
silently-wrong stress field, which is worse than not having one. Instead:
`Delta_CFS = K * M0_source / r^3 * lobe_pattern(azimuth, strike_diff) * mechanism_compat(regimes)`
-- moment/distance^3 scaling is textbook-correct dimensionally; the four-lobe angular pattern
(`cos(2*theta)`) is the standard qualitative approximation used throughout the popular and
technical stress-triggering literature; `K` is calibrated numerically (not by hand-arithmetic,
to avoid a unit slip) so a M7.0 event at 20km gives ~1 bar, the literature's typical range.

**Built-in `self_test()` — and it caught a real bug immediately**: my first attempt asserted the
*opposite* sign pattern from correct physics. The self-test's assertion, not the underlying
formula, had the sign backwards -- for two similarly-oriented parallel faults, Coulomb stress is
**encouraged along the source's own strike direction** (the real, well-documented mechanism
behind sequential ruptures like Landers 1992 -> Big Bear -> Hector Mine 1999, on this very fault
system) and **discouraged perpendicular to strike** (the classic "stress shadow"), with 45° as
the neutral zero-crossing -- not the other way around. Fixed the test to match correct physics,
verified: (a) 1/r³ decay confirmed exactly (ratio=1000 for a 10x distance change), (b) sign
pattern now correct, (c) segments >1500km apart correctly return near-zero cross-terms.

**First real result, with the graph wired to real catalogs**: Cascadia and Shumagin contribute
essentially nothing to California segments or each other (correctly negligible at >1000km) —
the California triad (SJC/Mojave/Parkfield, 50-150km apart) shows real, small, non-negligible
transfer, dominated by **SJC → Mojave** (net +0.0044 bar from 222 contributing M≥5 events) —
plausible given SJC is the closest segment with real M≥5 activity to the locked, quiet Mojave
section. All magnitudes stay well under 1 bar given current inter-segment distances (tens to
hundreds of km) — the mechanism is real and wired, its current numerical footprint on `C_state`
is small (`dC_from_stress_graph` sums to ~4e-06 for Mojave over 30 years), exactly as should be
expected until segments closer together or with larger sources are added.

**Wired into `pipeline.py`**: `build_state()` now reads `out/cfs_in_<slug>.csv` (per-segment
daily positive-only stress-step series, produced by `stress_graph.py`) and adds
`step_bar / STRESS_DROP_BAR` (STRESS_DROP_BAR=30 bar, a representative mid-range single-event
stress drop, not segment-specific -- flagged as an approximation) to `C`'s cumulative sum. Full
pipeline re-run on all 5 segments after wiring: no regressions, no crashes, numbers move by the
expected small amount.

**Honest limitations, stated for the next person (or session) to pick up**: no real focal
mechanisms (regime-level dip/rake defaults only, not per-event); no proper stress-tensor
resolution onto the receiver plane (the `strike_align` term is a coarse stand-in); segments are
treated as single point nodes, not sub-divided into along-strike patches (real Coulomb-stress
studies often resolve stress along many points on a receiver fault, not just its centroid); not
validated against any specific published numeric Coulomb-stress map, only against the
qualitative textbook pattern. Good enough to make the mechanism real and directionally correct;
not survey-grade.

## Status — F5 CSEP-style paired significance test ✅ 2026-07-03 (found and fixed a real bug along the way)

`src/csep_test.py` implements the Rhoades et al. (2011) T-test / W-test — the real statistical
comparison this project has been missing since F1 (every prior "+X.XX nats/event" number was a
point estimate on 2-16 holdout events, no significance test attached).

**Building it immediately surfaced a genuine structural bug, not noise**: T-test and W-test
disagreed sharply on every segment (W-test p=0.0, T-test p>0.1). Chasing the disagreement (not
picking whichever result looked better) found the cause: **G (the fuzzy gain) was mathematically
incapable of ever going below 1.0** -- confirmed empirically, G was ≥1.0 on 100% of holdout days,
0% at exactly 1.0, for every segment. Root cause: every consequent (hand-picked AND trained) was
constrained non-negative (softplus in `train_consequents.py`), so the weighted average `num/den`
-- a convex combination of nonnegative numbers -- could never be negative. R4 ("more anomaly
never lowers gain") had been silently over-implemented as "no consequent may be negative", which
is a materially stronger and wrong constraint: it meant MOCRE-1 could only ever elevate the ETAS
baseline rate, never suppress it, no matter how quiet a segment genuinely was. The earlier global
budget-renormalization (present since F0) was, in retrospect, quietly compensating for this
asymmetry rather than being a neutral convenience -- exactly why removing it (needed for an
honest CSEP test, see below) is what exposed the bug.

**Fixed at the root**: added `seis_bajo`/`seis_muy_bajo` rules (mirror of `seis_alto`/
`seis_muy_alto`, negative consequents) -- the low-anomaly end of the partition was simply
*missing* a rule, not contradicting the design. Monotonicity is now enforced correctly as an
**ordered chain** (`c(muy_bajo) <= c(bajo) <= 0 <= c(alto) <= c(muy_alto)`) in
`train_consequents.py`'s reparametrization, not "all consequents ≥0". Verified: G now ranges
0.45-6.6 in Parkfield's holdout, 12% of days below 1.0 (up from 0%).

**Second, subtler finding, deliberately left as a documented limitation rather than chased
further**: even after the fix, ~88% of days still show G>1 (not the ~50/50 a symmetric system
might suggest). Checked whether this comes from a genuinely skewed `a_seis` distribution --
it doesn't (median exactly 0, 38.6%/38.3% positive/negative, essentially balanced). The residual
asymmetry instead comes from the *other* 5 rules (`coincid_seis_gnss`, `seis_x_carga`,
`agitacion_x_carga`, `sse_solo`, `sse_x_seis`), which remain structurally one-sided by design --
defensible (a coincidence detector has no natural "anti-coincidence" counterpart; absence of
elevated evidence is what `seis_normal`/the anchor already represent), but it does mean
**the Wilcoxon test's persistent p=0.0 across all segments should NOT be read as clean evidence
of real skill** -- it's likely still partly driven by this residual structural pull, not
event-concentrated predictive power. **The T-test's verdict is the more trustworthy read right
now: at current sample sizes, MOCRE-1 is NOT statistically distinguishable from ETAS-only on any
segment.** That is the honest state of the model today, reported as such rather than smoothed
into a positive headline.

## Status — F5b rolling-origin walk-forward ✅ 2026-07-03

`src/rolling_csep.py`: instead of one train/holdout split, ETAS is refit at multiple sequential
origins (expanding window, strictly-prior data only) and each subsequent period is scored
independently with the same Rhoades T-test, then combined via Fisher's method. **First attempt
used fixed 4-year calendar epochs and badly underused the data** -- target events cluster in
time (16 of Parkfield's 27 events fall in one 4-year window around the 2004 sequence), so most
calendar bins were empty. Fixed: **event-count-based epochs** (5 consecutive target events per
epoch, boundaries adapt to each segment's actual event tempo) -- correct fix for irregular point
processes, not a parameter tweak to force a nicer-looking result.

**Real, nuanced results, reported as-is:**
- **Parkfield**: 4 usable epochs, split 2-and-2 (favors MOCRE-1 in 2, favors ETAS-only in 2),
  Fisher combined p=0.586 -- genuinely inconclusive, no directional signal across time.
- **Shumagin**: 4 usable epochs, **4-of-4 favor MOCRE-1** in direction (consistent, not one
  outlier epoch dominating), but Fisher combined p=0.199 -- not significant at p<0.05, but a
  consistent-direction result across independent replications is a meaningfully different
  (more encouraging) finding than a single inconclusive point estimate. Worth flagging as the
  most promising thread if data collection continues, without overstating it as a positive
  result today.
- SJC/Mojave/Cascadia: still too few events even with adaptive epoch sizing (1, 0, and 0 usable
  epochs respectively) -- confirms data volume, not methodology, is the remaining constraint.

## Status — F channel activated: 6th segment, Oklahoma induced seismicity ✅ 2026-07-03

**The audit's biggest finding, operationalized.** `segment_oklahoma.json` (Pawnee-Prague
corridor, regime=`induced`, the first non-tectonic regime in the project): real OCC UIC well
injection data (`src/ingest_wells.py`, monthly Vol/PSI per well, 2011-2024, ~3000 wells/month in
corridor) activates the $F$ (facilitación) state variable for the first time since the spec's
original S=(C,A,F,K) was written. Segment richness: **57 target events (M≥4.0), 36 in holdout —
the best-populated segment in the project**, and the catalog visibly shows the textbook induced
sequence (near-zero 2005-2008 → explosive growth to 2012-2016 → decline after regulatory
response), including Prague (2011-11-06 M5.7) and Pawnee (2016-09-03 M5.8, largest in Oklahoma
history) as real, correctly-timed target events.

**Found and fixed a second instance of the same bug class from F5's discovery**: first CSEP run
showed MOCRE-1 **significantly WORSE** than ETAS-only for Oklahoma (T-test p=0.00001) — chased
it (not smoothed over) and found `G` systematically inflated (+34% mean rate vs ETAS in holdout).
Root cause: `fluid_solo`/`fluid_x_seis` were introduced one-sided (only positive consequents),
the exact same missing-low-end-rule bug already fixed once for the seismicity ladder
(`seis_bajo`/`seis_muy_bajo`) -- reintroduced for a new variable instead of learned from. Added
`fluid_bajo`/`fluid_muy_bajo` (mirroring the seismicity fix) so the model can recognize
Oklahoma's real, regulatory-driven post-2016 injection decline as a genuine risk-lowering signal,
not just fail to react to it. Result: G inflation dropped from +34% to +4.7%, and the honest
verdict flipped from "significantly worse" to **"not statistically distinguishable from
ETAS-only"** — the same honest baseline state as every other well-populated segment, T-test and
W-test now agreeing (both non-significant) where before they visibly disagreed.

Retrained the 3 segments with saved consequents (SJC, Parkfield, Shumagin) after the shared
rule-set change broke their cached consequent files (`KeyError: 'fluid_bajo'`) -- all 6 segments
now run clean.

## Status — "todas las variables" pass: swarm/foreshock detectors + infrastructure fix ✅ 2026-07-03

Continuing systematically through the audited catalog (Matt: "te dije TODAS LAS VARIABLES", not
one and stop). Two new algorithmic features added, no new data source needed:

- **`a_swarm`/`E_state` (enjambre)**: sustained elevated 30d rate NOT decaying like Omori
  (gated by 14-day slope of the 30-day rate not being strongly negative) -- operationalizes the
  catalog's own distinction between a swarm and a normal aftershock sequence.
- **`a_foreshock` (foreshock_agudo)**: short-window (7d) rate accelerating relative to the
  medium-window (30d) baseline -- the "cascada de nucleación" signature, used directly as a rule
  input (no persistent EWMA state, it's inherently short-horizon).

**Hit the SAME bug class twice more in one pass, immediately** -- first version left
`enjambre_solo`/`foreshock_agudo` one-sided again (reasoning: "absence of swarm/acceleration is
just absence of evidence, not a negative-hazard state" -- wrong, disproven within the hour:
Parkfield's G-inflation ratio hit 1.8x, Oklahoma 1.36x, both flipped from "not distinguishable"
to "significantly worse" on CSEP). Added symmetric mirrors (`enjambre_bajo`/`foreshock_calmo`)
-- **that alone made it WORSE** (SJC's ratio hit 2.5x): the real second bug was in
`train_consequents.py`'s `unpack()`, which only special-cased the original seismicity chain by
hardcoded name -- every new `_bajo`-style rule (including `fluid_bajo` from the Oklahoma pass)
was silently trained back to positive by the generic softplus fallback, undoing the hand-picked
negative default the moment F3 ran.

**Fixed generally, not per-variable this time**: `unpack()` now reads from explicit
`NEGATIVE_CHAINS`/`NEGATIVE_SINGLES`/`POSITIVE_CHAINS` registries instead of hardcoded rule
names -- any future "_bajo" rule just needs to be added to a list, not require re-deriving the
softplus logic. This is infrastructure that should prevent this exact bug class recurring as
more of the 25-variable catalog gets wired in. Result after retraining all 4 populated segments:
SJC and Parkfield recovered clean ("not distinguishable", the healthy baseline); Shumagin
unaffected (was already clean); **Oklahoma still shows a smaller residual** (T-test p=0.017,
but W-test now healthy at 0.28, a large improvement from p=0.0/0.0 before) -- likely the
*deliberately* one-sided coincidence-type rules (`coincid_seis_gnss`, `seis_x_carga`, etc., kept
one-sided by design, see F5's notes) firing disproportionately given Oklahoma's unusually
volatile real dynamics. Documented as a known open item, not chased further this pass.

**Ambient noise (ruido sísmico ambiental)**: mechanism confirmed real (IRIS/EarthScope Mustang
PSD service works -- required discovering an undocumented `.M` quality-code suffix in the
target format, e.g. `IU.ANMO.00.BHZ.M`, verified with real 135KB PSD response) but **blocked for
our actual stations**: the CI network (Southern California, our own segments) returns no data
via this service after multiple format attempts -- flagged honestly as unresolved, not silently
dropped.

## Status — nivel de aguas subterráneas: deprioritized, not abandoned ✅ 2026-07-03

`src/ingest_groundwater.py` built against the real, confirmed-working `api.waterdata.usgs.gov`
endpoint (verified with real daily depth-to-water readings near San Jacinto). Wired into
`pipeline.py` (`a_water`, symmetric `agua_bajo`/`agua_alto` rules from the start this time,
learning from the enjambre/foreshock episode above) -- code is real and correct, degrades
gracefully to zero when no data file exists (verified: all 6 segments still run clean).

**What didn't work: the API itself.** Two full ingestion attempts (590s and 480s) both timed out
without completing even the closest ~10 candidate stations near San Jacinto -- confirmed HTTP
429 (rate-limited) during diagnosis, and the `daily` collection query appears slow per-request
even when not rate-limited. Rewrote with incremental per-station saving (so a timeout keeps
partial progress) and a 60-station distance cap -- still too slow to finish in reasonable time.
**Decision: deprioritize, not chase further right now.** This is the catalog's own lowest-
confidence active-candidate variable (mu_ev=0.35, "mecanismo real pero débil e inconsistente")
-- the engineering cost-to-value ratio here is poor compared to what's already been won this
session. Data access is confirmed real (not a dead end), just operationally slow to harvest;
revisit with a proper async/parallel client if this becomes worth the time later.

## Phases

F0 slice ✅ → F1 real ETAS ✅ → **F2 state machine: C,A,K live, C-saturation bug fixed** (F/fluid
pressure still inactive — no public well-pressure feed found for any segment yet) → **F3
consequent training ✅**, twice now (second pass fixed the one-sided-G bug) → **F2.5 ✅** (5
segments online; 37 declustered independent events; clustering pipeline correct) → **F4 ✅**
(Coulomb stress-transfer graph, self-tested, wired into C) → **F5 ✅** (Rhoades T-test/W-test —
found and fixed a real structural bug in G along the way; honest current verdict: MOCRE-1 is not
yet statistically distinguishable from ETAS-only on any segment) → **next: the model is now
honestly and rigorously benchmarked against its own null hypothesis for the first time — the
real next lever is DATA (more events per segment, more segments, real F-channel data) since the
pipeline, training, and testing machinery are all now correct and validated end-to-end; a true
CSEP submission (forecast registered before the evaluation window closes) is the remaining
process/infrastructure gap, not a math one** → F6 crowdsensing.

Regime gating (R3) is declared in spec but intentionally inert in code until F2.5 — a branch
with only one case to test is unverifiable dead code. Coverage direction: **more variables is
always better** here (25-variable catalog, not a 7-variable ceiling — see genealogy doc
§correlaciones) — next expansion candidates: InSAR (heavy processing, deferred), TEC via IGS
IONEX (feasible, not yet done), real historical catalog depth beyond ComCat's 1990 start.
# Errata de auditoría causal — 2026-07-11

Las entradas anteriores documentan el desarrollo histórico, no el estado científico actual. Una
auditoría completa encontró que el supuesto éxito de Oklahoma dependía de información disponible
el mismo día y de libros anuales de inyección retrocolocados por mes; además, el “W-test CSEP”
usaba meses como observaciones en vez de terremotos. Tras corregir causalidad, disponibilidad y la
fórmula por evento, Oklahoma obtiene IG −0.805/evento y ambos modelos fallan el N-test absoluto
(MOCRE 219 esperados, ETAS 183, observados 36). Shumagin queda positivo pero no significativo y
también falla el N-test. El global despliega Omori+tasa porque el incremento fuzzy mejora el Brier
solo ~10⁻⁶, por debajo del umbral material 10⁻⁴. IA3 no supera los baselines justos en magnitud,
cuándo ni dónde y esas salidas ya no se publican como predicciones. Toda afirmación histórica de
“validado”, “57%”, “Oklahoma gana” o p=0 debe leerse como revocada. La evaluación confirmatoria
empieza con el ledger prospectivo inmutable.

## MOCRE-2 — inferidor de estado crítico en sombra

Se implementó el rediseño jerárquico IA1→IA2 solicitado: disponibilidad explícita, evidencia por
clúster, noisy-OR dentro del clúster, coincidencias `min` entre mecanismos, memoria recurrente,
episodios declusterizados y controles emparejados. Se añadieron actividad regional, expansión
espacial, caída de b-value, concentración hacia la falla y aceleración de magnitud. Con 90 eventos
independientes, leave-one-region-out alcanza mediana AUC ~0.66, pero el test cronológico queda en
0.43, sensibilidad 16.7% y falsas alarmas 11.1%. El motor se abstiene (`promoted=false`) y queda
prospectivo en sombra. Esto localiza el siguiente cuello de botella en observabilidad física
(dv/v, strain y deformación espacial), no en más ajuste de reglas catalogales.
