"""MOCRE-1 FUZZYSET — the actual Mamdani inference engine (what Matt asked for, and what I wrongly
replaced with a logistic/AUC benchmark before). This is NOT a classifier. It is an expert engine:
give it the live variable state of any point on Earth, it fuzzifies, fires production rules,
aggregates, defuzzifies, and RETURNS THE OUTPUT VALUES:

  - ocurrencia  : universe [-1,1], partitions {no, sí}, PURE LINEAR RAMPS crossing at 0 (Matt's
                  literal spec: "de 0 a -1 es no, de 0 a 1 es sí"). Defuzzified by MEAN-OF-MAXIMUM,
                  not centroid -- verified 2026-07-05 that centroid on a symmetric-around-0 output
                  NEVER reaches +-1 even at 100% rule confidence (a "si" rule at w=1 centroids to
                  only +0.55, exactly the defect this project's own doc flags: "salidas simétricas
                  alrededor de 0, centroide sesga hacia 0" -- the same defect that made Avaltix's
                  IA2 double-defuzzify toward the middle). With linear ramps + MOM, verified
                  numerically: defuzzified value == rule weight w EXACTLY, no contraction.
  - magnitud    : universe Richter, partitions {moderada, fuerte, muy_fuerte}, centroid (a genuine
                  physical quantity, not a symmetric confidence -- centroid is the right default
                  here per the project's own doc).
  - (cuándo/dónde: scaffolded, added once occurrence+magnitude are calibrated.)

Same lineage as Avaltix IA2 (Mamdani classic): fuzzificación → reglas → implicación (recorte) →
agregación (max) → defuzzificación. Inputs share the universal [-1,1] anomaly space with the
5-partition ANOMALY_SETS (fuzzify_anomaly, Ruspini-corrected 2026-07-05, see common.py). Rule
firing strength includes mu_ev (per-variable catalog confidence, e.g. fluid_pressure=0.90,
TEC=0.25) as an explicit factor -- "definir la influencia de cada variable separadamente" (Matt) --
matching the spec already written in 02_motor_inferencia: w_r = mu_ev * T(mu_A(x)). The rule base
is DATA -- initially a small expert set (below), to be REPLACED/extended by the fuzzy-inverse
(Wang-Mendel extraction from real earthquakes, discover_rules.py). The engine itself does not care
where the rules come from.
"""
import numpy as np

from common import ANOMALY_SETS, fuzzify_variable, get_membership, sets_for_var

# mu_ev: per-variable evidence confidence from the audited 25-variable catalog (03_ejemplos/
# sismologia-prediccion-alerta.md) plus the honest tiers assigned to variables added later this
# session (ocean/insar/tec/noise, all "unaudited", see pipeline.py's DEFAULT_CONSEQUENTS comments
# for the same tiering). This is the "influencia de cada variable" Matt's spec asks for, applied
# as an explicit multiplicative factor in rule firing strength -- not folded silently into a
# learned weight where it can't be inspected separately.
MU_EV = {
    "a_seis": 0.50, "a_swarm": 0.55, "a_foreshock": 0.40, "a_gnss": 0.85, "a_sse": 0.60,
    "a_fluid": 0.90, "a_water": 0.35, "a_strain": 0.675, "a_ocean": 0.35, "a_insar": 0.15,
    "a_tec": 0.25, "a_noise": 0.35,
    # persistent states inherit their driving channel's confidence (A<-seis, K<-sse, F<-fluid,
    # E<-swarm); C_state is the recurrence/stress accumulator, mu_ev=0.50 per catalog cluster C.
    "A_state": 0.50, "C_state": 0.50, "K_state": 0.60, "E_state": 0.55, "F_state": 0.90,
}

# ---- output variables: universe grid + linguistic partitions ---------------------------------
# ocurrencia: bipolar linear ramps, Ruspini-exact (mu_si+mu_no=1 everywhere), crossing at x=0 with
# mu=0.5 each -- Matt's literal construction (his own doc's option A). Represented via the shared
# trimf machinery with a degenerate third param (b==c or a==b) so get_membership's existing
# clip-at-1.0 branch just returns the constant tail -- verified this reduces to the exact linear
# formula mu_si(x)=(x+1)/2, mu_no(x)=(1-x)/2.
OCC_GRID = np.linspace(-1, 1, 20001)   # fine grid: MOM needs a fine grid, coarse grids bias it
OCC_SETS = {
    "no": [-1.0, -1.0, 1.0],
    "si": [-1.0, 1.0, 1.0],
}
OCC_DEFUZZ = "mom"
# magnitud: Richter partitions on an absolute scale; a segment maps its own target_mag onto this.
# MEASURED 2026-07-05 (corr of transient state vs within-segment magnitude |r|<=0.08, r^2<0.6%):
# the transient anomaly state does NOT resolve magnitude -- magnitude is set by the fault's tectonic
# scale (target_mag from geometry / Wells&Coppersmith Mmax), a STATIC per-location input `mag_scale`
# in [-1,1]. So magnitud rules key off mag_scale, not the state channels; adding state->magnitud
# rules would be fitting 0.6% of variance = noise. This generalizes to any GEM point on Earth, which
# carries a target_mag from its fault length even with zero local instrumentation.
MAG_GRID = np.linspace(3.0, 8.5, 221)
# Ruspini-exact shared-foot chain (verified sum_mu==1 to 1e-9): trapezoidal extremes with REAL
# plateaus (moderada flat on [3,4], muy_fuerte flat on [7.5,8.5]) and a triangular center peaked at
# 5.75 -- literally Matt's spec "trd centrales triangulares, extremidades trapezoidales". The old
# 3-jul params ([3,3,4,5]/[4.5,5.5,6.5]/[6,7,8.5,8.5]) left a Ruspini gap at x=5 (sum=0.5), the same
# defect the audit caught in the input partition. Centroids land ~3.9/5.75/7.6 -> map cleanly onto
# the real per-segment median magnitudes (Parkfield 3.85->moderada, Chile/Shumagin 5.2-5.7->fuerte,
# megathrust tail->muy_fuerte).
MAG_SETS = {
    "moderada":   [3.0, 3.0, 4.0, 5.5],
    "fuerte":     [4.0, 5.5, 7.0],
    "muy_fuerte": [5.5, 7.0, 8.5, 8.5],
}
MAG_DEFUZZ = "centroid"

# cuándo: time-to-event over the operational horizon [0,30] days, Ruspini triangular chain
# (inminente/proximo/lejano, knots 0/15/30). MEASURED: transient state vs delta-t |r|<=0.05 -- the
# 30d-window label carries almost no day-resolution timing signal, so this output is honestly a
# near-uniform expectation (~15d) modulated only weakly. It exists to complete Matt's 4-axis spec
# and to carry the Poisson RATE (the world map's "delta tiempo" axis), NOT to claim a precise day.
CUANDO_GRID = np.linspace(0.0, 30.0, 301)
CUANDO_SETS = {
    "inminente": [0.0, 0.0, 15.0],
    "proximo":   [0.0, 15.0, 30.0],
    "lejano":    [15.0, 30.0, 30.0],
}
CUANDO_DEFUZZ = "centroid"

# dónde: normalized radial distance of the epicenter from the segment centroid, [0,1] (0=center,
# 1=edge=95th pct of historical epicenter spread). Ruspini chain centro/medio/borde. MEASURED: this
# is the ONE secondary axis with real transient signal -- C_state -0.24, F_state -0.16, A_state
# -0.15, a_strain +0.13 (high stress-accumulation -> epicenter nearer the segment core). The coarse
# location is the segment centroid itself (what the world map plots); this output refines the
# expected radius. Azimuth within-segment is NOT resolvable from segment-aggregated inputs (would
# need per-station spatial features) -- left honest/unmodeled rather than fabricated.
DONDE_GRID = np.linspace(0.0, 1.0, 201)
DONDE_SETS = {
    "centro": [0.0, 0.0, 0.5],
    "medio":  [0.0, 0.5, 1.0],
    "borde":  [0.5, 1.0, 1.0],
}
DONDE_DEFUZZ = "centroid"

OUTPUTS = {"ocurrencia": (OCC_GRID, OCC_SETS, OCC_DEFUZZ),
           "magnitud":   (MAG_GRID, MAG_SETS, MAG_DEFUZZ),
           "cuando":     (CUANDO_GRID, CUANDO_SETS, CUANDO_DEFUZZ),
           "donde":      (DONDE_GRID, DONDE_SETS, DONDE_DEFUZZ)}


class Rule:
    """SI (var_i es set_i) [Y ...] ENTONCES (salida es set), con peso.
    antecedent: list of (variable_name, anomaly_set_name).
    consequent: (output_name, output_set_name). weight in [0,1] (rule strength / confidence,
    learned or expert). Firing strength ALSO multiplies by each antecedent variable's mu_ev (the
    per-variable "influencia" Matt's spec asks for) -- a rule on a high-mu_ev variable (fluid
    pressure, 0.90) fires more strongly than the identical membership degree on a low-mu_ev one
    (TEC, 0.25), and this is visible/inspectable, not absorbed into the learned weight."""
    __slots__ = ("antecedent", "output", "out_set", "weight", "tnorm")

    def __init__(self, antecedent, consequent, weight=1.0, tnorm="min"):
        self.antecedent = antecedent
        self.output, self.out_set = consequent
        self.weight = float(weight)
        self.tnorm = tnorm

    def firing_strength(self, mu):
        """mu[var][set] = membership degree. w_r = mu_ev * T(mu_A(x)) * peso -- exactly the
        formula in 02_motor_inferencia/mamdani-sugeno.md: mu_ev multiplies the T-NORM RESULT
        (computed on raw memberships), it does not scale each antecedent clause before combining
        them (that would distort the min-as-binding-constraint semantics for coincidence rules).
        For a multi-variable rule, mu_ev is the weakest-link (min) confidence among its variables
        -- a coincidence is only as trustworthy as its least-trusted input."""
        if not self.antecedent:
            return 0.0
        degs = []
        for v, s in self.antecedent:
            if v not in mu or s not in mu[v]:
                return 0.0
            degs.append(mu[v][s])
        t = min(degs) if self.tnorm == "min" else float(np.prod(degs))
        rule_mu_ev = min(MU_EV.get(v, 1.0) for (v, s) in self.antecedent) if self.antecedent else 1.0
        return t * rule_mu_ev * self.weight


def fuzzify_inputs(x):
    """x: dict var-> value. Returns mu[var][set] using each variable's declared universe."""
    mu = {}
    for v, val in x.items():
        f = fuzzify_variable(v, np.array([float(val)]))
        mu[v] = {s: float(f[s][0]) for s in sets_for_var(v)}
    return mu


def defuzzify(grid, agg, method):
    """centroid = area-weighted mean (right default for a genuine physical-quantity output like
    magnitud). mom = mean-of-maximum: the mean of every x where the aggregated set reaches its own
    peak height -- required for ocurrencia specifically, since centroid contracts a symmetric-
    around-0 output toward the middle (verified: a 'si' rule at w=1 centroids to only +0.55, never
    +-1) while MOM returns the rule weight exactly (verified numerically, see module history)."""
    area = agg.sum()
    if area <= 1e-9:
        return None
    if method == "centroid":
        return float((grid * agg).sum() / area)
    peak = agg.max()
    at_peak = grid[np.isclose(agg, peak, atol=1e-6)]
    return float(at_peak.mean())


def infer(x, rules):
    """Full Mamdani inference. Returns {output_name: {value, confidence, activation}}.
    activation = total rule mass on that output (0 = no rule fired -> the engine abstains rather
    than inventing a value)."""
    mu = fuzzify_inputs(x)
    out = {}
    for oname, (grid, sets, defuzz) in OUTPUTS.items():
        agg = np.zeros_like(grid)               # aggregated (max) clipped consequent
        total_w = 0.0
        for r in rules:
            if r.output != oname:
                continue
            w = r.firing_strength(mu)
            if w <= 0:
                continue
            total_w += w
            clipped = np.minimum(w, get_membership(grid, sets[r.out_set]))  # Mamdani implication
            agg = np.maximum(agg, clipped)                                   # aggregation
        value = defuzzify(grid, agg, defuzz)
        out[oname] = {"value": value, "confidence": float(agg.max()) if value is not None else 0.0,
                     "activation": float(total_w)}
    return out


# ---- initial EXPERT rule base (placeholder until the fuzzy-inverse extracts real ones) --------
# One-variable and coincidence rules mirroring the physical priors already in the project. These
# exist so the engine runs end-to-end TODAY; discover_rules.py will replace them with rules mined
# from real earthquakes (Wang-Mendel), which is the actual fuzzy-inverse.
EXPERT_RULES = [
    Rule([("a_seis", "muy_alto")], ("ocurrencia", "si"), 0.9),
    Rule([("a_seis", "alto"), ("a_foreshock", "alto")], ("ocurrencia", "si"), 0.8),
    Rule([("a_swarm", "alto"), ("a_seis", "alto")], ("ocurrencia", "si"), 0.7),
    Rule([("C_state", "muy_alto"), ("a_seis", "alto")], ("ocurrencia", "si"), 0.75),
    Rule([("a_seis", "normal"), ("a_swarm", "normal"), ("a_foreshock", "normal")],
         ("ocurrencia", "no"), 0.6),
    Rule([("a_seis", "bajo")], ("ocurrencia", "no"), 0.5),
    Rule([("a_seis", "muy_alto")], ("magnitud", "fuerte"), 0.6),
    Rule([("a_seis", "muy_alto"), ("C_state", "muy_alto")], ("magnitud", "muy_fuerte"), 0.7),
    Rule([("a_seis", "alto")], ("magnitud", "moderada"), 0.6),
]


if __name__ == "__main__":
    # demo: three synthetic states -> the engine PRODUCES output values (not a class label)
    demos = {
        "calma total":        {"a_seis": -0.2, "a_swarm": 0.0, "a_foreshock": 0.0, "C_state": 0.3},
        "agitación creciente":{"a_seis": 0.6, "a_swarm": 0.5, "a_foreshock": 0.4, "C_state": 0.7},
        "crisis":             {"a_seis": 0.9, "a_swarm": 0.8, "a_foreshock": 0.7, "C_state": 0.95},
    }
    for name, x in demos.items():
        r = infer(x, EXPERT_RULES)
        oc = r["ocurrencia"]; mg = r["magnitud"]
        oc_txt = "sin regla activa" if oc["value"] is None else \
            f"{oc['value']:+.2f} ({'SÍ' if oc['value']>0 else 'NO'}, conf {abs(oc['value']):.2f})"
        mg_txt = "—" if mg["value"] is None else f"M~{mg['value']:.1f}"
        print(f"{name:<22} ocurrencia={oc_txt:<28} magnitud={mg_txt}")
