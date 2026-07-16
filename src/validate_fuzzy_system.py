#!/usr/bin/env python3
"""Systemic validation of MOCRE fuzzy engines.

This audit checks the scientific contract that every fuzzy input/output has an
explicit universe of discourse, complete membership sets, valid rule references,
and observable participation in the current calculation artifacts.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GLOBAL = SRC / "global"
OUT = ROOT / "out"
OUTG = OUT / "global"
VALIDATION = OUT / "validation"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC_GLOBAL))

from common import ANOMALY_SETS, domain_for_var, get_membership, membership_centroid, sets_for_var  # noqa: E402
import fuzzy_avx_engine  # noqa: E402
import fuzzy_engine  # noqa: E402
import global_ia2_faithful  # noqa: E402
import global_mamdani  # noqa: E402


STATUS_ORDER = {"FAIL": 0, "WARN": 1, "PASS": 2, "INFO": 3}
INPUT_CHANNELS_17 = [
    "a_seis", "a_swarm", "a_foreshock", "a_gnss", "a_sse", "a_fluid", "a_water", "a_strain",
    "a_ocean", "a_insar", "a_tec", "a_noise", "A_state", "C_state", "K_state", "E_state",
    "F_state",
]
ANOMALY_DOMAIN = [-1.0, 1.0]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class Audit:
    def __init__(self) -> None:
        self.findings: list[dict[str, Any]] = []
        self.engines: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}

    def add(self, status: str, code: str, title: str, detail: str, evidence: Any = None) -> None:
        self.findings.append({
            "status": status,
            "code": code,
            "title": title,
            "detail": detail,
            "evidence": _jsonable(evidence),
        })

    def counts(self) -> dict[str, int]:
        c = Counter(f["status"] for f in self.findings)
        return {s: int(c.get(s, 0)) for s in ["FAIL", "WARN", "PASS", "INFO"]}


def read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def membership(sets: dict[str, list[float]], grid: np.ndarray) -> dict[str, np.ndarray]:
    return {name: get_membership(grid, params) for name, params in sets.items()}


def set_centroid(params: list[float]) -> float:
    return float(membership_centroid(params))


def geometry_errors(sets: dict[str, list[float]]) -> list[str]:
    errors: list[str] = []
    for name, params in sets.items():
        if len(params) not in (3, 4):
            errors.append(f"{name}: {len(params)} parametros; se esperaban 3 o 4")
            continue
        vals = [float(v) for v in params]
        if any(not math.isfinite(v) for v in vals):
            errors.append(f"{name}: parametros no finitos {params}")
            continue
        if any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            errors.append(f"{name}: parametros no monotonos {params}")
        if vals[-1] - vals[0] <= 1e-12:
            errors.append(f"{name}: soporte colapsado {params}")
    return errors


def validate_partition(
    audit: Audit,
    label: str,
    sets: dict[str, list[float]],
    *,
    domain: list[float] | None = None,
    expect_ruspini: bool = True,
    expect_no_holes: bool = True,
    tol: float = 1e-3,
) -> dict[str, Any]:
    errs = geometry_errors(sets)
    if errs:
        audit.add("FAIL", "SET_GEOMETRY", f"{label}: geometria invalida", "Hay conjuntos mal formados.", errs)
    else:
        audit.add("PASS", "SET_GEOMETRY", f"{label}: geometria valida", "Todos los terminos son trimf/trapmf monotonos.")

    if domain is None:
        flat = [float(x) for p in sets.values() for x in p]
        domain = [min(flat), max(flat)]
    lo, hi = map(float, domain)
    grid = np.linspace(lo, hi, 5001)
    mu = membership(sets, grid)
    sums = np.sum(np.vstack(list(mu.values())), axis=0) if mu else np.zeros_like(grid)
    min_sum = float(sums.min()) if len(sums) else 0.0
    max_sum = float(sums.max()) if len(sums) else 0.0
    max_abs_ruspini_error = float(np.max(np.abs(sums - 1.0))) if len(sums) else 1.0
    holes = int((sums <= tol).sum())
    below = int((sums < 1.0 - tol).sum())
    above = int((sums > 1.0 + tol).sum())
    metrics = {
        "domain": [lo, hi],
        "n_terms": len(sets),
        "min_sum_mu": min_sum,
        "max_sum_mu": max_sum,
        "max_abs_sum_minus_1": max_abs_ruspini_error,
        "holes_grid_points": holes,
        "below_ruspini_grid_points": below,
        "above_ruspini_grid_points": above,
        "centroids": {k: set_centroid([float(v) for v in p]) for k, p in sets.items()},
    }
    if expect_no_holes and holes:
        audit.add("FAIL", "PARTITION_HOLES", f"{label}: huecos en el universo",
                  "Hay puntos del universo donde ningun termino tiene pertenencia positiva.", metrics)
    else:
        audit.add("PASS", "PARTITION_COVERAGE", f"{label}: universo cubierto",
                  "No hay huecos de pertenencia dentro del dominio declarado.", metrics)
    if expect_ruspini:
        status = "PASS" if max_abs_ruspini_error <= tol else "WARN"
        audit.add(status, "RUSPINI", f"{label}: suma de pertenencias",
                  "Particion Ruspini si la suma de pertenencias es 1 en todo el universo.", metrics)
    return metrics


def validate_rule_refs(
    audit: Audit,
    label: str,
    rules: list[dict[str, Any]],
    input_sets_by_var: dict[str, dict[str, list[float]]],
    output_sets_by_output: dict[str, dict[str, list[float]]] | None,
) -> dict[str, Any]:
    bad: list[str] = []
    empty = 0
    by_output: Counter[str] = Counter()
    by_var: Counter[str] = Counter()
    by_var_output: dict[str, Counter[str]] = defaultdict(Counter)
    arity: Counter[int] = Counter()
    output_terms_used: dict[str, set[str]] = defaultdict(set)

    for i, r in enumerate(rules):
        ant = r.get("antecedent") or r.get("cols") or []
        output = r.get("output", r.get("target", "ocurrencia"))
        out_set = r.get("out_set") or r.get("term") or r.get("out_term")
        if not ant:
            empty += 1
            bad.append(f"regla {i}: antecedente vacio")
        arity[len(ant)] += 1
        by_output[output] += 1
        if out_set:
            output_terms_used[output].add(out_set)
        for pair in ant:
            if len(pair) != 2:
                bad.append(f"regla {i}: antecedente mal formado {pair}")
                continue
            var, term = pair
            by_var[var] += 1
            by_var_output[output][var] += 1
            if var not in input_sets_by_var:
                bad.append(f"regla {i}: variable sin universo definido {var}")
                continue
            if term not in input_sets_by_var[var]:
                bad.append(f"regla {i}: termino inexistente {var}.{term}")
        if output_sets_by_output is not None:
            if output not in output_sets_by_output:
                bad.append(f"regla {i}: salida sin universo definido {output}")
            elif out_set and out_set not in output_sets_by_output[output]:
                bad.append(f"regla {i}: consecuente inexistente {output}.{out_set}")

    evidence = {
        "n_rules": len(rules),
        "empty_antecedents": empty,
        "arity": dict(arity),
        "rules_by_output": dict(by_output),
        "rules_by_var": dict(by_var),
        "rules_by_var_output": {k: dict(v) for k, v in by_var_output.items()},
        "output_terms_used": {k: sorted(v) for k, v in output_terms_used.items()},
        "bad_refs": bad[:100],
        "bad_refs_total": len(bad),
    }
    if bad:
        audit.add("FAIL", "RULE_REFS", f"{label}: referencias invalidas en reglas",
                  "Alguna regla apunta a variables, terminos o salidas no definidos.", evidence)
    else:
        audit.add("PASS", "RULE_REFS", f"{label}: reglas referencian universos definidos",
                  "Todas las reglas apuntan a variables, terminos y salidas existentes.", evidence)
    return evidence


def validate_code_contracts(audit: Audit) -> None:
    r = fuzzy_engine.Rule([("a", "alto"), ("b", "alto")], ("ocurrencia", "si"), 1.0)
    strength = r.firing_strength({"a": {"alto": 1.0}})
    if strength == 0.0:
        audit.add("PASS", "STRICT_ANTECEDENT", "Mamdani local no dispara reglas incompletas",
                  "Una regla con A y B devuelve fuerza 0 si B no esta presente.")
    else:
        audit.add("FAIL", "STRICT_ANTECEDENT", "Mamdani local dispara reglas incompletas",
                  "Una regla A y B se activo con solo A presente.", {"strength": strength})

    value, strengths, label = fuzzy_avx_engine.infer_value(
        {"a": 0.8},
        {"a": ANOMALY_SETS, "b": ANOMALY_SETS},
        [([("a", "alto"), ("b", "alto")], "t1")],
        {"t1": [0.0, 0.5, 1.0]},
    )
    if value is None and label is None and max(strengths.values()) == 0.0:
        audit.add("PASS", "STRICT_ANTECEDENT", "IA2/IA3 no dispara reglas incompletas",
                  "Una regla con A y B se abstiene si B no esta presente.")
    else:
        audit.add("FAIL", "STRICT_ANTECEDENT", "IA2/IA3 dispara reglas incompletas",
                  "Una regla A y B produjo salida con solo A presente.",
                  {"value": value, "strengths": strengths, "label": label})


def validate_data_ranges(
    audit: Audit,
    path: Path,
    label: str,
    required_columns: list[str],
    range_columns: list[str],
    domains: dict[str, list[float]] | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        audit.add("FAIL", "DATA_MISSING", f"{label}: archivo ausente", str(path))
        return None
    df = pd.read_csv(path)
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        audit.add("FAIL", "DATA_COLUMNS", f"{label}: columnas faltantes", "Faltan columnas requeridas.", missing_cols)
    else:
        audit.add("PASS", "DATA_COLUMNS", f"{label}: columnas requeridas presentes",
                  "El artefacto de datos contiene las columnas esperadas.")

    stats: dict[str, Any] = {"rows": int(len(df)), "columns": list(df.columns), "channels": {}}
    out_of_range: list[str] = []
    high_nan: list[str] = []
    for c in range_columns:
        if c not in df.columns:
            continue
        lo, hi = (domains or {}).get(c, [-1.0, 1.0])
        s = pd.to_numeric(df[c], errors="coerce")
        finite = s[np.isfinite(s)]
        nan_rate = float(s.isna().mean())
        col_stats = {
            "min": None if finite.empty else float(finite.min()),
            "max": None if finite.empty else float(finite.max()),
            "mean": None if finite.empty else float(finite.mean()),
            "nan_rate": nan_rate,
        }
        stats["channels"][c] = col_stats
        if not finite.empty and (finite.min() < lo - 0.0001 or finite.max() > hi + 0.0001):
            out_of_range.append(f"{c}: min={finite.min():.4f}, max={finite.max():.4f}")
        if nan_rate > 0.05:
            high_nan.append(f"{c}: nan_rate={nan_rate:.3f}")
    if out_of_range:
        audit.add("FAIL", "DATA_RANGE", f"{label}: canales fuera de su universo",
                  "Canales fuzzy salieron del universo declarado por variable.", out_of_range)
    else:
        audit.add("PASS", "DATA_RANGE", f"{label}: canales dentro de su universo",
                  "Las columnas fuzzy normalizadas respetan su universo.")
    if high_nan:
        audit.add("WARN", "DATA_NAN", f"{label}: NaN relevantes en canales",
                  "Hay canales con mas de 5% de valores NaN; el entrenamiento/inferencia puede rellenarlos.", high_nan)
    else:
        audit.add("PASS", "DATA_NAN", f"{label}: NaN bajo control",
                  "No hay canales con mas de 5% de NaN.")
    return stats


def rule_dict_from_fuzzy_rule(rule: fuzzy_engine.Rule) -> dict[str, Any]:
    return {
        "antecedent": [list(p) for p in rule.antecedent],
        "output": rule.output,
        "out_set": rule.out_set,
        "weight": rule.weight,
    }


def validate_local_mamdani(audit: Audit) -> None:
    input_sets = {v: sets_for_var(v) for v in fuzzy_engine.MU_EV}
    output_sets = {name: sets for name, (_grid, sets, _defuzz) in fuzzy_engine.OUTPUTS.items()}
    audit.manifest["local_mamdani"] = {
        "input_universes": {v: {"domain": domain_for_var(v), "sets": sets_for_var(v)} for v in fuzzy_engine.MU_EV},
        "output_universes": {
            "ocurrencia": {"domain": [-1.0, 1.0], "sets": fuzzy_engine.OCC_SETS, "defuzz": fuzzy_engine.OCC_DEFUZZ},
            "magnitud": {"domain": [3.0, 8.5], "sets": fuzzy_engine.MAG_SETS, "defuzz": fuzzy_engine.MAG_DEFUZZ},
            "cuando": {"domain": [0.0, 30.0], "sets": fuzzy_engine.CUANDO_SETS, "defuzz": fuzzy_engine.CUANDO_DEFUZZ},
            "donde": {"domain": [0.0, 1.0], "sets": fuzzy_engine.DONDE_SETS, "defuzz": fuzzy_engine.DONDE_DEFUZZ},
        },
    }
    validate_partition(audit, "local_mamdani.inputs.ANOMALY_SETS", ANOMALY_SETS, domain=ANOMALY_DOMAIN)
    validated_input_sets = {id(ANOMALY_SETS)}
    for var in fuzzy_engine.MU_EV:
        sets = sets_for_var(var)
        if id(sets) in validated_input_sets:
            continue
        validate_partition(audit, f"local_mamdani.input.{var}", sets, domain=domain_for_var(var))
        validated_input_sets.add(id(sets))
    for name, (grid, sets, _defuzz) in fuzzy_engine.OUTPUTS.items():
        validate_partition(audit, f"local_mamdani.output.{name}", sets,
                           domain=[float(grid.min()), float(grid.max())])

    expert_rules = [rule_dict_from_fuzzy_rule(r) for r in fuzzy_engine.EXPERT_RULES]
    expert = validate_rule_refs(audit, "local_mamdani.EXPERT_RULES", expert_rules, input_sets, output_sets)
    learned_path = OUT / "learned_rules.json"
    learned_rules = read_json(learned_path) if learned_path.exists() else []
    learned = validate_rule_refs(audit, "local_mamdani.out.learned_rules", learned_rules,
                                 input_sets | {"mag_scale": sets_for_var("mag_scale")}, output_sets)
    all_defined = set(input_sets) | {"mag_scale"}
    used_learned = set(learned["rules_by_var"])
    unused = sorted((set(INPUT_CHANNELS_17) | {"mag_scale"}) - used_learned)
    if unused:
        audit.add("WARN", "VARIABLE_COVERAGE", "learned_rules: variables definidas sin regla",
                  "Estas variables tienen universo definido pero no participan en learned_rules.json.", unused)
    else:
        audit.add("PASS", "VARIABLE_COVERAGE", "learned_rules: todas las variables esperadas participan",
                  "Todas las variables esperadas aparecen al menos una vez en las reglas aprendidas.")
    audit.engines["local_mamdani"] = {
        "expert_rules": expert,
        "learned_rules": learned,
        "defined_input_vars": sorted(all_defined),
    }


def ia3_output_raw(model: dict[str, Any], output: str, state: dict[str, float]) -> float | None:
    rules = model["outputs"][output]["rules"]
    sets = model["outputs"][output]["trained_sets"]
    strengths: dict[str, float] = {}
    for v, s, term in rules:
        val = state.get(v)
        if val is None:
            continue
        mu = float(get_membership(np.array([float(val)]), sets_for_var(v)[s])[0])
        if mu > strengths.get(term, 0.0):
            strengths[term] = mu
    num = den = 0.0
    for term, strength in strengths.items():
        if strength > 0 and term in sets:
            c = sum(sets[term]) / 3.0
            num += c * strength
            den += strength
    return float(num / den) if den > 0 else None


def validate_ia3_model(audit: Audit) -> None:
    path = OUT / "ia3_model.json"
    if not path.exists():
        audit.add("FAIL", "MODEL_MISSING", "ia3_model.json ausente", str(path))
        return
    model = read_json(path)
    model_channels = list(model.get("channels", []))
    expected_inputs = model_channels + ["mag_scale"]
    input_sets = {v: sets_for_var(v) for v in expected_inputs}
    output_sets = {name: entry["trained_sets"] for name, entry in model["outputs"].items()}
    audit.manifest["segment_ia3_4outputs"] = {
        "input_universes": {v: {"domain": domain_for_var(v), "sets": sets_for_var(v)} for v in expected_inputs},
        "output_universes": {
            name: {"domain": entry["universe"], "sets": entry["trained_sets"], "defuzz": "weighted_centroid"}
            for name, entry in model["outputs"].items()
        },
    }

    if sorted(model_channels) == sorted(INPUT_CHANNELS_17):
        audit.add("PASS", "MODEL_CHANNELS", "ia3_model: 17 canales dinamicos declarados",
                  "El modelo serializado contiene los 17 canales dinamicos esperados.", model_channels)
    else:
        audit.add("FAIL", "MODEL_CHANNELS", "ia3_model: canales dinamicos incompletos",
                  "El modelo serializado no coincide con los 17 canales esperados.",
                  {"expected": INPUT_CHANNELS_17, "actual": model_channels})

    all_rules: list[dict[str, Any]] = []
    per_output_used: dict[str, Any] = {}
    for name, entry in model["outputs"].items():
        validate_partition(audit, f"ia3_model.output.{name}", entry["trained_sets"],
                           domain=entry["universe"], expect_ruspini=False, expect_no_holes=True)
        rules = [{"antecedent": [[v, s]], "output": name, "out_set": term}
                 for v, s, term in entry["rules"]]
        all_rules.extend(rules)
        used = sorted({v for v, _s, _term in entry["rules"]})
        unused_for_output = sorted(set(expected_inputs) - set(used))
        per_output_used[name] = {
            "used_vars": used,
            "unused_vars_for_output": unused_for_output,
            "n_rules": len(entry["rules"]),
            "universe": entry["universe"],
            "mae_ia3": entry.get("mae_ia3"),
            "mae_baseline": entry.get("mae_baseline"),
            "brier_cal": entry.get("brier_cal"),
        }
        if name == "magnitud" and used == ["mag_scale"]:
            audit.add("INFO", "VARIABLE_COVERAGE", "ia3 magnitud usa solo mag_scale",
                      "Decision explicita del entrenamiento: la magnitud se atribuye a escala tectonica, no al estado transitorio.",
                      per_output_used[name])
        elif unused_for_output:
            audit.add("WARN", "VARIABLE_COVERAGE", f"ia3 {name}: variables no usadas por esta salida",
                      "Tienen universo definido, pero esta salida no tiene reglas que las usen.", per_output_used[name])
        else:
            audit.add("PASS", "VARIABLE_COVERAGE", f"ia3 {name}: todas las entradas esperadas usadas",
                      "Cada variable esperada aparece en al menos una regla de esta salida.", per_output_used[name])
        mae_ia3 = entry.get("mae_ia3")
        mae_baseline = entry.get("mae_baseline")
        if mae_ia3 is not None and mae_baseline is not None:
            if float(mae_ia3) <= float(mae_baseline):
                audit.add("PASS", "SKILL", f"ia3 {name}: MAE supera baseline",
                          "La salida entrenada reduce error frente a la media/base persistida.",
                          {"mae_ia3": mae_ia3, "mae_baseline": mae_baseline})
            else:
                audit.add("WARN", "SKILL", f"ia3 {name}: MAE no supera baseline",
                          "La salida existe y tiene universo/reglas, pero no demuestra skill frente a baseline.",
                          {"mae_ia3": mae_ia3, "mae_baseline": mae_baseline})
    refs = validate_rule_refs(audit, "ia3_model.rules", all_rules, input_sets, output_sets)
    brier = model.get("occ_brier_holdout")
    brier_base = model.get("occ_brier_base")
    if brier is not None and brier_base is not None:
        if float(brier) <= float(brier_base):
            audit.add("PASS", "SKILL", "ia3 ocurrencia: Brier supera base-rate",
                      "La probabilidad calibrada mejora el Brier frente a tasa base.",
                      {"occ_brier_holdout": brier, "occ_brier_base": brier_base})
        else:
            audit.add("WARN", "SKILL", "ia3 ocurrencia: Brier peor que base-rate",
                      "La probabilidad calibrada no debe presentarse como skill prospectiva en este artefacto.",
                      {"occ_brier_holdout": brier, "occ_brier_base": brier_base})

    sensitivity: dict[str, dict[str, float]] = {}
    for output in model["outputs"]:
        sensitivity[output] = {}
        for var in expected_inputs:
            var_rules = [r for r in model["outputs"][output]["rules"] if r[0] == var]
            if not var_rules:
                sensitivity[output][var] = 0.0
                continue
            vlo, vhi = map(float, domain_for_var(var))
            probes = {vlo, vhi, (vlo + vhi) / 2.0}
            for params in sets_for_var(var).values():
                probes.update(float(x) for x in params if vlo <= float(x) <= vhi)
            vals = []
            original_rules = model["outputs"][output]["rules"]
            model["outputs"][output]["rules"] = var_rules
            try:
                for probe in sorted(probes):
                    pred = ia3_output_raw(model, output, {var: probe})
                    if pred is not None:
                        vals.append(pred)
            finally:
                model["outputs"][output]["rules"] = original_rules
            if len(vals) < 2:
                diff = 0.0
            else:
                diff = max(vals) - min(vals)
            sensitivity[output][var] = float(diff)
        declared = set(per_output_used[output]["used_vars"])
        dead = sorted(v for v in declared if sensitivity[output].get(v, 0.0) <= 1e-9)
        if dead:
            audit.add("WARN", "SENSITIVITY", f"ia3 {output}: reglas sin sensibilidad aislada",
                      "La variable aparece en reglas, pero aislada no cambia la salida sobre su dominio declarado.",
                      {"dead_vars": dead, "sensitivity": sensitivity[output]})
        else:
            audit.add("PASS", "SENSITIVITY", f"ia3 {output}: variables usadas cambian la salida",
                      "Las variables declaradas por reglas tienen efecto observable en una prueba de perturbacion.",
                      sensitivity[output])
    audit.engines["segment_ia3_4outputs"] = {
        "model_channels": model_channels,
        "expected_inputs": expected_inputs,
        "rule_refs": refs,
        "per_output": per_output_used,
        "sensitivity": sensitivity,
        "occ_brier_holdout": model.get("occ_brier_holdout"),
        "occ_brier_base": model.get("occ_brier_base"),
    }


def validate_global_v2(audit: Audit) -> None:
    path = OUTG / "global_engine_v2.json"
    if not path.exists():
        audit.add("FAIL", "MODEL_MISSING", "global_engine_v2.json ausente", str(path))
        return
    eng = read_json(path)
    channels = eng.get("channels", [])
    sets5 = eng.get("sets5", global_mamdani.SETS5)
    input_sets = {v: sets5 for v in channels}
    output_sets = {"ocurrencia": {"no": [-1.0, -1.0, 1.0], "si": [-1.0, 1.0, 1.0]}}
    audit.manifest["global_v2_mamdani"] = {
        "input_universes": {v: {"domain": ANOMALY_DOMAIN, "sets": sets5} for v in channels},
        "output_universes": {
            "ocurrencia": {"domain": [-1.0, 1.0], "sets": output_sets["ocurrencia"], "defuzz": "mom"},
            "magnitud": {"domain": [3.0, 8.5], "sets": eng.get("mag_sets", {}), "defuzz": "weighted_centroid"},
        },
    }
    validate_partition(audit, "global_v2.inputs.SETS5", sets5, domain=ANOMALY_DOMAIN)
    if set(channels) == set(global_mamdani.CH):
        audit.add("PASS", "MODEL_CHANNELS", "global_v2: canales declarados coinciden con codigo",
                  "El JSON persistido declara los canales que global_mamdani.CH espera.", channels)
    else:
        audit.add("FAIL", "MODEL_CHANNELS", "global_v2: canales no coinciden con codigo",
                  "El JSON persistido no coincide con global_mamdani.CH.",
                  {"json": channels, "code": global_mamdani.CH})

    rules = [{"antecedent": r["cols"], "output": "ocurrencia", "out_set": "si" if r.get("lift", 1.0) > 1 else "no"}
             for r in eng.get("rules", [])]
    refs = validate_rule_refs(audit, "global_v2.rules", rules, input_sets, output_sets)
    unused = sorted(set(channels) - set(refs["rules_by_var"]))
    if unused:
        audit.add("WARN", "VARIABLE_COVERAGE", "global_v2: canales sin reglas",
                  "Canales con universo definido pero sin reglas persistidas.", unused)
    else:
        audit.add("PASS", "VARIABLE_COVERAGE", "global_v2: todos los canales participan",
                  "Cada canal declarado aparece en al menos una regla persistida.")

    mu_ev = eng.get("mu_ev", {})
    bad_mu_ev = {c: mu_ev.get(c) for c in channels if c not in mu_ev or not (0.0 <= float(mu_ev.get(c, -1)) <= 1.0)}
    if bad_mu_ev:
        audit.add("FAIL", "MU_EV", "global_v2: mu_ev incompleto o fuera de rango",
                  "La influencia por variable debe existir y estar en [0,1].", bad_mu_ev)
    else:
        audit.add("PASS", "MU_EV", "global_v2: mu_ev completo",
                  "Todas las variables tienen influencia medida en [0,1].", mu_ev)

    if "mag_sets" in eng:
        validate_partition(audit, "global_v2.output.magnitud.mag_sets", eng["mag_sets"],
                           domain=[3.0, 8.5], expect_ruspini=False, expect_no_holes=False)
        cents = {k: set_centroid(v) for k, v in eng["mag_sets"].items()}
        if any(not (3.0 <= v <= 8.5) for v in cents.values()):
            audit.add("FAIL", "OUTPUT_UNIVERSE", "global_v2 magnitud: centroides fuera de Richter",
                      "Los centroides IA3 de magnitud salieron del universo [3,8.5].", cents)
        else:
            audit.add("PASS", "OUTPUT_UNIVERSE", "global_v2 magnitud: centroides dentro de Richter",
                      "Los centroides IA3 de magnitud quedan dentro del universo fisico.", cents)
    else:
        audit.add("FAIL", "OUTPUT_UNIVERSE", "global_v2: mag_sets ausente",
                  "El motor global emite magnitud pero el JSON no trae sus conjuntos.")

    audit.engines["global_v2_mamdani"] = {
        "channels": channels,
        "rule_refs": refs,
        "mu_ev": mu_ev,
        "walkforward": eng.get("walkforward", []),
        "target": eng.get("target"),
    }


def validate_global_v3(audit: Audit) -> None:
    path = OUTG / "global_engine_v3.json"
    if not path.exists():
        audit.add("FAIL", "MODEL_MISSING", "global_engine_v3.json ausente", str(path))
        return
    eng = read_json(path)
    var_sets = eng.get("var_sets", {})
    channels = list(var_sets)
    audit.manifest["global_v3_ia2_ia3"] = {
        "input_universes": {
            v: {"domain": [min(x for p in sets.values() for x in p), max(x for p in sets.values() for x in p)],
                "sets": sets}
            for v, sets in var_sets.items()
        },
        "output_universes": {
            "ocurrencia": {
                "domain": [0.0, 1.0],
                "terms": eng.get("out_names", []),
                "centroids_ia3": eng.get("out_centroids", []),
                "defuzz": "weighted_centroid",
            }
        },
    }
    expected = set(global_ia2_faithful.CH_ALL)
    if set(channels) == expected:
        audit.add("PASS", "MODEL_CHANNELS", "global_v3: var_sets completos",
                  "El motor fiel trae universos por variable para IA1 y los 7 canales raw.", channels)
    else:
        audit.add("FAIL", "MODEL_CHANNELS", "global_v3: var_sets incompletos",
                  "Faltan universos por variable en el motor fiel.",
                  {"expected": sorted(expected), "actual": sorted(channels)})
    for v, sets in var_sets.items():
        flat = [float(x) for p in sets.values() for x in p]
        validate_partition(audit, f"global_v3.input.{v}.var_sets", sets,
                           domain=[min(flat), max(flat)], expect_ruspini=True)

    input_sets = {v: sets for v, sets in var_sets.items()}
    rules = [{"antecedent": r["cols"], "output": "ocurrencia", "out_set": None} for r in eng.get("rules", [])]
    refs = validate_rule_refs(audit, "global_v3.rules", rules, input_sets, None)
    unused = sorted(set(channels) - set(refs["rules_by_var"]))
    if unused:
        audit.add("WARN", "VARIABLE_COVERAGE", "global_v3: variables sin reglas",
                  "Variables con universo definido pero sin reglas persistidas.", unused)
    else:
        audit.add("PASS", "VARIABLE_COVERAGE", "global_v3: todas las variables participan",
                  "Cada variable con universo aparece en al menos una regla.")
    cents = eng.get("out_centroids", [])
    if len(cents) == len(eng.get("out_names", [])) and all(0.0 <= float(c) <= 1.0 for c in cents):
        ordered = all(float(cents[i]) <= float(cents[i + 1]) for i in range(len(cents) - 1))
        audit.add("PASS" if ordered else "FAIL", "OUTPUT_UNIVERSE", "global_v3 salida: centroides IA3",
                  "Los centroides de salida deben estar en [0,1] y mantener orden linguistico.",
                  {"out_names": eng.get("out_names", []), "out_centroids": cents, "ordered": ordered})
    else:
        audit.add("FAIL", "OUTPUT_UNIVERSE", "global_v3 salida: centroides invalidos",
                  "Faltan centroides o estan fuera de [0,1].",
                  {"out_names": eng.get("out_names", []), "out_centroids": cents})
    audit.engines["global_v3_ia2_ia3"] = {
        "channels": channels,
        "rule_refs": refs,
        "out_names": eng.get("out_names", []),
        "out_centroids": cents,
        "walkforward": eng.get("walkforward", []),
        "base": eng.get("base"),
    }


def validate_live_global_data(audit: Audit) -> None:
    path = OUTG / "cells_live.csv"
    if not path.exists():
        audit.add("FAIL", "DATA_MISSING", "global cells_live ausente", str(path))
        return
    live = pd.read_csv(path)
    required = ["rate_bg", "n7", "n30", "n90", "n365", "t_last", "mag_max", "omori", "la", "lo", "slip"]
    missing = [c for c in required if c not in live.columns]
    if missing:
        audit.add("FAIL", "DATA_COLUMNS", "cells_live: columnas faltantes", "Faltan columnas raw para derivar canales.", missing)
        return
    audit.add("PASS", "DATA_COLUMNS", "cells_live: columnas raw presentes",
              "Las columnas necesarias para derivar global v2/v3 estan presentes.",
              {"rows": len(live), "columns": list(live.columns)})
    chan_v2 = global_mamdani.derive_channels(live)
    range_bad = {}
    for c in global_mamdani.CH:
        s = chan_v2[c]
        if s.min() < -1.0001 or s.max() > 1.0001 or not np.isfinite(s).all():
            range_bad[c] = {"min": float(s.min()), "max": float(s.max()), "finite": bool(np.isfinite(s).all())}
    if range_bad:
        audit.add("FAIL", "DATA_RANGE", "global v2 live: canales derivados fuera de universo",
                  "derive_channels produjo valores fuera de [-1,1] o no finitos.", range_bad)
    else:
        audit.add("PASS", "DATA_RANGE", "global v2 live: canales derivados validos",
                  "Todos los canales derivados caen dentro de [-1,1].",
                  {c: {"min": float(chan_v2[c].min()), "max": float(chan_v2[c].max())} for c in global_mamdani.CH})

    eng3_path = OUTG / "global_engine_v3.json"
    if eng3_path.exists():
        eng3 = read_json(eng3_path)
        chan_v3, _taus, _cls = global_ia2_faithful.derive_channels_classed(live, taus=eng3["taus"])
        bad = {}
        for c in global_ia2_faithful.CH_RAW:
            s = chan_v3[c]
            if s.min() < -1.0001 or s.max() > 1.0001 or not np.isfinite(s).all():
                bad[c] = {"min": float(s.min()), "max": float(s.max()), "finite": bool(np.isfinite(s).all())}
        if bad:
            audit.add("FAIL", "DATA_RANGE", "global v3 live: canales derivados fuera de universo",
                      "derive_channels_classed produjo valores fuera de [-1,1] o no finitos.", bad)
        else:
            audit.add("PASS", "DATA_RANGE", "global v3 live: canales derivados validos",
                      "Todos los canales raw de v3 caen dentro de [-1,1].",
                      {c: {"min": float(chan_v3[c].min()), "max": float(chan_v3[c].max())}
                       for c in global_ia2_faithful.CH_RAW})

    globe_path = ROOT / "globe" / "globe_data.json"
    if globe_path.exists():
        globe = read_json(globe_path)
        fields = globe.get("fields", [])
        expected_fields = ["a_seis", "a_swarm", "a_fore", "A_state", "C_state", "mag_scale", "a_tect", "ia1", "omori", "rate_bg", "n30"]
        missing_fields = [c for c in expected_fields if c not in fields]
        if missing_fields:
            audit.add("FAIL", "GLOBE_FIELDS", "globe_data: entradas no expuestas",
                      "El panel/globo no expone todos los campos de verificacion esperados.", missing_fields)
        else:
            audit.add("PASS", "GLOBE_FIELDS", "globe_data: entradas fuzzy expuestas",
                      "El JSON del globo incluye los canales de entrada usados por el motor global.", fields)


def write_reports(audit: Audit) -> tuple[Path, Path, Path]:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {
        "generated_utc": generated,
        "summary": audit.counts(),
        "findings": audit.findings,
        "engines": audit.engines,
        "universe_manifest": audit.manifest,
    }
    json_path = VALIDATION / "fuzzy_system_validation.json"
    manifest_path = VALIDATION / "fuzzy_universe_manifest.json"
    md_path = VALIDATION / "fuzzy_system_validation.md"
    json_path.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")
    manifest_path.write_text(json.dumps(_jsonable(audit.manifest), indent=2, ensure_ascii=False) + "\n")

    lines: list[str] = []
    counts = audit.counts()
    lines.append("# Validacion sistemica de motores difusos MOCRE")
    lines.append("")
    lines.append(f"Generado UTC: {generated}")
    lines.append("")
    lines.append(f"Resumen: FAIL={counts['FAIL']} | WARN={counts['WARN']} | PASS={counts['PASS']} | INFO={counts['INFO']}")
    lines.append("")
    lines.append("## Hallazgos")
    for status in ["FAIL", "WARN", "PASS", "INFO"]:
        items = [f for f in audit.findings if f["status"] == status]
        if not items:
            continue
        lines.append("")
        lines.append(f"### {status}")
        for f in items:
            lines.append(f"- **{f['title']}** (`{f['code']}`): {f['detail']}")
    lines.append("")
    lines.append("## Cobertura por motor")
    for name, engine in audit.engines.items():
        lines.append("")
        lines.append(f"### {name}")
        if "defined_input_vars" in engine:
            lines.append(f"- Variables definidas: {', '.join(engine['defined_input_vars'])}")
        if "expected_inputs" in engine:
            lines.append(f"- Entradas esperadas: {', '.join(engine['expected_inputs'])}")
        if "channels" in engine:
            lines.append(f"- Canales: {', '.join(engine['channels'])}")
        refs = engine.get("rule_refs") or engine.get("learned_rules")
        if refs:
            lines.append(f"- Reglas: {refs.get('n_rules')} | aridad: {refs.get('arity')} | variables usadas: {', '.join(refs.get('rules_by_var', {}).keys())}")
        if "per_output" in engine:
            for out_name, entry in engine["per_output"].items():
                lines.append(f"- {out_name}: reglas={entry['n_rules']} | usadas={', '.join(entry['used_vars'])}")
        if "walkforward" in engine:
            lines.append(f"- Walk-forward persistido: {len(engine['walkforward'])} epocas")
    lines.append("")
    lines.append("## Artefactos")
    lines.append(f"- JSON completo: `{json_path.relative_to(ROOT)}`")
    lines.append(f"- Manifest de universos: `{manifest_path.relative_to(ROOT)}`")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path, manifest_path


def main() -> int:
    audit = Audit()
    validate_code_contracts(audit)
    validate_local_mamdani(audit)
    validate_ia3_model(audit)
    validate_global_v2(audit)
    validate_global_v3(audit)
    validate_data_ranges(
        audit,
        OUT / "dense_training.csv",
        "dense_training",
        INPUT_CHANNELS_17 + ["mag_scale", "occ_30d", "mag_next", "days_next", "radial_next", "block"],
        INPUT_CHANNELS_17 + ["mag_scale"],
        {c: domain_for_var(c) for c in INPUT_CHANNELS_17 + ["mag_scale"]},
    )
    validate_data_ranges(
        audit,
        OUT / "cases_enriched.csv",
        "cases_enriched",
        INPUT_CHANNELS_17 + ["lam_etas", "f_30d", "f_90d", "segment", "date", "block"],
        INPUT_CHANNELS_17,
        {c: domain_for_var(c) for c in INPUT_CHANNELS_17},
    )
    validate_live_global_data(audit)
    json_path, md_path, manifest_path = write_reports(audit)
    counts = audit.counts()
    print(f"validation written: {json_path}")
    print(f"markdown report:    {md_path}")
    print(f"universe manifest:  {manifest_path}")
    print(f"summary: FAIL={counts['FAIL']} WARN={counts['WARN']} PASS={counts['PASS']} INFO={counts['INFO']}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
