"""MOCRE-1 canonical variable inventory GENERATOR (created 2026-07-04).

WHY this exists instead of a hand-written markdown: the static docs (01_fundamentos/universo-
discurso-particiones.md = 2-jul, 03_ejemplos/sismologia-prediccion-alerta.md = 3-jul) fell out of
sync with the code within ONE day of heavy work (K_state gate, ocean/TEC/InSAR channels, WSM
prior, tau recalibration all landed 4-jul). A markdown "map of variables" is a lie the moment the
next channel is added. This reads the ACTUAL sources of truth -- pipeline.py's RULE_NAMES /
DEFAULT_CONSEQUENTS, each config's tau_*, and the real files on disk per segment -- and emits the
live state. Run `python3 inventory.py` for the console table, `python3 inventory.py --md > path`
to refresh the readable snapshot. The snapshot is a CONVENIENCE cache; this script is the truth.

The 25-variable catalog (see the FuzzyCore seismology design notes, audited 2026-07-03) is the
design target. Each catalog variable maps to at most one code channel; some code
constructs (C_state, ΔCFS, historial, geometría) are internal/static and realize a catalog
variable without a dedicated a_X anomaly channel. Ocean loading is a genuine ADDITION not in the
original 25 (found+added 4-jul), flagged as such.
"""
import json
import os

import numpy as np

from common import ROOT, load_config, segment_paths
from pipeline import (DEFAULT_CONSEQUENTS, RULE_NAMES, build_daily, build_state, load_catalog,
                      load_gnss)

SEGMENTS = ["segment_sjc.json", "segment_parkfield.json", "segment_mojave.json",
            "segment_cascadia.json", "segment_shumagin.json", "segment_oklahoma.json",
            "segment_chile_central.json", "segment_italy_apennines.json", "segment_japan_nankai.json"]
N_SEGMENTS = len(SEGMENTS)

# Catalog variable registry. Each entry: (catalog_name, cluster, mu_ev, channel, kind, tau_key,
# rules, status, note). channel=None means no dedicated a_X. status is the honest current state.
#   kind:   'dynamic'  -> a_X(t) anomaly channel fed to fuzzify_anomaly (shares ANOMALY_SETS)
#           'state'    -> persistent EWMA/accumulator state (A/C/K/E/F), not directly fuzzified raw
#           'static'   -> fixed per-segment parameter/prior (no time axis)
#           'absent'   -> catalog variable with no implementation
REGISTRY = [
    # MOCRE-2 regional preparation features (outside original catalog)
    ("Actividad sísmica regional", "R", None, "a_regional_rate", "dynamic", None, [], "active",
     "MOCRE-2: anomalía Poisson, bbox regional, causal"),
    ("Expansión espacial regional", "R", None, "a_regional_spread", "dynamic", None, [], "active",
     "MOCRE-2: ocupación espacial 30d normalizada por actividad"),
    ("Caída de b-value", "R", None, "a_bvalue_drop", "dynamic", None, [], "active",
     "MOCRE-2: diferencia MLE reciente vs baseline"),
    ("Concentración hacia la falla", "R", None, "a_fault_concentration", "dynamic", None, [], "active",
     "MOCRE-2: cambio de distancia radial media"),
    ("Aceleración de magnitud regional", "R", None, "a_mag_accel", "dynamic", None, [], "active",
     "MOCRE-2: magnitud media reciente vs baseline"),
    # cluster A -- seismicity & rupture
    ("Microsismicidad (tasa M<3)", "A", 0.50, "a_seis", "dynamic", "tau_seismicity",
     ["seis_bajo", "seis_muy_bajo", "seis_normal", "seis_alto", "seis_muy_alto"], "active", ""),
    ("Foreshocks", "A", 0.40, "a_foreshock", "dynamic", "tau_foreshock",
     ["foreshock_calmo", "foreshock_agudo"], "active", "algorithmic (7d/30d accel), no new data"),
    ("Enjambres sísmicos", "A", 0.55, "a_swarm", "dynamic", "tau_swarm",
     ["enjambre_bajo", "enjambre_solo"], "active", "+ E_state EWMA; algorithmic, no new data"),
    ("Cambios Vp/Vs", "A", 0.475, None, "absent", None, [], "wall_infra",
     "VERIFIED: no published time series exists; dv/v pivot needs Julia+GB waveform "
     "cross-correlation infra -- a signal-processing project, not data integration"),
    ("Ruido sísmico ambiental", "A", 0.35, "a_noise", "dynamic", "tau_noise",
     ["noise_bajo", "noise_alto"], "active",
     "NEW 4-jul (3-jul 'blocked' was a bad-metric error): MUSTANG sample_rms, 6 segs, "
     "level-proxy not dv/v, water tier"),
    # cluster B -- crustal deformation
    ("GNSS/GPS interseísmico", "B", 0.85, "a_gnss", "dynamic", "tau_gnss",
     ["coincid_seis_gnss"], "active", "tau recalibrated 4-jul 2.0->7.5"),
    ("InSAR", "B", 0.85, "a_insar", "dynamic", "tau_insar",
     ["insar_bajo", "insar_alto"], "active", "5/9; LOWEST tier, atmo-noise-limited"),
    ("Inclinómetros/extensómetros (strain)", "B", 0.675, "a_strain", "dynamic", "tau_strain",
     ["strain_bajo", "strain_alto", "coincid_seis_strain"], "active",
     "4 segments w/ real PB net; tau recalibrated 4-jul 3.0->6.3"),
    ("Slow Slip Events", "B", 0.60, "a_sse", "dynamic", "tau_sse",
     ["sse_solo", "sse_x_seis"], "active",
     "+ K_state EWMA; gated OFF for regime=induced (4-jul); tau recal 3.0->10.9"),
    # cluster C -- stress & mechanical state
    ("Acumulación de esfuerzo", "C", 0.50, "C_state", "state", None,
     ["seis_x_carga", "agitacion_x_carga"], "active",
     "internal accumulator from recurrence + GNSS loading + ΔCFS"),
    ("Transferencia Coulomb (ΔCFS)", "C", 0.70, "C_state", "state", None, [], "active",
     "stress_graph.py point-source approx, feeds C_state"),
    ("Presión de fluidos subterráneos", "C", 0.90, "a_fluid", "dynamic", "tau_fluid",
     ["fluid_bajo", "fluid_muy_bajo", "fluid_solo", "fluid_x_seis"], "active",
     "+ F_state; only regime=induced (Oklahoma OCC wells)"),
    ("Acumulación esfuerzo — orientación (World Stress Map)", "C", 0.50, None, "static", None,
     [], "prior_only", "NEW 4-jul: static SHmax/regime prior per segment (data/stress_map/), "
     "NOT a temporal channel; diagnostic in stress_graph.py, not yet in G"),
    # cluster D -- geochemistry & fluids
    ("Radón", "D", 0.25, None, "absent", None, [], "wall_data",
     "VERIFIED wall 4-jul: WQP grab-samples end 2014-2018, IRON=Italy-only frozen 2021, "
     "Shumagin=0. No continuous open series exists for these segments -- needs field hardware"),
    ("Helio/metano/hidrógeno", "D", 0.30, None, "absent", None, [], "wall_data",
     "VERIFIED wall 4-jul: only a static point-compilation (Zenodo HEDB), 0 continuous series; "
     "California springs sampled once in 1983/1997/2013, Shumagin/Oklahoma=0"),
    ("Nivel de aguas subterráneas", "D", 0.35, "a_water", "dynamic", "tau_water",
     ["agua_bajo", "agua_alto"], "code_ready_no_data",
     "code done, USGS OGC API rate-limited, no data harvested"),
    ("Gases volcánicos (SO2, CO2)", "D", 0.70, None, "absent", None, [], "not_applicable",
     "volcanic context only -- none of the 9 segments"),
    # cluster E -- remote/satellite geophysics
    ("Campo gravitatorio", "E", 0.30, None, "absent", None, [], "low_value",
     "VERIFIED 4-jul: GRACE-FO mascon open+current but ~300km res groups Anza+Mojave into 1 cell, "
     "monthly cadence, measures water-equiv not tectonic stress -- scale/mechanism mismatch"),
    ("TEC ionosférico", "E", 0.25, "a_tec", "dynamic", "tau_tec",
     ["tec_bajo", "tec_alto"], "active",
     "NEW 4-jul: JPL GIM, 6 segments; real data, contested mechanism"),
    ("Temperatura superficial (satélite)", "E", 0.20, None, "absent", None, [], "low_value",
     "VERIFIED 4-jul: MODIS LST open+current but mu_ev=0.20 lowest, mechanism non-reproducible, "
     "Shumagin ~97% cloud-blind -- lowest value/effort of the catalog"),
    ("Variaciones electromagnéticas", "E", 0.20, None, "absent", None, [], "low_value",
     "VERIFIED 4-jul: only INTERMAGNET geomag observatories 100-860km away (measure global field, "
     "not local EM); QuakeFinder frozen 2010+closed. Touches only 2/9 segments marginally"),
    # cluster F -- structure & history (also feeds "where")
    ("Historial sísmico", "F", 0.55, None, "static", None, [], "active",
     "config T_rec_years / years_since_last_major, feeds C_state initial condition"),
    ("Geometría de fallas", "F", 0.75, None, "static", None, [], "active",
     "real CA-GIS traces + PCA strike; feeds dist_to_trace + stress_graph"),
    # ADDITION not in the original 25-variable catalog
    ("Carga oceánica no-tidal", "+", None, "a_ocean", "dynamic", "tau_ocean",
     ["ocean_bajo", "ocean_alto", "coincid_seis_ocean"], "active",
     "NEW 4-jul, NOT in the audited 25; EOST/ECCO2, water-tier confidence (unaudited mechanism)"),
]

STATUS_MARK = {
    "active": "OK", "prior_only": "PRIOR", "code_ready_no_data": "WAIT",
    "blocked": "BLOCKED", "not_built": "--", "not_applicable": "N/A",
    "wall_data": "WALL", "wall_infra": "WALL", "low_value": "LOWVAL",
}


# Count-derived channels: their raw signal is an event-rate (or ratio of rates). robust_anomaly
# (median/MAD) is statistically ill-posed for these when the base rate is near zero -- MAD
# collapses to 0 over an all-zeros window and the tanh saturates. This is a STRUCTURAL limit of
# median/MAD-on-counts, NOT a tau miscalibration (verified 4-jul: Cascadia a_seis is 98% saturated
# because 98.45% of days have zero events; the clean z p95 is 0.67). Fixing it properly needs a
# count model (Poisson anomaly) for these channels -- flagged, deliberately not rushed.
COUNT_CHANNELS = {"a_seis", "a_regional_rate"}


def channel_health():
    """Real per-channel health across every configured segment: coverage and saturation.
    and (for count channels) the zero-day fraction that drives MAD-collapse saturation.
    Saturation = fraction of nonzero output with |value|>0.95. >5% is flagged; for count channels
    the flag is annotated with WHY (low-rate vs recalibratable)."""
    health = {}
    for cfgname in SEGMENTS:
        cfg = load_config(cfgname)
        seg = cfg["segment_id"]
        cat = load_catalog(cfg)
        stations, gnss = load_gnss(cfg)
        daily = build_daily(cfg, cat, stations, gnss)
        zero_frac = float((daily["n_seis"] == 0).mean()) if "n_seis" in daily else None
        daily = build_state(cfg, daily)
        for col in daily.columns:
            if not col.startswith("a_"):
                continue
            v = daily[col]
            nn = v[(v != 0) & v.notna()].to_numpy()
            entry = health.setdefault(col, {})
            if len(nn) == 0:
                entry[seg] = None
            else:
                entry[seg] = (len(nn), float((np.abs(nn) > 0.95).mean()), zero_frac)
    return health


def get_tau(tau_key):
    if tau_key is None:
        return None
    vals = set()
    for cfgname in SEGMENTS:
        cfg = load_config(cfgname)
        v = cfg["state"].get(tau_key)
        if v is not None:
            vals.add(v)
    if not vals:
        return "(default in code)"
    return sorted(vals)[0] if len(vals) == 1 else f"varies {sorted(vals)}"


def build_rows():
    health = channel_health()
    rows = []
    for (name, cluster, mu, chan, kind, tau_key, rules, status, note) in REGISTRY:
        tau = get_tau(tau_key)
        segs_with_data, max_sat, flagged = 0, 0.0, []
        if chan and chan in health:
            for seg, h in health[chan].items():
                if h is not None:
                    segs_with_data += 1
                    n, sat, zero_frac = h
                    max_sat = max(max_sat, sat)
                    if sat > 0.05:
                        why = ""
                        if chan in COUNT_CHANNELS and zero_frac is not None and zero_frac > 0.90:
                            why = f"(low-rate:{zero_frac:.0%}0days)"
                        flagged.append(f"{seg}={sat:.0%}{why}")
        rows.append({
            "name": name, "cluster": cluster, "mu_ev": mu, "channel": chan, "kind": kind,
            "tau": tau, "rules": rules, "status": status, "note": note,
            "segs_with_data": segs_with_data, "max_sat": max_sat, "sat_flags": flagged,
        })
    return rows


def print_console(rows):
    print(f"{'#':<3}{'STATUS':<8}{'CLU':<4}{'μ_ev':<6}{'VARIABLE':<42}{'CHANNEL':<11}"
          f"{'KIND':<9}{'τ':<7}{'DATA':<6}{'SAT'}")
    print("-" * 118)
    for i, r in enumerate(rows, 1):
        mu = f"{r['mu_ev']:.2f}" if r["mu_ev"] is not None else "-"
        tau = str(r["tau"]) if r["tau"] is not None else "-"
        data = (f"{r['segs_with_data']}/{N_SEGMENTS}"
                if r["channel"] and r["kind"] == "dynamic" else "-")
        sat = ",".join(r["sat_flags"]) if r["sat_flags"] else ("clean" if r["segs_with_data"] else "-")
        print(f"{i:<3}{STATUS_MARK.get(r['status'], '?'):<8}{r['cluster']:<4}{mu:<6}"
              f"{r['name'][:41]:<42}{(r['channel'] or '-'):<11}{r['kind']:<9}{tau:<7}{data:<6}{sat}")
    print("-" * 118)
    # summary
    active = sum(1 for r in rows if r["status"] == "active")
    dyn = sum(1 for r in rows if r["kind"] == "dynamic" and r["segs_with_data"] > 0)
    total_catalog = sum(1 for r in rows if r["cluster"] not in {"+", "R"})
    print(f"\nUNIVERSO DE DISCURSO: 1 solo, compartido -- [-1,1] (espacio tanh), partición "
          f"ANOMALY_SETS de 5 términos. Ver common.py:ANOMALY_SETS.")
    print(f"  Todas las variables dinámicas comparten ese universo/partición; lo que varía por "
          f"variable es SOLO el mapeo señal_cruda->[-1,1] vía tau_X (calibración de escala).")
    print(f"\nRECUENTO REAL:")
    print(f"  - {active} variables ACTIVAS (con implementación viva), de {total_catalog} del "
          f"catálogo de 25 + 6 adiciones (carga oceánica + 5 regionales MOCRE-2).")
    print(f"  - {dyn} canales de anomalía a_X con datos reales en disco.")
    print(f"  - Estados persistentes: A_state, C_state, K_state, E_state, F_state (5).")
    sat_channels = [r["name"] for r in rows if r["sat_flags"]]
    if sat_channels:
        print(f"\n  ⚠ SATURACIÓN RESIDUAL >5%:")
        for r in rows:
            if r["sat_flags"]:
                kind = "LÍMITE ESTRUCTURAL (count/MAD, no tau -- necesita anomalía Poisson)" \
                    if r["channel"] in COUNT_CHANNELS and any("low-rate" in f for f in r["sat_flags"]) \
                    else "recalibrable (distribución difiere del target uniforme)"
                print(f"      {r['channel']}: {', '.join(r['sat_flags'])}  -> {kind}")


def emit_markdown(rows):
    lines = ["# MOCRE-1 — Inventario canónico de variables (autogenerado)", ""]
    lines.append("> Generado por `src/inventory.py` desde código+configs+datos reales. "
                 "NO editar a mano — se regenera. Este fichero es un snapshot de conveniencia; "
                 "la verdad viva es `python3 src/inventory.py`.")
    lines.append("")
    lines.append("## Universo de discurso")
    lines.append("**Uno solo, compartido:** `[-1,1]` (espacio tanh), partición `ANOMALY_SETS` de "
                 "5 términos (`muy_bajo/bajo/normal/alto/muy_alto`), en `common.py`. Todas las "
                 "variables dinámicas usan la MISMA partición; lo único que varía por variable es "
                 "el mapeo señal_cruda→[-1,1] vía `tau_X`.")
    lines.append("")
    lines.append("## Tabla de variables")
    lines.append("")
    lines.append("| # | Estado | Clú | μ_ev | Variable | Canal | Tipo | τ | Datos | Saturación | Nota |")
    lines.append("|---|--------|-----|------|----------|-------|------|---|-------|-----------|------|")
    for i, r in enumerate(rows, 1):
        mu = f"{r['mu_ev']:.2f}" if r["mu_ev"] is not None else "—"
        tau = str(r["tau"]) if r["tau"] is not None else "—"
        data = (f"{r['segs_with_data']}/{N_SEGMENTS}"
                if r["channel"] and r["kind"] == "dynamic" else "—")
        sat = ", ".join(r["sat_flags"]) if r["sat_flags"] else ("limpio" if r["segs_with_data"] else "—")
        lines.append(f"| {i} | {STATUS_MARK.get(r['status'], '?')} | {r['cluster']} | {mu} | "
                     f"{r['name']} | `{r['channel'] or '—'}` | {r['kind']} | {tau} | {data} | "
                     f"{sat} | {r['note']} |")
    lines.append("")
    lines.append("Leyenda estado: `OK`=activa · `PRIOR`=prior estático (no canal temporal) · "
                 "`WAIT`=código listo sin datos (solo cosecha) · `WALL`=muro real verificado "
                 "(no existe fuente/infra viable) · `LOWVAL`=accesible pero baja relación "
                 "valor/esfuerzo (verificado) · `N/A`=no aplica a estos segmentos.")
    lines.append("")
    # recuento
    active = sum(1 for r in rows if r["status"] == "active")
    dyn = sum(1 for r in rows if r["kind"] == "dynamic" and r["segs_with_data"] > 0)
    total_catalog = sum(1 for r in rows if r["cluster"] not in {"+", "R"})
    lines.append("## Recuento")
    lines.append(f"- **{active} variables activas** de {total_catalog} del catálogo de 25 "
                 f"+ 6 adiciones fuera de catálogo (carga oceánica + 5 regionales MOCRE-2).")
    lines.append(f"- **{dyn} canales de anomalía `a_X`** con datos reales en disco.")
    lines.append("- 5 estados persistentes: `A_state, C_state, K_state, E_state, F_state`.")
    lines.append("")
    lines.append("## Qué NO está listo (señalizado honestamente)")
    lines.append("")
    lines.append("**Conteos corregidos:** `a_seis` y `a_regional_rate` usan anomalía Poisson "
                 "causal. `a_swarm` y `a_foreshock` son razones de ventanas activas y conservan "
                 "normalización robusta con gate de actividad mínima.")
    lines.append("")
    lines.append("**Residuos menores recalibrables** (distribución del segmento difiere del "
                 "target uniforme, un τ por-segmento lo bajaría): `a_gnss` Shumagin ~9%, "
                 "`a_insar` Shumagin ~18% (además muestra pequeña, n=9 bins), `a_strain` "
                 "SJC/Parkfield ~5-6%.")
    lines.append("")
    lines.append("**Muros REALES verificados por descarga real 4-jul (no corazonada — no existe "
                 "fuente/infra viable, ningún esfuerzo de ingeniería lo crea):**")
    lines.append("- Radón (μ_ev 0.25): solo grab-samples que paran en 2014-2018, red continua "
                 "única (IRON) es Italia congelada 2021, Shumagin sin dato. Requiere hardware de "
                 "campo propio.")
    lines.append("- Helio/metano/H2 (μ_ev 0.30): solo compilado estático de puntos, 0 series "
                 "continuas; California muestreada 1 vez en 1983/1997/2013.")
    lines.append("- Vp/Vs (μ_ev 0.475): no existe serie publicada; el pivote dv/v es un proyecto "
                 "de procesamiento de señal (Julia + cross-correlation waveform GB-scale), no "
                 "integración de datos.")
    lines.append("")
    lines.append("**Accesibles pero baja relación valor/esfuerzo (verificado 4-jul — se dejan "
                 "sin construir por decisión, no por muro):**")
    lines.append("- Campo gravitatorio (μ_ev 0.30): GRACE-FO abierto y actual, pero ~300km de "
                 "resolución agrupa Anza+Mojave en una celda, cadencia mensual, mide agua no "
                 "esfuerzo tectónico — desajuste de escala y mecanismo.")
    lines.append("- Temperatura superficial (μ_ev 0.20, el más bajo): MODIS LST abierto, pero "
                 "mecanismo no reproducible y Shumagin ~97% ciego por nube.")
    lines.append("- Variaciones EM (μ_ev 0.20, el más disputado): solo observatorios "
                 "geomagnéticos a 100-860km (campo global, no EM local); QuakeFinder congelado "
                 "2010 y cerrado. Roza 2/9 segmentos.")
    lines.append("")
    lines.append("**Solo cosecha (código listo, alcanzable):** nivel de aguas subterráneas "
                 "(`a_water`, API USGS rate-limited — puro tiempo de cosecha, no código).")
    lines.append("")
    lines.append("**N/A a estos segmentos:** gases volcánicos (μ_ev 0.70, solo contexto "
                 "volcánico — ninguno de los 9 es un volcán).")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    rows = build_rows()
    if "--md" in sys.argv:
        print(emit_markdown(rows))
    else:
        print_console(rows)
