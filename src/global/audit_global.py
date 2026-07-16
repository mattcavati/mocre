"""MOCRE-GLOBAL — AUDITORÍA E2E de universos de discurso, cobertura mundial, entradas y salidas.
Todo medido sobre los artefactos REALES de producción (cells_live, engines, globe_data), no sobre
supuestos. Cada sección emite PASS/WARN/FAIL.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTG = os.path.join(ROOT, "out", "global")
GLOBE = os.path.join(ROOT, "globe")

from global_mamdani import CH, SETS5, TERMS, derive_channels, memb, mamdani_occurrence  # noqa
from global_ia2_faithful import (CH_ALL, derive_channels_classed, ia1_apply, apply_v3,   # noqa
                                 activity_class)

ISSUES = []


def verdict(ok, warn, label, detail=""):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    if tag != "PASS":
        ISSUES.append((tag, label, detail))
    print(f"  [{tag}] {label}" + (f" — {detail}" if detail else ""))


# ================================================================================================
print("=" * 96)
print("A. UNIVERSOS DE DISCURSO — ENTRADAS (motor v2, partición compartida SETS5)")
print("=" * 96)
live = pd.read_csv(os.path.join(OUTG, "cells_live.csv"))
dense = pd.read_csv(os.path.join(OUTG, "cellmonths.csv.gz"), parse_dates=["date"])
chan_live = derive_channels(live)
chan_hist = derive_channels(dense)

# Ruspini exacto de SETS5 (la copia local de global_mamdani, no la de common)
grid_u = np.linspace(-1, 1, 4001)
tot = np.zeros_like(grid_u)
for p in SETS5.values():
    tot += memb(grid_u, p)
err = float(np.abs(tot - 1).max())
verdict(err < 1e-5, False, f"Ruspini SETS5 (suma pertenencias=1): err_max={err:.2e}")

print(f"\n  {'canal':<10} {'min':>7} {'p5':>7} {'p50':>7} {'p95':>7} {'max':>7} "
      f"{'%|x|>0.95':>9} {'%==0':>6}  cobertura de términos (vivo)")
for c in CH:
    xv = chan_live[c].to_numpy()
    xh = chan_hist[c].to_numpy()
    sat = 100 * (np.abs(xh) > 0.95).mean()
    zero = 100 * (xv == 0).mean()
    cov = []
    for t in TERMS:
        m = memb(xv, SETS5[t])
        cov.append(f"{t[:4]}:{100*(m>0.5).mean():.0f}%")
    q = np.percentile(xv, [0, 5, 50, 95, 100])
    print(f"  {c:<10} {q[0]:>7.2f} {q[1]:>7.2f} {q[2]:>7.2f} {q[3]:>7.2f} {q[4]:>7.2f} "
          f"{sat:>8.1f}% {zero:>5.1f}%  {' '.join(cov)}")
    # veredictos: canal vivo (no todo ceros), sin saturación patológica, universo ocupado
    verdict(zero < 95, zero < 99.5, f"{c}: canal VIVO en producción", f"{zero:.1f}% celdas==0")
    if c == "a_tect":
        # canal ESTÁTICO bimodal por diseño (sin falla GEM=-1, falla rápida=+1): la métrica de
        # saturación de anomalías z no aplica; mu_ev ya lo descuenta (0.77, el más bajo)
        print(f"        (a_tect: estático bimodal por diseño — check de saturación no aplicable)")
    else:
        verdict(sat < 15, sat < 30, f"{c}: saturación histórica sana", f"{sat:.1f}% |x|>0.95")
    span = q[3] - q[1]
    verdict(span > 0.15, span > 0.05, f"{c}: universo ocupado (p95-p5)", f"span={span:.2f}")

# ================================================================================================
print("\n" + "=" * 96)
print("B. UNIVERSOS DE DISCURSO — v3 fiel (sets POR variable) + SALIDAS de ambos motores")
print("=" * 96)
eng3 = json.load(open(os.path.join(OUTG, "global_engine_v3.json")))
chan3_all, _, _ = derive_channels_classed(live, taus=eng3["taus"])
for v, sets in eng3["var_sets"].items():
    knots = sorted({p for prm in sets.values() for p in prm})
    lo_, hi_ = knots[0], knots[-1]
    g = np.linspace(lo_, hi_, 2001)
    tot = np.zeros_like(g)
    for prm in sets.values():
        tot += memb(g, prm)
    err = float(np.abs(tot - 1).max())
    # término estrecho: ¿detector de MASA PUNTUAL (>=3% de datos vivos en su soporte) o muerto?
    # La revalidación walk-forward del 6-jul demostró que los detectores de masa puntual son
    # ESENCIALES (deduplicarlos colapsó la cola extrema de IA3, 57% real). Solo un término
    # estrecho SIN masa es inerte (y no daña).
    xv = chan3_all[v].to_numpy() if v in chan3_all.columns else None
    detectors, dead = 0, 0
    for prm in sets.values():
        if (prm[-1] - prm[0]) < 1e-3:
            if xv is not None and np.mean(np.abs(xv - prm[1]) < 1e-3) >= 0.03:
                detectors += 1
            else:
                dead += 1
    verdict(err < 1e-3 and dead <= 2, err < 0.05,
            f"v3 sets '{v}': Ruspini err={err:.1e}, detectores-masa-puntual={detectors} "
            f"(por diseño), slivers inertes={dead} (no dañan), n_terms={len(sets)}")

print("\n  SALIDAS:")
eng2 = json.load(open(os.path.join(OUTG, "global_engine_v2.json")))
cal_t = np.array(eng2["calib"]["t"])
print(f"  v2 P calibrada: universo [0,1], tabla 3D {cal_t.shape}, "
      f"rango [{cal_t.min():.4f}, {cal_t.max():.4f}]")
verdict(0 <= cal_t.min() and cal_t.max() <= 1, False, "v2 salida P dentro de [0,1]")
cents = eng3["out_centroids"]
print(f"  v3 ocurrencia: 7 términos {eng3['out_names']}, centroides IA3 "
      f"{[round(c,4) for c in cents]}")
verdict(all(cents[i] <= cents[i+1] + 1e-9 for i in range(6)), False,
        "v3 centroides ORDENADOS (nulo<=...<=extremo)")
mag_sets = eng2["mag_sets"]
mc = {k: sum(v) / 3 for k, v in mag_sets.items()}
print(f"  v2 magnitud (IA3): centroides {[f'{k}={v:.2f}' for k, v in mc.items()]}")
verdict(all(4.0 < v < 9.0 for v in mc.values()), False, "magnitud dentro del universo Richter")

# ================================================================================================
print("\n" + "=" * 96)
print("C. COBERTURA MUNDIAL — las 20 zonas sísmicas mayores del planeta")
print("=" * 96)
gd = json.load(open(os.path.join(GLOBE, "globe_data.json")))
cells = pd.DataFrame(gd["cells"], columns=gd["fields"])
ZONAS = {
    "Japón": (30, 45, 128, 146), "Indonesia": (-11, 6, 95, 141),
    "Chile": (-45, -17, -75, -66), "Himalaya/Nepal": (26, 36, 75, 95),
    "Turquía/Anatolia": (36, 42, 26, 45), "Irán/Zagros": (25, 40, 44, 63),
    "California": (32, 42, -125, -114), "Alaska/Aleutianas": (50, 65, -180, -130),
    "Nueva Zelanda": (-47, -34, 166, 179), "Mediterráneo(It/Gr)": (35, 46, 8, 30),
    "Andes norte": (-18, 5, -82, -70), "México/CAm": (8, 20, -105, -85),
    "Caribe": (10, 20, -85, -60), "Kamchatka/Kuriles": (44, 60, 145, 165),
    "Filipinas": (5, 20, 119, 127), "Taiwán": (21, 26, 119, 123),
    "China(Sichuan/Yunnan)": (22, 35, 97, 106), "Asia central": (36, 45, 65, 80),
    "Cáucaso": (38, 44, 40, 50), "Tonga/Fiji": (-25, -14, -180, -170),
}
print(f"  total micro-puntos: {len(cells)} | con P>1%: {(cells.p30>0.01).sum()} | "
      f"alertas: {int(cells.alert.sum())}")
for z, (la0, la1, lo0, lo1) in ZONAS.items():
    m = (cells.la >= la0) & (cells.la <= la1) & (cells.lo >= lo0) & (cells.lo <= lo1)
    n = int(m.sum())
    pmax = cells.p30[m].max() * 100 if n else 0
    pmed = cells.p30[m].mean() * 100 if n else 0
    verdict(n >= 10, n >= 3, f"{z:<22} {n:>4} celdas | P media {pmed:5.1f}% máx {pmax:5.1f}%")

# hemisferios / océanos (dorsales)
for name, m in [("hemisferio N", cells.la >= 0), ("hemisferio S", cells.la < 0),
                ("dorsales Atlántico (lon -45..-5, |lat|<60)",
                 (cells.lo > -45) & (cells.lo < -5) & (cells.la.abs() < 60) &
                 (cells.n365 >= 0))]:
    print(f"  {name}: {int(m.sum())} celdas")

# ================================================================================================
print("\n" + "=" * 96)
print("D. ENTRADAS → FUNCIÓN: ¿cada canal llega VIVO al motor y con reglas que lo usan?")
print("=" * 96)
# v2: reglas por canal + fuerza media en vivo
MU_live = {}
for c in CH:
    x = chan_live[c].to_numpy()
    for t in TERMS:
        MU_live[(c, t)] = memb(x, SETS5[t])
rules2 = eng2["rules"]
print(f"  v2: {len(rules2)} reglas")
for c in CH:
    nr = sum(1 for r in rules2 if c in r["chans"])
    # fuerza media de las pertenencias del canal en vivo (¿el canal 'dispara' algo?)
    mu_mean = np.mean([MU_live[(c, t)].mean() for t in TERMS])
    fire = 0.0
    for r in rules2:
        if c not in r["chans"]:
            continue
        m = MU_live[tuple(r["cols"][0])]
        for cc in r["cols"][1:]:
            m = np.minimum(m, MU_live[tuple(cc)])
        fire = max(fire, float(m.max()))
    verdict(nr > 0 and fire > 0.01, nr > 0,
            f"v2 canal {c:<10} reglas={nr:>3} | mu_ev={eng2['mu_ev'][c]:.2f} | "
            f"máx disparo vivo={fire:.2f}")
# v3: ia1 y canales
chan3, _, cls = derive_channels_classed(live, taus=eng3["taus"])
ia1_m = {"om": [-np.inf] + eng3["ia1"]["om"] + [np.inf],
         "rt": [-np.inf] + eng3["ia1"]["rt"] + [np.inf],
         "t": np.array(eng3["ia1"]["t"]), "n": eng3["ia1"]["n"]}
s_ia1 = ia1_apply(live.omori.to_numpy(), live.rate_bg.to_numpy(), ia1_m)
print(f"\n  v3: ia1 en vivo: rango [{s_ia1.min():.4f}, {s_ia1.max():.4f}], "
      f"media {s_ia1.mean():.4f}")
verdict(s_ia1.max() > s_ia1.min(), False, "v3 ia1 NO degenerada")
missing_taus = [k for c in ["a_seis", "a_swarm", "a_fore", "A_state", "C_state"]
                for k in [f"{c}|{q}" for q in range(4)] if k not in eng3["taus"]]
verdict(not missing_taus, False, f"v3 taus por clase completos (20)", str(missing_taus))
print(f"  clases de actividad en vivo: {np.bincount(cls, minlength=4).tolist()} "
      f"(rala/moderada/activa/muy_activa)")

# ================================================================================================
print("\n" + "=" * 96)
print("E. SALIDAS EN VIVO — distribución, degeneración, campos del globo")
print("=" * 96)
for f in gd["fields"]:
    col = cells[f]
    nnan = int(col.isna().sum())
    print(f"  {f:<10} rango [{col.min():.4g}, {col.max():.4g}]  NaN/null={nnan}")
verdict(cells.p30.between(0, 1).all(), False, "p30 en [0,1] en las 7250")
verdict(cells.p30.nunique() > 20, cells.p30.nunique() > 5,
        f"p30 GRADUADA (valores distintos={cells.p30.nunique()})")
verdict(cells.mag_exp.between(4.0, 9.5).all(), False, "mag_exp en universo Richter")
m_hist = cells.mag_max > 0                                    # 0 = celda GEM sin historia M>=5
verdict((cells.mag_max[m_hist] >= cells.mag_exp[m_hist] - 2).all(), True,
        f"mag_max coherente con mag_exp (celdas con historia; sin historia={int((~m_hist).sum())}, "
        f"el globo muestra '—')")
mnull = int(cells.mean_days.isna().sum())
verdict(mnull == 0, mnull < 100, f"mean_days poblado (null={mnull})")
verdict(cells.p3.between(0, 1).all(), False, "p3 (motor fiel) en [0,1]")
verdict(0 < int(cells.alert.sum()) < 200, True,
        f"alertas extremas en rango razonable ({int(cells.alert.sum())})")
print(f"\n  distribución p30: p50={cells.p30.quantile(.5)*100:.2f}% "
      f"p90={cells.p30.quantile(.9)*100:.1f}% p99={cells.p30.quantile(.99)*100:.1f}% "
      f"max={cells.p30.max()*100:.1f}%")
print(f"  radar: {len(gd.get('recent', []))} sismos reales últimos 7d")

# ================================================================================================
print("\n" + "=" * 96)
print(f"RESUMEN: {len(ISSUES)} incidencias")
for tag, label, det in ISSUES:
    print(f"  [{tag}] {label} {('— ' + det) if det else ''}")
if not ISSUES:
    print("  TODO PASS")
