"""Extrae estados de investigación por segmento para el mapa público.

Las probabilidades son horizontes Poisson; magnitud y tiempo no se presentan
como predicciones cuando sus motores no superan un baseline justo.
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from common import ROOT, load_config, segment_paths, _load_polyline_trace
from predict_ia3 import predict as ia3_predict, load_model
from pipeline import load_catalog

OUT = os.path.join(ROOT, "out")
# smoothing: the single last-day lambda is noisy (a recent event inflates ETAS's Omori tail);
# the model's *current standing* hazard is better read as the trailing-90-day mean intensity.
SMOOTH_DAYS = 90

# same global anchor as discover_rules.py: target_mag -> mag_scale in [-1,1] (3.5->-1, 7.5->+1)
MAG_LO, MAG_HI = 3.5, 7.5
# the 17 anomaly/state channels the fuzzy engine reads, all already in series_*.csv
FUZZY_CHANNELS = ["a_seis", "a_swarm", "a_foreshock", "a_gnss", "a_sse", "a_fluid", "a_water",
                  "a_strain", "a_ocean", "a_insar", "a_tec", "a_noise", "A_state", "C_state",
                  "K_state", "E_state", "F_state"]

PLACE_LABELS = {
    "AAK-SHUMAGIN": {
        "city": "Sand Point",
        "region": "Shumagin Islands, Alaska Peninsula",
        "country": "United States",
        "zone": "Alaska-Aleutian subduction zone",
    },
    "CHILE_CENTRAL": {
        "city": "Coquimbo / La Serena",
        "region": "Central Chile",
        "country": "Chile",
        "zone": "Peru-Chile subduction margin",
    },
    "CSZ-CENTRAL-OR": {
        "city": "Newport",
        "region": "Central Oregon coast",
        "country": "United States",
        "zone": "Cascadia Subduction Zone",
    },
    "ITALY_APENNINES": {
        "city": "L'Aquila / Amatrice",
        "region": "Central Apennines",
        "country": "Italy",
        "zone": "Apennine normal-fault belt",
    },
    "JAPAN_NANKAI": {
        "city": "Kochi",
        "region": "Shikoku, southwest Japan",
        "country": "Japan",
        "zone": "Nankai Trough",
    },
    "OK-PAWNEE-PRAGUE": {
        "city": "Pawnee / Prague",
        "region": "Central Oklahoma",
        "country": "United States",
        "zone": "Induced seismicity corridor",
    },
    "SAF-MOJAVE": {
        "city": "Mojave / Palmdale",
        "region": "Southern California",
        "country": "United States",
        "zone": "San Andreas fault, Mojave section",
    },
    "SAF-PARKFIELD": {
        "city": "Parkfield",
        "region": "Central California",
        "country": "United States",
        "zone": "San Andreas fault, Parkfield section",
    },
    "SJC-ANZA": {
        "city": "Anza / San Jacinto",
        "region": "Southern California",
        "country": "United States",
        "zone": "San Jacinto fault zone",
    },
}


def fuzzy_forecast(tail, target_mag):
    """Motor FUZZYSET de Cícero (IA2+IA3 port), sobre el estado ACTUAL del segmento (media móvil
    SMOOTH_DAYS de cada canal) + mag_scale estático. ``predict_ia3`` suprime las salidas que no
    superan su baseline condicionado; actualmente ocurrencia, magnitud, cuándo y dónde no tienen
    skill publicable."""
    state = {c: float(tail[c].mean()) if c in tail.columns and tail[c].notna().any() else 0.0
             for c in FUZZY_CHANNELS}
    mag_scale = float(np.clip(2 * (target_mag - MAG_LO) / (MAG_HI - MAG_LO) - 1, -1, 1))
    return ia3_predict(state, mag_scale)


def segment_centroid_and_trace(cfg):
    trace = cfg["fault_trace"]
    polylines = _load_polyline_trace(cfg)
    if polylines is not None:
        pts = [(lat, lon) for line in polylines for lon, lat in line]
        lats = [p[0] for p in pts]
        lons = [p[1] for p in pts]
        centroid = (float(np.mean(lats)), float(np.mean(lons)))
        # downsample the trace for drawing (a few hundred vertices is plenty on a world map)
        step = max(1, len(pts) // 60)
        trace_pts = [[round(lon, 4), round(lat, 4)] for lat, lon in pts[::step]]
    else:
        p1, p2 = trace["p1"], trace["p2"]
        centroid = ((p1["lat"] + p2["lat"]) / 2, (p1["lon"] + p2["lon"]) / 2)
        trace_pts = [[p1["lon"], p1["lat"]], [p2["lon"], p2["lat"]]]
    return centroid, trace_pts


def poisson_window(lam_daily, days):
    """P(at least one M>=target event within `days`), homogeneous-Poisson approx of the current
    standing rate. Honest for a 'what is the near-term chance' readout; NOT a claim the rate is
    constant over decades."""
    return 1.0 - math.exp(-lam_daily * days)


def csep_verdicts():
    path = os.path.join(OUT, "csep_paired_test_results.json")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in json.load(open(path)):
        if r.get("status") != "ok":
            out[r["segment"]] = f"sin potencia ({r.get('n_events', 0)} eventos); exploratorio"
            continue
        ig = r["information_gain_per_event_nats"]
        tp = r["t_test"]["p_value"]
        wp = r["w_test"]["p_value"]
        absolute = r["n_test_mocre"]["passes_95pct"]
        out[r["segment"]] = (f"IG/evento={ig:+.3f}; T p={tp:.3g}; W p={wp:.3g}; "
                              f"N-test={'pasa' if absolute else 'falla'}; exploratorio")
    return out


def main():
    verdicts = csep_verdicts()
    critical_path = os.path.join(OUT, "critical_shadow.json")
    critical = ({s["segment"]: s for s in json.load(open(critical_path)).get("segments", [])}
                if os.path.exists(critical_path) else {})
    has_ia3 = os.path.exists(os.path.join(OUT, "ia3_model.json"))
    if has_ia3:
        load_model()   # precarga
    forecasts = []
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        cfg = load_config(os.path.basename(cfg_path))
        slug = segment_paths(cfg)["slug"]
        series_path = os.path.join(OUT, f"series_{slug}.csv")
        summary_path = os.path.join(OUT, f"summary_{slug}.json")
        if not (os.path.exists(series_path) and os.path.exists(summary_path)):
            continue
        series = pd.read_csv(series_path, index_col=0, parse_dates=True)
        summary = json.load(open(summary_path))
        centroid, trace_pts = segment_centroid_and_trace(cfg)

        tail = series.tail(SMOOTH_DAYS)
        lam_mocre = float(tail["lam_mocre"].mean())
        lam_etas = float(tail["lam_etas"].mean())
        G_now = float(tail["G"].mean())
        target_mag = summary["target_mag"]

        # 4-AXIS FUZZY ENGINE (motor de Cícero IA2+IA3) sobre el estado ACTUAL — las 4 salidas de la
        # spec por punto, entrenadas a la manera del padre (error, no AUC vs ETAS).
        fuzzy = fuzzy_forecast(tail, target_mag) if has_ia3 else None

        # DELTA TIEMPO: expected time to next M>=target, and P within honest windows.
        mean_years = (1.0 / lam_mocre / 365.25) if lam_mocre > 0 else None
        windows = {"30d": poisson_window(lam_mocre, 30),
                   "1yr": poisson_window(lam_mocre, 365),
                   "10yr": poisson_window(lam_mocre, 3652)}
        # honest temporal bucket (catalog's own partition): the field cannot do 'hours/days'
        # short-term with this evidence -- everything here is background-rate / long-range.
        if mean_years is None:
            tiempo_bucket = "indeterminado (sin tasa)"
        elif mean_years > 50:
            tiempo_bucket = "plazo muy largo (>50 años tasa media)"
        elif mean_years > 10:
            tiempo_bucket = "plazo largo (décadas)"
        else:
            tiempo_bucket = "plazo elevado (años)"

        # short distinctive label for the map (segment_id's cluster prefix collides: both SAF
        # segments would show "SAF") -- use the locality part instead.
        short_label = {
            "AAK-SHUMAGIN": "Shumagin", "CSZ-CENTRAL-OR": "Cascadia",
            "CHILE_CENTRAL": "Chile", "ITALY_APENNINES": "Apeninos",
            "JAPAN_NANKAI": "Nankai",
            "OK-PAWNEE-PRAGUE": "Oklahoma", "SAF-PARKFIELD": "Parkfield",
            "SAF-MOJAVE": "Mojave", "SJC-ANZA": "Anza",
        }.get(cfg["segment_id"], cfg["segment_id"].split("-")[0])

        forecasts.append({
            "slug": slug,
            "segment_id": cfg["segment_id"],
            "short_label": short_label,
            "name": cfg["name"],
            "regime": cfg["regime"],
            # UBICACIÓN
            "lat": round(centroid[0], 4),
            "lon": round(centroid[1], 4),
            "place": PLACE_LABELS.get(cfg["segment_id"], {
                "city": short_label,
                "region": cfg["name"],
                "country": "—",
                "zone": cfg["regime"],
            }),
            "trace": trace_pts,
            # VALOR (Richter)
            "target_mag": target_mag,
            # DELTA TIEMPO
            "lam_mocre_daily": lam_mocre,
            "lam_etas_daily": lam_etas,
            "gain_now": round(G_now, 3),
            "mean_years_to_next": round(mean_years, 1) if mean_years else None,
            "p_30d": round(windows["30d"], 5),
            "p_1yr": round(windows["1yr"], 4),
            "p_10yr": round(windows["10yr"], 3),
            "tiempo_bucket": tiempo_bucket,
            # 4-AXIS FUZZY OUTPUTS (motor FUZZYSET Mamdani, learned_rules.json)
            "fuzzy": fuzzy,
            "critical_shadow": critical.get(slug),
            # honesty
            "csep_verdict": verdicts.get(slug, "sin test"),
            "state_as_of": str(series.index[-1].date()),
            "catalog_cutoff": str(load_catalog(cfg)["date"].max().date()),
            "last_data": str(series.index[-1].date()),
            "n_target_events_hist": summary["n_target_events"],
        })

    out = {
        "generated_from": "MOCRE-1 series_*.csv (trailing-90d mean intensity)",
        "model_note": ("lam_mocre = ETAS causal × ganancia fuzzy experimental, sin "
                       "renormalización futura. El T/W-test se calcula por evento y el N-test "
                       "comprueba la tasa absoluta. Con la corrección causal, Oklahoma empeora "
                       "frente a ETAS y ambos modelos fallan el N-test; Shumagin es positivo pero "
                       "no significativo y también falla el N-test. Todos estos resultados son "
                       "exploratorios porque el holdout fue reutilizado. Las esperas mostradas "
                       "son 1/λ de Poisson, no fechas ni ventanas previstas."),
        "segments": forecasts,
    }
    with open(os.path.join(OUT, "forecast.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(forecasts)} segment forecasts -> {os.path.join(OUT, 'forecast.json')}")
    for fc in forecasts:
        print(f"  {fc['segment_id']:<18} M>={fc['target_mag']} | {fc['lat']:.2f},{fc['lon']:.2f} "
              f"| P(1yr)={fc['p_1yr']:.3f} | {fc['tiempo_bucket']} | {fc['csep_verdict'][:40]}")


if __name__ == "__main__":
    main()
