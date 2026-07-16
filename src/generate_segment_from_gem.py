"""MOCRE-1 Vía A: auto-generate a segment config from the GEM Global Active Faults Database.
This is the mechanism that lets the model cover the WHOLE world instead of 6 hand-written US
segments -- pick a region, take its most active GEM fault, and derive a runnable segment config
(trace, regime, recurrence, target magnitude) automatically.

HONESTY / SCOPE: these configs are AUTO-DERIVED, deliberately lower-confidence than the 6
hand-curated US segments. Recurrence comes from slip_rate via a Wells&Coppersmith scaling (a real
approximation, NOT paleoseismology); regime from GEM's slip_type; target magnitude from fault
length. Each generated config records "auto-GEM" provenance in its comments so it's never confused
with a sourced one. This is Vía A (broad coverage); Vía B (hand-curated regions) is separate.

Usage: python3 generate_segment_from_gem.py <slug> <name> <minlat> <maxlat> <minlon> <maxlon>
  picks the highest-activity fault (slip_rate x length) in that bbox and writes
  config/segment_<slug>.json + config/<slug>_trace.json.
"""
import json
import math
import os
import re
import sys

from common import ROOT

GEM_PATH = os.path.join(ROOT, "data", "gem", "gem_active_faults.geojson")

# GEM slip_type -> MOCRE-1 regime. The model's regime only really gates behavior at two points:
# induced (turns K/SSE off) and everything-else (K on, F off unless wells exist). So thrust/
# subduction vs strike-slip vs normal all map to non-induced regimes with the same channel gating;
# the label is kept descriptive for the stress-graph mechanism-compat and for honesty.
REGIME_MAP = {
    "Subduction_Thrust": "subduction", "Blind Thrust": "subduction", "Reverse": "subduction",
    "Reverse-Strike-Slip": "subduction", "Reverse-Dextral": "subduction",
    "Reverse-Sinistral": "subduction", "Anticline": "subduction",
    "Dextral": "transform", "Sinistral": "transform", "Strike-Slip": "transform",
    "Dextral_Transform": "transform", "Sinistral_Transform": "transform",
    "Dextral-Reverse": "transform", "Sinistral-Reverse": "transform",
    "Dextral-Normal": "transform", "Sinistral-Normal": "transform",
    "Normal": "normal", "Normal-Dextral": "normal", "Normal-Sinistral": "normal",
    "Normal-Strike-Slip": "normal", "Spreading_Ridge": "normal",
}


def parse_slip_rate(s):
    """GEM net_slip_rate is a string '(best,min,max)' in mm/yr; return best or None."""
    if not s:
        return None
    m = re.match(r"\(([\d.]+)", str(s))
    return float(m.group(1)) if m else None


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def fault_length_km(coords):
    return sum(haversine_km(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
              for i in range(len(coords) - 1))


def azimuth_deg(coords):
    """Overall strike from first to last vertex (deg from north, 0-360)."""
    lon1, lat1 = coords[0]
    lon2, lat2 = coords[-1]
    x = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    y = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.cos(math.radians(lon2 - lon1)))
    return math.degrees(math.atan2(x, y)) % 360.0


def main():
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)
    slug, name = sys.argv[1], sys.argv[2]
    minlat, maxlat, minlon, maxlon = map(float, sys.argv[3:7])

    gem = json.load(open(GEM_PATH))
    # candidate faults: any vertex inside the bbox
    cands = []
    for f in gem["features"]:
        coords = f["geometry"]["coordinates"]
        if not coords or f["geometry"]["type"] != "LineString":
            continue
        if any(minlon <= c[0] <= maxlon and minlat <= c[1] <= maxlat for c in coords):
            sr = parse_slip_rate(f["properties"].get("net_slip_rate"))
            L = fault_length_km(coords)
            activity = (sr or 0.5) * L          # slip_rate x length = "how active" proxy
            cands.append((activity, sr, L, f))
    if not cands:
        print(f"no GEM faults in bbox for {slug}")
        sys.exit(1)
    cands.sort(key=lambda t: -t[0])
    activity, sr, L, fault = cands[0]
    coords = fault["geometry"]["coordinates"]
    props = fault["properties"]
    slip_type = props.get("slip_type") or "Strike-Slip"
    regime = REGIME_MAP.get(slip_type, "transform")
    fault_name = props.get("name") or f"GEM fault ({slip_type})"

    # target magnitude from length (Wells & Coppersmith 1994, M vs surface rupture length),
    # then floored/capped so the catalog actually has M>=target events to score against.
    m_char = 5.08 + 1.16 * math.log10(max(L, 1.0))
    target_mag = round(min(max(m_char - 2.0, 4.0), 5.5) * 2) / 2  # to nearest 0.5

    # recurrence from slip rate: T_rec = D_char / slip_rate. D_char (avg coseismic displacement)
    # from Wells&Coppersmith AD scaling: log10(AD_m) = -4.80 + 0.69*M_char.
    slip = sr if sr else 2.0                    # mm/yr fallback if GEM has no rate
    ad_m = 10 ** (-4.80 + 0.69 * m_char)
    t_rec = max(ad_m * 1000.0 / slip, 50.0)     # years, floor 50

    centroid_lat = sum(c[1] for c in coords) / len(coords)
    centroid_lon = sum(c[0] for c in coords) / len(coords)
    az = azimuth_deg(coords)
    # NGL plate frame by rough region -- IGS20 global works everywhere as a fallback.
    frame = "IGS20"
    pad = 1.2
    bbox = {"minlat": round(min(c[1] for c in coords) - pad, 2),
            "maxlat": round(max(c[1] for c in coords) + pad, 2),
            "minlon": round(min(c[0] for c in coords) - pad, 2),
            "maxlon": round(max(c[0] for c in coords) + pad, 2)}

    prov = (f"AUTO-GENERATED from GEM Global Active Faults DB (fault '{fault_name}', slip_type="
            f"{slip_type}, net_slip_rate={sr} mm/yr, length={L:.0f}km). Lower-confidence than the "
            f"hand-curated US segments: regime from slip_type, recurrence from slip_rate via "
            f"Wells&Coppersmith AD scaling (NOT paleoseismology), target_mag from length. Vía A.")

    trace = {"polylines": [[[round(c[0], 4), round(c[1], 4)] for c in coords]]}
    trace_file = f"{slug}_trace.json"
    with open(os.path.join(ROOT, "config", trace_file), "w") as fh:
        json.dump(trace, fh)

    cfg = {
        "segment_id": slug.upper(),
        "name": name,
        "regime": regime,
        "fault_trace": {
            "comment": prov,
            "polyline_file": trace_file,
            "p1": {"lat": round(coords[0][1], 4), "lon": round(coords[0][0], 4)},
            "p2": {"lat": round(coords[-1][1], 4), "lon": round(coords[-1][0], 4)},
        },
        "corridor_half_width_km": 40.0,
        "catalog": {"start": "1990-01-01", "min_magnitude": 2.5, "bbox": bbox},
        "gnss": {"frame": frame, "max_dist_km": 80.0, "min_years": 6.0,
                 "fault_azimuth_deg": round(az, 1),
                 "fault_azimuth_comment": "from GEM fault trace endpoints (auto)"},
        "state": {
            "v_ref_mm_yr": round(slip, 1),
            "v_ref_comment": "= GEM net_slip_rate (auto proxy for GNSS loading rate)",
            "T_rec_years": round(t_rec, 0),
            "years_since_last_major": round(t_rec * 0.5, 0),   # unknown -> mid-cycle prior
            "recurrence_source": prov,
            "lambda_A_daily": 0.95, "tau_seismicity": 4.0, "tau_gnss": 7.5, "tau_sse": 10.9,
            "tau_foreshock": 4.1, "tau_strain": 6.3, "baseline_window_days": 365,
        },
        "etas_lite": {
            "target_mag": target_mag, "b_value": 1.0,
            "decluster_ref_mag": max(target_mag - 1.5, 2.5),
            "omori_c_days": 0.02, "omori_p": 1.1, "alpha": 0.9,
            "trigger_min_mag": max(target_mag - 1.0, 3.0),
            "train_end": "2015-01-01",
        },
        "gain": {"log10_min": -1.0, "log10_max": 1.7},
    }
    out = os.path.join(ROOT, "config", f"segment_{slug}.json")
    with open(out, "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out}")
    print(f"  fault: {fault_name} | slip_type={slip_type} -> regime={regime}")
    print(f"  length={L:.0f}km slip={slip}mm/yr -> M_char={m_char:.1f} target_mag={target_mag} "
          f"T_rec={t_rec:.0f}yr")
    print(f"  centroid {centroid_lat:.2f},{centroid_lon:.2f} az={az:.0f} bbox={bbox}")


if __name__ == "__main__":
    main()
