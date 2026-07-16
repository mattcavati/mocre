"""MOCRE-1 F0: select NGL GNSS stations near the fault trace and download tenv3 series."""
import os
import sys
import time
import urllib.request
from datetime import date

from common import _load_polyline_trace, dist_to_trace_km, load_config, segment_paths

HOLDINGS = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
TENV3 = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/{frame}/{sta}.{frame}.tenv3"


def get(url, timeout=120):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry {attempt+1} {url} ({e})", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    print(f"segment: {cfg['segment_id']} ({paths['slug']})")
    g = cfg["gnss"]

    # Cheap bbox pre-filter BEFORE the expensive real-polyline distance calc. Real traces can
    # have thousands of vertices (Mojave: 1990 polylines) -- calling dist_to_trace_km on all
    # ~23k global NGL stations one-by-one against that is a multi-hour hang, not a bug in the
    # geometry itself. Padded bbox from the catalog config (already sized for this segment).
    b = cfg["catalog"]["bbox"]
    pad = (g["max_dist_km"] / 111.0) * 1.5  # deg, generous
    bbox = (b["minlat"] - pad, b["maxlat"] + pad, b["minlon"] - pad, b["maxlon"] + pad)

    text = get(HOLDINGS)
    lines = text.strip().splitlines()[1:]
    candidates = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 11:
            continue
        sta, lat, lon = parts[0], float(parts[1]), float(parts[2])
        if lon > 180:
            lon -= 360
        if not (bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]):
            continue
        try:
            beg, fin = date.fromisoformat(parts[7]), date.fromisoformat(parts[8])
        except ValueError:
            continue
        years = (fin - beg).days / 365.25
        if years < g["min_years"]:
            continue
        candidates.append((sta, lat, lon, years, parts[8]))
    print(f"{len(candidates)} candidates survive bbox pre-filter (of {len(lines)} global stations)")

    picked = []
    for sta, lat, lon, years, end in candidates:
        dist, side = dist_to_trace_km(lat, lon, cfg)
        if float(dist) > g["max_dist_km"]:
            continue
        picked.append({"sta": sta, "lat": lat, "lon": lon, "dist_km": float(dist),
                       "side": int(side), "years": round(years, 1), "end": end})
    print(f"{len(picked)} candidate stations within {g['max_dist_km']} km, >={g['min_years']} yr")
    ok = []
    for p in sorted(picked, key=lambda r: r["dist_km"]):
        url = TENV3.format(frame=g["frame"], sta=p["sta"])
        try:
            data = get(url)
        except Exception:
            print(f"  {p['sta']}: no series, skipped")
            continue
        path = os.path.join(paths["gnss_dir"], f"{p['sta']}.tenv3")
        with open(path, "w") as f:
            f.write(data)
        n = len(data.splitlines()) - 1
        p["n_days"] = n
        ok.append(p)
        print(f"  {p['sta']}: side={'NE' if p['side']>0 else 'SW'} dist={p['dist_km']:.1f}km "
              f"{p['years']}yr {n} days -> saved")
    import json
    with open(paths["stations_json"], "w") as f:
        json.dump(ok, f, indent=1)
    print(f"saved {len(ok)} stations")


if __name__ == "__main__":
    main()
