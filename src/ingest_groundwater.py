"""MOCRE-1 "todas las variables" pass: ingest real USGS groundwater level data (Cluster D,
mu_ev=0.35, "poroelastic stress response, mechanism real but weak as a lone precursor" per the
catalog). Verified 2026-07-03: the LEGACY nwis/gwlevels API is decommissioned (Feb-2026); the
current API (api.waterdata.usgs.gov/ogcapi/v0) works, confirmed with real daily depth-to-water
readings. Parameter 72019 = depth to water level, feet below land surface (positive anomaly in
this ingestion = water table unusually DEEP/low, not high -- sign convention documented, not
inverted to "intuitive" since the pipeline's tanh normalization is sign-agnostic anyway).

REWRITE 2026-07-03: first version timed out after 590s with ZERO output written (only saved at
the end of a fully sequential double-request-per-candidate loop) -- lost the whole run. Root
cause partly the API's own rate limiting (HTTP 429 confirmed while diagnosing), partly no
incremental progress. Fixed: (a) cap to the N closest candidate stations by distance, not every
station in a large bbox, (b) single request per candidate (drop the redundant
has-data-then-fetch two-call pattern -- just fetch and treat empty as "no data"), (c) small
delay between requests to respect the rate limit, (d) write partial results after every station
so a timeout or interrupt keeps whatever was already fetched instead of losing everything.
"""
import os
import time
import urllib.parse
import urllib.request
import json

import numpy as np
import pandas as pd

from common import ROOT, dist_to_trace_km, load_config, segment_paths

API = "https://api.waterdata.usgs.gov/ogcapi/v0"
PARAM_DEPTH_TO_WATER = "72019"
MAX_STATIONS = 60
REQUEST_DELAY_S = 0.25


def get_json(url, timeout=25, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"    429 rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def find_stations(cfg, limit=2000):
    b = cfg["catalog"]["bbox"]
    bbox = f"{b['minlon']},{b['minlat']},{b['maxlon']},{b['maxlat']}"
    params = {"bbox": bbox, "site_type_code": "GW", "limit": limit, "f": "json"}
    url = f"{API}/collections/monitoring-locations/items?" + urllib.parse.urlencode(params)
    d = get_json(url)
    return d.get("features", []) if d else []


def fetch_daily(station_id):
    params = {"monitoring_location_id": station_id, "parameter_code": PARAM_DEPTH_TO_WATER,
             "limit": 10000, "f": "json"}
    url = f"{API}/collections/daily/items?" + urllib.parse.urlencode(params)
    d = get_json(url)
    if not d:
        return []
    rows = []
    for f in d.get("features", []):
        p = f["properties"]
        rows.append({"station": station_id, "date": p.get("time"), "depth_ft": p.get("value")})
    return rows


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    print(f"segment: {cfg['segment_id']} ({paths['slug']})")
    gw_dir = os.path.join(ROOT, "data", "groundwater", paths["slug"])
    os.makedirs(gw_dir, exist_ok=True)
    raw_path = os.path.join(gw_dir, "daily_raw.csv")
    agg_path = os.path.join(gw_dir, "daily_aggregate.csv")

    candidates = find_stations(cfg)
    print(f"{len(candidates)} GW monitoring locations in bbox (uncapped)")
    if not candidates:
        pd.DataFrame(columns=["station", "date", "depth_ft"]).to_csv(raw_path, index=False)
        pd.DataFrame(columns=["mean_z_depth", "n_stations"]).to_csv(agg_path)
        print("no candidates -- wrote empty files")
        return

    lats = [c["geometry"]["coordinates"][1] for c in candidates]
    lons = [c["geometry"]["coordinates"][0] for c in candidates]
    dist, _ = dist_to_trace_km(lats, lons, cfg)
    order = np.argsort(dist)[:MAX_STATIONS]
    picked = [candidates[i] for i in order]
    print(f"capped to {len(picked)} closest stations (max dist {dist[order[-1]]:.1f}km)")

    all_rows = []
    n_with_data = 0
    for i, feat in enumerate(picked):
        sid = feat["properties"]["id"]
        try:
            rows = fetch_daily(sid)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(picked)}] {sid}: failed ({e})")
            time.sleep(REQUEST_DELAY_S)
            continue
        if rows:
            n_with_data += 1
            all_rows.extend(rows)
        if (i + 1) % 10 == 0 or rows:
            print(f"  [{i+1}/{len(picked)}] {sid}: {len(rows)} readings "
                  f"({n_with_data} stations with data so far)")
            # incremental save -- a later interrupt/timeout keeps this, doesn't lose the run
            pd.DataFrame(all_rows).to_csv(raw_path, index=False)
        time.sleep(REQUEST_DELAY_S)

    df = pd.DataFrame(all_rows)
    df.to_csv(raw_path, index=False)
    print(f"{n_with_data}/{len(picked)} stations had real daily series, "
          f"{len(df)} total readings -> {raw_path}")

    if len(df) == 0:
        pd.DataFrame(columns=["mean_z_depth", "n_stations"]).to_csv(agg_path)
        return

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["depth_ft"] = pd.to_numeric(df["depth_ft"], errors="coerce")
    df["z"] = df.groupby("station")["depth_ft"].transform(
        lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median() + 1e-6))
    agg = df.groupby("date").agg(mean_z_depth=("z", "mean"), n_stations=("station", "nunique"))
    agg.to_csv(agg_path)
    print(f"wrote {len(agg)} daily aggregate rows -> {agg_path}")
    print(agg.tail(5))


if __name__ == "__main__":
    main()
