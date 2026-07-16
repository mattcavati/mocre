"""MOCRE-1 "todas las variables" pass, cont.: real borehole strain data (Cluster B/deformation,
inclinometros/extensometros in the 25-variable catalog, mu_ev 0.60->0.65-0.70 per the 3-jul audit
-- "red PBO/EarthScope tan abierta como el GNSS ya usado, y esos mismos pozos tienen sensores de
presion de poro" -- verified for real before writing this, not taken on faith from the audit note:

  - IRIS FDSN station service (net=PB, cha=BS1, level=channel) gives real Gladwin Tensor
    Strainmeter station metadata (lat/lon), same free/no-auth pattern as everything else in this
    project.
  - EarthScope/UNAVCO already publish a Level-2 PROCESSED product per station-year at
    bsm.unavco.org/bsm/level2/<STA>/<STA>.<YEAR>.bsm.level2.tar -- tide, ocean-load, barometric
    and instrument-drift corrections already applied (verified by downloading and inspecting a
    real file, B072/2013: columns are date/Eee+Enn(mstrain)/s_offset/strain_quality/tide_c/
    detrend_c/atmp_c/...). This is the areal (dilatational) strain Eee+Enn member specifically --
    NOT the raw 20Hz gauge channels, which would need tidal-model + calibration-matrix processing
    from scratch (a much bigger, genuinely separate geodetic-processing project). Using the
    pre-corrected product is the honest low-friction path, same spirit as using NGL's tenv3
    products instead of raw GNSS RINEX.
  - Real station coverage checked per segment before committing to this (net=PB has NO stations
    at all near Mojave or Shumagin -- that is a true "channel not present here", not a bug, exactly
    like F(fluid) being zero outside induced-regime segments): SJC-Anza 9, SAF-Parkfield 8,
    CSZ-Central-OR 18, OK-Pawnee-Prague 1 (station AVN2, installed 2016-09-10, specifically
    monitoring the induced-seismicity zone). SAF-Mojave and AAK-Shumagin get a_strain=0 always.

Sign convention: Eee+Enn > 0 is dilatation (expansion), < 0 is contraction -- physically
meaningful in both directions (dilatancy-diffusion models predict volumetric expansion before
rupture from microcracking; contraction from pore-pressure/poroelastic loading is also a real
precursor candidate) and the catalog does not establish which direction should dominate. Symmetric
bajo/alto rules from the very start (this project has hit the one-sided-rule bug three times
already for skipping this on a variable's first pass -- see train_consequents.py NEGATIVE_SINGLES
comment).
"""
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd

from common import ROOT, dist_to_trace_km, load_config, segment_paths

STATION_SVC = ("https://service.iris.edu/fdsnws/station/1/query?net=PB&cha=BS1&format=text"
               "&level=channel&minlat={minlat}&maxlat={maxlat}&minlon={minlon}&maxlon={maxlon}")
LEVEL2_DIR = "http://bsm.unavco.org/bsm/level2/{sta}/"
LEVEL2_TAR = "http://bsm.unavco.org/bsm/level2/{sta}/{sta}.{year}.bsm.level2.tar"
TAR_CACHE = os.path.join(ROOT, "data", "strain", "_tar_cache")
MAX_STATIONS = 15


def get(url, timeout=60, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt+1} {url} ({e})", file=sys.stderr)
            time.sleep(3 * (attempt + 1))


def find_candidate_stations(cfg):
    b = cfg["catalog"]["bbox"]
    strain_cfg = cfg.get("strain", {})
    max_dist_km = strain_cfg.get("max_dist_km", cfg.get("gnss", {}).get("max_dist_km", 40.0))
    pad = (max_dist_km / 111.0) * 1.5
    url = STATION_SVC.format(minlat=b["minlat"] - pad, maxlat=b["maxlat"] + pad,
                             minlon=b["minlon"] - pad, maxlon=b["maxlon"] + pad)
    text = get(url).decode()
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return [], max_dist_km
    candidates = []
    for ln in lines[1:]:
        p = ln.split("|")
        sta, lat, lon = p[1], float(p[4]), float(p[5])
        candidates.append((sta, lat, lon))
    if not candidates:
        return [], max_dist_km
    lats = [c[1] for c in candidates]
    lons = [c[2] for c in candidates]
    dist, _ = dist_to_trace_km(lats, lons, cfg)
    dist = np.atleast_1d(dist)
    picked = [(sta, lat, lon, float(d)) for (sta, lat, lon), d in zip(candidates, dist)
              if d <= max_dist_km]
    picked.sort(key=lambda r: r[3])
    return picked[:MAX_STATIONS], max_dist_km


def list_years(sta):
    try:
        html = get(LEVEL2_DIR.format(sta=sta), timeout=30).decode()
    except Exception as e:  # noqa: BLE001
        print(f"  {sta}: no level2 directory ({e})")
        return []
    return sorted(set(re.findall(rf'{sta}\.(\d{{4}})\.bsm\.level2\.tar', html)))


def fetch_areal_strain(sta, year):
    """Download (or reuse cached) station-year tar, return the Eee+Enn member as a DataFrame,
    or None if that member/tar isn't available (some early install-year tars are near-empty)."""
    cache_dir = os.path.join(TAR_CACHE, sta)
    os.makedirs(cache_dir, exist_ok=True)
    tar_path = os.path.join(cache_dir, f"{sta}.{year}.bsm.level2.tar")
    if not os.path.exists(tar_path):
        try:
            data = get(LEVEL2_TAR.format(sta=sta, year=year), timeout=90)
        except Exception as e:  # noqa: BLE001
            print(f"    {sta} {year}: download failed ({e})")
            return None
        with open(tar_path, "wb") as f:
            f.write(data)
    try:
        with tarfile.open(tar_path) as tf:
            member = next((m for m in tf.getnames() if m.endswith("Eee+Enn.txt.gz")), None)
            if member is None:
                return None
            raw = tf.extractfile(member).read()
    except tarfile.ReadError:
        return None
    text = gzip.decompress(raw).decode(errors="replace")
    # NOT read via the file's own header row: verified (B072/2013, B086/2006) that EarthScope's
    # own header line is missing a tab between the last two column names ("version atmp" is one
    # header cell but two real data fields) -- a genuine quirk of their exporter, not a mistake
    # here. Every data row has 14 tab-separated fields regardless of station/year, so column
    # identity is hardcoded from direct inspection rather than trusted from the header.
    COLS = ["strain", "date", "doy", "MJD", "areal_mstrain", "s_offset", "quality",
            "tide_c", "detrend_c", "atmp_c", "atmp_c_quality", "level", "version", "atmp"]
    df = pd.read_csv(io.StringIO(text), sep="\t", skiprows=1, names=COLS, usecols=range(14))
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip())
    df["quality"] = df["quality"].astype(str).str.strip()
    # keep good('g') and interpolated('i') -- both are physically real signal with a short gap
    # filled in per EarthScope's own processing; drop bad('b') only, same spirit as GNSS's own
    # gap-interpolation (limit=30 days) rather than discarding anything not perfectly clean.
    df = df[df["quality"].isin(["g", "i"])]
    return df[["date", "areal_mstrain"]]


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    print(f"segment: {cfg['segment_id']} ({paths['slug']})")
    picked, max_dist_km = find_candidate_stations(cfg)
    print(f"{len(picked)} PB strainmeter stations within {max_dist_km}km (capped to {MAX_STATIONS})")
    strain_dir = os.path.join(ROOT, "data", "strain", paths["slug"])
    os.makedirs(strain_dir, exist_ok=True)
    raw_path = os.path.join(strain_dir, "daily_raw.csv")
    agg_path = os.path.join(strain_dir, "daily_aggregate.csv")

    if not picked:
        pd.DataFrame(columns=["station", "date", "areal_mstrain"]).to_csv(raw_path, index=False)
        pd.DataFrame(columns=["mean_z_areal", "n_stations"]).to_csv(agg_path)
        print("no PB strainmeter stations in range -- wrote empty files (honest: real network "
              "gap here, not a fetch failure)")
        return

    all_rows = []
    for sta, lat, lon, dist_km in picked:
        years = list_years(sta)
        if not years:
            continue
        n_rows_sta = 0
        for year in years:
            df = fetch_areal_strain(sta, year)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["station"] = sta
            n_rows_sta += len(df)
            all_rows.append(df[["station", "date", "areal_mstrain"]])
        print(f"  {sta}: dist={dist_km:.1f}km {len(years)} station-years, {n_rows_sta} readings")
        # incremental save
        if all_rows:
            pd.concat(all_rows, ignore_index=True).to_csv(raw_path, index=False)

    if not all_rows:
        pd.DataFrame(columns=["station", "date", "areal_mstrain"]).to_csv(raw_path, index=False)
        pd.DataFrame(columns=["mean_z_areal", "n_stations"]).to_csv(agg_path)
        print("stations found but no usable Level-2 data -- wrote empty files")
        return

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(raw_path, index=False)
    print(f"{df['station'].nunique()} stations, {len(df)} total 5-min readings -> {raw_path}")

    # daily mean per station first (5-min -> daily), then robust z per station, then cross-station
    # average -- same recipe as ingest_groundwater.py's aggregation, for the same reason (puts
    # stations with very different absolute strain offsets on a common comparable scale before
    # averaging).
    df["day"] = df["date"].dt.normalize()
    daily_sta = df.groupby(["station", "day"])["areal_mstrain"].mean().reset_index()
    daily_sta["z"] = daily_sta.groupby("station")["areal_mstrain"].transform(
        lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median() + 1e-6))
    agg = daily_sta.groupby("day").agg(mean_z_areal=("z", "mean"),
                                       n_stations=("station", "nunique"))
    agg.index.name = "date"
    agg.to_csv(agg_path)
    print(f"wrote {len(agg)} daily aggregate rows -> {agg_path}")
    print(agg.tail(5))


if __name__ == "__main__":
    main()
