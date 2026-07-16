"""MOCRE-1 "ocean loading" pass (4-jul-2026, continuation): real non-tidal ocean loading data as
a genuinely NEW variable -- NOT part of the 25-variable audited catalog, proposed and verified
this session after Matt asked about ocean data / global-vs-local stations. Treat with MORE
skepticism than the audited channels, not less (see DEFAULT_CONSEQUENTS tier below).

Two candidate sources checked for real, no-auth access before writing any code (same discipline
as strain/groundwater):
  - NASA International Mass Loading Service (massloading.sciencecloud.nasa.gov) -- REJECTED after
    verifying actual content, not just reachability. curl -I gave 200 OK, but the precomputed
    per-station series and even the "on-demand" custom-computation form are BOTH bounded to
    19800101-20171130 (page's own text, last modified 2017-12-09) -- frozen 9 years stale, useless
    as a live anomaly-state input for holdouts that run through 2026.
  - EOST/ITES Loading Service (loading.u-strasbg.fr), ECCO2 model -- ACCEPTED. Verified by
    downloading a real station file (P297), not trusting the directory-listing mtime: daily NEU
    (North/East/Up, mm) displacement from 1992-01-04 through 2024-12-31 -- ~1.5yr stale as of
    2026-07, same order of staleness already accepted for Oklahoma's wells data, not the NASA
    source's 9-year freeze. Real coverage check against stations already selected by ingest_gnss.py
    (reused directly, no new station search): SJC 84/113, Parkfield 44/47, Mojave 70/88, Cascadia
    36/48, Shumagin 6/6, Oklahoma 23/39 -- broad, real overlap, not a handful of lucky matches.

Physical target: vertical (Up) displacement from non-tidal ocean bottom pressure loading --
literature (referenced in EOST's own page) ties this to modulation of tremor/SSE rate specifically
in subduction zones (Cascadia is the motivating case Matt asked about), but this project has NOT
independently audited that mechanism the way the 25-variable catalog audit did for strain/fluids/
etc. Symmetric bajo/alto from the very first line (lesson already re-learned 3+ times in this
project for skipping this on a variable's first pass) -- direction is not established either way.
"""
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

from common import ROOT, load_config, segment_paths

LIST_URL = "http://loading.u-strasbg.fr/listdata.php?dirn=dicf"
FILE_URL = "http://loading.u-strasbg.fr/ITRF/CF/ECCO2/{fname}"
DIRLIST_CACHE = os.path.join(ROOT, "data", "ocean_load", "_dirlist_cache.html")
FILE_CACHE = os.path.join(ROOT, "data", "ocean_load", "_file_cache")
MJD_EPOCH = pd.Timestamp("1858-11-17")
MAX_STATIONS = 20


def get(url, timeout=90, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt+1} ({e})", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def load_code_to_files():
    """Code -> list of ECCO2 filenames, from the (large, cached) EOST directory listing.

    Cached to disk: the source listing is ~40MB combining all models (ATMIB/ECCO/ECCO2/GLDAS2/
    GLORYS/GRACE) for every ITRF station -- fine to fetch once, wasteful to refetch per segment."""
    os.makedirs(os.path.dirname(DIRLIST_CACHE), exist_ok=True)
    if not os.path.exists(DIRLIST_CACHE):
        html = get(LIST_URL).decode(errors="replace")
        with open(DIRLIST_CACHE, "w") as f:
            f.write(html)
    else:
        with open(DIRLIST_CACHE) as f:
            html = f.read()
    start = html.find(">ECCO2<")
    end = html.find("<b>", start + 10)
    section = html[start:end] if start >= 0 else ""
    files = re.findall(r"([A-Z0-9]{4})_([A-Z0-9]+)_NEU\.ecco2", section)
    code_to_files = {}
    for code, domes in files:
        code_to_files.setdefault(code, []).append(f"{code}_{domes}_NEU.ecco2")
    return code_to_files


def fetch_series(fname):
    """Download (or reuse cached) one station's ECCO2 NEU file, return daily Up-component
    DataFrame indexed by real date (converted from MJD)."""
    os.makedirs(FILE_CACHE, exist_ok=True)
    path = os.path.join(FILE_CACHE, fname)
    if not os.path.exists(path):
        try:
            data = get(FILE_URL.format(fname=fname))
        except Exception as e:  # noqa: BLE001
            print(f"    {fname}: download failed ({e})")
            return None
        with open(path, "wb") as f:
            f.write(data)
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["mjd", "n_mm", "e_mm", "u_mm"])
    except Exception as e:  # noqa: BLE001
        print(f"    {fname}: parse failed ({e})")
        return None
    df["date"] = MJD_EPOCH + pd.to_timedelta(df["mjd"], unit="D")
    df["date"] = df["date"].dt.normalize()
    # ECCO2 is native daily (unlike ATMIB's 6-hourly) -- one row/day already, but a defensive
    # groupby mean handles any duplicate-day rows without assuming the source never changes.
    daily = df.groupby("date")["u_mm"].mean()
    return daily


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    print(f"segment: {cfg['segment_id']} ({paths['slug']})")

    # Reuse the GNSS stations ingest_gnss.py already selected for this segment -- real, already
    # distance-filtered against the fault corridor, no need to run a second station search against
    # a different network. A true prerequisite: run ingest_gnss.py for this segment first.
    gnss_dir = os.path.join(ROOT, "data", "gnss", paths["slug"])
    ocean_dir = os.path.join(ROOT, "data", "ocean_load", paths["slug"])
    os.makedirs(ocean_dir, exist_ok=True)
    raw_path = os.path.join(ocean_dir, "daily_raw.csv")
    agg_path = os.path.join(ocean_dir, "daily_aggregate.csv")

    if not os.path.isdir(gnss_dir):
        pd.DataFrame(columns=["station", "date", "u_mm"]).to_csv(raw_path, index=False)
        pd.DataFrame(columns=["mean_z_up", "n_stations"]).to_csv(agg_path)
        print("no GNSS stations ingested yet for this segment -- run ingest_gnss.py first; "
              "wrote empty files")
        return

    codes = sorted(f[:-6] for f in os.listdir(gnss_dir) if f.endswith(".tenv3"))
    code_to_files = load_code_to_files()
    matched = [(c, code_to_files[c][0]) for c in codes if c in code_to_files][:MAX_STATIONS]
    print(f"{len(matched)}/{len(codes)} segment GNSS stations have real ECCO2 ocean-loading "
          f"coverage (capped to {MAX_STATIONS})")

    if not matched:
        pd.DataFrame(columns=["station", "date", "u_mm"]).to_csv(raw_path, index=False)
        pd.DataFrame(columns=["mean_z_up", "n_stations"]).to_csv(agg_path)
        print("no ECCO2 coverage for any station in this segment -- real network gap, wrote "
              "empty files")
        return

    all_rows = []
    for code, fname in matched:
        s = fetch_series(fname)
        if s is None or s.empty:
            continue
        d = s.reset_index()
        d.columns = ["date", "u_mm"]
        d["station"] = code
        all_rows.append(d[["station", "date", "u_mm"]])
        print(f"  {code}: {len(d)} daily readings ({d['date'].min().date()} -> "
              f"{d['date'].max().date()})")
        if all_rows:
            pd.concat(all_rows, ignore_index=True).to_csv(raw_path, index=False)

    if not all_rows:
        pd.DataFrame(columns=["station", "date", "u_mm"]).to_csv(raw_path, index=False)
        pd.DataFrame(columns=["mean_z_up", "n_stations"]).to_csv(agg_path)
        print("stations matched but no usable data -- wrote empty files")
        return

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(raw_path, index=False)
    print(f"{df['station'].nunique()} stations, {len(df)} total daily readings -> {raw_path}")

    # Per-station robust z first (puts stations with different absolute Up-displacement offsets
    # on a comparable scale), then cross-station daily mean -- same recipe as strain/groundwater.
    df["z"] = df.groupby("station")["u_mm"].transform(
        lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median() + 1e-6))
    agg = df.groupby("date").agg(mean_z_up=("z", "mean"), n_stations=("station", "nunique"))
    agg.index.name = "date"
    agg.to_csv(agg_path)
    print(f"wrote {len(agg)} daily aggregate rows -> {agg_path}")
    print(f"data currency: last real reading {df['date'].max().date()} -- STALE past that date, "
          f"same honest caveat as Oklahoma wells data going stale past 2024-12.")
    print(agg.tail(5))


if __name__ == "__main__":
    main()
