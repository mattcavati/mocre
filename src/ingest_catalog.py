"""MOCRE-1 F0: download USGS ComCat catalog for the segment bbox (recursive time split)."""
import csv
import io
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

from common import load_config, segment_paths

API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
LIMIT = 20000


def fetch(start, end, cfg):
    b = cfg["catalog"]["bbox"]
    params = {
        "format": "csv",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "minlatitude": b["minlat"],
        "maxlatitude": b["maxlat"],
        "minlongitude": b["minlon"],
        "maxlongitude": b["maxlon"],
        "minmagnitude": cfg["catalog"]["min_magnitude"],
        "orderby": "time-asc",
        "limit": LIMIT,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                text = r.read().decode()
            rows = list(csv.DictReader(io.StringIO(text)))
            return rows
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry {attempt+1} ({e})", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def fetch_recursive(start, end, cfg, out):
    rows = fetch(start, end, cfg)
    if len(rows) >= LIMIT:  # window saturated: split
        mid = start + (end - start) / 2
        print(f"  {start}..{end}: saturated, splitting")
        fetch_recursive(start, mid, cfg, out)
        fetch_recursive(mid, end, cfg, out)
    else:
        out.extend(rows)
        print(f"  {start}..{end}: {len(rows)} events")


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    print(f"segment: {cfg['segment_id']} ({paths['slug']})")
    start = date.fromisoformat(cfg["catalog"]["start"])
    end = date.today()
    all_rows = []
    step = timedelta(days=730)
    t = start
    while t < end:
        t2 = min(t + step, end)
        fetch_recursive(t, t2, cfg, all_rows)
        t = t2
    # dedupe by event id, keep chronological
    seen, rows = set(), []
    for r in all_rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            rows.append(r)
    out_path = paths["catalog_csv"]
    keep = ["time", "latitude", "longitude", "depth", "mag", "magType", "id", "place"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} events -> {out_path}")


if __name__ == "__main__":
    main()
