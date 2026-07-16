"""Actualización incremental y atómica de los catálogos regionales ComCat."""
from __future__ import annotations

import glob
import os
import tempfile
from datetime import date, timedelta

import pandas as pd

from common import ROOT, load_config, segment_paths
from ingest_catalog import fetch_recursive


def merge_rows(path, rows):
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    new = pd.DataFrame(rows)
    both = pd.concat([old, new], ignore_index=True)
    if "id" in both:
        identified = both[both.id.notna() & both.id.astype(str).ne("")].drop_duplicates("id", keep="last")
        unidentified = both[~(both.id.notna() & both.id.astype(str).ne(""))]
        both = pd.concat([identified, unidentified], ignore_index=True)
    both = both.drop_duplicates(["time", "latitude", "longitude"], keep="last")
    both["time"] = pd.to_datetime(both.time, utc=True, errors="coerce")
    both = both.sort_values("time")
    both["time"] = both.time.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    fd, tmp = tempfile.mkstemp(prefix=".events-", suffix=".csv", dir=os.path.dirname(path))
    try:
        os.close(fd); both.to_csv(tmp, index=False); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return len(old), len(both)


def main():
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "config", "segment_*.json"))):
        cfg = load_config(os.path.basename(cfg_path)); paths = segment_paths(cfg)
        old = pd.read_csv(paths["catalog_csv"], usecols=["time"])
        last = pd.to_datetime(old.time, utc=True, errors="coerce").max()
        start = max(date.fromisoformat(cfg["catalog"]["start"]),
                    (last - pd.Timedelta(days=8)).date()) if pd.notna(last) else date.fromisoformat(cfg["catalog"]["start"])
        rows = []
        fetch_recursive(start, date.today() + timedelta(days=1), cfg, rows)
        before, after = merge_rows(paths["catalog_csv"], rows)
        print(f"{paths['slug']}: {before}->{after} ({after-before:+d})")


if __name__ == "__main__":
    main()
