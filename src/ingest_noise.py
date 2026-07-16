"""MOCRE-1 "ruido sísmico ambiental" pass (4-jul-2026): seismic background-noise level as a
channel. In the audited 25-variable catalog (mu_ev=0.35). The 3-jul audit marked it BLOCKED
("MUSTANG doesn't cover the CI network") -- REOPENED after a 4-jul verification found that claim
was FALSE: it was a bad metric-name error (metric=sample_rate doesn't exist in MUSTANG -> a
misleading 404). With a valid metric (sample_rms / pct_below_nlnm) CI returns data to today.

Source verified real (downloaded actual content, not HTTP 200): EarthScope MUSTANG summary metrics
(service.earthscope.org/mustang/measurements/1/query), open, no account, daily values current to
~2-3 days ago (verified 2026-07-02 across all 6 segments). One fresh broadband station per segment
was confirmed to have data <=3 days old.

HONEST SCOPE / CONFIDENCE (same discipline as InSAR/ocean): the real earthquake-noise precursor
mechanism is dv/v (relative seismic velocity change from ambient-noise cross-correlation), which
needs the raw continuous waveform + a cross-correlation/stretching pipeline (fase-2, not built).
What THIS channel uses is `sample_rms`, the daily RMS of the background signal -- a proxy for the
noise LEVEL, not a velocity change. Two honest caveats, both handled:
  1. sample_rms spikes on days with real earthquakes (verified: ~4000-7700 vs ~700-900 baseline)
     -- so this channel is PARTIALLY correlated with seismicity itself. Mitigations: (a) log10
     transform + robust_anomaly's median/MAD baseline is outlier-robust so event days don't poison
     the baseline; (b) NO coincidence rule with a_seis (would be spurious self-confirmation) --
     only noise_bajo/noise_alto direct rules, same choice as InSAR's missing coincid rule.
  2. background noise is strongly seasonal (ocean microseism) -- the 365-day robust baseline
     window absorbs the annual cycle, same as every other channel here.
Confidence tier: LOW (water tier), for the mechanism-proxy gap above -- not a data-quality issue
(the data is clean and current), an interpretation-strength one.
"""
import io
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

from common import ROOT, load_config, segment_paths

# one fresh broadband station per segment, verified 2026-07-04 to have sample_rms <=3 days old.
# (net, sta, cha) -- location code is resolved by trying candidates (see fetch).
STATION_TABLE = {
    "sjc-anza": ("AZ", "FRD", "BHZ"),
    "saf-parkfield": ("CI", "PLM", "BHZ"),
    "saf-mojave": ("CI", "ADO", "BHZ"),
    "csz-central-or": ("IU", "COR", "BHZ"),
    "aak-shumagin": ("AV", "VNSO", "BHZ"),
    "ok-pawnee-prague": ("OK", "BLOK", "HHZ"),
}
MUSTANG = ("https://service.earthscope.org/mustang/measurements/1/query"
           "?metric=sample_rms&net={net}&sta={sta}&loc={loc}&cha={cha}&format=text"
           "&timewindow={start},{end}")
LOC_CANDIDATES = ["", "00", "10"]
BACKFILL_DAYS = 731  # ~2 years, same real bounded window as TEC


def get(url, timeout=60, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "mocre1/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def parse_measurements(text):
    """MUSTANG text format: a title line, a header line, then quoted CSV rows
    value,target,start,end,lddate. Returns DataFrame[date, sample_rms] or None."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    body = "\n".join(lines[1:])  # drop the human title line, keep header + rows
    df = pd.read_csv(io.StringIO(body))
    if "value" not in df.columns or "start" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["start"], format="%Y/%m/%d %H:%M:%S").dt.normalize()
    df = df[["date", "value"]].rename(columns={"value": "sample_rms"})
    df = df[df["sample_rms"] > 0]
    return df.groupby("date")["sample_rms"].median().reset_index()


def fetch_station(net, sta, cha, start, end):
    for loc in LOC_CANDIDATES:
        url = MUSTANG.format(net=net, sta=sta, loc=(loc or "--"), cha=cha,
                             start=start, end=end)
        try:
            text = get(url)
        except Exception:  # noqa: BLE001
            continue
        df = parse_measurements(text)
        if df is not None and len(df) > 30:
            return df, loc
    return None, None


def main():
    cfg = load_config()
    paths = segment_paths(cfg)
    slug = paths["slug"]
    print(f"segment: {cfg['segment_id']} ({slug})")

    out_dir = os.path.join(ROOT, "data", "noise", slug)
    os.makedirs(out_dir, exist_ok=True)
    agg_path = os.path.join(out_dir, "daily_aggregate.csv")

    if slug not in STATION_TABLE:
        pd.DataFrame(columns=["log_rms", "n"]).to_csv(agg_path)
        print(f"{slug}: no station assigned -- wrote empty file")
        return

    net, sta, cha = STATION_TABLE[slug]
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=BACKFILL_DAYS)
    fmt = "%Y-%m-%dT00:00:00"
    print(f"station: {net}.{sta}..{cha} | window {start.date()} -> {end.date()}")

    df, loc = fetch_station(net, sta, cha, start.strftime(fmt), end.strftime(fmt))
    if df is None:
        pd.DataFrame(columns=["log_rms", "n"]).to_csv(agg_path)
        print(f"{slug}: no usable MUSTANG sample_rms for {net}.{sta} (tried loc {LOC_CANDIDATES}) "
              f"-- wrote empty file (real gap or station renamed)")
        return

    df = df.set_index("date").sort_index()
    # log10 compresses the earthquake-day spikes (RMS ~700 baseline vs ~7700 on event days) so the
    # downstream robust median/MAD baseline is not distorted; the anomaly itself still fires on a
    # genuinely elevated day. This is the noise LEVEL proxy, not dv/v -- see module docstring.
    df["log_rms"] = np.log10(df["sample_rms"])
    out = df[["log_rms"]].copy()
    out["n"] = 1
    out.to_csv(agg_path)
    print(f"loc='{loc}' | {len(out)} daily rms samples -> {agg_path}")
    print(f"real coverage: {out.index.min().date()} -> {out.index.max().date()} "
          f"(latency ~2-3 days, fine for a 2026-07 holdout)")
    print(out.tail(5))


if __name__ == "__main__":
    main()
