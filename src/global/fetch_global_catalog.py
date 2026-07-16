"""Descarga reproducible del catálogo mundial ComCat M>=4.5.

Se conservan los identificadores y la fecha de revisión de ComCat. Esos campos son
necesarios para distinguir una revisión de un terremoto de un terremoto nuevo.
"""
import io
import os
import sys
import time
import urllib.request

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = os.path.join(ROOT, "data", "global")
os.makedirs(OUTDIR, exist_ok=True)
BASE = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
        "&minmagnitude=4.5&orderby=time-asc&starttime={a}&endtime={b}")

# chunks de 2 años (media ~11.5k eventos/chunk, bajo el límite de 20k; los años sísmicos fuertes
# como 2011 caben porque el conteo real 2010-2012 ~ 17k)
YEARS = list(range(1980, 2027, 2))


def fetch(a, b, path, tries=4):
    url = BASE.format(a=a, b=b)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mocre-research/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read()
            df = pd.read_csv(
                io.BytesIO(raw),
                usecols=["id", "time", "updated", "latitude", "longitude", "depth", "mag", "net"],
            )
            if len(df) >= 20000:
                # chunk saturado: partir en dos mitades anuales y concatenar
                mid = f"{(int(a[:4]) + int(b[:4])) // 2}-01-01"
                p1 = path + ".h1"; p2 = path + ".h2"
                fetch(a, mid, p1); fetch(mid, b, p2)
                df = pd.concat([pd.read_csv(p1), pd.read_csv(p2)], ignore_index=True)
                os.remove(p1); os.remove(p2)
            df.to_csv(path, index=False)
            print(f"  {a}..{b}: {len(df)} eventos", flush=True)
            return
        except Exception as e:
            print(f"  {a}..{b}: intento {k+1} fallo ({e}); reintento en {15*(k+1)}s", flush=True)
            time.sleep(15 * (k + 1))
    raise RuntimeError(f"chunk {a}..{b} irrecuperable")


def main():
    parts = []
    for i, y in enumerate(YEARS[:-1]):
        a, b = f"{y}-01-01", f"{YEARS[i+1]}-01-01"
        path = os.path.join(OUTDIR, f"chunk_{y}.csv")
        if not (os.path.exists(path) and os.path.getsize(path) > 1000):
            fetch(a, b, path)
        parts.append(path)
    # tramo final hasta hoy
    last = os.path.join(OUTDIR, "chunk_tail.csv")
    fetch(f"{YEARS[-1]}-01-01", "2030-01-01", last)
    parts.append(last)

    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.dropna(subset=["latitude", "longitude", "mag"])
    if "id" in df:
        with_id = df[df.id.notna() & df.id.astype(str).ne("")].sort_values("updated")
        without_id = df[~(df.id.notna() & df.id.astype(str).ne(""))]
        df = pd.concat([with_id.drop_duplicates("id", keep="last"), without_id], ignore_index=True)
    df = df.drop_duplicates(subset=["time", "latitude", "longitude"], keep="last")
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601").dt.tz_convert(None)
    df = df.sort_values("time").reset_index(drop=True)
    out = os.path.join(OUTDIR, "catalog_global.csv")
    df.to_csv(out, index=False)
    print(f"\nTOTAL: {len(df)} eventos {df.time.min()} .. {df.time.max()} -> {out}", flush=True)


if __name__ == "__main__":
    main()
