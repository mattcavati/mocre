"""Registro prospectivo inmutable de las salidas globales de MOCRE.

Cada intervalo produce como máximo un fichero nuevo. Un proceso repetido nunca
reescribe el pronóstico ya emitido, condición mínima para una prueba prospectiva.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "out" / "prospective"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_hashes():
    out = {}
    for name in ("global_engine_v2.json", "global_engine_v3.json"):
        path = ROOT / "out" / "global" / name
        if path.exists():
            out[name] = sha256_file(path)
    return out


def archive_forecast(forecast, ledger_dir=None, issued_at=None, interval_minutes=None):
    """Archiva *forecast* sin sobrescribir. Devuelve la ruta o ``None`` si existe."""
    issued = issued_at or datetime.now(timezone.utc)
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)
    issued = issued.astimezone(timezone.utc)
    minutes = interval_minutes or int(os.getenv("MOCRE_LEDGER_INTERVAL_MINUTES", "60"))
    slot_minute = (issued.minute // minutes) * minutes if minutes <= 60 else 0
    slot = issued.replace(minute=slot_minute, second=0, microsecond=0)
    base = Path(ledger_dir or os.getenv("MOCRE_LEDGER_DIR", DEFAULT_LEDGER))
    folder = base / slot.strftime("%Y/%m/%d")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"forecast-{slot:%H%M}Z.json.gz"

    record = dict(forecast)
    record["ledger"] = {
        "schema": 1,
        "issued_at": issued.isoformat(timespec="seconds"),
        "slot": slot.isoformat(timespec="seconds"),
        "immutable": True,
        "model_sha256": model_hashes(),
    }
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return None
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=int(issued.timestamp())) as gz:
                gz.write(json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return str(path)
