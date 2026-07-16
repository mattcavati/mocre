import gzip
import json
from datetime import datetime, timezone

import pandas as pd

import forecast_ledger
import update_realtime


def _row(event_id, time, updated, lat=10.0, lon=20.0, mag=5.0):
    return {"id": event_id, "time": time, "updated": updated, "latitude": lat,
            "longitude": lon, "depth": 12.0, "mag": mag, "net": "us"}


def test_merge_persists_revision_even_when_row_count_does_not_grow(tmp_path, monkeypatch):
    monkeypatch.setattr(update_realtime, "DATA", str(tmp_path))
    old = pd.DataFrame([_row(None, "2026-01-01T00:00:00", None, mag=4.9)])
    old.to_csv(tmp_path / "catalog_global.csv", index=False)
    revision = pd.DataFrame([_row("us123", "2026-01-01T00:00:02",
                                  "2026-01-01T01:00:00", mag=5.1)])

    merged = update_realtime.merge_catalog(revision)
    persisted = pd.read_csv(tmp_path / "catalog_global.csv")

    assert len(merged) == len(persisted) == 1
    assert persisted.loc[0, "id"] == "us123"
    assert persisted.loc[0, "mag"] == 5.1


def test_ledger_is_immutable_per_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_ledger, "model_hashes", lambda: {"model": "abc"})
    issued = datetime(2026, 7, 11, 12, 7, tzinfo=timezone.utc)
    first = forecast_ledger.archive_forecast({"cells": [[1]], "fields": ["x"]},
                                             tmp_path, issued)
    second = forecast_ledger.archive_forecast({"cells": [[2]], "fields": ["x"]},
                                              tmp_path, issued)

    assert first is not None
    assert second is None
    with gzip.open(first, "rt") as f:
        stored = json.load(f)
    assert stored["cells"] == [[1]]
    assert stored["ledger"]["immutable"] is True
