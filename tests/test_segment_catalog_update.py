import pandas as pd

from update_segment_catalogs import merge_rows


def test_incremental_catalog_replaces_revision_atomically(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame([{"time": "2020-01-01T00:00:00Z", "latitude": 1, "longitude": 2,
                   "depth": 3, "mag": 4.0, "id": "x"}]).to_csv(path, index=False)
    before, after = merge_rows(path, [{"time": "2020-01-01T00:00:01Z", "latitude": 1,
                                      "longitude": 2, "depth": 3, "mag": 4.2, "id": "x"}])
    stored = pd.read_csv(path)
    assert before == after == 1
    assert stored.loc[0, "mag"] == 4.2
