import numpy as np
import pandas as pd

from csep_test import comparison_test


def test_comparison_uses_one_sample_per_event_and_count_correction():
    idx = pd.date_range("2020-01-01", periods=3)
    series = pd.DataFrame({"lam_mocre": [2.0, 2.0, 3.0],
                           "lam_etas": [1.0, 1.0, 1.0]}, index=idx)
    events = pd.DataFrame({"date": [idx[0], idx[2]]})
    got = comparison_test(series, events)

    expected = (np.log(2.0) + np.log(3.0)) / 2.0 - (7.0 - 3.0) / 2.0
    assert got["n_events"] == 2
    assert np.isclose(got["information_gain_per_event_nats"], expected)
    assert got["low_power"] is True


def test_n_test_exposes_absolute_rate_failure():
    idx = pd.date_range("2020-01-01", periods=100)
    mocre = np.ones(100) * 2
    mocre[:3] = [1, 2, 3]
    series = pd.DataFrame({"lam_mocre": mocre,
                           "lam_etas": np.ones(100) * 0.01}, index=idx)
    events = pd.DataFrame({"date": idx[:3]})
    got = comparison_test(series, events)
    assert got["n_test_mocre"]["passes_95pct"] is False
