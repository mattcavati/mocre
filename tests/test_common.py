import numpy as np
import pandas as pd

from common import causal_reindex, membership_centroid


def test_membership_centroid_triangle_and_shoulders():
    assert membership_centroid([0, 3, 6]) == 3
    assert np.isclose(membership_centroid([0, 0, 1, 2]), 7 / 9)
    assert np.isclose(membership_centroid([0, 1, 2, 2]), 11 / 9)


def test_causal_reindex_never_uses_future_endpoint():
    source = pd.Series([1.0, 9.0], index=pd.to_datetime(["2020-01-01", "2020-01-04"]))
    days = pd.date_range("2020-01-01", "2020-01-04")
    got = causal_reindex(source, days)
    assert got.tolist() == [1.0, 1.0, 1.0, 9.0]
