import numpy as np
import pandas as pd

from critical_state import RULES, derive_cluster_evidence, infer_series, recurrent_state, rule_firings


def _frame(n=100):
    idx = pd.date_range("2020-01-01", periods=n)
    cols = ["a_seis", "a_swarm", "a_foreshock", "a_noise", "a_gnss", "a_sse",
            "a_strain", "a_insar", "C_state", "a_fluid", "F_state", "a_water",
            "a_ocean", "a_tec"]
    df = pd.DataFrame(0.0, index=idx, columns=cols)
    for c in ["gnss", "strain", "insar", "fluid", "water", "ocean", "tec", "noise"]:
        df[f"has_{c}_data"] = 1.0
    return df


def test_multicluster_coincidence_exceeds_single_cluster():
    one = _frame(); both = _frame()
    one.loc[one.index[-30]:, ["a_seis", "a_swarm"]] = 0.9
    both.loc[both.index[-30]:, ["a_seis", "a_swarm", "a_gnss", "a_strain"]] = 0.9
    r1 = rule_firings(derive_cluster_evidence(one)).iloc[-1]
    r2 = rule_firings(derive_cluster_evidence(both)).iloc[-1]
    assert r1["seismic_deformation"] == 0
    assert r2["seismic_deformation"] > 0.5


def test_missing_sensor_cannot_vote_as_evidence():
    df = _frame()
    df["has_gnss_data"] = 0
    df["has_strain_data"] = 0
    df["has_insar_data"] = 0
    df[["a_gnss", "a_sse", "a_strain", "a_insar"]] = 1.0
    evidence = derive_cluster_evidence(df)
    assert evidence.iloc[-1]["deformation_level"] == 0
    assert evidence.iloc[-1]["deformation_coverage"] == 0


def test_recurrent_state_has_memory_but_decays():
    x = np.r_[np.zeros(10), 1.0, np.zeros(30)]
    state = recurrent_state(x)
    assert state[10] > 0.5
    assert state[11] > 0
    assert state[-1] < state[11]


def test_future_rows_do_not_change_past_inference():
    df = _frame(120)
    model = {"intercept": -2.0, "weights": [1.0] * len(RULES)}
    before = infer_series(df.iloc[:100], model)[2]
    df.iloc[100:, df.columns.get_loc("a_seis")] = 1.0
    after = infer_series(df, model)[2].iloc[:100]
    assert np.allclose(before, after)
