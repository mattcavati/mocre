"""Inferencia fuzzy jerárquica y causal del estado preparatorio de una falla.

IA1 transforma canales observados en evidencia por clúster físico. IA2 combina
coincidencia, cobertura, tendencia y memoria para inferir un estado latente.
No contiene etiquetas ni calibra probabilidades: esa responsabilidad pertenece
al entrenamiento externo y permite usar exactamente la misma ruta en vivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import get_membership, sets_for_var


CLUSTERS = {
    "regional": ["a_regional_rate", "a_regional_spread", "a_bvalue_drop",
                 "a_fault_concentration", "a_mag_accel"],
    "seismic": ["a_seis", "a_swarm", "a_foreshock", "a_noise"],
    "deformation": ["a_gnss", "a_sse", "a_strain", "a_insar"],
    "mechanical": ["C_state", "a_fluid", "F_state"],
    "hydro": ["a_water", "a_fluid"],
    "remote": ["a_ocean", "a_tec"],
}

AVAILABILITY = {
    "a_regional_rate": "has_regional_catalog_data",
    "a_regional_spread": "has_regional_catalog_data",
    "a_bvalue_drop": "has_regional_catalog_data",
    "a_fault_concentration": "has_regional_catalog_data",
    "a_mag_accel": "has_regional_catalog_data",
    "a_gnss": "has_gnss_data", "a_sse": "has_gnss_data",
    "a_fluid": "has_fluid_data", "a_water": "has_water_data",
    "a_strain": "has_strain_data", "a_ocean": "has_ocean_data",
    "a_insar": "has_insar_data", "a_tec": "has_tec_data",
    "a_noise": "has_noise_data",
}

RULES = [
    ("regional_expansion", ("regional_level", "regional_persistence")),
    ("regional_local_confirmation", ("regional_level", "seismic_level")),
    ("regional_deformation", ("regional_trend", "deformation_level")),
    ("regional_loading", ("regional_level", "mechanical_level")),
    ("regional_quiescence_loading", ("regional_quiet", "mechanical_level")),
    ("seismic_deformation", ("seismic_level", "deformation_level")),
    ("deformation_loading", ("deformation_trend", "mechanical_level")),
    ("seismic_loading", ("seismic_trend", "mechanical_level")),
    ("local_quiescence_deformation", ("seismic_quiet", "deformation_level")),
    ("persistent_multiphysics", ("seismic_persistence", "deformation_persistence")),
    ("triple_cluster", ("seismic_level", "deformation_level", "mechanical_level")),
    ("hydro_mechanical", ("hydro_trend", "mechanical_level")),
    ("remote_confirmation", ("remote_level", "deformation_level")),
    ("coherent_acceleration", ("seismic_coherence", "deformation_coherence")),
]


def _availability(df: pd.DataFrame, channel: str) -> np.ndarray:
    name = AVAILABILITY.get(channel)
    if name and name in df:
        return df[name].fillna(0).to_numpy(float) > 0.5
    if channel in df:
        return df[channel].notna().to_numpy()
    return np.zeros(len(df), dtype=bool)


def _term_membership(values, channel, terms) -> np.ndarray:
    x = np.nan_to_num(np.asarray(values, dtype=float))
    sets = sets_for_var(channel)
    return np.maximum.reduce([get_membership(x, sets[t]) for t in terms])


def derive_cluster_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """IA1: evidencia causal [0,1] por clúster, sin convertir missing en neutral."""
    out = pd.DataFrame(index=df.index)
    for cluster, channels in CLUSTERS.items():
        levels, quiets, trends, available = [], [], [], []
        for channel in channels:
            if channel not in df:
                continue
            x = df[channel].astype(float)
            av = _availability(df, channel)
            level = _term_membership(x.to_numpy(), channel, ("alto", "muy_alto"))
            quiet = _term_membership(x.to_numpy(), channel, ("bajo", "muy_bajo"))
            # Diferencia de medias exclusivamente retrospectivas; el valor del
            # día ya fue causalizado por pipeline.py.
            fast = x.rolling(7, min_periods=3).mean()
            slow = x.rolling(60, min_periods=20).mean()
            trend = np.clip((fast - slow).fillna(0).to_numpy() / 0.35, -1, 1)
            trend = _term_membership(trend, "anomaly", ("alto", "muy_alto"))
            levels.append(np.where(av, level, np.nan))
            quiets.append(np.where(av, quiet, np.nan))
            trends.append(np.where(av, trend, np.nan))
            available.append(av.astype(float))
        if not levels:
            for suffix in ("level", "quiet", "trend", "persistence", "coherence", "coverage"):
                out[f"{cluster}_{suffix}"] = 0.0
            continue
        lev = np.vstack(levels).T
        qui = np.vstack(quiets).T
        tre = np.vstack(trends).T
        avm = np.vstack(available).T
        # Evidencias independientes se acumulan con noisy-OR; no se pierden
        # detrás del máximo de una única cláusula.
        out[f"{cluster}_level"] = 1 - np.nanprod(1 - lev, axis=1)
        out[f"{cluster}_quiet"] = 1 - np.nanprod(1 - qui, axis=1)
        out[f"{cluster}_trend"] = 1 - np.nanprod(1 - tre, axis=1)
        n_valid = np.sum(np.isfinite(lev), axis=1)
        daily_high = np.divide(np.nansum(lev, axis=1), n_valid,
                               out=np.zeros(len(lev)), where=n_valid > 0)
        out[f"{cluster}_persistence"] = pd.Series(daily_high, index=df.index).rolling(
            30, min_periods=10).mean().fillna(0)
        coherent = np.sum((lev >= 0.5) & np.isfinite(lev), axis=1)
        out[f"{cluster}_coherence"] = np.divide(coherent, n_valid,
                                                 out=np.zeros(len(lev)), where=n_valid > 0)
        out[f"{cluster}_coverage"] = np.mean(avm, axis=1)
    return out.fillna(0).clip(0, 1)


def rule_firings(evidence: pd.DataFrame) -> pd.DataFrame:
    """T-norma min por regla; la cobertura limita la fuerza de su inferencia."""
    out = pd.DataFrame(index=evidence.index)
    for name, antecedents in RULES:
        fire = np.minimum.reduce([evidence[a].to_numpy(float) for a in antecedents])
        clusters = {a.rsplit("_", 1)[0] for a in antecedents}
        coverage = np.minimum.reduce([evidence[f"{c}_coverage"].to_numpy(float) for c in clusters])
        out[name] = fire * coverage
    return out


def recurrent_state(raw_score, decay=0.94) -> np.ndarray:
    """Memoria IA2: ataque rápido y relajación controlada, siempre causal."""
    x = np.clip(np.asarray(raw_score, dtype=float), 0, 1)
    state = np.zeros(len(x), dtype=float)
    for i in range(1, len(x)):
        attack = 0.65 if x[i] > state[i - 1] else 1 - decay
        state[i] = state[i - 1] + attack * (x[i] - state[i - 1])
    return np.clip(state, 0, 1)


def infer_series(df: pd.DataFrame, model: dict | None = None):
    evidence = derive_cluster_evidence(df)
    firings = rule_firings(evidence)
    weights = np.ones(len(RULES), dtype=float) / max(len(RULES), 1)
    intercept = -2.0
    if model:
        weights = np.asarray(model["weights"], dtype=float)
        intercept = float(model["intercept"])
    z = intercept + firings.to_numpy() @ weights
    raw = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    state = recurrent_state(raw)
    return evidence, firings, pd.Series(state, index=df.index, name="critical_state")
