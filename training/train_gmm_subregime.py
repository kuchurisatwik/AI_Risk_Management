"""
GMM Sub-Regime Clustering — Local Microstructure Quality Separation.

Within each HMM macro regime, a Gaussian Mixture Model clusters candles
into local quality states. This separates high-quality setups from noise.

Trend Branch GMM: trend_initiation, trend_continuation, trend_exhaustion
MeanRev Branch GMM: clean_reversion, slow_fade, false_signal
"""

import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Features for trend sub-regime clustering
TREND_GMM_FEATURES = [
    'ema_20_slope', 'momentum_score', 'momentum_exhaustion_score',
    'breakout_distance', 'volume_ratio'
]

# Features for mean-reversion sub-regime clustering
MEANREV_GMM_FEATURES = [
    'normalized_vwap_distance', 'rsi_extremity', 'bb_width_percentile',
    'trade_imbalance', 'wick_imbalance_ratio'
]

# Regime names for each branch
TREND_SUBREGIME_NAMES = ['trend_initiation', 'trend_continuation', 'trend_exhaustion']
MEANREV_SUBREGIME_NAMES = ['clean_reversion', 'slow_fade', 'false_signal']


def _auto_map_trend_clusters(model, scaler, feature_names):
    """Map GMM clusters to trend sub-regime names based on centroids."""
    means_scaled = model.means_
    means = scaler.inverse_transform(means_scaled)

    # Index mapping: ema_20_slope=0, momentum_score=1, exhaustion=2, breakout=3, volume=4
    momentum_idx = feature_names.index('momentum_score')
    exhaustion_idx = feature_names.index('momentum_exhaustion_score')
    breakout_idx = feature_names.index('breakout_distance')

    mapping = {}
    used = set()

    # Sort by momentum strength (strongest first)
    order = np.argsort(-np.abs(means[:, momentum_idx]))

    for state in order:
        mom = np.abs(means[state, momentum_idx])
        exh = means[state, exhaustion_idx]
        brk = np.abs(means[state, breakout_idx])

        if 'trend_initiation' not in used and brk > np.median(np.abs(means[:, breakout_idx])):
            mapping[state] = 'trend_initiation'
            used.add('trend_initiation')
        elif 'trend_exhaustion' not in used and exh > np.median(means[:, exhaustion_idx]):
            mapping[state] = 'trend_exhaustion'
            used.add('trend_exhaustion')
        elif 'trend_continuation' not in used:
            mapping[state] = 'trend_continuation'
            used.add('trend_continuation')
        else:
            remaining = set(TREND_SUBREGIME_NAMES) - used
            if remaining:
                mapping[state] = remaining.pop()
                used.add(mapping[state])
            else:
                mapping[state] = f'trend_cluster_{state}'

    return mapping


def _auto_map_meanrev_clusters(model, scaler, feature_names):
    """Map GMM clusters to mean-reversion sub-regime names based on centroids."""
    means_scaled = model.means_
    means = scaler.inverse_transform(means_scaled)

    vwap_idx = feature_names.index('normalized_vwap_distance')
    rsi_idx = feature_names.index('rsi_extremity')

    mapping = {}
    used = set()

    # Sort by VWAP distance (most extreme first)
    order = np.argsort(-np.abs(means[:, vwap_idx]))

    for state in order:
        vwap_dist = np.abs(means[state, vwap_idx])
        rsi_ext = means[state, rsi_idx]

        if 'clean_reversion' not in used and vwap_dist > np.median(np.abs(means[:, vwap_idx])) and rsi_ext > np.median(means[:, rsi_idx]):
            mapping[state] = 'clean_reversion'
            used.add('clean_reversion')
        elif 'false_signal' not in used and vwap_dist < np.median(np.abs(means[:, vwap_idx])):
            mapping[state] = 'false_signal'
            used.add('false_signal')
        elif 'slow_fade' not in used:
            mapping[state] = 'slow_fade'
            used.add('slow_fade')
        else:
            remaining = set(MEANREV_SUBREGIME_NAMES) - used
            if remaining:
                mapping[state] = remaining.pop()
                used.add(mapping[state])
            else:
                mapping[state] = f'meanrev_cluster_{state}'

    return mapping


def fit_gmm_subregime(df, branch='trend', n_components=3, random_state=42):
    """
    Fit a GMM on the appropriate feature set.

    Args:
        df: DataFrame (should be pre-filtered to the correct macro regime)
        branch: 'trend' or 'meanrev'
        n_components: Number of GMM components
        random_state: Random seed

    Returns:
        scaler, gmm_model, mapping dict
    """
    if branch == 'trend':
        features = TREND_GMM_FEATURES
        map_fn = _auto_map_trend_clusters
    else:
        features = MEANREV_GMM_FEATURES
        map_fn = _auto_map_meanrev_clusters

    # Filter to available features and drop NaNs
    available = [f for f in features if f in df.columns]
    if len(available) < 3:
        logger.warning(f"GMM {branch}: only {len(available)} features available, skipping")
        return None, None, None

    X = df[available].dropna()
    if len(X) < n_components * 10:
        logger.warning(f"GMM {branch}: only {len(X)} samples, need {n_components * 10}, skipping")
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        n_init=5,
        random_state=random_state
    )
    gmm.fit(X_scaled)

    mapping = map_fn(gmm, scaler, available)

    logger.info(f"GMM {branch}: fitted {n_components} components on {len(X)} samples")
    logger.info(f"  BIC: {gmm.bic(X_scaled):.1f} | AIC: {gmm.aic(X_scaled):.1f}")
    for k, v in mapping.items():
        count = np.sum(gmm.predict(X_scaled) == k)
        logger.info(f"  Cluster {k} -> {v} ({count} samples)")

    return scaler, gmm, mapping


def assign_gmm_labels(df, scaler, gmm, mapping, branch='trend'):
    """Assign GMM sub-regime labels to a DataFrame."""
    if scaler is None or gmm is None:
        df['gmm_subregime'] = 'unknown'
        return df

    if branch == 'trend':
        features = TREND_GMM_FEATURES
    else:
        features = MEANREV_GMM_FEATURES

    available = [f for f in features if f in df.columns]
    df = df.copy()

    # Fill NaN in feature columns for prediction
    X = df[available].fillna(0).values
    X_scaled = scaler.transform(X)

    clusters = gmm.predict(X_scaled)
    df['gmm_subregime'] = [mapping.get(c, 'unknown') for c in clusters]
    df['gmm_confidence'] = gmm.predict_proba(X_scaled).max(axis=1)

    return df


def save_gmm_model(scaler, gmm, mapping, output_dir, branch='trend'):
    """Save GMM components to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f'gmm_{branch}'
    joblib.dump(scaler, output_dir / f'{prefix}_scaler.pkl')
    joblib.dump(gmm, output_dir / f'{prefix}_model.pkl')
    joblib.dump(mapping, output_dir / f'{prefix}_mapping.pkl')

    logger.info(f"Saved GMM {branch} model to {output_dir}")
