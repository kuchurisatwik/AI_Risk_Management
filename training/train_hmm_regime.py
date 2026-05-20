"""
HMM Regime Labeler — Gaussian Hidden Markov Model for Regime Detection.

Uses a 5-state Gaussian HMM with 4 stationary features to detect macro
market regimes. Includes temporal persistence filtering to prevent
costly regime-whipsawing during transitions.
"""

import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Stationary features for HMM (4 features for richer regime separation)
HMM_FEATURES = ['log_return', 'realized_volatility', 'volume_ratio', 'atr_expansion_ratio']

REGIME_NAMES = [
    'trending_low_vol',
    'trending_high_vol',
    'sideways_low_vol',
    'choppy_high_vol',
    'crash_mode'
]

# Minimum consecutive bars a regime must persist before the router
# activates its branch. Prevents costly whipsawing during transitions.
MIN_REGIME_DWELL = 3


def prepare_hmm_features(df):
    """
    Compute log returns and ensure all 4 stationary features for HMM.
    """
    df = df.copy()

    # Calculate log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return'] = df['log_return'].fillna(0)

    if 'realized_volatility' not in df.columns:
        df['realized_volatility'] = df['log_return'].rolling(20).std().fillna(0)

    if 'volume_ratio' not in df.columns:
        vol_mean = df['volume'].rolling(20).mean()
        df['volume_ratio'] = (df['volume'] / vol_mean).fillna(1.0)

    if 'atr_expansion_ratio' not in df.columns:
        if 'atr_14' in df.columns:
            atr_mean = df['atr_14'].rolling(50).mean()
            df['atr_expansion_ratio'] = (df['atr_14'] / atr_mean).fillna(1.0)
        else:
            df['atr_expansion_ratio'] = 1.0

    return df


def auto_map_hmm_states(model, scaler):
    """
    Map hidden states to regime names based on means.

    HMM outputs state means: [log_return, realized_vol, volume_ratio, atr_expansion]
    5-state mapping: trending_low/high_vol, sideways_low_vol, choppy_high_vol, crash_mode
    """
    means_scaled = model.means_
    means = scaler.inverse_transform(means_scaled)

    # Feature indices: 0=log_return, 1=realized_vol, 2=volume_ratio, 3=atr_expansion
    vols = means[:, 1]       # realized volatility
    drift = np.abs(means[:, 0])  # absolute drift
    vol_ratios = means[:, 2]  # volume spikes
    atr_exp = means[:, 3]     # ATR expansion

    mapping = {}
    used_names = set()
    n_states = len(means)

    # Crash mode: extreme volatility + extreme volume spike + extreme ATR expansion
    vol_threshold = np.percentile(vols, 80) if n_states >= 5 else np.max(vols)
    vol_ratio_threshold = np.percentile(vol_ratios, 80) if n_states >= 5 else np.max(vol_ratios)

    # Sort states by volatility (highest first)
    vol_order = vols.argsort()[::-1]

    for state in vol_order:
        vol = vols[state]
        dr = drift[state]
        vr = vol_ratios[state]
        ae = atr_exp[state]

        if 'crash_mode' not in used_names and vol >= vol_threshold and vr >= vol_ratio_threshold:
            mapping[state] = 'crash_mode'
            used_names.add('crash_mode')
        elif 'choppy_high_vol' not in used_names and vol > np.median(vols) and dr <= np.median(drift):
            mapping[state] = 'choppy_high_vol'
            used_names.add('choppy_high_vol')
        elif 'trending_high_vol' not in used_names and vol > np.median(vols):
            mapping[state] = 'trending_high_vol'
            used_names.add('trending_high_vol')
        elif 'sideways_low_vol' not in used_names and vol <= np.median(vols) and dr <= np.median(drift):
            mapping[state] = 'sideways_low_vol'
            used_names.add('sideways_low_vol')
        elif 'trending_low_vol' not in used_names:
            mapping[state] = 'trending_low_vol'
            used_names.add('trending_low_vol')
        else:
            remaining = set(REGIME_NAMES) - used_names
            if remaining:
                mapping[state] = remaining.pop()
                used_names.add(mapping[state])
            else:
                mapping[state] = f'regime_{state}'

    return mapping


def fit_hmm_model(train_df, n_components=5, random_state=42):
    """
    Fit 5-state Gaussian HMM on training data ONLY.

    Returns:
        scaler, hmm_model, state-to-name mapping
    """
    logger.info(f"Fitting Gaussian HMM with {n_components} components...")

    df = prepare_hmm_features(train_df)
    X_train = df[HMM_FEATURES].dropna().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        tol=0.01
    )

    model.fit(X_scaled)

    # Auto-map states
    mapping = auto_map_hmm_states(model, scaler)

    logger.info(f"HMM Converged: {model.monitor_.converged}")
    logger.info("Transition Matrix:")
    logger.info("\n" + str(np.round(model.transmat_, 3)))

    return scaler, model, mapping


def apply_temporal_persistence(labels, min_dwell=MIN_REGIME_DWELL):
    """
    Apply minimum-dwell temporal persistence filter.

    If a regime appears for fewer than `min_dwell` consecutive bars,
    it is replaced by the previous persistent regime. This prevents
    costly whipsawing during HMM transition uncertainty.
    """
    result = labels.copy()
    n = len(result)

    i = 0
    while i < n:
        j = i
        while j < n and result[j] == result[i]:
            j += 1
        run_length = j - i

        if run_length < min_dwell and i > 0:
            # Replace short run with previous regime
            result[i:j] = result[i - 1]

        i = j

    return result


def assign_hmm_labels(df, scaler, model, mapping, apply_persistence=True):
    """
    Assign HMM regime labels and probabilities to a DataFrame.
    Includes optional temporal persistence filtering.
    """
    df = prepare_hmm_features(df)
    X = df[HMM_FEATURES].fillna(0).values
    X_scaled = scaler.transform(X)

    df_out = df.copy()

    # Predict most likely state sequence (Viterbi)
    hidden_states = model.predict(X_scaled)
    df_out['regime_cluster'] = hidden_states
    df_out['regime_label'] = df_out['regime_cluster'].map(mapping)

    # Apply temporal persistence filter
    if apply_persistence:
        df_out['regime_label'] = apply_temporal_persistence(
            df_out['regime_label'].values, MIN_REGIME_DWELL
        )

    # Predict state probabilities
    state_probs = model.predict_proba(X_scaled)
    df_out['regime_confidence'] = state_probs.max(axis=1)

    # Clean up temp columns
    if 'log_return' in df_out.columns:
        df_out = df_out.drop(columns=['log_return'])

    return df_out


def save_hmm_model(scaler, model, mapping, output_dir):
    """Save HMM components to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, output_dir / 'regime_hmm_scaler.pkl')
    joblib.dump(model, output_dir / 'regime_hmm_model.pkl')
    joblib.dump(mapping, output_dir / 'regime_hmm_mapping.pkl')
    
    logger.info(f"Saved HMM regime model to {output_dir}")
