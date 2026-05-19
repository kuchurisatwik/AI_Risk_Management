"""
HMM Regime Labeler — Gaussian Hidden Markov Model for Regime Detection.

Replaces the Euclidean KMeans model with a temporal HMM that models 
the transition probabilities between market states. Fits on stationary
returns and realized volatility.
"""

import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Stationary features for HMM
HMM_FEATURES = ['log_return', 'realized_volatility']

REGIME_NAMES = [
    'trending_low_vol',
    'trending_high_vol',
    'sideways_low_vol',
    'choppy_high_vol'
]


def prepare_hmm_features(df):
    """
    Compute log returns and ensure stationary features for HMM.
    """
    df = df.copy()
    
    # Calculate log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Fill NAs
    df['log_return'] = df['log_return'].fillna(0)
    if 'realized_volatility' not in df.columns:
        # Fallback if realized_volatility wasn't passed properly
        df['realized_volatility'] = df['log_return'].rolling(20).std().fillna(0)
        
    return df


def auto_map_hmm_states(model, scaler):
    """
    Map hidden states to regime names based on means and variances.
    
    HMM outputs state means: [log_return_mean, volatility_mean]
    """
    means_scaled = model.means_
    means = scaler.inverse_transform(means_scaled)
    
    # Extract volatility for each state (index 1 is realized_volatility)
    vols = means[:, 1]
    
    # Extract directional drift (absolute log return)
    drift = np.abs(means[:, 0])
    
    mapping = {}
    used_names = set()
    
    # Sort states by volatility (highest first)
    vol_order = vols.argsort()[::-1]
    
    for state in vol_order:
        vol = vols[state]
        dr = drift[state]
        
        # Heuristics based on relative ranking of states
        if 'choppy_high_vol' not in used_names and vol > np.median(vols) and dr <= np.median(drift):
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
            # Fallback
            remaining = set(REGIME_NAMES) - used_names
            if remaining:
                mapping[state] = remaining.pop()
                used_names.add(mapping[state])
            else:
                mapping[state] = f'regime_{state}'
                
    return mapping


def fit_hmm_model(train_df, n_components=4, random_state=42):
    """
    Fit Gaussian HMM on training data ONLY.
    
    Returns:
        scaler, hmm_model, state-to-name mapping
    """
    logger.info(f"Fitting Gaussian HMM with {n_components} components...")
    
    df = prepare_hmm_features(train_df)
    X_train = df[HMM_FEATURES].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=100,
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


def assign_hmm_labels(df, scaler, model, mapping):
    """
    Assign HMM regime labels and probabilities to a DataFrame.
    """
    df = prepare_hmm_features(df)
    X = df[HMM_FEATURES].values
    X_scaled = scaler.transform(X)
    
    df_out = df.copy()
    
    # Predict most likely state sequence (Viterbi)
    hidden_states = model.predict(X_scaled)
    df_out['regime_cluster'] = hidden_states
    df_out['regime_label'] = df_out['regime_cluster'].map(mapping)
    
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
