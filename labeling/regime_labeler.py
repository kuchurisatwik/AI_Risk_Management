"""
Regime Labeler — Unsupervised KMeans Clustering on normalized features.

Fits KMeans ONLY on training data to prevent leakage.
Assigns regime_cluster and regime_label to all splits.
Saves scaler + kmeans model to models/regime/.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Per roadmap Section 5.2 — cluster ONLY on normalized ratio/percentile features
REGIME_FEATURES = [
    'atr_expansion_ratio',
    'volatility_percentile',
    'ema_20_slope',
    'trend_alignment_score',
    'volume_ratio',
    'bb_width_percentile'
]

# Mapping of cluster centroids to human-readable regime names
# This gets auto-mapped based on centroid characteristics
REGIME_NAMES = [
    'trending_low_vol',
    'trending_high_vol',
    'sideways_low_vol',
    'choppy_high_vol',
    'crash_mode'
]


def auto_map_clusters(km, scaler, feature_names):
    """
    Automatically map cluster IDs to regime names based on centroid characteristics.
    
    Logic:
    - High ema_20_slope + low atr_expansion → trending_low_vol
    - High ema_20_slope + high atr_expansion → trending_high_vol
    - Low abs(ema_20_slope) + low atr_expansion → sideways_low_vol
    - Low abs(ema_20_slope) + high atr_expansion → choppy_high_vol
    - Extreme atr_expansion + extreme volatility_percentile → crash_mode
    """
    # Get centroids in original feature space
    centroids_scaled = km.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    
    centroid_df = pd.DataFrame(centroids, columns=feature_names)
    
    mapping = {}
    used_names = set()
    
    # Sort clusters by volatility (atr_expansion_ratio) to assign crash_mode first
    vol_order = centroid_df['atr_expansion_ratio'].argsort()[::-1]
    
    for cluster_id in vol_order:
        row = centroid_df.iloc[cluster_id]
        atr_exp = row['atr_expansion_ratio']
        vol_pct = row['volatility_percentile']
        ema_slope = abs(row['ema_20_slope'])
        
        if 'crash_mode' not in used_names and atr_exp > 2.0 and vol_pct > 0.8:
            mapping[cluster_id] = 'crash_mode'
            used_names.add('crash_mode')
        elif 'choppy_high_vol' not in used_names and atr_exp > 1.3 and ema_slope < 0.3:
            mapping[cluster_id] = 'choppy_high_vol'
            used_names.add('choppy_high_vol')
        elif 'trending_high_vol' not in used_names and atr_exp > 1.0 and ema_slope > 0.2:
            mapping[cluster_id] = 'trending_high_vol'
            used_names.add('trending_high_vol')
        elif 'sideways_low_vol' not in used_names and atr_exp < 1.0 and ema_slope < 0.2:
            mapping[cluster_id] = 'sideways_low_vol'
            used_names.add('sideways_low_vol')
        elif 'trending_low_vol' not in used_names:
            mapping[cluster_id] = 'trending_low_vol'
            used_names.add('trending_low_vol')
        else:
            # Fallback for any unmapped
            remaining = set(REGIME_NAMES) - used_names
            if remaining:
                mapping[cluster_id] = remaining.pop()
                used_names.add(mapping[cluster_id])
            else:
                mapping[cluster_id] = f'regime_{cluster_id}'
    
    return mapping, centroid_df


def fit_regime_model(train_df, n_clusters=5, random_state=42):
    """
    Fit KMeans on training data ONLY.
    
    Returns:
        scaler, kmeans model, cluster-to-name mapping, centroid DataFrame
    """
    X_train = train_df[REGIME_FEATURES].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        max_iter=500
    )
    km.fit(X_scaled)
    
    # Evaluate cluster quality
    sil_score = silhouette_score(X_scaled, km.labels_, sample_size=min(5000, len(X_scaled)))
    logger.info(f"Regime KMeans silhouette score: {sil_score:.4f}")
    
    # Auto-map clusters to regime names
    mapping, centroid_df = auto_map_clusters(km, scaler, REGIME_FEATURES)
    
    return scaler, km, mapping, centroid_df, sil_score


def assign_regime_labels(df, scaler, km, mapping):
    """
    Assign regime_cluster and regime_label to a DataFrame.
    Uses scaler.transform() (NOT fit_transform) to prevent leakage.
    """
    X = df[REGIME_FEATURES].values
    X_scaled = scaler.transform(X)
    
    df = df.copy()
    df['regime_cluster'] = km.predict(X_scaled)
    df['regime_label'] = df['regime_cluster'].map(mapping)
    
    # Regime confidence = inverse of distance to nearest centroid
    distances = km.transform(X_scaled).min(axis=1)
    df['regime_confidence'] = 1.0 / (1.0 + distances)
    
    return df


def save_regime_model(scaler, km, mapping, output_dir):
    """Save scaler, kmeans, and mapping to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, output_dir / 'regime_scaler.pkl')
    joblib.dump(km, output_dir / 'regime_kmeans.pkl')
    joblib.dump(mapping, output_dir / 'regime_mapping.pkl')
    
    logger.info(f"Saved regime model to {output_dir}")
