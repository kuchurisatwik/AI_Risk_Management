"""
Behavioral Labeler — Composite anomaly detection label.

Binary classification target:
  1 = Trader in emotionally compromised state (anomaly)
  0 = Normal trading behavior

Uses only current-row features (no future data).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_behavioral(df, threshold=0.6):
    """
    Generate behavioral anomaly labels.
    
    Uses the pre-computed emotional_risk_score composite.
    Threshold of 0.6 gives more training examples than the policy
    engine's hard block at 0.8, allowing the model to learn gradations.
    
    Args:
        df: DataFrame with 'emotional_risk_score' column
        threshold: Score above which behavior is labeled anomalous
    
    Returns:
        numpy array of binary labels (0/1)
    """
    labels = (df['emotional_risk_score'] > threshold).astype(int).values
    
    n_anomaly = labels.sum()
    n_normal = len(labels) - n_anomaly
    logger.info(
        f"Behavioral labels: {n_anomaly} anomaly ({n_anomaly/len(labels)*100:.1f}%) / "
        f"{n_normal} normal ({n_normal/len(labels)*100:.1f}%)"
    )
    
    return labels
