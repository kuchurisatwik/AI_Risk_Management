"""
Drift Detector — Online distribution monitoring.

Detects when live feature distributions diverge from training,
signaling potential model degradation or regime shift.
Uses PSI (Population Stability Index) and rolling statistics.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Drift detection results."""
    features_drifted: List[str] = field(default_factory=list)
    psi_scores: Dict[str, float] = field(default_factory=dict)
    overall_drift: bool = False
    retrain_recommended: bool = False


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    
    PSI < 0.10: No significant shift
    PSI 0.10 - 0.25: Moderate shift, monitor
    PSI > 0.25: Significant shift, retrain
    """
    # Create bins from expected distribution
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1
    
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


class DriftDetector:
    """
    Monitors feature and prediction drift.
    
    Stores reference distributions from the training set
    and compares live data against them.
    """
    
    PSI_WARNING = 0.10
    PSI_CRITICAL = 0.25
    
    def __init__(self):
        self.reference_stats = {}  # {feature: {mean, std, q25, q75}}
        self.reference_arrays = {}  # {feature: np.array}
        self._fitted = False
    
    def fit(self, train_df: pd.DataFrame, feature_cols: list):
        """
        Store reference distributions from training data.
        """
        for col in feature_cols:
            if col in train_df.columns:
                vals = train_df[col].dropna().values
                self.reference_stats[col] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'q25': float(np.percentile(vals, 25)),
                    'q75': float(np.percentile(vals, 75)),
                }
                self.reference_arrays[col] = vals
        
        self._fitted = True
        logger.info(f"Drift detector fitted on {len(self.reference_arrays)} features")
    
    def check(self, live_df: pd.DataFrame, feature_cols: list = None) -> DriftReport:
        """
        Check live data against reference distributions.
        
        Args:
            live_df: Recent live data (e.g., last 500 candles)
            feature_cols: Features to check (None = all fitted)
        
        Returns:
            DriftReport
        """
        if not self._fitted:
            raise RuntimeError("DriftDetector not fitted. Call .fit() first.")
        
        if feature_cols is None:
            feature_cols = list(self.reference_arrays.keys())
        
        report = DriftReport()
        
        for col in feature_cols:
            if col not in live_df.columns or col not in self.reference_arrays:
                continue
            
            live_vals = live_df[col].dropna().values
            if len(live_vals) < 10:
                continue
            
            ref_vals = self.reference_arrays[col]
            psi = compute_psi(ref_vals, live_vals)
            report.psi_scores[col] = psi
            
            if psi > self.PSI_CRITICAL:
                report.features_drifted.append(col)
                report.overall_drift = True
            elif psi > self.PSI_WARNING:
                report.features_drifted.append(col)
        
        # Recommend retrain if >30% of features show significant drift
        drift_ratio = len(report.features_drifted) / max(1, len(feature_cols))
        if drift_ratio > 0.30:
            report.retrain_recommended = True
        
        return report
