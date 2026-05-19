"""
Adaptive Threshold Engine — Regime-aware momentum thresholds.

Thresholds are calibrated RELATIVE to the actual model output distribution.
A calibrated probability of 0.32 in a trending market may represent
strong edge when the distribution range is 0.28-0.35.
"""

import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Regime-aware threshold PERCENTILES within the momentum probability distribution.
# e.g., trending_low_vol uses the 60th percentile of probabilities as threshold,
# meaning we take the top 40% of signals.
REGIME_PERCENTILE_THRESHOLDS = {
    'trending_low_vol':  0.55,   # take top 45% of signals
    'trending_high_vol': 0.65,   # take top 35% of signals
    'sideways_low_vol':  0.75,   # take top 25% of signals
    'choppy_high_vol':   0.90,   # take top 10% of signals
    'crash_mode':        1.00,   # block all
    'unknown':           0.80,
}


@dataclass
class ThresholdState:
    """Current adaptive threshold with metadata."""
    base_threshold: float
    adjusted_threshold: float
    regime: str
    adjustments: dict


class AdaptiveThresholdEngine:
    """
    Calibrates thresholds based on the actual training probability distribution.
    
    Instead of fixed absolute thresholds (which fail when calibrated
    probabilities land in a narrow range), we compute percentile-based
    thresholds from the training set's momentum_probability distribution.
    """
    
    def __init__(self):
        self._percentiles = {}  # cached percentile lookup
        self._fitted = False
    
    def fit(self, train_momentum_probs: np.ndarray):
        """
        Fit the threshold engine to the training probability distribution.
        
        Precomputes percentile values so inference is O(1).
        """
        valid = train_momentum_probs[~np.isnan(train_momentum_probs)]
        
        # Precompute percentile lookup at 1% granularity
        for pct in range(0, 101):
            self._percentiles[pct / 100.0] = float(np.percentile(valid, pct))
        
        logger.info(
            f"Threshold engine fitted: "
            f"p25={self._percentiles[0.25]:.3f} "
            f"p50={self._percentiles[0.50]:.3f} "
            f"p75={self._percentiles[0.75]:.3f} "
            f"p95={self._percentiles[0.95]:.3f}"
        )
        self._fitted = True
    
    def get_threshold(
        self,
        regime_label: str,
        volatility_percentile: float = 0.5,
        recent_drawdown: float = 0.0,
        strategy_health_score: float = 1.0,
        regime_confidence: float = 0.5,
    ) -> ThresholdState:
        """
        Compute the adaptive momentum threshold.
        
        Uses percentile-based lookup from the training distribution,
        then applies soft adjustments for adverse conditions.
        """
        if not self._fitted:
            # Fallback to absolute threshold if not fitted
            return ThresholdState(
                base_threshold=0.31,
                adjusted_threshold=0.31,
                regime=regime_label,
                adjustments={}
            )
        
        # Get the regime's percentile target
        target_pct = REGIME_PERCENTILE_THRESHOLDS.get(regime_label, 0.80)
        
        # Look up the actual probability value at that percentile
        pct_key = round(target_pct, 2)
        base = self._percentiles.get(pct_key, self._percentiles.get(0.75, 0.32))
        
        adjustments = {}
        adjusted = base
        
        # Soft adjustments: shift percentile upward for adverse conditions
        
        # High volatility: be more selective
        if volatility_percentile > 0.7:
            pct_shift = (volatility_percentile - 0.7) * 0.10
            higher_pct = min(1.0, round(target_pct + pct_shift, 2))
            adjusted = self._percentiles.get(higher_pct, adjusted)
            adjustments['vol_shift'] = pct_shift
        
        # Drawdown: tighten filter
        if recent_drawdown > 0.03:
            pct_shift = min(recent_drawdown * 1.5, 0.10)
            higher_pct = min(1.0, round(target_pct + pct_shift, 2))
            adjusted = self._percentiles.get(higher_pct, adjusted)
            adjustments['dd_shift'] = pct_shift
        
        # Low strategy health: tighten
        if strategy_health_score < 0.5:
            pct_shift = (0.5 - strategy_health_score) * 0.10
            higher_pct = min(1.0, round(target_pct + pct_shift, 2))
            adjusted = self._percentiles.get(higher_pct, adjusted)
            adjustments['health_shift'] = pct_shift
        
        return ThresholdState(
            base_threshold=base,
            adjusted_threshold=adjusted,
            regime=regime_label,
            adjustments=adjustments
        )


# Module-level convenience function for backward compatibility
_global_engine = AdaptiveThresholdEngine()

def fit_threshold_engine(train_momentum_probs: np.ndarray):
    """Fit the global threshold engine."""
    _global_engine.fit(train_momentum_probs)

def compute_adaptive_threshold(
    regime_label: str,
    volatility_percentile: float = 0.5,
    recent_drawdown: float = 0.0,
    strategy_health_score: float = 1.0,
    regime_confidence: float = 0.5,
) -> ThresholdState:
    """Compute threshold using the global engine."""
    return _global_engine.get_threshold(
        regime_label, volatility_percentile,
        recent_drawdown, strategy_health_score, regime_confidence
    )
