"""
Adaptive Threshold Engine v2 — Regime Probability Floors + Ranking.

Replaced percentile-based thresholds with simple regime-specific
probability floors. The actual selectivity comes from the Daily
Opportunity Ranker in trade_decision.py, not from these thresholds.
These floors act as minimum-edge safety gates.
"""

import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Regime-specific probability floors.
# These are intentionally close to 0.50 — real selectivity comes
# from the Daily Opportunity Ranker (top-N per day).
REGIME_PROBABILITY_FLOORS = {
    'trending_low_vol':  0.52,   # Slight edge is sufficient
    'trending_high_vol': 0.55,
    'sideways_low_vol':  0.54,
    'choppy_high_vol':   0.60,   # Require stronger conviction
    'crash_mode':        1.00,   # Unreachable → block
    'unknown':           0.60,
}


@dataclass
class ThresholdState:
    """Current adaptive threshold with metadata."""
    base_threshold: float
    adjusted_threshold: float
    regime: str
    branch: str
    adjustments: dict


class AdaptiveThresholdEngine:
    """
    Regime-aware probability floor gating.

    Instead of percentile-based thresholds (which failed with compressed
    post-purge distributions), this uses simple probability floors as
    safety gates. The Daily Opportunity Ranker handles selectivity.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, train_momentum_probs: np.ndarray):
        """
        Fit the threshold engine (lightweight — just records stats).
        """
        valid = train_momentum_probs[~np.isnan(train_momentum_probs)]
        if len(valid) > 0:
            logger.info(
                f"Threshold engine fitted: "
                f"p25={np.percentile(valid, 25):.3f} "
                f"p50={np.percentile(valid, 50):.3f} "
                f"p75={np.percentile(valid, 75):.3f} "
                f"p95={np.percentile(valid, 95):.3f}"
            )
        self._fitted = True

    def get_threshold(
        self,
        regime_label: str,
        branch: str = 'trend',
        volatility_percentile: float = 0.5,
        recent_drawdown: float = 0.0,
        strategy_health_score: float = 1.0,
        regime_confidence: float = 0.5,
    ) -> ThresholdState:
        """
        Compute the regime-specific probability floor.

        This is a MINIMUM gate, not the primary selection mechanism.
        The Daily Opportunity Ranker handles top-N selection.
        """
        base = REGIME_PROBABILITY_FLOORS.get(regime_label, 0.60)

        adjustments = {}
        adjusted = base

        # Tighten floor slightly during adverse conditions
        if recent_drawdown > 0.05:
            shift = min(recent_drawdown * 0.5, 0.05)
            adjusted += shift
            adjustments['dd_shift'] = shift

        if strategy_health_score < 0.4:
            shift = (0.4 - strategy_health_score) * 0.05
            adjusted += shift
            adjustments['health_shift'] = shift

        # Cap at 0.70 (never require extreme confidence)
        adjusted = min(adjusted, 0.70)

        return ThresholdState(
            base_threshold=base,
            adjusted_threshold=adjusted,
            regime=regime_label,
            branch=branch,
            adjustments=adjustments
        )


# Module-level convenience functions
_global_engine = AdaptiveThresholdEngine()

def fit_threshold_engine(train_momentum_probs: np.ndarray):
    """Fit the global threshold engine."""
    _global_engine.fit(train_momentum_probs)

def compute_adaptive_threshold(
    regime_label: str,
    branch: str = 'trend',
    volatility_percentile: float = 0.5,
    recent_drawdown: float = 0.0,
    strategy_health_score: float = 1.0,
    regime_confidence: float = 0.5,
) -> ThresholdState:
    """Compute threshold using the global engine."""
    return _global_engine.get_threshold(
        regime_label, branch, volatility_percentile,
        recent_drawdown, strategy_health_score, regime_confidence
    )
