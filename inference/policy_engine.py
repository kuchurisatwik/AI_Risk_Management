"""
Policy Engine v2 — Regime-Routed Dual-Branch Authority Layer.

Key changes from v1:
  1. Removed hard blocks for sideways_low_vol and choppy_high_vol
     (these regimes now have their own model branches)
  2. Added daily trade budget enforcement
  3. Added minimum inter-trade spacing (4 bars = 1 hour)
  4. Regime routing now uses REGIME_RISK_CONFIG from config.py
"""

import logging
from dataclasses import dataclass, field
from typing import List
import pandas as pd

from inference.model_ensemble import ModelOutputs
from training.config import REGIME_RISK_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    """Output of the Policy Engine."""
    allow_trade: bool = False
    risk_percent: float = 0.0
    sl_multiplier: float = 1.5
    tp_multiplier: float = 1.5
    block_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    regime_action: str = 'normal'  # normal, reduce, block
    active_branch: str = 'trend'   # trend, meanrev


class PolicyEngine:
    """
    Institutional-grade policy engine with regime routing.

    Applies in order:
    1. Hard blocks (instant rejection, no override)
    2. Daily trade budget check
    3. Inter-trade spacing check
    4. Regime-specific risk modification
    5. Soft adjustments (risk reduction based on conditions)
    6. Final authorization
    """

    # Hard block thresholds
    EMOTIONAL_BLOCK_THRESHOLD = 0.80
    STRATEGY_DISABLE_THRESHOLD = 0.30
    ILLIQUIDITY_BLOCK_THRESHOLD = 0.15
    MAX_CONSECUTIVE_LOSSES = 5

    # Daily trade budget
    GLOBAL_MAX_DAILY_TRADES = 3

    # Minimum bars between entries (12 bars = 1 hour on 5m)
    MIN_INTER_TRADE_BARS = 12

    def __init__(self, exec_params: dict = None):
        self._daily_trade_count = 0
        self._current_utc_day = None
        self._bars_since_last_trade = 999  # Start with no restriction
        self.exec_params = exec_params or {}
        
        # Override defaults with optimized params if available
        self.GLOBAL_MAX_DAILY_TRADES = self.exec_params.get('max_daily_trades', 3)
        self.MIN_INTER_TRADE_BARS = self.exec_params.get('min_inter_trade_bars', 12)

    def reset_daily_budget(self, utc_day):
        """Reset the daily trade counter for a new UTC day."""
        if utc_day != self._current_utc_day:
            self._daily_trade_count = 0
            self._current_utc_day = utc_day

    def record_trade(self):
        """Record that a trade was executed."""
        self._daily_trade_count += 1
        self._bars_since_last_trade = 0

    def tick(self):
        """Advance the bar counter (called every candle)."""
        self._bars_since_last_trade += 1

    def evaluate(self, model_outputs: ModelOutputs, features: dict) -> PolicyDecision:
        """
        Run the full policy evaluation.
        """
        decision = PolicyDecision(allow_trade=True, risk_percent=1.0)
        decision.active_branch = model_outputs.active_branch

        # Track current UTC day for budget
        open_time = features.get('open_time', None)
        if open_time is not None:
            if isinstance(open_time, pd.Timestamp):
                utc_day = open_time.date()
            else:
                try:
                    utc_day = pd.Timestamp(open_time).date()
                except Exception:
                    utc_day = self._current_utc_day
            self.reset_daily_budget(utc_day)

        # ============================
        # STAGE 1: HARD BLOCKS
        # ============================

        # Rule 1: Crash mode → block everything
        if model_outputs.regime_label == 'crash_mode':
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append('HARD_BLOCK: crash_mode regime detected')
            decision.regime_action = 'block'
            return decision

        # Rule 2: No active branch (block or unknown)
        if model_outputs.active_branch == 'none':
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append('HARD_BLOCK: no active model branch')
            return decision

        # Rule 3: Emotional risk → block
        emotional_score = features.get('emotional_risk_score', 0.0)
        if emotional_score > self.EMOTIONAL_BLOCK_THRESHOLD:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: emotional_risk_score={emotional_score:.2f} > {self.EMOTIONAL_BLOCK_THRESHOLD}'
            )
            return decision

        # Rule 4: Behavioral anomaly from Isolation Forest
        if model_outputs.behavioral_anomaly:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append('HARD_BLOCK: behavioral anomaly detected by IsolationForest')
            return decision

        # Rule 5: Liquidity collapse
        illiquidity = features.get('amihud_illiquidity', 0.0)
        if illiquidity > self.ILLIQUIDITY_BLOCK_THRESHOLD:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: amihud_illiquidity={illiquidity:.4f} > {self.ILLIQUIDITY_BLOCK_THRESHOLD}'
            )
            return decision

        # Rule 6: Strategy completely broken
        health = features.get('strategy_health_score', 1.0)
        if health < self.STRATEGY_DISABLE_THRESHOLD:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: strategy_health={health:.2f} < {self.STRATEGY_DISABLE_THRESHOLD}'
            )
            return decision

        # Rule 7: Excessive loss streak
        consec_losses = features.get('consecutive_losses', 0)
        if consec_losses >= self.MAX_CONSECUTIVE_LOSSES:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: consecutive_losses={consec_losses} >= {self.MAX_CONSECUTIVE_LOSSES}'
            )
            return decision

        # ============================
        # STAGE 2: DAILY TRADE BUDGET
        # ============================

        regime = model_outputs.regime_label
        regime_cfg = REGIME_RISK_CONFIG.get(regime, {})
        regime_max_trades = regime_cfg.get('max_daily_trades', self.GLOBAL_MAX_DAILY_TRADES)
        effective_max = min(regime_max_trades, self.GLOBAL_MAX_DAILY_TRADES)

        if self._daily_trade_count >= effective_max:
            decision.allow_trade = False
            decision.block_reasons.append(
                f'BUDGET: daily_trades={self._daily_trade_count} >= max={effective_max}'
            )
            return decision

        # ============================
        # STAGE 3: INTER-TRADE SPACING
        # ============================

        if self._bars_since_last_trade < self.MIN_INTER_TRADE_BARS:
            decision.allow_trade = False
            decision.block_reasons.append(
                f'SPACING: bars_since_last={self._bars_since_last_trade} < min={self.MIN_INTER_TRADE_BARS}'
            )
            return decision

        # ============================
        # STAGE 4: REGIME ROUTING
        # ============================

        if regime == 'trending_low_vol':
            decision.regime_action = 'normal'
            decision.sl_multiplier = self.exec_params.get('sl_mult_trending_low', 2.5)
            decision.tp_multiplier = self.exec_params.get('tp_mult_trending_low', 4.0)
        elif regime == 'trending_high_vol':
            decision.regime_action = 'reduce'
            decision.sl_multiplier = self.exec_params.get('sl_mult_trending_high', 3.0)
            decision.tp_multiplier = self.exec_params.get('tp_mult_trending_high', 3.0)
            decision.risk_percent *= 0.5
            decision.warnings.append('REGIME: trending_high_vol -> 50% sizing')
        elif regime == 'sideways_low_vol':
            decision.regime_action = 'normal'
            decision.sl_multiplier = self.exec_params.get('sl_mult_sideways', 2.0)
            decision.tp_multiplier = self.exec_params.get('tp_mult_sideways', 2.0)
            decision.risk_percent *= 0.6
            decision.warnings.append('REGIME: sideways_low_vol -> 60% sizing (mean-reversion)')
        elif regime == 'choppy_high_vol':
            decision.regime_action = 'reduce'
            decision.sl_multiplier = self.exec_params.get('sl_mult_choppy', 4.0)
            decision.tp_multiplier = self.exec_params.get('tp_mult_choppy', 2.0)
            decision.risk_percent *= 0.2
            decision.warnings.append('REGIME: choppy_high_vol -> 20% sizing (strict filter)')

        # ============================
        # STAGE 5: SOFT ADJUSTMENTS
        # ============================

        # Risk model: NO_TRADE as soft reduction (not hard block)
        if model_outputs.risk_level == 'NO_TRADE':
            decision.risk_percent *= 0.10
            decision.warnings.append('SOFT: risk_model=NO_TRADE -> 90% risk reduction')

        # Trend alignment penalty (for trend branch only)
        if decision.active_branch == 'trend':
            ema_50_slope = features.get('ema_50_slope', 0.0)
            direction = model_outputs.predicted_direction
            if direction == -1 and ema_50_slope > 0:
                decision.risk_percent *= 0.5
                decision.warnings.append('TREND: SHORT against bullish ema_50 -> 50% reduction')
            elif direction == 1 and ema_50_slope < 0:
                decision.risk_percent *= 0.5
                decision.warnings.append('TREND: LONG against bearish ema_50 -> 50% reduction')

        # GMM sub-regime exhaustion penalty
        if model_outputs.gmm_subregime == 'trend_exhaustion':
            decision.risk_percent *= 0.4
            decision.warnings.append('GMM: trend_exhaustion -> 60% risk reduction')
        elif model_outputs.gmm_subregime == 'false_signal':
            decision.risk_percent *= 0.3
            decision.warnings.append('GMM: false_signal -> 70% risk reduction')

        # Consecutive losses: progressive reduction
        if consec_losses >= 3:
            reduction = 0.5
            decision.risk_percent *= reduction
            decision.warnings.append(f'SOFT: {consec_losses} consecutive losses -> 50% risk reduction')

        # Recent drawdown: protect capital
        dd = features.get('recent_drawdown', 0.0)
        if dd > 0.03:
            dd_factor = max(0.3, 1.0 - dd * 5.0)
            decision.risk_percent *= dd_factor
            decision.warnings.append(f'SOFT: drawdown {dd:.2%} -> {dd_factor:.0%} risk factor')

        # High volatility: widen stops, reduce size
        vol_pct = features.get('volatility_percentile', 0.5)
        if vol_pct > 0.75:
            vol_factor = max(0.5, 1.0 - (vol_pct - 0.75))
            decision.risk_percent *= vol_factor
            decision.sl_multiplier *= 1.2
            decision.warnings.append(f'SOFT: vol_percentile {vol_pct:.2f} -> widened SL, reduced size')

        # Risk model downgrades
        if model_outputs.risk_level == 'HIGH_RISK':
            decision.risk_percent *= 0.5
            decision.warnings.append('SOFT: risk_model=HIGH_RISK -> 50% risk reduction')
        elif model_outputs.risk_level == 'MEDIUM_RISK':
            decision.risk_percent *= 0.75
            decision.warnings.append('SOFT: risk_model=MEDIUM_RISK -> 25% risk reduction')

        # Clamp risk percent to [0.25%, 1.5%]
        # Floor raised from 0.1% to 0.25% to prevent multiplicative crushing
        # from stacking 5+ soft reductions to near-zero position sizes
        decision.risk_percent = max(0.25, min(1.5, decision.risk_percent))

        return decision
