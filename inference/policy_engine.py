"""
Policy Engine — The Final Authority Layer.

Models output probabilities. The Policy Engine makes decisions.
It applies hard safety constraints, soft scoring adjustments,
and regime-aware risk modifications.

The Policy Engine ALWAYS has authority over model outputs.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from inference.model_ensemble import ModelOutputs

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


class PolicyEngine:
    """
    Institutional-grade policy engine.
    
    Applies in order:
    1. Hard blocks (instant rejection, no override)
    2. Regime routing (regime-specific behavior)
    3. Soft adjustments (risk reduction based on conditions)
    4. Final authorization
    """
    
    # Hard block thresholds
    EMOTIONAL_BLOCK_THRESHOLD = 0.80
    STRATEGY_DISABLE_THRESHOLD = 0.30
    ILLIQUIDITY_BLOCK_THRESHOLD = 0.15
    MAX_CONSECUTIVE_LOSSES = 5
    
    def evaluate(self, model_outputs: ModelOutputs, features: dict) -> PolicyDecision:
        """
        Run the full policy evaluation.
        
        Args:
            model_outputs: Predictions from ModelEnsemble
            features: Raw feature dict for additional checks
        
        Returns:
            PolicyDecision with final trade authorization
        """
        decision = PolicyDecision(allow_trade=True, risk_percent=1.0)
        
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
        
        # Rule 2: Emotional risk → block
        emotional_score = features.get('emotional_risk_score', 0.0)
        if emotional_score > self.EMOTIONAL_BLOCK_THRESHOLD:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: emotional_risk_score={emotional_score:.2f} > {self.EMOTIONAL_BLOCK_THRESHOLD}'
            )
            return decision
        
        # Rule 3: Behavioral anomaly from Isolation Forest
        if model_outputs.behavioral_anomaly:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append('HARD_BLOCK: behavioral anomaly detected by IsolationForest')
            return decision
        
        # Rule 4: Liquidity collapse
        illiquidity = features.get('amihud_illiquidity', 0.0)
        if illiquidity > self.ILLIQUIDITY_BLOCK_THRESHOLD:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: amihud_illiquidity={illiquidity:.4f} > {self.ILLIQUIDITY_BLOCK_THRESHOLD}'
            )
            return decision
        
        # Rule 5: Strategy completely broken
        health = features.get('strategy_health_score', 1.0)
        if health < self.STRATEGY_DISABLE_THRESHOLD:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: strategy_health={health:.2f} < {self.STRATEGY_DISABLE_THRESHOLD}'
            )
            return decision
        
        # Rule 6: Excessive loss streak
        consec_losses = features.get('consecutive_losses', 0)
        if consec_losses >= self.MAX_CONSECUTIVE_LOSSES:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: consecutive_losses={consec_losses} >= {self.MAX_CONSECUTIVE_LOSSES}'
            )
            return decision
        
        # Rule 7: Risk model says NO_TRADE — treat as strong soft reduction
        # (The hard blocks above already catch the truly dangerous cases.
        # NO_TRADE from the ML model is a secondary signal, not an absolute veto.)
        if model_outputs.risk_level == 'NO_TRADE':
            decision.risk_percent *= 0.10
            decision.warnings.append('SOFT: risk_model=NO_TRADE -> 90% risk reduction')
        
        # ============================
        # STAGE 2: REGIME ROUTING
        # ============================
        
        regime = model_outputs.regime_label
        
        if regime == 'trending_low_vol':
            decision.regime_action = 'normal'
            decision.sl_multiplier = 1.5
            decision.tp_multiplier = 1.5  # 1:1 RR
        
        elif regime == 'trending_high_vol':
            decision.regime_action = 'reduce'
            decision.sl_multiplier = 2.0
            decision.tp_multiplier = 2.0  # 1:1 RR
            decision.risk_percent *= 0.7
            decision.warnings.append('REGIME: trending_high_vol - reduced sizing')
        
        elif regime == 'sideways_low_vol':
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.regime_action = 'block'
            decision.block_reasons.append('HARD_BLOCK: sideways_low_vol regime - insufficient edge')
            return decision
        
        elif regime == 'choppy_high_vol':
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.regime_action = 'block'
            decision.block_reasons.append('HARD_BLOCK: choppy_high_vol regime - historically destructive')
            return decision
        
        # ============================
        # STAGE 3: TREND ALIGNMENT
        # ============================
        
        # Penalize counter-trend trades (Bug 5 fix)
        ema_50_slope = features.get('ema_50_slope', 0.0)
        direction = model_outputs.predicted_direction
        if direction == -1 and ema_50_slope > 0:
            # Shorting in a bullish trend
            decision.risk_percent *= 0.5
            decision.warnings.append('TREND: SHORT against bullish ema_50_slope -> 50% risk reduction')
        elif direction == 1 and ema_50_slope < 0:
            # Going long in a bearish trend
            decision.risk_percent *= 0.5
            decision.warnings.append('TREND: LONG against bearish ema_50_slope -> 50% risk reduction')
        
        # ============================
        # STAGE 4: SOFT ADJUSTMENTS
        # ============================
        
        # Consecutive losses: progressive reduction
        if consec_losses >= 3:
            reduction = 0.5
            decision.risk_percent *= reduction
            decision.warnings.append(f'SOFT: {consec_losses} consecutive losses -> {reduction:.0%} risk reduction')
        
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
        
        # Oversized trade detection
        oversized = features.get('oversized_trade_score', 0.0)
        if oversized > 0.5:
            decision.risk_percent = min(decision.risk_percent, 0.5)
            decision.warnings.append(f'SOFT: oversized_trade_score={oversized:.2f} -> capped risk')
        
        # Clamp risk percent to [0.1%, 1.5%]
        decision.risk_percent = max(0.1, min(1.5, decision.risk_percent))
        
        return decision
