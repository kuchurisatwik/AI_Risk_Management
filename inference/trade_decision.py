"""
Trade Decision Orchestrator — Daily Opportunity Ranker.

Replaces static probability thresholds with cross-sectional 
daily ranking. Selects the top N best opportunities per UTC day,
gated by regime probability floors.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from inference.model_ensemble import ModelEnsemble, ModelOutputs
from inference.policy_engine import PolicyEngine, PolicyDecision
from inference.threshold_engine import compute_adaptive_threshold
from inference.risk_sizer import compute_adaptive_sizing, SizingResult

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Complete trade decision output."""
    action: str = 'NO_TRADE'   # LONG, SHORT, NO_TRADE
    
    # Confidence
    meta_probability: float = 0.0
    meta_margin: float = 0.0
    daily_rank: int = 0
    
    # Risk parameters
    risk_percent: float = 0.0
    sl_distance: float = 0.0
    tp_distance: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    reward_risk_ratio: float = 0.0
    position_size_usd: float = 0.0
    
    # Context
    regime: str = 'unknown'
    regime_confidence: float = 0.0
    active_branch: str = 'none'
    
    # Governance
    block_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    policy_allowed: bool = False


class TradeDecisionEngine:
    """
    Institutional Daily Opportunity Ranker Orchestrator.
    """
    
    def __init__(self, models_dir: str, exec_params: dict = None):
        self.exec_params = exec_params or {}
        self.ensemble = ModelEnsemble(models_dir)
        self.policy = PolicyEngine(exec_params=self.exec_params)
        self._loaded = False
        
        self._current_utc_day = None
        self._daily_signals = []  # List of margin scores seen today
    
    def load(self):
        """Load all models."""
        self.ensemble.load()
        self._loaded = True

    def _update_daily_tracker(self, utc_day, margin):
        """Track margins seen today to determine relative rank."""
        if utc_day != self._current_utc_day:
            self._current_utc_day = utc_day
            self._daily_signals = []
            
        self._daily_signals.append(margin)
        
        # Calculate rank (1 = highest margin seen today)
        # Sort descending
        sorted_margins = sorted(self._daily_signals, reverse=True)
        # Rank is the index + 1
        rank = sorted_margins.index(margin) + 1
        return rank

    def decide(self, features: dict, equity: float = 10000.0, precomputed_outputs: ModelOutputs = None) -> TradeDecision:
        decision = TradeDecision()
        self.policy.tick() # Advance bar counter
        
        # 1. Model Inference
        if precomputed_outputs is not None:
            outputs = precomputed_outputs
        else:
            outputs = self.ensemble.predict(features)
            
        decision.meta_probability = outputs.meta_probability
        decision.meta_margin = outputs.meta_margin
        decision.regime = outputs.regime_label
        decision.regime_confidence = outputs.regime_confidence
        decision.active_branch = outputs.active_branch
        
        # 2. Track current UTC day for ranking
        open_time = features.get('open_time', None)
        if open_time is not None:
            if isinstance(open_time, pd.Timestamp):
                utc_day = open_time.date()
            else:
                try:
                    utc_day = pd.Timestamp(open_time).date()
                except Exception:
                    utc_day = self._current_utc_day
        else:
            utc_day = self._current_utc_day
            
        decision.daily_rank = self._update_daily_tracker(utc_day, outputs.meta_margin)
        
        # 3. Policy Engine Gating (Hard blocks, budgets, spacing, soft modifiers)
        policy = self.policy.evaluate(outputs, features)
        decision.policy_allowed = policy.allow_trade
        decision.block_reasons = policy.block_reasons
        decision.warnings = policy.warnings
        
        if not policy.allow_trade:
            decision.action = 'NO_TRADE'
            return decision

        # 4. Probability Floor Check
        vol_pct = features.get('volatility_percentile', 0.5)
        recent_dd = features.get('recent_drawdown', 0.0)
        health = features.get('strategy_health_score', 1.0)
        
        thresh_state = compute_adaptive_threshold(
            regime_label=outputs.regime_label,
            branch=outputs.active_branch,
            volatility_percentile=vol_pct,
            recent_drawdown=recent_dd,
            strategy_health_score=health,
            regime_confidence=outputs.regime_confidence
        )
        
        # Override baseline threshold with optimized floor if available
        floor_map = {
            'trending_low_vol': self.exec_params.get('prob_floor_trending_low', None),
            'trending_high_vol': self.exec_params.get('prob_floor_trending_high', None),
            'sideways_low_vol': self.exec_params.get('prob_floor_sideways', None),
            'choppy_high_vol': self.exec_params.get('prob_floor_choppy', None),
        }
        custom_floor = floor_map.get(outputs.regime_label)
        if custom_floor is not None:
            thresh_state.adjusted_threshold = max(thresh_state.adjusted_threshold, custom_floor)
        
        if decision.meta_probability < thresh_state.adjusted_threshold:
            decision.action = 'NO_TRADE'
            decision.block_reasons.append(
                f'FLOOR: prob={decision.meta_probability:.3f} < floor={thresh_state.adjusted_threshold:.3f}'
            )
            return decision
            
        # 5. Daily Opportunity Ranker Check
        # Default N=3 trades per day. If a signal isn't in the top 3 seen *so far* today,
        # we reject it, expecting a better one might come (or we already traded the best).
        # We also look at the policy max_daily_trades.
        from training.config import REGIME_RISK_CONFIG
        regime_cfg = REGIME_RISK_CONFIG.get(outputs.regime_label, {})
        max_trades = regime_cfg.get('max_daily_trades', 3)
        
        if decision.daily_rank > max_trades:
            decision.action = 'NO_TRADE'
            decision.block_reasons.append(
                f'RANK: daily_rank={decision.daily_rank} > max_allowed={max_trades}'
            )
            return decision
        
        # 6. Action Direction
        direction = outputs.predicted_direction
        decision.action = 'LONG' if direction == 1 else 'SHORT'
        
        # 7. Adaptive Sizing
        atr = features.get('atr_14', 0.0)
        entry_price = features.get('close', 0.0)
        pred_vol = outputs.predicted_volatility
        realized_vol = features.get('realized_volatility', 0.0)
        vwap = features.get('vwap', 0.0)
        
        if atr > 0 and entry_price > 0:
            sizing = compute_adaptive_sizing(
                equity=equity,
                entry_price=entry_price,
                direction=direction,
                regime=outputs.regime_label,
                active_branch=outputs.active_branch,
                predicted_volatility=pred_vol,
                realized_volatility=realized_vol,
                atr_14=atr,
                vwap_price=vwap,
                sl_multiplier=policy.sl_multiplier,
                tp_multiplier=policy.tp_multiplier,
                policy_risk_modifier=policy.risk_percent
            )
            
            decision.risk_percent = sizing.risk_percent
            decision.sl_distance = sizing.sl_distance
            decision.tp_distance = sizing.tp_distance
            decision.sl_price = sizing.sl_price
            decision.tp_price = sizing.tp_price
            decision.reward_risk_ratio = sizing.reward_risk_ratio
            decision.position_size_usd = sizing.position_size_usd
            
            if sizing.risk_percent <= 0:
                decision.action = 'NO_TRADE'
                decision.block_reasons.append(f'SIZING: risk_percent={sizing.risk_percent:.2f}%')
            else:
                # Successfully passing all gates -> record trade in policy
                self.policy.record_trade()
        
        return decision
