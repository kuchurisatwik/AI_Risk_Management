"""
Risk Sizing Engine — Regime-Adaptive Volatility Targeting.

Replaces fixed risk percent with dynamic sizing based on volatility targeting,
modulated by regime-specific parameters from REGIME_RISK_CONFIG.
"""

import logging
from dataclasses import dataclass

from training.config import REGIME_RISK_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    risk_percent: float
    sl_distance: float
    tp_distance: float
    sl_price: float
    tp_price: float
    reward_risk_ratio: float
    position_size_usd: float


def compute_adaptive_sizing(
    equity: float,
    entry_price: float,
    direction: int,
    regime: str,
    active_branch: str,
    predicted_volatility: float,
    realized_volatility: float,
    atr_14: float,
    vwap_price: float,
    sl_multiplier: float = 1.5,
    tp_multiplier: float = 1.5,
    policy_risk_modifier: float = 1.0,
) -> SizingResult:
    """
    Computes position size using Regime-Adaptive Volatility Targeting.
    """
    # 1. Get Regime Configuration
    cfg = REGIME_RISK_CONFIG.get(regime, REGIME_RISK_CONFIG.get('unknown', {
        'target_vol': 0.05, 'max_risk_pct': 1.0, 'kelly_frac': 0.3
    }))
    
    target_vol = cfg.get('target_vol', 0.05)
    max_risk_pct = cfg.get('max_risk_pct', 1.0)
    kelly_frac = cfg.get('kelly_frac', 0.3)

    # 2. Distances and Prices based on Branch
    if active_branch == 'meanrev' and vwap_price > 0:
        # MeanRev Branch: Target is VWAP
        tp_price = vwap_price
        tp_dist = abs(entry_price - vwap_price)
        
        # SL is 1.5x the VWAP distance from entry, adjusted by multiplier
        sl_dist = tp_dist * sl_multiplier
        if direction == 1:
            sl_price = entry_price - sl_dist
        else:
            sl_price = entry_price + sl_dist
    else:
        # Trend Branch (or fallback): ATR-based
        sl_dist = atr_14 * sl_multiplier
        tp_dist = atr_14 * tp_multiplier
        
        if direction == 1:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

    # Odds = Reward / Risk
    b = tp_dist / sl_dist if sl_dist > 0 else 1.0

    # 3. Volatility Targeting
    # raw_leverage = target_vol / realized_volatility
    eff_vol = max(realized_volatility, predicted_volatility, 0.01) # Use max to be conservative
    raw_leverage = target_vol / eff_vol

    # We convert leverage to a risk % based on stop loss distance.
    # If we use raw_leverage, our position size = equity * raw_leverage
    # Risk % = Position Size * (SL Distance / Entry Price) / Equity
    # Risk % = raw_leverage * (SL Distance / Entry Price)
    
    sl_pct = sl_dist / entry_price if entry_price > 0 else 0.01
    vol_target_risk_pct = raw_leverage * sl_pct * 100.0

    # 4. Final Risk Percent
    # Apply Kelly fraction cap and policy modifier
    raw_risk_pct = vol_target_risk_pct * kelly_frac * policy_risk_modifier
    
    # Hard cap risk
    risk_percent = min(raw_risk_pct, max_risk_pct)
    
    # 5. Position Size USD
    risk_amount = equity * (risk_percent / 100.0)
    
    if sl_pct > 0:
        position_size_usd = risk_amount / sl_pct
    else:
        position_size_usd = 0.0
        
    return SizingResult(
        risk_percent=risk_percent,
        sl_distance=sl_dist,
        tp_distance=tp_dist,
        sl_price=sl_price,
        tp_price=tp_price,
        reward_risk_ratio=b,
        position_size_usd=position_size_usd
    )
