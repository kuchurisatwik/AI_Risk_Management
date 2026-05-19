"""
Risk Labeler — Rule-based risk classification.

Generates 4-class labels: LOW_RISK, MEDIUM_RISK, HIGH_RISK, NO_TRADE.
Uses only current-row features (no future data).
Requires regime_label to be pre-assigned.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_risk_single(row):
    """
    Rule-based risk scoring for a single row.
    Adapted from roadmap Section 6.3 with our proxy column names.
    """
    score = 0
    
    # Volatility conditions
    if row.get('volatility_percentile', 0) > 0.8:
        score += 2
    if row.get('atr_expansion_ratio', 0) > 1.8:
        score += 1
    
    # Liquidity conditions (using our proxy for spread_ratio)
    if row.get('amihud_illiquidity', 0) > 0.1:
        score += 2
    
    # Strategy health conditions
    if row.get('consecutive_losses', 0) >= 3:
        score += 2
    if row.get('recent_drawdown', 0) > 0.05:
        score += 2
    if row.get('last_5_trade_winrate', 1.0) < 0.3:
        score += 1
    if row.get('strategy_health_score', 1.0) < 0.4:
        score += 2
    
    # Behavioral conditions
    if row.get('revenge_trade_score', 0) > 0.6:
        score += 3
    if row.get('overtrading_score', 0) > 0.7:
        score += 2
    if row.get('oversized_trade_score', 0) > 0.5:
        score += 2
    
    # Regime conditions
    regime = row.get('regime_label', '')
    if regime == 'choppy_high_vol':
        score += 3
    if regime == 'crash_mode':
        score += 10
    
    if score == 0:
        return 'LOW_RISK'
    elif score <= 3:
        return 'MEDIUM_RISK'
    elif score <= 7:
        return 'HIGH_RISK'
    else:
        return 'NO_TRADE'


def label_risk(df):
    """
    Generate risk labels for the entire DataFrame.
    Requires 'regime_label' column to already exist.
    
    Returns:
        numpy array of risk label strings
    """
    labels = df.apply(label_risk_single, axis=1).values
    
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    logger.info(f"Risk label distribution: {dist}")
    
    return labels
