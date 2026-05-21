"""
Paper Trader — Event-driven simulation loop.

Processes historical candles one-by-one through the full
inference pipeline, simulating realistic execution with
slippage, fees, and order tracking.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from inference.trade_decision import TradeDecisionEngine, TradeDecision

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single simulated trade with full diagnostic audit trail."""
    entry_time: pd.Timestamp
    entry_price: float
    direction: int            # +1 LONG, -1 SHORT
    sl_price: float
    tp_price: float
    risk_percent: float
    position_size_usd: float
    regime: str
    confidence: float
    active_branch: str = 'unknown'
    gmm_subregime: str = 'unknown'
    # Filled on exit
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ''
    # ---- Diagnostic fields (institutional audit trail) ----
    meta_probability: float = 0.0
    meta_margin: float = 0.0
    regime_confidence: float = 0.0
    risk_level: str = 'unknown'
    behavioral_anomaly: bool = False
    sl_distance_pct: float = 0.0
    tp_distance_pct: float = 0.0
    designed_rr: float = 0.0
    predicted_volatility: float = 0.0
    realized_volatility: float = 0.0
    atr_14: float = 0.0
    bars_held: int = 0
    max_adverse_excursion: float = 0.0   # MAE: worst unrealized loss
    max_favorable_excursion: float = 0.0 # MFE: best unrealized profit
    policy_warnings: str = ''            # comma-joined policy warnings


@dataclass
class PaperTradingResult:
    """Full paper trading simulation results."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_rr_realized: float = 0.0
    trade_frequency_pct: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    regime_performance: dict = field(default_factory=dict)
    block_reasons_summary: dict = field(default_factory=dict)


class PaperTrader:
    """
    Event-driven paper trading simulator.
    
    Processes each candle through the full inference pipeline:
    features -> models -> policy -> threshold -> sizing -> execution
    """
    
    SLIPPAGE_PCT = 0.0001   # 0.01% slippage per trade
    FEE_PCT = 0.0004        # 0.04% maker+taker fee
    
    def __init__(self, models_dir: str, initial_equity: float = 10000.0, exec_params: dict = None):
        self.exec_params = exec_params or {}
        self.engine = TradeDecisionEngine(models_dir, exec_params=self.exec_params)
        self.initial_equity = initial_equity
    
    def load(self):
        """Load inference models."""
        self.engine.load()
    
    def run(self, df: pd.DataFrame, precomputed_outputs: list = None) -> PaperTradingResult:
        """
        Run the full paper trading simulation.
        
        Args:
            df: DataFrame with all features (test set)
            precomputed_outputs: List of ModelOutputs corresponding to each row
        
        Returns:
            PaperTradingResult with comprehensive metrics
        """
        result = PaperTradingResult()
        equity = self.initial_equity
        peak_equity = equity
        open_trade: Optional[Trade] = None
        all_returns = []
        block_counter = {}
        bars_held = 0  # Bug 7 fix: initialize before use
        
        columns = list(df.columns)
        
        for i in range(len(df)):
            row = df.iloc[i]
            features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in columns}
            current_time = row.get('open_time', pd.Timestamp.now())
            current_close = row['close']
            
            # ---- Check open trade for SL/TP hit ----
            if open_trade is not None:
                hit = False
                if open_trade.direction == 1:  # LONG
                    if row['low'] <= open_trade.sl_price:
                        open_trade.exit_price = open_trade.sl_price
                        open_trade.exit_reason = 'SL_HIT'
                        hit = True
                    elif row['high'] >= open_trade.tp_price:
                        open_trade.exit_price = open_trade.tp_price
                        open_trade.exit_reason = 'TP_HIT'
                        hit = True
                else:  # SHORT
                    if row['high'] >= open_trade.sl_price:
                        open_trade.exit_price = open_trade.sl_price
                        open_trade.exit_reason = 'SL_HIT'
                        hit = True
                    elif row['low'] <= open_trade.tp_price:
                        open_trade.exit_price = open_trade.tp_price
                        open_trade.exit_reason = 'TP_HIT'
                        hit = True
                
                if hit:
                    # Apply slippage and fees (Bug 3 fix: charge fees on BOTH legs)
                    entry_fee = open_trade.entry_price * self.FEE_PCT
                    exit_fee = open_trade.exit_price * self.FEE_PCT
                    slippage = open_trade.exit_price * self.SLIPPAGE_PCT
                    
                    raw_pnl = open_trade.direction * (open_trade.exit_price - open_trade.entry_price)
                    pnl_per_unit = raw_pnl - slippage - entry_fee - exit_fee
                    
                    # Scale to position size
                    units = open_trade.position_size_usd / open_trade.entry_price if open_trade.entry_price > 0 else 0
                    open_trade.pnl = pnl_per_unit * units
                    open_trade.exit_time = current_time
                    open_trade.bars_held = bars_held
                    
                    equity += open_trade.pnl
                    all_returns.append(open_trade.pnl / peak_equity if peak_equity > 0 else 0)
                    
                    result.trades.append(open_trade)
                    open_trade = None
                    bars_held = 0
                else:
                    bars_held += 1
                    # Time barrier
                    time_barrier = self.exec_params.get('time_barrier_bars', 36)
                    if bars_held >= time_barrier:
                        open_trade.exit_price = current_close
                        open_trade.exit_reason = 'TIME_BARRIER'
                        
                        entry_fee = open_trade.entry_price * self.FEE_PCT
                        exit_fee = open_trade.exit_price * self.FEE_PCT
                        slippage = open_trade.exit_price * self.SLIPPAGE_PCT
                        
                        raw_pnl = open_trade.direction * (open_trade.exit_price - open_trade.entry_price)
                        pnl_per_unit = raw_pnl - slippage - entry_fee - exit_fee
                        
                        units = open_trade.position_size_usd / open_trade.entry_price if open_trade.entry_price > 0 else 0
                        open_trade.pnl = pnl_per_unit * units
                        open_trade.exit_time = current_time
                        open_trade.bars_held = bars_held
                        
                        equity += open_trade.pnl
                        all_returns.append(open_trade.pnl / peak_equity if peak_equity > 0 else 0)
                        
                        result.trades.append(open_trade)
                        open_trade = None
                        bars_held = 0
            
            # ---- Track MAE/MFE for open trade ----
            if open_trade is not None and open_trade.exit_reason == '':
                unrealized = open_trade.direction * (current_close - open_trade.entry_price)
                if unrealized < open_trade.max_adverse_excursion:
                    open_trade.max_adverse_excursion = unrealized
                if unrealized > open_trade.max_favorable_excursion:
                    open_trade.max_favorable_excursion = unrealized
            
            # ---- Make new decision if no open trade ----
            if open_trade is None:
                precomp = precomputed_outputs[i] if precomputed_outputs else None
                if not precomp:
                    precomp = self.engine.ensemble.predict(features)
                    
                decision = self.engine.decide(features, equity, precomputed_outputs=precomp)
                
                if decision.action in ('LONG', 'SHORT'):
                    direction = 1 if decision.action == 'LONG' else -1
                    entry_with_slip = current_close + direction * current_close * self.SLIPPAGE_PCT
                    
                    # Capture full inference state for diagnostics
                    sl_dist_pct = abs(decision.sl_price - entry_with_slip) / entry_with_slip * 100 if entry_with_slip > 0 else 0
                    tp_dist_pct = abs(decision.tp_price - entry_with_slip) / entry_with_slip * 100 if entry_with_slip > 0 else 0
                    
                    open_trade = Trade(
                        entry_time=current_time,
                        entry_price=entry_with_slip,
                        direction=direction,
                        sl_price=decision.sl_price,
                        tp_price=decision.tp_price,
                        risk_percent=decision.risk_percent,
                        position_size_usd=decision.position_size_usd,
                        regime=decision.regime,
                        confidence=decision.meta_probability,
                        active_branch=decision.active_branch,
                        gmm_subregime=precomp.gmm_subregime,
                        # Diagnostic fields
                        meta_probability=decision.meta_probability,
                        meta_margin=decision.meta_margin,
                        regime_confidence=decision.regime_confidence,
                        risk_level=getattr(precomp, 'risk_level', 'unknown'),
                        sl_distance_pct=sl_dist_pct,
                        tp_distance_pct=tp_dist_pct,
                        designed_rr=decision.reward_risk_ratio,
                        predicted_volatility=features.get('predicted_volatility', 0.0),
                        realized_volatility=features.get('realized_volatility', 0.0),
                        atr_14=features.get('atr_14', 0.0),
                        policy_warnings=','.join(decision.warnings) if decision.warnings else '',
                    )
                    bars_held = 0
                else:
                    all_returns.append(0.0)
                    # Track block reasons
                    for reason in decision.block_reasons:
                        tag = reason.split(':')[0].strip()
                        block_counter[tag] = block_counter.get(tag, 0) + 1
            else:
                all_returns.append(0.0)
            
            # Track equity
            if equity > peak_equity:
                peak_equity = equity
            result.equity_curve.append(equity)
        
        # ---- Close any remaining open trade at last close ----
        if open_trade is not None:
            open_trade.exit_price = df.iloc[-1]['close']
            open_trade.exit_reason = 'END_OF_DATA'
            raw_pnl = open_trade.direction * (open_trade.exit_price - open_trade.entry_price)
            units = open_trade.position_size_usd / open_trade.entry_price if open_trade.entry_price > 0 else 0
            open_trade.pnl = raw_pnl * units
            open_trade.exit_time = df.iloc[-1].get('open_time', pd.Timestamp.now())
            equity += open_trade.pnl
            result.trades.append(open_trade)
            result.equity_curve.append(equity)
        
        # ---- Compute Final Metrics ----
        result.total_trades = len(result.trades)
        result.total_return = (equity - self.initial_equity) / self.initial_equity
        result.trade_frequency_pct = result.total_trades / len(df) * 100 if len(df) > 0 else 0
        
        if result.total_trades > 0:
            wins = sum(1 for t in result.trades if t.pnl > 0)
            result.win_rate = wins / result.total_trades
            avg_win = np.mean([t.pnl for t in result.trades if t.pnl > 0]) if wins > 0 else 0
            losses = result.total_trades - wins
            avg_loss = abs(np.mean([t.pnl for t in result.trades if t.pnl <= 0])) if losses > 0 else 1
            result.avg_rr_realized = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Sharpe & Sortino (annualized for 5m bars: 105,120 bars/year)
        returns = np.array(all_returns)
        if len(returns) > 1 and returns.std() > 0:
            result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(105120)
            downside = returns[returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                result.sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(105120)
        
        # Max Drawdown
        eq = np.array(result.equity_curve)
        if len(eq) > 0:
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak
            result.max_drawdown = float(dd.min())
        
        # Regime performance
        for t in result.trades:
            if t.regime not in result.regime_performance:
                result.regime_performance[t.regime] = {'trades': 0, 'pnl': 0.0, 'wins': 0}
            result.regime_performance[t.regime]['trades'] += 1
            result.regime_performance[t.regime]['pnl'] += t.pnl
            if t.pnl > 0:
                result.regime_performance[t.regime]['wins'] += 1
        
        result.block_reasons_summary = block_counter
        
        return result
