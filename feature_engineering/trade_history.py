"""
Synthetic Trade Simulator & Behavioral Features.
Simulates a realistic trading strategy over historical data to generate
the required trade history and behavioral features (Phase 2).
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SyntheticTradeGenerator:
    """
    Simulates an EMA crossover strategy with ATR-based TP/SL to generate
    a realistic trade ledger. Then computes behavioral/strategy features
    at each candle based strictly on past trades (causal).
    """
    
    def __init__(self, atr_multiplier_tp=2.0, atr_multiplier_sl=1.0):
        self.atr_tp = atr_multiplier_tp
        self.atr_sl = atr_multiplier_sl
        
    def _compute_trade_features_for_candle(self, closed_trades: list, current_equity: float, peak_equity: float, current_idx: int) -> dict:
        """Computes the 7 required trade/behavioral features given the trade history."""
        n_trades = len(closed_trades)
        
        # Defaults if no trades
        if n_trades == 0:
            return {
                "strategy_recent_accuracy": 0.5,
                "strategy_avg_rr": 1.0,
                "last_5_trade_winrate": 0.5,
                "consecutive_losses": 0,
                "recent_drawdown": 0.0,
                "strategy_health_score": 0.5,
                "revenge_trade_score": 0.0
            }
            
        recent_20 = closed_trades[-20:]
        recent_5 = closed_trades[-5:]
        
        # Advanced Behavioral Defaults
        oversized = 0.0
        overtrading = 0.0
        time_since_loss = 100.0
        loss_recovery = 0.0
        fomo = 0.0
        panic = 0.0
        
        # Winrates
        wins_20 = sum(1 for t in recent_20 if t["pnl"] > 0)
        winrate_20 = wins_20 / len(recent_20)
        
        wins_5 = sum(1 for t in recent_5 if t["pnl"] > 0)
        winrate_5 = wins_5 / len(recent_5)
        
        # Avg RR
        avg_win = np.mean([t["pnl"] for t in recent_20 if t["pnl"] > 0]) if wins_20 > 0 else 0
        losses_20 = [t["pnl"] for t in recent_20 if t["pnl"] <= 0]
        avg_loss = abs(np.mean(losses_20)) if losses_20 else 1.0
        avg_rr = avg_win / avg_loss if avg_loss != 0 else avg_win
        
        # Consecutive losses and time since last loss
        cons_losses = 0
        last_loss_idx = None
        for t in reversed(closed_trades):
            if t["pnl"] < 0:
                cons_losses += 1
                if last_loss_idx is None:
                    last_loss_idx = t.get("exit_idx", 0)
            else:
                break
                
        if last_loss_idx is not None and current_idx > last_loss_idx:
            time_since_loss = current_idx - last_loss_idx
            
        # Overtrading (trades per rolling window vs baseline)
        if len(closed_trades) > 10:
            recent_trades_idx = [t["exit_idx"] for t in recent_20 if "exit_idx" in t]
            if len(recent_trades_idx) >= 2:
                avg_gap = (recent_trades_idx[-1] - recent_trades_idx[0]) / len(recent_trades_idx)
                # If avg gap is very small (lots of trades in short time), overtrading is high
                if avg_gap > 0:
                    overtrading = min(10.0 / avg_gap, 1.0)
                    
        # Oversized & Loss Recovery Aggression
        if len(closed_trades) > 5:
            avg_size = np.mean([t.get("position_size", 1.0) for t in recent_20])
            last_size = closed_trades[-1].get("position_size", 1.0)
            if avg_size > 0:
                oversized = min(last_size / avg_size, 2.0) - 1.0
                oversized = max(oversized, 0.0)
                
            if cons_losses > 0 and last_size > avg_size:
                loss_recovery = min((last_size / avg_size) - 1.0, 1.0)
                
        # FOMO / Panic (Simulated proxies)
        # If entry was on extremely high RSI (Long) or low RSI (Short)
        last_trade = closed_trades[-1]
        if last_trade.get("rsi_at_entry", 50) > 75 and last_trade["side"] == "LONG":
            fomo = 1.0
        elif last_trade.get("rsi_at_entry", 50) < 25 and last_trade["side"] == "SHORT":
            fomo = 1.0
                
        # Drawdown
        drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        
        # Strategy Health Score (Custom composite 0.0 to 1.0)
        # Higher is healthier. Weight winrate and inverse drawdown.
        health = (winrate_20 * 0.5) + (min(avg_rr, 3.0) / 3.0 * 0.3) + ((1.0 - min(drawdown, 0.5)*2) * 0.2)
        
        # Revenge Trade Score (0.0 to 1.0)
        revenge_score = min(cons_losses / 4.0, 1.0) * (1.0 if time_since_loss < 5 else 0.5)
        
        # Emotional Risk Score (Composite)
        emotional_risk = min((revenge_score * 0.4) + (oversized * 0.3) + (overtrading * 0.2) + (fomo * 0.1), 1.0)
        discipline = 1.0 - emotional_risk
        
        return {
            "strategy_recent_accuracy": winrate_20,
            "strategy_avg_rr": avg_rr,
            "last_5_trade_winrate": winrate_5,
            "consecutive_losses": cons_losses,
            "recent_drawdown": drawdown,
            "strategy_health_score": health,
            "revenge_trade_score": revenge_score,
            "oversized_trade_score": oversized,
            "overtrading_score": overtrading,
            "emotional_risk_score": emotional_risk,
            "discipline_score": discipline,
            "panic_exit_score": panic,
            "fomo_score": fomo,
            "time_since_last_loss": time_since_loss,
            "loss_recovery_aggression": loss_recovery
        }

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the simulation step-by-step to generate features.
        Takes a DataFrame containing open, high, low, close, ema_9, ema_20, atr_14.
        Returns the DataFrame with trade history features appended.
        """
        required = ["open", "high", "low", "close", "ema_9", "ema_20", "atr_14"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            # Note: For the MVP we expect ema_9 and ema_20 to exist. 
            # If they don't, we'll create temporary ones for the simulation.
            df = df.copy()
            if "ema_9" not in df.columns:
                df["ema_9"] = df["close"].ewm(span=9).mean()
            if "ema_20" not in df.columns:
                df["ema_20"] = df["close"].ewm(span=20).mean()
            if "atr_14" not in df.columns:
                df["atr_14"] = df["high"] - df["low"] # Rough fallback
                
        features_list = []
        
        open_trade = None
        closed_trades = []
        initial_equity = 10000.0
        equity = initial_equity
        peak_equity = initial_equity
        
        # For fast iteration, we use itertuples
        for row in df.itertuples():
            # Process open trade
            if open_trade is not None:
                is_closed = False
                pnl = 0
                
                # Check SL/TP hits
                if open_trade["side"] == "LONG":
                    if row.low <= open_trade["sl"]:
                        pnl = open_trade["sl"] - open_trade["entry_price"]
                        is_closed = True
                    elif row.high >= open_trade["tp"]:
                        pnl = open_trade["tp"] - open_trade["entry_price"]
                        is_closed = True
                else: # SHORT
                    if row.high >= open_trade["sl"]:
                        pnl = open_trade["entry_price"] - open_trade["sl"]
                        is_closed = True
                    elif row.low <= open_trade["tp"]:
                        pnl = open_trade["entry_price"] - open_trade["tp"]
                        is_closed = True
                        
                if is_closed:
                    # Normalize PnL to a fixed 1% risk per trade
                    risk_amt = equity * 0.01
                    price_risk = abs(open_trade["entry_price"] - open_trade["sl"])
                    position_size = risk_amt / price_risk if price_risk > 0 else 0
                    
                    actual_pnl = pnl * position_size
                    equity += actual_pnl
                    if equity > peak_equity:
                        peak_equity = equity
                        
                    closed_trades.append({
                        "side": open_trade["side"],
                        "pnl": actual_pnl,
                        "position_size": position_size,
                        "exit_idx": row.Index,
                        "rsi_at_entry": open_trade.get("rsi", 50)
                    })
                    open_trade = None
            
            # Entry Logic (EMA crossover)
            # Only enter if no open trade
            if open_trade is None and row.Index > 20:
                prev_ema9 = df.at[df.index[row.Index-1], "ema_9"]
                prev_ema20 = df.at[df.index[row.Index-1], "ema_20"]
                current_rsi = getattr(row, "rsi_14", 50)
                
                # Long crossover
                if prev_ema9 <= prev_ema20 and row.ema_9 > row.ema_20:
                    open_trade = {
                        "side": "LONG",
                        "entry_price": row.close,
                        "sl": row.close - (row.atr_14 * self.atr_sl),
                        "tp": row.close + (row.atr_14 * self.atr_tp),
                        "rsi": current_rsi
                    }
                # Short crossover
                elif prev_ema9 >= prev_ema20 and row.ema_9 < row.ema_20:
                    open_trade = {
                        "side": "SHORT",
                        "entry_price": row.close,
                        "sl": row.close + (row.atr_14 * self.atr_sl),
                        "tp": row.close - (row.atr_14 * self.atr_tp),
                        "rsi": current_rsi
                    }
                    
            # Compute features for this candle based on history
            feats = self._compute_trade_features_for_candle(closed_trades, equity, peak_equity, row.Index)
            features_list.append(feats)
            
        # Add features to dataframe
        feats_df = pd.DataFrame(features_list, index=df.index)
        
        # Drop temporary columns if we added them
        result = pd.concat([df, feats_df], axis=1)
        
        logger.info(f"Generated synthetic trade features. Total simulated trades: {len(closed_trades)}")
        logger.info(f"Final Simulated Equity: ${equity:.2f} (from ${initial_equity})")
        
        return result

def add_trade_history_features(symbol="BTCUSDT", base_tf="15m"):
    """Loads liquidity dataset and appends synthetic trade history features."""
    project_root = Path(__file__).resolve().parent.parent
    features_dir = project_root / "data" / "features" / symbol
    
    # We should merge onto the liquidity dataset if it exists, else base dataset
    liq_path = features_dir / f"liquidity_merged_{base_tf}.parquet"
    base_path = features_dir / f"base_features_{base_tf}.parquet"
    
    in_path = liq_path if liq_path.exists() else base_path
    
    if not in_path.exists():
        logger.error(f"Input features not found at {in_path}")
        return None
        
    df = pd.read_parquet(in_path)
    
    simulator = SyntheticTradeGenerator()
    result_df = simulator.generate_features(df)
    
    return result_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    final_df = add_trade_history_features()
    if final_df is not None:
        print(f"Columns: {list(final_df.columns)}")
