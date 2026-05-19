"""
RUN PHASE 6: Institutional Redesign
Executes HMM Regime Detection, Meta-Ensemble Training, and Z-Score Execution.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from training.train_hmm_regime import fit_hmm_model, assign_hmm_labels, save_hmm_model
from training.train_meta_ensemble import train_meta_ensemble
from training.train_volatility import train_volatility
from training.train_risk import train_risk
from training.train_behavioral import train_behavioral
from execution.paper_trader import PaperTrader


def main():
    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 6: Institutional Overhaul (HMM, Meta-Labeling)")
    print("#" * 60 + "\n")
    
    data_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    models_dir = PROJECT_ROOT / "models"
    
    # ---- 1. Load Data ----
    print("STEP 1: LOAD DATA")
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    
    print(f"  Train: {len(train_df)} rows")
    
    # ---- 2. Train HMM Regime Model ----
    print("\nSTEP 2: HMM REGIME DETECTION")
    scaler, hmm_model, mapping = fit_hmm_model(train_df, n_components=4)
    
    train_df = assign_hmm_labels(train_df, scaler, hmm_model, mapping)
    val_df = assign_hmm_labels(val_df, scaler, hmm_model, mapping)
    test_df = assign_hmm_labels(test_df, scaler, hmm_model, mapping)
    
    save_hmm_model(scaler, hmm_model, mapping, models_dir / "regime")
    
    # ---- 3. Train Meta-Ensemble (Primary + Confidence) ----
    print("\nSTEP 3: META-ENSEMBLE TRAINING (Triple-Barrier)")
    # train_meta_ensemble handles labeling predictions natively
    prim, meta, val_met, test_met = train_meta_ensemble(
        train_df, val_df, test_df, models_dir / "momentum"
    )
    
    # ---- 4. Train Downstream Models ----
    print("\nSTEP 4: TRAIN DOWNSTREAM MODELS (Vol, Risk, Behavioral)")
    from training.config import VOLATILITY_FEATURES, RISK_FEATURES, BEHAVIORAL_FEATURES, LABEL_VOLATILITY, LABEL_RISK, LABEL_BEHAVIORAL
    train_volatility(train_df, val_df, test_df, models_dir / "volatility")
    train_risk(train_df, val_df, test_df, models_dir / "risk")
    train_behavioral(train_df, val_df, test_df, models_dir / "behavioral")
    
    # ---- 5. Paper Trading (Z-Score + Kelly) ----
    print("\nSTEP 5: PAPER TRADING (Z-Score Ranking + Kelly Sizing)")
    
    trader = PaperTrader(
        models_dir=str(models_dir),
        initial_equity=10000.0
    )
    
    # Re-warm margin window with validation set to get valid Z-scores at test start
    print("  Warming up Z-Score window with validation set...")
    trader.engine.load()
    for _, row in val_df.iterrows():
        features = {col: row[col] for col in val_df.columns}
        out = trader.engine.ensemble.predict(features)
        trader.engine.margin_window.append(out.meta_margin)
        
    print("  Running simulation on test set...")
    result = trader.run(test_df)
    
    # ---- 6. Results ----
    print("\n" + "=" * 60)
    print("STEP 6: RESULTS")
    print("=" * 60)
    
    print(f"\n  PERFORMANCE METRICS:")
    print(f"    Total Return:     {result.total_return * 100:.2f}%")
    print(f"    Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"    Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"    Max Drawdown:     {result.max_drawdown * 100:.2f}%")
    print(f"    Total Trades:     {result.total_trades}")
    
    if result.total_trades > 0:
        freq = result.total_trades / len(test_df)
        print(f"    Trade Frequency:  {freq:.1%} of bars")
        print(f"    Win Rate:         {result.win_rate:.1%}")
        print(f"    Avg RR Realized:  {result.avg_rr_realized:.2f}")
    
    print(f"\n  REGIME BREAKDOWN:")
    for reg, stats in result.regime_performance.items():
        if stats['trades'] > 0:
            wr = (stats['wins'] / stats['trades']) * 100
            print(f"    {reg:<20}: {stats['trades']:>3} trades | PnL: ${stats['pnl']:>7.2f} | WR: {wr:.0f}%")
            
    # Save trades to CSV
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    if result.trades:
        trade_data = []
        for t in result.trades:
            trade_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position_size_usd': t.position_size_usd,
                'pnl': t.pnl,
                'regime': t.regime,
                'confidence': t.confidence,
                'exit_reason': t.exit_reason
            })
        trades_df = pd.DataFrame(trade_data)
        trades_file = results_dir / "test_trades.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"\n  Saved {len(result.trades)} detailed trades to {trades_file}")

    # Exit with code for pipelining
    sys.exit(0 if result.sharpe_ratio >= 0 else 1)

if __name__ == "__main__":
    main()
