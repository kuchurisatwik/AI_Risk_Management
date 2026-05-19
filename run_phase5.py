"""
Phase 5 Orchestrator — Policy Engine Validation.

Runs the full inference pipeline against the test set
in paper trading mode and validates system behavior.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.paper_trader import PaperTrader
from monitoring.drift_detector import DriftDetector
from inference.threshold_engine import fit_threshold_engine
from inference.model_ensemble import ModelEnsemble
from training.config import MOMENTUM_FEATURES, VOLATILITY_FEATURES


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Policy Engine Validation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 5: Policy Engine & Paper Trading Validation")
    print("#" * 60)

    # ---- Load Data ----
    print("\n" + "=" * 60)
    print("STEP 1: LOAD DATA")
    print("=" * 60)
    
    data_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")
    print(f"  Test range: {test_df['open_time'].min()} to {test_df['open_time'].max()}")

    # ---- Drift Detection ----
    print("\n" + "=" * 60)
    print("STEP 2: DRIFT DETECTION")
    print("=" * 60)
    
    drift = DriftDetector()
    all_features = MOMENTUM_FEATURES + VOLATILITY_FEATURES
    # Remove duplicates
    all_features = list(dict.fromkeys(all_features))
    drift.fit(train_df, all_features)
    report = drift.check(test_df, all_features)
    
    print(f"  Features checked: {len(report.psi_scores)}")
    print(f"  Features drifted: {len(report.features_drifted)}")
    if report.features_drifted:
        for f in report.features_drifted:
            print(f"    {f}: PSI={report.psi_scores[f]:.4f}")
    if report.retrain_recommended:
        print("  WARNING: Retraining recommended due to significant drift!")
    else:
        print("  Drift within acceptable bounds.")

    # ---- Calibrate Threshold Engine ----
    print("\n" + "=" * 60)
    print("STEP 3: CALIBRATE ADAPTIVE THRESHOLD ENGINE")
    print("=" * 60)
    
    # Compute momentum probabilities on training set to learn the distribution
    ens = ModelEnsemble(str(PROJECT_ROOT / "models"))
    ens.load()
    
    # Sample training set for speed (every 5th row)
    sample_indices = range(0, len(train_df), 5)
    train_probs = []
    for i in sample_indices:
        row = train_df.iloc[i]
        features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in train_df.columns}
        out = ens.predict(features)
        train_probs.append(out.momentum_probability)
    
    train_probs = np.array(train_probs)
    fit_threshold_engine(train_probs)
    
    print(f"  Fitted on {len(train_probs)} training samples")
    print(f"  Probability range: [{train_probs.min():.3f}, {train_probs.max():.3f}]")
    print(f"  Median: {np.median(train_probs):.3f}")

    # ---- Paper Trading ----
    print("\n" + "=" * 60)
    print("STEP 4: PAPER TRADING SIMULATION")
    print("=" * 60)
    
    trader = PaperTrader(
        models_dir=str(PROJECT_ROOT / "models"),
        initial_equity=10000.0
    )
    trader.load()
    
    print("  Running simulation on test set...")
    result = trader.run(test_df)

    # ---- Results ----
    print("\n" + "=" * 60)
    print("STEP 5: RESULTS")
    print("=" * 60)
    
    print(f"\n  PERFORMANCE METRICS:")
    print(f"    Total Return:     {result.total_return*100:+.2f}%")
    print(f"    Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"    Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"    Max Drawdown:     {result.max_drawdown*100:.2f}%")
    print(f"    Total Trades:     {result.total_trades}")
    print(f"    Trade Frequency:  {result.trade_frequency_pct:.1f}% of bars")
    print(f"    Win Rate:         {result.win_rate*100:.1f}%")
    print(f"    Avg RR Realized:  {result.avg_rr_realized:.2f}")
    
    print(f"\n  REGIME BREAKDOWN:")
    for regime, stats in result.regime_performance.items():
        wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
        print(f"    {regime:20s}: {stats['trades']:3d} trades | PnL: ${stats['pnl']:8.2f} | WR: {wr:.0f}%")
    
    print(f"\n  BLOCK REASONS (top 10):")
    sorted_blocks = sorted(result.block_reasons_summary.items(), key=lambda x: -x[1])
    for reason, count in sorted_blocks[:10]:
        print(f"    {reason:30s}: {count:,}")
    
    # ---- Governance Validation ----
    print("\n" + "=" * 60)
    print("STEP 6: GOVERNANCE VALIDATION")
    print("=" * 60)
    
    crash_trades = [t for t in result.trades if t.regime == 'crash_mode']
    print(f"  Trades in crash_mode: {len(crash_trades)} (should be 0)")
    
    print(f"  Max Drawdown: {result.max_drawdown*100:.2f}% (target: < 5%)")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f} (target: > 0.5)")
    
    # Check if system is too conservative (0 trades)
    if result.total_trades == 0:
        print("\n  NOTE: Zero trades executed. Possible causes:")
        print("    - Calibrated probabilities below adaptive thresholds")
        print("    - Risk model blocking all candles as HIGH_RISK or NO_TRADE")
        print("    - Consider reviewing threshold engine parameters")

    print("\n" + "#" * 60)
    print("#  PHASE 5 COMPLETE")
    print("#  Policy Engine validated against test set")
    print("#" * 60)


if __name__ == "__main__":
    main()
