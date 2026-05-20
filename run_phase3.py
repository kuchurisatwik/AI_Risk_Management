"""
Phase 3 Orchestrator — Labeling & Dataset Preparation Pipeline.

Steps:
    1. Load master features, drop warm-up NaN rows
    2. Time-based split (70/15/15)
    3. Fit KMeans regime model on train only
    4. Generate all labels (momentum, volatility, risk, behavioral)
    5. Run leakage validation
    6. Save labeled splits to data/labeled/
    7. Print dataset statistics

Usage:
    python run_phase3.py
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def run_pipeline_for_symbol(symbol, base_tf):
    print(f"\n" + "*" * 60)
    print(f"* PROCESSING SYMBOL: {symbol} (Base TF: {base_tf})")
    print(f"*" * 60)
    
    # ---------------------------------------------------------
    # STEP 1: LOAD & PURGED TIME-BASED SPLIT (70/15/15)
    # ---------------------------------------------------------
    master_path = PROJECT_ROOT / "data" / "features" / symbol / f"master_features_{base_tf}.parquet"
    if not master_path.exists():
        print(f"  [!] Master features not found for {symbol} at {master_path}. Skipping.")
        return
        
    df = pd.read_parquet(master_path)
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    purge_size = 12
    embargo_size = 24

    train_end = int(n * 0.70)
    val_start = train_end + embargo_size
    val_end = int(n * 0.85)
    test_start = val_end + embargo_size

    train_df = df.iloc[:train_end - purge_size].copy()
    val_df = df.iloc[val_start:val_end - purge_size].copy()
    test_df = df.iloc[test_start:].copy()

    print(f"  Total usable rows: {n:,}")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    # ---------------------------------------------------------
    # STEP 2: REGIME DETECTION
    # ---------------------------------------------------------
    from labeling.regime_labeler import fit_regime_model, assign_regime_labels, save_regime_model
    scaler, km, mapping, centroid_df, sil_score = fit_regime_model(train_df)

    print(f"  Silhouette Score: {sil_score:.4f}")
    
    # Assign to all splits
    train_df = assign_regime_labels(train_df, scaler, km, mapping)
    val_df = assign_regime_labels(val_df, scaler, km, mapping)
    test_df = assign_regime_labels(test_df, scaler, km, mapping)

    # Save model independently per symbol
    save_regime_model(scaler, km, mapping, PROJECT_ROOT / "models" / symbol / "regime")

    # ---------------------------------------------------------
    # STEP 3: GENERATE LABELS
    # ---------------------------------------------------------
    from labeling.momentum_labeler import label_momentum
    from labeling.meanrev_labeler import label_meanrev
    from labeling.volatility_labeler import label_volatility
    from labeling.risk_labeler import label_risk
    from labeling.behavioral_labeler import label_behavioral

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df['label_momentum'] = label_momentum(split_df, regime_aware=True)
        split_df['label_meanrev'] = label_meanrev(split_df)
        split_df['label_volatility'] = label_volatility(split_df)
        split_df['label_risk'] = label_risk(split_df)
        split_df['label_behavioral'] = label_behavioral(split_df)

    # ---------------------------------------------------------
    # STEP 4: LEAKAGE VALIDATION
    # ---------------------------------------------------------
    from labeling.leakage_check import validate_all_labels, structural_checks
    MINIMUM_FEATURES = [
        'ema_20_slope', 'atr_14', 'atr_expansion_ratio', 'rsi_velocity',
        'vwap_distance', 'amihud_illiquidity', 'trade_imbalance',
        'volume_delta', 'strategy_health_score', 'strategy_recent_accuracy',
        'strategy_avg_rr', 'last_5_trade_winrate', 'consecutive_losses',
        'recent_drawdown', 'revenge_trade_score'
    ]

    feature_sets = {
        'label_momentum': MINIMUM_FEATURES,
        'label_meanrev': MINIMUM_FEATURES,
        'label_volatility': MINIMUM_FEATURES,
        'label_risk': MINIMUM_FEATURES,
        'label_behavioral': MINIMUM_FEATURES,
    }

    issues = validate_all_labels(train_df, feature_sets)
    structural_checks(train_df, val_df, test_df)

    if issues:
        print(f"  WARNING: {sum(len(v) for v in issues.values())} potential leakage issues found!")
    else:
        print("  Leakage checks PASSED.")

    # ---------------------------------------------------------
    # STEP 5: SAVE LABELED DATASETS
    # ---------------------------------------------------------
    out_dir = PROJECT_ROOT / "data" / "labeled" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = out_dir / f"{name}.parquet"
        split_df.to_parquet(path, engine="pyarrow", index=False)
        print(f"  Saved {name}: {len(split_df):,} rows -> {path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Labeling & Dataset Preparation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 3: Labeling & Dataset Preparation (Multi-Asset)")
    print("#" * 60)

    config_path = PROJECT_ROOT / "config" / "data_sources.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    base_tf = config.get("base_timeframe", "5m")

    for symbol in symbols:
        run_pipeline_for_symbol(symbol, base_tf)

    print("\n" + "#" * 60)
    print("#  PHASE 3 COMPLETE")
    print("#  Ready for Phase 4: Model Training")
    print("#" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
