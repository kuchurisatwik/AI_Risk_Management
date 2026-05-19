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


def step_load_and_split():
    """Step 1-2: Load master features, dropna, time-based split."""
    print("\n" + "=" * 60)
    print("STEP 1: LOAD & TIME-BASED SPLIT (70/15/15)")
    print("=" * 60)

    master_path = PROJECT_ROOT / "data" / "features" / "BTCUSDT" / "master_features_15m.parquet"
    df = pd.read_parquet(master_path)
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"  Total usable rows: {n:,}")
    print(f"  Train: {len(train_df):,} rows ({train_df['open_time'].min()} to {train_df['open_time'].max()})")
    print(f"  Val:   {len(val_df):,} rows ({val_df['open_time'].min()} to {val_df['open_time'].max()})")
    print(f"  Test:  {len(test_df):,} rows ({test_df['open_time'].min()} to {test_df['open_time'].max()})")

    return train_df, val_df, test_df


def step_regime(train_df, val_df, test_df):
    """Step 3: Fit KMeans on train, assign regime labels to all splits."""
    from labeling.regime_labeler import (
        fit_regime_model, assign_regime_labels, save_regime_model, REGIME_FEATURES
    )

    print("\n" + "=" * 60)
    print("STEP 2: REGIME DETECTION (KMeans on Train Only)")
    print("=" * 60)

    scaler, km, mapping, centroid_df, sil_score = fit_regime_model(train_df)

    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Cluster Mapping:")
    for cid, name in sorted(mapping.items()):
        count = (km.labels_ == cid).sum()
        print(f"    Cluster {cid} -> {name} ({count:,} train samples)")

    print(f"\n  Centroid Values (original scale):")
    centroid_df.index = [mapping[i] for i in range(len(centroid_df))]
    print(centroid_df.round(3).to_string())

    # Assign to all splits (transform only, no fit)
    train_df = assign_regime_labels(train_df, scaler, km, mapping)
    val_df = assign_regime_labels(val_df, scaler, km, mapping)
    test_df = assign_regime_labels(test_df, scaler, km, mapping)

    # Save model
    save_regime_model(scaler, km, mapping, PROJECT_ROOT / "models" / "regime")

    return train_df, val_df, test_df


def step_labels(train_df, val_df, test_df):
    """Step 4: Generate all labels."""
    from labeling.momentum_labeler import label_momentum
    from labeling.volatility_labeler import label_volatility
    from labeling.risk_labeler import label_risk
    from labeling.behavioral_labeler import label_behavioral

    print("\n" + "=" * 60)
    print("STEP 3: GENERATE LABELS")
    print("=" * 60)

    # Momentum labels (forward-looking, applied to each split independently)
    print("\n  [Momentum]")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df['label_momentum'] = label_momentum(split_df)

    # Volatility labels (forward-looking)
    print("\n  [Volatility]")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df['label_volatility'] = label_volatility(split_df)

    # Risk labels (uses regime_label, current features only)
    print("\n  [Risk]")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df['label_risk'] = label_risk(split_df)

    # Behavioral labels (current features only)
    print("\n  [Behavioral]")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df['label_behavioral'] = label_behavioral(split_df)

    return train_df, val_df, test_df


def step_leakage(train_df, val_df, test_df):
    """Step 5: Run leakage validation."""
    from labeling.leakage_check import validate_all_labels, structural_checks

    print("\n" + "=" * 60)
    print("STEP 4: LEAKAGE VALIDATION")
    print("=" * 60)

    # Define feature sets per model (roadmap minimum 15 features)
    MINIMUM_FEATURES = [
        'ema_20_slope', 'atr_14', 'atr_expansion_ratio', 'rsi_velocity',
        'vwap_distance', 'amihud_illiquidity', 'trade_imbalance',
        'volume_delta', 'strategy_health_score', 'strategy_recent_accuracy',
        'strategy_avg_rr', 'last_5_trade_winrate', 'consecutive_losses',
        'recent_drawdown', 'revenge_trade_score'
    ]

    feature_sets = {
        'label_momentum': MINIMUM_FEATURES,
        'label_volatility': MINIMUM_FEATURES,
        'label_risk': MINIMUM_FEATURES,
        'label_behavioral': MINIMUM_FEATURES,
    }

    print("\n  Feature-Label Correlation Check:")
    issues = validate_all_labels(train_df, feature_sets)

    print("\n  Time-Split Integrity Check:")
    structural_checks(train_df, val_df, test_df)

    if issues:
        print(f"\n  WARNING: {sum(len(v) for v in issues.values())} potential leakage issues found!")
    else:
        print("\n  All leakage checks PASSED.")

    return issues


def step_save(train_df, val_df, test_df):
    """Step 6: Save labeled splits."""
    print("\n" + "=" * 60)
    print("STEP 5: SAVE LABELED DATASETS")
    print("=" * 60)

    out_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = out_dir / f"{name}.parquet"
        split_df.to_parquet(path, engine="pyarrow", index=False)
        print(f"  Saved {name}: {len(split_df):,} rows -> {path}")

    return out_dir


def step_summary(train_df, val_df, test_df):
    """Step 7: Print dataset statistics."""
    print("\n" + "=" * 60)
    print("STEP 6: DATASET SUMMARY")
    print("=" * 60)

    print(f"\n  Total columns: {len(train_df.columns)}")
    print(f"  Label columns: label_momentum, label_volatility, label_risk, label_behavioral")
    print(f"  Regime columns: regime_cluster, regime_label, regime_confidence")

    # Momentum distribution
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        valid = df['label_momentum'].dropna()
        pos = (valid == 1).sum()
        neg = (valid == 0).sum()
        print(f"\n  {name} Momentum: {pos} positive ({pos/(pos+neg)*100:.1f}%) / {neg} negative")

    # Risk distribution
    print(f"\n  Train Risk Distribution:")
    risk_dist = train_df['label_risk'].value_counts()
    for label, count in risk_dist.items():
        print(f"    {label}: {count:,} ({count/len(train_df)*100:.1f}%)")

    # Regime distribution
    print(f"\n  Train Regime Distribution:")
    regime_dist = train_df['regime_label'].value_counts()
    for label, count in regime_dist.items():
        print(f"    {label}: {count:,} ({count/len(train_df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Labeling & Dataset Preparation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 3: Labeling & Dataset Preparation")
    print("#" * 60)

    # Step 1-2: Load and split
    train_df, val_df, test_df = step_load_and_split()

    # Step 3: Regime detection
    train_df, val_df, test_df = step_regime(train_df, val_df, test_df)

    # Step 4: Generate all labels
    train_df, val_df, test_df = step_labels(train_df, val_df, test_df)

    # Step 5: Leakage validation
    step_leakage(train_df, val_df, test_df)

    # Step 6: Save
    step_save(train_df, val_df, test_df)

    # Step 7: Summary
    step_summary(train_df, val_df, test_df)

    print("\n" + "#" * 60)
    print("#  PHASE 3 COMPLETE")
    print("#  Labeled datasets saved to data/labeled/BTCUSDT/")
    print("#  Ready for Phase 4: Model Training")
    print("#" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
