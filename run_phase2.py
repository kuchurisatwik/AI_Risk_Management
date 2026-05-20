"""
Phase 2 Orchestrator — Feature Engineering Pipeline (Efficient Proxy Approach).

Steps:
    1. Compute efficient liquidity proxies (trade imbalance, Amihud illiquidity)
    2. Run synthetic trade simulation for behavioral features
    3. Save final master features dataset

Usage:
    python run_phase2.py
"""

import sys
import argparse
import logging
from pathlib import Path

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

def step_multi_tf(symbols, base_tf):
    """Step 1: Align multi-timeframe features."""
    from feature_engineering.multi_tf_features import align_and_compute_multi_tf
    print("\n" + "=" * 60)
    print("STEP 1: ALIGN MULTI-TIMEFRAME FEATURES")
    print("=" * 60)
    
    for symbol in symbols:
        df = align_and_compute_multi_tf(symbol=symbol, base_tf=base_tf)
        if df is not None:
            out_path = PROJECT_ROOT / "data" / "features" / symbol / f"multi_tf_merged_{base_tf}.parquet"
            df.to_parquet(out_path, engine="pyarrow", index=False)
            print(f"  [{symbol}] Saved Multi-TF merged features: {len(df)} rows")

def step_liquidity(symbols, base_tf):
    """Step 2: Compute efficient liquidity proxies."""
    from feature_engineering.liquidity_features import build_liquidity_features
    print("\n" + "=" * 60)
    print("STEP 2: COMPUTE LIQUIDITY PROXIES (O(1) Time Complexity)")
    print("=" * 60)
    
    for symbol in symbols:
        df = build_liquidity_features(symbol=symbol, base_tf=base_tf)
        if df is not None:
            out_path = PROJECT_ROOT / "data" / "features" / symbol / f"liquidity_merged_{base_tf}.parquet"
            df.to_parquet(out_path, engine="pyarrow", index=False)
            print(f"  [{symbol}] Saved liquidity merged features: {len(df)} rows")
        
def step_trade_history(symbols, base_tf):
    """Step 3: Generate Synthetic Trade/Behavioral Features."""
    from feature_engineering.trade_history import add_trade_history_features
    print("\n" + "=" * 60)
    print("STEP 3: SYNTHETIC TRADE HISTORY & BEHAVIORAL FEATURES")
    print("=" * 60)
    
    for symbol in symbols:
        df = add_trade_history_features(symbol=symbol, base_tf=base_tf)
        if df is not None:
            out_path = PROJECT_ROOT / "data" / "features" / symbol / f"master_features_{base_tf}.parquet"
            df.to_parquet(out_path, engine="pyarrow", index=False)
            print(f"  [{symbol}] Saved final master features: {len(df)} rows")
            print(f"  [{symbol}] Total columns ready for Phase 3: {len(df.columns)}")

def main():
    import yaml
    parser = argparse.ArgumentParser(description="Phase 2: Efficient Feature Engineering Pipeline")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 2: Feature Engineering Pipeline (Efficient Approach)")
    print("#" * 60)

    config_path = PROJECT_ROOT / "config" / "data_sources.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    base_tf = config.get("base_timeframe", "5m")

    step_multi_tf(symbols, base_tf)
    step_liquidity(symbols, base_tf)
    step_trade_history(symbols, base_tf)

    print("\n" + "#" * 60)
    print("#  [*] PHASE 2 COMPLETE")
    print("#  Master Features Dataset is ready for Phase 3 (Regime Modeling)!")
    print("#" * 60)
    sys.exit(0)

if __name__ == "__main__":
    main()
