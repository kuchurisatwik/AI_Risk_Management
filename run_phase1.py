"""
Phase 1 Orchestrator — Run the complete data foundation pipeline.

Steps:
    1. Download klines from Binance Vision
    2. Convert raw CSVs -> consolidated Parquet
    3. Run data quality validation
    4. Compute basic indicators (optional, for verification)

Usage:
    python run_phase1.py                    # Full pipeline
    python run_phase1.py --download-only    # Just download
    python run_phase1.py --validate-only    # Just validate existing data
    python run_phase1.py --skip-download    # Skip download, do consolidation + validation
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


def step_download():
    """Step 1: Download klines from Binance Vision."""
    from data_pipeline.collectors.binance_downloader import BinanceDownloader
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOAD DATA FROM BINANCE VISION")
    print("=" * 60)
    dl = BinanceDownloader()
    results = dl.download_all()
    return results


def step_consolidate():
    """Step 2: Consolidate daily CSVs into Parquet."""
    from data_pipeline.storage.parquet_writer import ParquetWriter
    print("\n" + "=" * 60)
    print("STEP 2: CONSOLIDATE CSVs -> PARQUET")
    print("=" * 60)
    writer = ParquetWriter()
    results = writer.consolidate_all()
    return results


def step_validate():
    """Step 3: Run data quality validation."""
    from data_pipeline.cleaning.data_quality import DataQualityValidator
    print("\n" + "=" * 60)
    print("STEP 3: DATA QUALITY VALIDATION")
    print("=" * 60)
    validator = DataQualityValidator()
    reports, gate_pass = validator.validate_all()
    return reports, gate_pass


def step_compute_indicators():
    """Step 4: Compute basic indicators on 5m base timeframe."""
    import pandas as pd
    import yaml
    from feature_engineering.base_indicators import compute_all_base_indicators

    print("\n" + "=" * 60)
    print("STEP 4: COMPUTE BASE INDICATORS (5m TIMEFRAME)")
    print("=" * 60)

    config_path = PROJECT_ROOT / "config" / "data_sources.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    timeframes = config.get("timeframes", ["5m", "15m", "1h"])

    for symbol in symbols:
        features_dir = PROJECT_ROOT / "data" / "features" / symbol
        features_dir.mkdir(parents=True, exist_ok=True)
        
        for tf in timeframes:
            cleaned_path = PROJECT_ROOT / "data" / "cleaned" / symbol / f"{tf}.parquet"
            if not cleaned_path.exists():
                print(f"  ⚠ {tf}.parquet not found for {symbol} — skipping indicator computation")
                continue

            df = pd.read_parquet(cleaned_path)
            print(f"\n  [{symbol}] Computing {tf}... Loaded: {len(df):,} rows")

            df_with_indicators = compute_all_base_indicators(df)

            out_path = features_dir / f"base_features_{tf}.parquet"
            df_with_indicators.to_parquet(out_path, engine="pyarrow", index=False)

            n_indicators = len(df_with_indicators.columns) - len(df.columns)
            print(f"  [*] Computed {n_indicators} indicator columns")
            print(f"  [*] Saved to {out_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Data Foundation Pipeline"
    )
    parser.add_argument("--download-only", action="store_true",
                        help="Only download data, skip consolidation")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing data")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, run consolidation + validation")
    parser.add_argument("--with-indicators", action="store_true",
                        help="Also compute base indicators after validation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 1: Data Foundation Pipeline")
    print("#" * 60)

    if args.validate_only:
        reports, gate_pass = step_validate()
        sys.exit(0 if gate_pass else 1)

    if args.download_only:
        step_download()
        sys.exit(0)

    # Full pipeline
    if not args.skip_download:
        step_download()

    step_consolidate()
    reports, gate_pass = step_validate()

    if args.with_indicators:
        step_compute_indicators()

    # Final summary
    print("\n" + "#" * 60)
    if gate_pass:
        print("#  [PASS] PHASE 1 COMPLETE - Gate condition met!")
        print("#  Next: Run with --with-indicators to compute features")
        print("#  Then proceed to Phase 2: Feature Engineering")
    else:
        print("#  ❌ PHASE 1 INCOMPLETE — Gate condition NOT met")
        print("#  Check missing candle percentages above")
        print("#  May need to download more data or investigate gaps")
    print("#" * 60)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
