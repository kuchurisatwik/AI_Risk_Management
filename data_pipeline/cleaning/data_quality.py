"""
Data Quality Validator — Section 3.3 of the roadmap.
Validates OHLCV data integrity, detects missing candles, and produces quality reports.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from utils.time_utils import TIMEFRAME_SECONDS, find_missing_timestamps

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    Validates cleaned Parquet data against roadmap quality rules:
    - Remove duplicate timestamps (keep last)
    - Detect missing candles (gap analysis)
    - Forward-fill missing candles for indicators only (flag them)
    - Reject bad ticks: price spike > 5 ATR from rolling mean
    - OHLCV integrity: high >= max(open,close), low <= min(open,close)
    - Timezone consistency (UTC)
    """

    def __init__(self, config_path=None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        if config_path is None:
            config_path = self.project_root / "config" / "assets.yaml"
        with open(config_path) as f:
            self.assets_cfg = yaml.safe_load(f)
        self.cleaned_dir = self.project_root / "data" / "cleaned"

    def validate_ohlcv_integrity(self, df):
        """
        Check: high >= max(open, close) and low <= min(open, close)
        Returns mask of invalid rows.
        """
        max_oc = df[["open", "close"]].max(axis=1)
        min_oc = df[["open", "close"]].min(axis=1)
        bad_high = df["high"] < max_oc - 1e-10
        bad_low = df["low"] > min_oc + 1e-10
        invalid = bad_high | bad_low
        return invalid

    def detect_missing_candles(self, df, timeframe):
        """Find missing candle timestamps based on expected frequency."""
        missing = find_missing_timestamps(df, timeframe, "open_time")
        return missing

    def detect_spikes(self, df, atr_threshold=5.0, rolling_window=50):
        """
        Detect price spikes > N ATR from rolling mean.
        Uses a simple rolling ATR approximation for spike detection.
        """
        close = df["close"].copy()
        high = df["high"].copy()
        low = df["low"].copy()

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        rolling_atr = tr.rolling(window=rolling_window, min_periods=10).mean()
        rolling_mean = close.rolling(window=rolling_window, min_periods=10).mean()

        deviation = (close - rolling_mean).abs()
        spikes = deviation > (atr_threshold * rolling_atr)

        # Don't flag first N rows (not enough history)
        spikes.iloc[:rolling_window] = False
        return spikes

    def check_timezone(self, df):
        """Verify open_time is UTC."""
        if hasattr(df["open_time"].dt, "tz"):
            tz = df["open_time"].dt.tz
            return str(tz) == "UTC" if tz else False
        return False

    def forward_fill_gaps(self, df, timeframe):
        """
        Forward-fill missing candles. Fills price with last close,
        volume with 0. Adds 'is_filled' flag column.
        """
        if df.empty:
            return df

        interval = pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
        full_idx = pd.date_range(
            start=df["open_time"].min(),
            end=df["open_time"].max(),
            freq=interval
        )

        df = df.set_index("open_time")
        df = df.reindex(full_idx)
        df.index.name = "open_time"

        # Mark filled rows
        df["is_filled"] = df["close"].isna()

        # Forward-fill prices (never for actual trading — indicators only)
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].ffill()

        # Zero-fill volume columns
        vol_cols = ["volume", "quote_volume", "taker_buy_volume",
                    "taker_buy_quote_volume"]
        for col in vol_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Fill count and close_time
        if "count" in df.columns:
            df["count"] = df["count"].fillna(0).astype(int)
        if "close_time" in df.columns:
            df["close_time"] = df["close_time"].ffill()

        df = df.reset_index()
        return df

    def validate_timeframe(self, symbol, timeframe):
        """Run all quality checks on one timeframe. Returns a report dict."""
        parquet_path = self.cleaned_dir / symbol / f"{timeframe}.parquet"
        if not parquet_path.exists():
            return {"timeframe": timeframe, "status": "file_not_found"}

        df = pd.read_parquet(parquet_path)
        total_rows = len(df)

        # 1. OHLCV integrity
        invalid_ohlcv = self.validate_ohlcv_integrity(df)
        n_invalid = invalid_ohlcv.sum()

        # 2. Missing candles
        missing = self.detect_missing_candles(df, timeframe)
        n_missing = len(missing)
        expected = total_rows + n_missing
        missing_pct = (n_missing / expected * 100) if expected > 0 else 0

        # 3. Spike detection
        asset_cfg = self.assets_cfg.get(symbol, {})
        quality_cfg = asset_cfg.get("quality", {})
        atr_thresh = quality_cfg.get("spike_atr_threshold", 5.0)
        spike_window = quality_cfg.get("spike_rolling_window", 50)
        spikes = self.detect_spikes(df, atr_thresh, spike_window)
        n_spikes = spikes.sum()

        # 4. Timezone check
        tz_ok = self.check_timezone(df)

        # 5. Duplicates check (should be 0 after parquet_writer)
        n_dupes = df.duplicated(subset=["open_time"]).sum()

        # Gate check: < 0.5% missing
        max_missing = quality_cfg.get("max_missing_candle_pct", 0.5)
        gate_pass = missing_pct < max_missing

        report = {
            "timeframe": timeframe,
            "total_rows": total_rows,
            "date_range": f"{df['open_time'].min()} -> {df['open_time'].max()}",
            "missing_candles": n_missing,
            "missing_pct": round(missing_pct, 4),
            "invalid_ohlcv": int(n_invalid),
            "spikes_detected": int(n_spikes),
            "duplicates": int(n_dupes),
            "timezone_utc": tz_ok,
            "gate_pass": gate_pass,
            "status": "validated"
        }
        return report

    def validate_all(self, symbol="BTCUSDT"):
        """Validate all timeframes for a symbol. Print quality report."""
        ds_cfg_path = self.project_root / "config" / "data_sources.yaml"
        with open(ds_cfg_path) as f:
            ds_cfg = yaml.safe_load(f)
        timeframes = ds_cfg["timeframes"]

        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"  Symbol: {symbol}")

        reports = {}
        all_pass = True

        for tf in timeframes:
            r = self.validate_timeframe(symbol, tf)
            reports[tf] = r

            if r["status"] == "file_not_found":
                print(f"\n  {tf}: [!] FILE NOT FOUND")
                all_pass = False
                continue

            gate = "[PASS]" if r["gate_pass"] else "[FAIL]"
            if not r["gate_pass"]:
                all_pass = False

            print(f"\n  {tf}: {gate}")
            print(f"    Rows:       {r['total_rows']:,}")
            print(f"    Range:      {r['date_range']}")
            print(f"    Missing:    {r['missing_candles']:,} ({r['missing_pct']:.2f}%)")
            print(f"    Bad OHLCV:  {r['invalid_ohlcv']}")
            print(f"    Spikes:     {r['spikes_detected']}")
            print(f"    Duplicates: {r['duplicates']}")
            print(f"    UTC:        {'YES' if r['timezone_utc'] else 'NO'}")

        print(f"\n{'=' * 60}")
        overall = "[PASS] PHASE 1 GATE MET" if all_pass else "[FAIL] PHASE 1 GATE NOT MET"
        print(f"  {overall}")
        print(f"{'=' * 60}")

        return reports, all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validator = DataQualityValidator()
    validator.validate_all()
