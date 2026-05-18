"""
Parquet Writer — Consolidates daily kline CSVs into clean Parquet files.
Input:  data/raw/{symbol}/{timeframe}/*.csv
Output: data/cleaned/{symbol}/{timeframe}.parquet
"""

import logging
from pathlib import Path
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

KEEP_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume"
]

FLOAT_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"
]


class ParquetWriter:
    def __init__(self, config_path=None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        if config_path is None:
            config_path = self.project_root / "config" / "data_sources.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.raw_dir = self.project_root / "data" / "raw"
        self.cleaned_dir = self.project_root / "data" / "cleaned"
        self.symbols = self.config["symbols"]
        self.timeframes = self.config["timeframes"]

    def _read_daily_csvs(self, symbol, timeframe):
        csv_dir = self.raw_dir / symbol / timeframe
        if not csv_dir.exists():
            logger.warning(f"Directory not found: {csv_dir}")
            return pd.DataFrame()
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files in {csv_dir}")
            return pd.DataFrame()
        logger.info(f"Reading {len(csv_files)} CSVs from {csv_dir}")
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
                for col in FLOAT_COLUMNS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {f.name}: {e}")
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"  Combined: {len(combined):,} rows")
        return combined

    def _clean_dataframe(self, df):
        if df.empty:
            return df
        df = df[KEEP_COLUMNS].copy()
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
        df = df.dropna(subset=["open_time"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.sort_values("open_time").reset_index(drop=True)
        before = len(df)
        df = df.drop_duplicates(subset=["open_time"], keep="last")
        dupes = before - len(df)
        if dupes > 0:
            logger.info(f"  Removed {dupes:,} duplicate timestamps")
        return df.reset_index(drop=True)

    def consolidate_timeframe(self, symbol, timeframe):
        logger.info(f"Consolidating {symbol}/{timeframe}")
        df = self._read_daily_csvs(symbol, timeframe)
        if df.empty:
            return {"timeframe": timeframe, "rows": 0, "status": "empty"}
        df = self._clean_dataframe(df)
        out_dir = self.cleaned_dir / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{timeframe}.parquet"
        df.to_parquet(out_path, engine="pyarrow", index=False)
        size_mb = round(out_path.stat().st_size / (1024*1024), 2)
        result = {
            "timeframe": timeframe, "rows": len(df),
            "date_range": f"{df['open_time'].min()} -> {df['open_time'].max()}",
            "file_path": str(out_path), "file_size_mb": size_mb,
            "status": "success"
        }
        logger.info(f"  [*] {timeframe}: {len(df):,} rows | {size_mb:.1f} MB")
        return result

    def consolidate_all(self):
        print("\n" + "=" * 60)
        print("PARQUET CONSOLIDATION")
        print("=" * 60)
        all_results = {}
        for symbol in self.symbols:
            print(f"\n  Symbol: {symbol}")
            sym_res = {}
            for tf in self.timeframes:
                sym_res[tf] = self.consolidate_timeframe(symbol, tf)
            all_results[symbol] = sym_res
        # Summary
        print(f"\n{'=' * 60}")
        print("CONSOLIDATION SUMMARY")
        print(f"{'=' * 60}")
        for symbol, tf_res in all_results.items():
            total = sum(r["rows"] for r in tf_res.values())
            print(f"\n  {symbol}: {total:,} total rows")
            for tf, r in tf_res.items():
                if r["status"] == "success":
                    print(f"    {tf:>4s}: {r['rows']:>10,} rows | {r['file_size_mb']:>6.1f} MB")
                else:
                    print(f"    {tf:>4s}: {r['status']}")
        return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ParquetWriter().consolidate_all()
