"""
Multi-Timeframe Alignment — Section 3.3 / 4.2.8.
Aligns higher-timeframe candle values to base timeframe close timestamps.
Ensures higher-TF values are available at the correct base-TF bar.
"""

import logging
from pathlib import Path
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class MultiTimeframeAligner:
    """
    Aligns higher-timeframe data to the base timeframe (15m).

    For each base-TF bar, the aligned higher-TF value is the LAST COMPLETED
    higher-TF candle at or before that base-TF bar's close time.
    This prevents forward-looking bias.
    """

    def __init__(self, config_path=None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        if config_path is None:
            config_path = self.project_root / "config" / "data_sources.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.cleaned_dir = self.project_root / "data" / "cleaned"
        self.base_tf = self.config.get("base_timeframe", "15m")

    def load_timeframe(self, symbol, timeframe):
        """Load a cleaned Parquet file for a timeframe."""
        path = self.cleaned_dir / symbol / f"{timeframe}.parquet"
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df = df.sort_values("open_time").reset_index(drop=True)
        return df

    def align_to_base(self, base_df, higher_df, higher_tf,
                       columns=None):
        """
        Align higher-TF data to base-TF timestamps using asof merge.

        Args:
            base_df: Base timeframe DataFrame (must have 'open_time')
            higher_df: Higher timeframe DataFrame
            higher_tf: String label for suffix (e.g. "1h")
            columns: Which columns to bring from higher_df.
                     If None, brings OHLCV.

        Returns:
            base_df with additional columns suffixed by _{higher_tf}
        """
        if higher_df.empty:
            return base_df

        if columns is None:
            columns = ["open", "high", "low", "close", "volume"]

        # Prepare higher-TF: use close_time as the "available at" timestamp
        htf = higher_df[["close_time"] + columns].copy()
        htf = htf.sort_values("close_time")

        # Rename columns with suffix
        rename_map = {c: f"{c}_{higher_tf}" for c in columns}
        htf = htf.rename(columns=rename_map)

        # asof merge: for each base bar, find the last higher-TF bar
        # whose close_time <= base bar's open_time (strictly causal)
        result = pd.merge_asof(
            base_df.sort_values("open_time"),
            htf,
            left_on="open_time",
            right_on="close_time",
            direction="backward"
        )

        # Drop the merged close_time column
        if "close_time_y" in result.columns:
            result = result.drop(columns=["close_time_y"])
            if "close_time_x" in result.columns:
                result = result.rename(columns={"close_time_x": "close_time"})

        return result

    def build_multi_tf_dataset(self, symbol="BTCUSDT"):
        """
        Build a multi-timeframe aligned dataset.
        Base = 15m, aligned with 1h, 4h, 1d.

        Returns the base DataFrame with higher-TF columns merged.
        """
        base_df = self.load_timeframe(symbol, self.base_tf)
        if base_df.empty:
            logger.error(f"No base timeframe data for {symbol}/{self.base_tf}")
            return pd.DataFrame()

        higher_tfs = [tf for tf in self.config["timeframes"]
                      if tf != self.base_tf and tf not in ["5m"]]

        logger.info(f"Building multi-TF dataset: base={self.base_tf}, "
                     f"higher={higher_tfs}")

        for htf in higher_tfs:
            htf_df = self.load_timeframe(symbol, htf)
            if htf_df.empty:
                logger.warning(f"Skipping {htf}: no data")
                continue
            base_df = self.align_to_base(base_df, htf_df, htf)
            logger.info(f"  Aligned {htf}: {len(base_df)} rows")

        # Also align lower TF (5m) if available
        lower_tf = "5m"
        lower_df = self.load_timeframe(symbol, lower_tf)
        if not lower_df.empty:
            base_df = self.align_to_base(base_df, lower_df, lower_tf)
            logger.info(f"  Aligned {lower_tf}: {len(base_df)} rows")

        return base_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    aligner = MultiTimeframeAligner()
    df = aligner.build_multi_tf_dataset()
    if not df.empty:
        print(f"Multi-TF dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
