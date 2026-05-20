"""
Binance Vision Kline Downloader
================================
Downloads daily kline ZIP files from https://data.binance.vision/
for BTCUSDT futures (USDT-M) across multiple timeframes.

File pattern: {base_url}/{symbol}/{timeframe}/{symbol}-{timeframe}-{date}.zip
Each ZIP contains a single CSV with 12 columns (no header).

Usage:
    from data_pipeline.collectors.binance_downloader import BinanceDownloader
    dl = BinanceDownloader()
    dl.download_all()
"""

import os
import sys
import zipfile
import hashlib
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class BinanceDownloader:
    """
    Downloads daily kline data from Binance Vision public S3 bucket.

    Features:
        - Concurrent downloads with configurable parallelism
        - Resume support: skips already-downloaded files
        - SHA256 checksum validation
        - Progress tracking with tqdm
        - Configurable date range and timeframes
    """

    def __init__(self, config_path: str = None, data_dir: str = None):
        """
        Args:
            config_path: Path to data_sources.yaml. Defaults to config/data_sources.yaml
            data_dir: Override root data directory. Defaults to data/raw/
        """
        # Resolve project root
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Load config
        if config_path is None:
            config_path = self.project_root / "config" / "data_sources.yaml"
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = self.project_root / "data" / "raw"

        # Extract settings
        binance_cfg = self.config["binance"]
        self.base_url = binance_cfg["base_url"]
        self.checksum_suffix = binance_cfg["checksum_suffix"]

        dl_cfg = self.config["download"]
        self.max_concurrent = dl_cfg["max_concurrent"]
        self.retry_attempts = dl_cfg["retry_attempts"]
        self.retry_delay = dl_cfg["retry_delay_seconds"]
        self.timeout = dl_cfg["timeout_seconds"]
        self.verify_checksum = dl_cfg["verify_checksum"]

        self.symbols = self.config["symbols"]
        self.timeframes = self.config["timeframes"]

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AdaptiveRiskSystem/0.1"
        })

    def _get_date_range(self) -> list:
        """Generate date strings based on config."""
        dr = self.config["date_range"]
        start = dr.get("start")
        end = dr.get("end")

        today = datetime.now(timezone.utc).date()
        if end is None:
            end_date = today - timedelta(days=1)
        else:
            end_date = datetime.strptime(str(end), "%Y-%m-%d").date()

        if start is None:
            start_date = end_date - timedelta(days=365)  # 1 year
        else:
            start_date = datetime.strptime(str(start), "%Y-%m-%d").date()

        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        return dates

    def _build_url(self, symbol: str, timeframe: str, date_str: str) -> str:
        """Build the download URL for a single daily file."""
        filename = f"{symbol}-{timeframe}-{date_str}.zip"
        return f"{self.base_url}/{symbol}/{timeframe}/{filename}"

    def _build_checksum_url(self, symbol: str, timeframe: str,
                             date_str: str) -> str:
        """Build the checksum URL for a single daily file."""
        filename = f"{symbol}-{timeframe}-{date_str}.zip{self.checksum_suffix}"
        return f"{self.base_url}/{symbol}/{timeframe}/{filename}"

    def _get_output_dir(self, symbol: str, timeframe: str) -> Path:
        """Get output directory for a symbol/timeframe pair."""
        out_dir = self.data_dir / symbol / timeframe
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _download_single(self, symbol: str, timeframe: str,
                          date_str: str) -> dict:
        """
        Download a single daily kline ZIP file.

        Returns:
            dict with keys: date, status, path, message
        """
        out_dir = self._get_output_dir(symbol, timeframe)
        zip_filename = f"{symbol}-{timeframe}-{date_str}.zip"
        csv_filename = f"{symbol}-{timeframe}-{date_str}.csv"
        zip_path = out_dir / zip_filename
        csv_path = out_dir / csv_filename

        # Skip if CSV already extracted
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return {
                "date": date_str, "status": "skipped",
                "path": str(csv_path), "message": "already exists"
            }

        url = self._build_url(symbol, timeframe, date_str)

        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)

                if resp.status_code == 404:
                    return {
                        "date": date_str, "status": "not_found",
                        "path": None,
                        "message": "file not available on Binance Vision"
                    }

                resp.raise_for_status()

                # Write ZIP
                with open(zip_path, "wb") as f:
                    f.write(resp.content)

                # Verify checksum if enabled
                if self.verify_checksum:
                    if not self._verify_checksum(symbol, timeframe,
                                                  date_str, zip_path):
                        zip_path.unlink(missing_ok=True)
                        if attempt == self.retry_attempts:
                            return {
                                "date": date_str, "status": "checksum_fail",
                                "path": None,
                                "message": "checksum mismatch after retries"
                            }
                        continue

                # Extract CSV from ZIP
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(out_dir)
                except zipfile.BadZipFile:
                    try:
                        zip_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    if attempt == self.retry_attempts:
                        return {
                            "date": date_str, "status": "bad_zip",
                            "path": None, "message": "corrupt ZIP file"
                        }
                    continue

                # Clean up ZIP after successful extraction
                try:
                    zip_path.unlink(missing_ok=True)
                except OSError:
                    pass

                return {
                    "date": date_str, "status": "downloaded",
                    "path": str(csv_path), "message": "success"
                }

            except requests.exceptions.RequestException as e:
                if attempt == self.retry_attempts:
                    return {
                        "date": date_str, "status": "error",
                        "path": None, "message": str(e)
                    }
                import time
                time.sleep(self.retry_delay * attempt)

        return {
            "date": date_str, "status": "error",
            "path": None, "message": "exhausted retries"
        }

    def _verify_checksum(self, symbol: str, timeframe: str,
                          date_str: str, zip_path: Path) -> bool:
        """Verify SHA256 checksum of downloaded ZIP."""
        checksum_url = self._build_checksum_url(symbol, timeframe, date_str)

        try:
            resp = self.session.get(checksum_url, timeout=self.timeout)
            if resp.status_code != 200:
                # No checksum file available — skip verification
                logger.debug(f"No checksum for {date_str}, skipping")
                return True

            # Checksum file format: "sha256_hash  filename"
            expected_hash = resp.text.strip().split()[0].lower()

            # Compute actual hash
            sha256 = hashlib.sha256()
            with open(zip_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest().lower()

            if actual_hash != expected_hash:
                logger.warning(
                    f"Checksum mismatch for {date_str}: "
                    f"expected={expected_hash[:16]}... "
                    f"actual={actual_hash[:16]}..."
                )
                return False

            return True

        except requests.exceptions.RequestException:
            # Network issue fetching checksum — pass
            return True

    def download_timeframe(self, symbol: str, timeframe: str,
                            dates: list = None) -> dict:
        """
        Download all daily files for a single symbol/timeframe.

        Args:
            symbol: e.g. "BTCUSDT"
            timeframe: e.g. "15m"
            dates: list of date strings. If None, uses config range.

        Returns:
            Summary dict with counts by status.
        """
        if dates is None:
            dates = self._get_date_range()

        logger.info(
            f"Downloading {symbol}/{timeframe}: "
            f"{len(dates)} days ({dates[0]} -> {dates[-1]})"
        )

        results = {"downloaded": 0, "skipped": 0, "not_found": 0, "error": 0}
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {
                executor.submit(
                    self._download_single, symbol, timeframe, d
                ): d for d in dates
            }

            pbar = tqdm(
                as_completed(futures), total=len(futures),
                desc=f"  {timeframe}", unit="file", leave=True
            )

            for future in pbar:
                result = future.result()
                status = result["status"]

                if status in results:
                    results[status] += 1
                else:
                    results["error"] += 1

                if status == "error":
                    errors.append(result)

                # Update progress bar
                pbar.set_postfix(
                    ok=results["downloaded"],
                    skip=results["skipped"],
                    miss=results["not_found"],
                    err=results["error"]
                )

        if errors:
            logger.warning(f"  {len(errors)} errors for {timeframe}:")
            for e in errors[:5]:
                logger.warning(f"    {e['date']}: {e['message']}")

        return results

    def download_all(self) -> dict:
        """
        Download all configured symbols and timeframes.

        Returns:
            Nested dict: {symbol: {timeframe: results_dict}}
        """
        dates = self._get_date_range()

        print("=" * 60)
        print("BINANCE VISION DATA DOWNLOADER")
        print("=" * 60)
        print(f"  Symbols:    {', '.join(self.symbols)}")
        print(f"  Timeframes: {', '.join(self.timeframes)}")
        print(f"  Date range: {dates[0]} -> {dates[-1]} ({len(dates)} days)")
        print(f"  Output:     {self.data_dir}")
        print(f"  Workers:    {self.max_concurrent}")
        print("=" * 60)

        all_results = {}

        for symbol in self.symbols:
            print(f"\n{'-' * 40}")
            print(f"Symbol: {symbol}")
            print(f"{'-' * 40}")

            symbol_results = {}
            for tf in self.timeframes:
                res = self.download_timeframe(symbol, tf, dates)
                symbol_results[tf] = res

            all_results[symbol] = symbol_results

        # Print summary
        print(f"\n{'=' * 60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'=' * 60}")
        for symbol, tf_results in all_results.items():
            print(f"\n  {symbol}:")
            for tf, res in tf_results.items():
                total = sum(res.values())
                print(
                    f"    {tf:>4s}: {res['downloaded']:>4d} downloaded | "
                    f"{res['skipped']:>4d} skipped | "
                    f"{res['not_found']:>4d} not found | "
                    f"{res['error']:>4d} errors  "
                    f"({total} total)"
                )

        return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    downloader = BinanceDownloader()
    results = downloader.download_all()
