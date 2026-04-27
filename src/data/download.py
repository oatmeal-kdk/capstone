"""Download raw OHLCV market data with yfinance.

Example:
    from src.data.download import download_multiple_tickers

    download_multiple_tickers(
        tickers=["SPY", "QQQ", "GLD"],
        start_date="2010-01-01",
        end_date="2024-12-31",
        interval="1d",
    )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import yfinance as yf

PathLike = str | Path
MissingDataStrategy = Literal["drop", "ffill"]

DEFAULT_RAW_DATA_DIR = Path("data") / "raw"
logger = logging.getLogger(__name__)


def _ensure_directory(directory: PathLike) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a DataFrame with flattened yfinance column labels."""

    flattened = data.copy()

    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            str(level_0)
            if str(level_0) != "Price"
            else str(level_1)
            for level_0, level_1 in flattened.columns.to_flat_index()
        ]

    return flattened


def _apply_missing_data_strategy(
    data: pd.DataFrame,
    strategy: MissingDataStrategy,
) -> pd.DataFrame:
    """Handle missing values according to the requested strategy."""

    if strategy == "ffill":
        return data.ffill().dropna(how="any")

    if strategy == "drop":
        return data.dropna(how="any")

    raise ValueError(f"Unsupported missing data strategy: {strategy}")


def _save_dataframe(data: pd.DataFrame, output_path: PathLike) -> None:
    """Persist a DataFrame to CSV after ensuring the parent directory exists."""

    output_file = Path(output_path)
    _ensure_directory(output_file.parent)
    data.to_csv(output_file)


def download_single_ticker(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
    raw_data_dir: PathLike = DEFAULT_RAW_DATA_DIR,
    missing_data_strategy: MissingDataStrategy = "drop",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    auto_adjust: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """Download, clean, and save raw OHLCV data for a single ticker.

    Args:
        ticker: Market symbol understood by yfinance.
        start_date: Inclusive start date in a yfinance-compatible format.
        end_date: Exclusive end date in a yfinance-compatible format.
        interval: Sampling interval such as ``"1d"`` or ``"1h"``.
        raw_data_dir: Output directory for raw CSV files.
        missing_data_strategy: Missing-value handling method.
        max_retries: Number of retry attempts for download failures.
        retry_delay: Delay in seconds between retry attempts.
        auto_adjust: Whether to let yfinance auto-adjust prices.
        progress: Whether to show yfinance progress output.

    Returns:
        A cleaned OHLCV DataFrame indexed by timestamp.

    Raises:
        RuntimeError: If the download fails after all retry attempts.
        ValueError: If the downloaded dataset is empty.
    """

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=progress,
                auto_adjust=auto_adjust,
                threads=False,
                group_by="column",
            )

            if data.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'.")

            cleaned = _flatten_columns(data)
            cleaned = cleaned.sort_index()
            cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
            cleaned.index.name = "Date"
            cleaned = _apply_missing_data_strategy(cleaned, missing_data_strategy)

            output_path = Path(raw_data_dir) / f"{ticker}.csv"
            _save_dataframe(cleaned, output_path)
            logger.info("Saved raw data for %s to %s", ticker, output_path)
            return cleaned

        except Exception as error:
            last_error = error
            logger.warning(
                "Download attempt %s/%s failed for %s: %s",
                attempt,
                max_retries,
                ticker,
                error,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Failed to download data for ticker '{ticker}' after {max_retries} attempts."
    ) from last_error


def download_multiple_tickers(
    tickers: Sequence[str],
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
    raw_data_dir: PathLike = DEFAULT_RAW_DATA_DIR,
    missing_data_strategy: MissingDataStrategy = "drop",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    auto_adjust: bool = False,
    progress: bool = False,
    skip_failed: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download, clean, and save raw OHLCV data for multiple tickers.

    Args:
        tickers: Sequence of market symbols understood by yfinance.
        start_date: Inclusive start date in a yfinance-compatible format.
        end_date: Exclusive end date in a yfinance-compatible format.
        interval: Sampling interval such as ``"1d"`` or ``"1h"``.
        raw_data_dir: Output directory for raw CSV files.
        missing_data_strategy: Missing-value handling method.
        max_retries: Number of retry attempts for each ticker.
        retry_delay: Delay in seconds between retry attempts.
        auto_adjust: Whether to let yfinance auto-adjust prices.
        progress: Whether to show yfinance progress output.
        skip_failed: Whether to continue processing when a ticker fails.

    Returns:
        A mapping from ticker symbols to downloaded DataFrames.

    Raises:
        RuntimeError: If a download fails and ``skip_failed`` is ``False``.
    """

    downloaded_data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            downloaded_data[ticker] = download_single_ticker(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                raw_data_dir=raw_data_dir,
                missing_data_strategy=missing_data_strategy,
                max_retries=max_retries,
                retry_delay=retry_delay,
                auto_adjust=auto_adjust,
                progress=progress,
            )
        except Exception:
            if not skip_failed:
                raise
            logger.exception("Skipping ticker %s after repeated download failures.", ticker)

    return downloaded_data
