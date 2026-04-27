"""Preprocess raw OHLCV CSV files into model-ready feature datasets.

Example:
    from src.data.preprocess import preprocess_multiple_tickers

    preprocess_multiple_tickers(
        tickers=["SPY", "QQQ", "GLD"],
        normalize=True,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

PathLike = str | Path
MissingDataStrategy = Literal["drop", "ffill"]

DEFAULT_RAW_DATA_DIR = Path("data") / "raw"
DEFAULT_PROCESSED_DATA_DIR = Path("data") / "processed"


def _ensure_directory(directory: PathLike) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_raw_csv(ticker: str, raw_data_dir: PathLike) -> pd.DataFrame:
    """Load a raw CSV file for a ticker into a DataFrame."""

    input_path = Path(raw_data_dir) / f"{ticker}.csv"
    data = pd.read_csv(input_path, index_col=0, parse_dates=True)
    data.index.name = "Date"
    return data.sort_index()


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


def _generate_return_features(
    data: pd.DataFrame,
    price_column: str,
) -> pd.DataFrame:
    """Add percentage and log returns derived from a price column."""

    if price_column not in data.columns:
        raise KeyError(f"Price column '{price_column}' was not found in the raw data.")

    processed = data.copy()
    processed["pct_return"] = processed[price_column].pct_change()
    processed["log_return"] = np.log(processed[price_column] / processed[price_column].shift(1))
    return processed


def _normalize_numeric_columns(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    suffix: str = "_zscore",
) -> pd.DataFrame:
    """Append z-score normalized versions of numeric columns."""

    processed = data.copy()
    target_columns = list(columns) if columns is not None else processed.select_dtypes(include=["number"]).columns.tolist()

    for column in target_columns:
        std = processed[column].std(ddof=0)
        if std == 0 or pd.isna(std):
            processed[f"{column}{suffix}"] = 0.0
            continue
        mean = processed[column].mean()
        processed[f"{column}{suffix}"] = (processed[column] - mean) / std

    return processed


def _save_processed_csv(data: pd.DataFrame, output_path: PathLike) -> None:
    """Persist a processed DataFrame to CSV after ensuring the directory exists."""

    output_file = Path(output_path)
    _ensure_directory(output_file.parent)
    data.to_csv(output_file)


def preprocess_single_ticker(
    ticker: str,
    raw_data_dir: PathLike = DEFAULT_RAW_DATA_DIR,
    processed_data_dir: PathLike = DEFAULT_PROCESSED_DATA_DIR,
    price_column: str = "Close",
    missing_data_strategy: MissingDataStrategy = "ffill",
    normalize: bool = False,
    normalization_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load, preprocess, and save a single ticker dataset.

    Args:
        ticker: Market symbol whose raw CSV should be processed.
        raw_data_dir: Directory containing raw ticker CSV files.
        processed_data_dir: Output directory for processed CSV files.
        price_column: Column used to compute return features.
        missing_data_strategy: Missing-value handling method.
        normalize: Whether to append normalized numeric features.
        normalization_columns: Optional subset of columns to normalize.

    Returns:
        A processed DataFrame indexed by timestamp.
    """

    raw_data = _load_raw_csv(ticker=ticker, raw_data_dir=raw_data_dir)
    processed = _generate_return_features(data=raw_data, price_column=price_column)
    processed = _apply_missing_data_strategy(processed, missing_data_strategy)

    if normalize:
        processed = _normalize_numeric_columns(
            data=processed,
            columns=normalization_columns,
        )

    output_path = Path(processed_data_dir) / f"{ticker}.csv"
    _save_processed_csv(processed, output_path)
    return processed


def preprocess_multiple_tickers(
    tickers: Sequence[str],
    raw_data_dir: PathLike = DEFAULT_RAW_DATA_DIR,
    processed_data_dir: PathLike = DEFAULT_PROCESSED_DATA_DIR,
    price_column: str = "Close",
    missing_data_strategy: MissingDataStrategy = "ffill",
    normalize: bool = False,
    normalization_columns: Sequence[str] | None = None,
    skip_failed: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load, preprocess, and save multiple ticker datasets.

    Args:
        tickers: Sequence of market symbols whose raw CSV files should be processed.
        raw_data_dir: Directory containing raw ticker CSV files.
        processed_data_dir: Output directory for processed CSV files.
        price_column: Column used to compute return features.
        missing_data_strategy: Missing-value handling method.
        normalize: Whether to append normalized numeric features.
        normalization_columns: Optional subset of columns to normalize.
        skip_failed: Whether to continue processing when a ticker fails.

    Returns:
        A mapping from ticker symbols to processed DataFrames.
    """

    processed_data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            processed_data[ticker] = preprocess_single_ticker(
                ticker=ticker,
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
                price_column=price_column,
                missing_data_strategy=missing_data_strategy,
                normalize=normalize,
                normalization_columns=normalization_columns,
            )
        except Exception:
            if not skip_failed:
                raise

    return processed_data
