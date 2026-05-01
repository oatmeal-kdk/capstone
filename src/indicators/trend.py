"""Paper-based trend indicators for the ESN + GA project MVP.

The paper's Moving Average system uses the Simple Moving Average (SMA).
Additional moving average variants are stored in
``src/indicators/experimental/trend_extra.py`` for future ablation studies
or extended quant experiments.
"""

from __future__ import annotations

import pandas as pd


def _validate_price_column(df: pd.DataFrame, price_col: str) -> None:
    """Validate that the input DataFrame contains the requested price column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' was not found in the DataFrame.")


def _validate_window(window: int) -> None:
    """Validate that a rolling window is a positive integer."""

    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be a positive integer.")

    if window <= 0:
        raise ValueError("window must be a positive integer.")


def sma(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.Series:
    """Compute the simple moving average for a price series.

    Args:
        df: Input OHLCV DataFrame.
        window: Rolling lookback window.
        price_col: Name of the price column to average.

    Returns:
        A pandas Series containing the simple moving average values.
    """

    _validate_price_column(df, price_col)
    _validate_window(window)

    result = df[price_col].rolling(window=window, min_periods=window).mean()
    return result.rename(f"sma_{window}")

