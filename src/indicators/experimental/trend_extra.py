"""Experimental trend indicators excluded from the paper-based MVP.

These indicators are not used in the paper-based MVP pipeline. They are kept
for future ablation studies or extended quant experiments and should not be
imported by the default indicator pipeline unless explicitly needed.
"""

from __future__ import annotations

import numpy as np
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


def ema(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.Series:
    """Compute the exponential moving average for a price series.

    Args:
        df: Input OHLCV DataFrame.
        window: Exponential averaging span.
        price_col: Name of the price column to average.

    Returns:
        A pandas Series containing the exponential moving average values.
    """

    _validate_price_column(df, price_col)
    _validate_window(window)

    result = df[price_col].ewm(span=window, adjust=False).mean()
    return result.rename(f"ema_{window}")


def wma(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.Series:
    """Compute the weighted moving average for a price series.

    Args:
        df: Input OHLCV DataFrame.
        window: Rolling lookback window.
        price_col: Name of the price column to average.

    Returns:
        A pandas Series containing the weighted moving average values.
    """

    _validate_price_column(df, price_col)
    _validate_window(window)

    weights = np.arange(1, window + 1, dtype=float)
    denominator = weights.sum()

    result = df[price_col].rolling(window=window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights) / denominator),
        raw=True,
    )
    return result.rename(f"wma_{window}")


def dema(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.Series:
    """Compute the double exponential moving average for a price series.

    Args:
        df: Input OHLCV DataFrame.
        window: Exponential averaging span.
        price_col: Name of the price column to average.

    Returns:
        A pandas Series containing the double exponential moving average values.
    """

    _validate_price_column(df, price_col)
    _validate_window(window)

    ema_1 = df[price_col].ewm(span=window, adjust=False).mean()
    ema_2 = ema_1.ewm(span=window, adjust=False).mean()
    result = (2 * ema_1) - ema_2
    return result.rename(f"dema_{window}")


def tema(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.Series:
    """Compute the triple exponential moving average for a price series.

    Args:
        df: Input OHLCV DataFrame.
        window: Exponential averaging span.
        price_col: Name of the price column to average.

    Returns:
        A pandas Series containing the triple exponential moving average values.
    """

    _validate_price_column(df, price_col)
    _validate_window(window)

    ema_1 = df[price_col].ewm(span=window, adjust=False).mean()
    ema_2 = ema_1.ewm(span=window, adjust=False).mean()
    ema_3 = ema_2.ewm(span=window, adjust=False).mean()
    result = (3 * ema_1) - (3 * ema_2) + ema_3
    return result.rename(f"tema_{window}")
