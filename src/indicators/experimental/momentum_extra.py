"""Experimental momentum indicators excluded from the paper-based MVP.

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


def _validate_window(window: int, name: str = "window") -> None:
    """Validate that a window parameter is a positive integer."""

    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError(f"{name} must be a positive integer.")

    if window <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def momentum(df: pd.DataFrame, window: int = 10, price_col: str = "Close") -> pd.Series:
    """Compute momentum for a price series.

    Args:
        df: Input OHLCV DataFrame.
        window: Lookback window used in the momentum difference.
        price_col: Name of the price column to evaluate.

    Returns:
        A pandas Series containing the momentum values.
    """

    _validate_price_column(df, price_col)
    _validate_window(window)

    result = df[price_col] - df[price_col].shift(window)
    return result.rename(f"momentum_{window}")


def macd(
    df: pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    price_col: str = "Close",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the MACD line, signal line, and histogram.

    Args:
        df: Input OHLCV DataFrame.
        fast_window: Span for the fast EMA.
        slow_window: Span for the slow EMA.
        signal_window: Span for the MACD signal EMA.
        price_col: Name of the price column to evaluate.

    Returns:
        A tuple containing the MACD line, signal line, and histogram.
    """

    _validate_price_column(df, price_col)
    _validate_window(fast_window, name="fast_window")
    _validate_window(slow_window, name="slow_window")
    _validate_window(signal_window, name="signal_window")

    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window.")

    fast_ema = df[price_col].ewm(span=fast_window, adjust=False).mean()
    slow_ema = df[price_col].ewm(span=slow_window, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    return (
        macd_line.rename(f"macd_{fast_window}_{slow_window}"),
        signal_line.rename(f"macd_signal_{signal_window}"),
        histogram.rename(f"macd_hist_{fast_window}_{slow_window}_{signal_window}"),
    )


def ppo(
    df: pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    price_col: str = "Close",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the PPO line, signal line, and histogram.

    Args:
        df: Input OHLCV DataFrame.
        fast_window: Span for the fast EMA.
        slow_window: Span for the slow EMA.
        signal_window: Span for the PPO signal EMA.
        price_col: Name of the price column to evaluate.

    Returns:
        A tuple containing the PPO line, signal line, and histogram.
    """

    _validate_price_column(df, price_col)
    _validate_window(fast_window, name="fast_window")
    _validate_window(slow_window, name="slow_window")
    _validate_window(signal_window, name="signal_window")

    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window.")

    fast_ema = df[price_col].ewm(span=fast_window, adjust=False).mean()
    slow_ema = df[price_col].ewm(span=slow_window, adjust=False).mean()

    ppo_line = (fast_ema - slow_ema).div(slow_ema.replace(0.0, np.nan)) * 100.0
    signal_line = ppo_line.ewm(span=signal_window, adjust=False).mean()
    histogram = ppo_line - signal_line

    return (
        ppo_line.rename(f"ppo_{fast_window}_{slow_window}"),
        signal_line.rename(f"ppo_signal_{signal_window}"),
        histogram.rename(f"ppo_hist_{fast_window}_{slow_window}_{signal_window}"),
    )
