"""Stochastic signal system for the ESN + GA project MVP.

This module generates stochastic-oscillator buy, sell, and hold signals only.
It does not implement turning points, GA optimization, ESN logic, or
backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.oscillator import stochastic


def _validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Validate that the input DataFrame contains the required columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        missing_str = ", ".join(f"'{col}'" for col in missing_cols)
        raise KeyError(f"Required column(s) not found in the DataFrame: {missing_str}.")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_positive_number(value: float, name: str) -> None:
    """Validate that a value is a strictly positive number."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a positive number.")

    if value <= 0:
        raise ValueError(f"{name} must be a positive number.")


def _validate_numeric(value: float, name: str) -> None:
    """Validate that a value is numeric."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a numeric value.")


def stochastic_signal(
    df: pd.DataFrame,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
    buy_k_threshold: float = 30.0,
    buy_diff_threshold: float = 5.0,
    sell_k_threshold: float = 70.0,
    sell_diff_threshold: float = 5.0,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.Series:
    """Generate stochastic crossover signals.

    Args:
        df: Input OHLCV DataFrame.
        window: Lookback window for the stochastic oscillator.
        smooth_k: Smoothing window for %K.
        smooth_d: Smoothing window for %D.
        buy_k_threshold: Upper %K threshold for buy signals.
        buy_diff_threshold: Maximum allowed ``%K - %D`` at a buy signal.
        sell_k_threshold: Lower %K threshold for sell signals.
        sell_diff_threshold: Maximum allowed ``%D - %K`` at a sell signal.
        high_col: High-price column name.
        low_col: Low-price column name.
        close_col: Close-price column name.

    Returns:
        A signal Series containing ``1`` for buy, ``-1`` for sell, and
        ``0`` for hold.
    """

    _validate_columns(df, [high_col, low_col, close_col])
    _validate_positive_int(window, "window")
    _validate_positive_int(smooth_k, "smooth_k")
    _validate_positive_int(smooth_d, "smooth_d")
    _validate_numeric(buy_k_threshold, "buy_k_threshold")
    _validate_numeric(buy_diff_threshold, "buy_diff_threshold")
    _validate_numeric(sell_k_threshold, "sell_k_threshold")
    _validate_numeric(sell_diff_threshold, "sell_diff_threshold")

    if not 0 <= buy_k_threshold <= 100:
        raise ValueError("buy_k_threshold must be between 0 and 100.")

    if not 0 <= sell_k_threshold <= 100:
        raise ValueError("sell_k_threshold must be between 0 and 100.")

    if buy_diff_threshold < 0:
        raise ValueError("buy_diff_threshold must be non-negative.")

    if sell_diff_threshold < 0:
        raise ValueError("sell_diff_threshold must be non-negative.")

    k_series, d_series = stochastic(
        df,
        window=window,
        smooth_k=smooth_k,
        smooth_d=smooth_d,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
    )

    previous_k = k_series.shift(1)
    previous_d = d_series.shift(1)

    bullish_crossover = (previous_k <= previous_d) & (k_series > d_series)
    bearish_crossover = (previous_k >= previous_d) & (k_series < d_series)

    buy_mask = (
        bullish_crossover
        & (k_series < buy_k_threshold)
        & ((k_series - d_series) < buy_diff_threshold)
    )
    sell_mask = (
        bearish_crossover
        & (k_series > sell_k_threshold)
        & ((d_series - k_series) < sell_diff_threshold)
    )

    signal = pd.Series(0, index=df.index, dtype=np.int64)
    signal = signal.mask(buy_mask & ~sell_mask, 1)
    signal = signal.mask(sell_mask & ~buy_mask, -1)

    return signal.rename("stochastic_signal")


__all__ = ["stochastic_signal"]
