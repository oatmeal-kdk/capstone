"""ROC signal system for the ESN + GA project MVP.

This module generates ROC-based buy, sell, and hold signals only. It does not
implement turning points, GA optimization, ESN logic, or backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.momentum import roc


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


def roc_signal(
    df: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 25,
    upper_bound: float = 110.0,
    lower_bound: float = 90.0,
    equilibrium: float = 100.0,
    equilibrium_band: float = 2.0,
    price_col: str = "Close",
) -> pd.Series:
    """Generate ROC signals from long-term extremes near equilibrium.

    Args:
        df: Input price DataFrame.
        short_window: Lookback window for the short ROC.
        long_window: Lookback window for the long ROC.
        upper_bound: Long ROC threshold for sell signals.
        lower_bound: Long ROC threshold for buy signals.
        equilibrium: Equilibrium level for short ROC.
        equilibrium_band: Allowed distance from equilibrium for short ROC.
        price_col: Price column used in the ROC calculations.

    Returns:
        A signal Series containing ``1`` for buy, ``-1`` for sell, and ``0``
        for hold.
    """

    _validate_columns(df, [price_col])
    _validate_positive_int(short_window, "short_window")
    _validate_positive_int(long_window, "long_window")
    _validate_numeric(upper_bound, "upper_bound")
    _validate_numeric(lower_bound, "lower_bound")
    _validate_numeric(equilibrium, "equilibrium")
    _validate_numeric(equilibrium_band, "equilibrium_band")

    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window.")

    if not lower_bound < equilibrium < upper_bound:
        raise ValueError("Parameters must satisfy lower_bound < equilibrium < upper_bound.")

    if equilibrium_band < 0:
        raise ValueError("equilibrium_band must be non-negative.")

    short_roc = roc(df, window=short_window, price_col=price_col)
    long_roc = roc(df, window=long_window, price_col=price_col)

    near_equilibrium = short_roc.sub(equilibrium).abs() <= equilibrium_band
    buy_mask = (long_roc <= lower_bound) & near_equilibrium
    sell_mask = (long_roc >= upper_bound) & near_equilibrium

    signal = pd.Series(0, index=df.index, dtype=np.int64)
    signal = signal.mask(buy_mask & ~sell_mask, 1)
    signal = signal.mask(sell_mask & ~buy_mask, -1)

    return signal.rename("roc_signal")


__all__ = ["roc_signal"]
