"""Candle-pattern signal system for the ESN + GA project MVP.

This module aggregates candle-pattern indicator outputs into rule-based buy,
sell, and hold signals. It does not implement turning points, GA optimization,
ESN logic, or backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.candle import candle_patterns


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


def _validate_bool(value: bool, name: str) -> None:
    """Validate that a value is boolean."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def candle_signal(
    df: pd.DataFrame,
    use_hammer_hanging_man: bool = True,
    use_dark_cloud_cover: bool = True,
    use_piercing_line: bool = True,
    use_engulfing_pattern: bool = True,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.Series:
    """Aggregate selected candle patterns into one signal series.

    Args:
        df: Input OHLCV DataFrame.
        use_hammer_hanging_man: Whether to include hammer/hanging-man patterns.
        use_dark_cloud_cover: Whether to include dark cloud cover patterns.
        use_piercing_line: Whether to include piercing line patterns.
        use_engulfing_pattern: Whether to include engulfing patterns.
        open_col: Open-price column name.
        high_col: High-price column name.
        low_col: Low-price column name.
        close_col: Close-price column name.

    Returns:
        A signal Series containing ``1`` for buy, ``-1`` for sell, and ``0``
        for hold.
    """

    _validate_columns(df, [open_col, high_col, low_col, close_col])
    _validate_bool(use_hammer_hanging_man, "use_hammer_hanging_man")
    _validate_bool(use_dark_cloud_cover, "use_dark_cloud_cover")
    _validate_bool(use_piercing_line, "use_piercing_line")
    _validate_bool(use_engulfing_pattern, "use_engulfing_pattern")

    patterns = candle_patterns(
        df,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
    )

    selected_columns: list[str] = []
    if use_hammer_hanging_man:
        selected_columns.append("hammer_hanging_man")
    if use_dark_cloud_cover:
        selected_columns.append("dark_cloud_cover")
    if use_piercing_line:
        selected_columns.append("piercing_line")
    if use_engulfing_pattern:
        selected_columns.append("engulfing_pattern")

    signal = pd.Series(0, index=df.index, dtype=np.int64)
    if not selected_columns:
        return signal.rename("candle_signal")

    selected_patterns = patterns[selected_columns]
    bullish_mask = (selected_patterns > 0).any(axis=1)
    bearish_mask = (selected_patterns < 0).any(axis=1)

    signal = signal.mask(bullish_mask & ~bearish_mask, 1)
    signal = signal.mask(bearish_mask & ~bullish_mask, -1)

    return signal.rename("candle_signal")


__all__ = ["candle_signal"]
