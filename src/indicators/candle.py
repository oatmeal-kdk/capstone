"""Paper-based candle chart patterns for the ESN + GA project MVP.

This module implements only the candle pattern occurrence values required by
the paper's candle chart system. Final trading interpretation, trend context,
turning point logic, GA optimization, and ESN logic are intentionally
excluded and will be implemented separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Validate that the input DataFrame contains the required columns.

    Args:
        df: Input OHLCV DataFrame.
        required_cols: Column names that must be present in ``df``.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        KeyError: If any required column is missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        missing_str = ", ".join(f"'{col}'" for col in missing_cols)
        raise KeyError(f"Required column(s) not found in the DataFrame: {missing_str}.")


def _validate_positive_number(value: float, name: str) -> None:
    """Validate that a numeric parameter is strictly positive.

    Args:
        value: Numeric value to validate.
        name: Parameter name used in error messages.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not strictly positive.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a positive number.")

    if value <= 0:
        raise ValueError(f"{name} must be a positive number.")


def _body(df: pd.DataFrame, open_col: str, close_col: str) -> pd.Series:
    """Compute the absolute candle body size."""

    return df[close_col].sub(df[open_col]).abs()


def _upper_shadow(
    df: pd.DataFrame,
    open_col: str,
    high_col: str,
    close_col: str,
) -> pd.Series:
    """Compute the upper shadow size for each candle."""

    return df[high_col].sub(df[[open_col, close_col]].max(axis=1))


def _lower_shadow(
    df: pd.DataFrame,
    open_col: str,
    low_col: str,
    close_col: str,
) -> pd.Series:
    """Compute the lower shadow size for each candle."""

    return df[[open_col, close_col]].min(axis=1).sub(df[low_col])


def _candle_range(df: pd.DataFrame, high_col: str, low_col: str) -> pd.Series:
    """Compute the full candle range from high to low."""

    return df[high_col].sub(df[low_col])


def hammer_hanging_man(
    df: pd.DataFrame,
    shadow_body_ratio: float = 2.0,
    upper_shadow_ratio: float = 0.5,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.Series:
    """Detect hammer-like or hanging-man-like candle shapes.

    This function identifies only the single-candle shape. Trend context and
    final reversal interpretation are intentionally excluded.

    Args:
        df: Input OHLCV DataFrame.
        shadow_body_ratio: Minimum lower-shadow-to-body ratio.
        upper_shadow_ratio: Maximum upper-shadow-to-body ratio.
        open_col: Name of the open-price column.
        high_col: Name of the high-price column.
        low_col: Name of the low-price column.
        close_col: Name of the close-price column.

    Returns:
        A pandas Series containing ``1`` for bullish hammer-like candles,
        ``-1`` for bearish hammer-like candles, and ``0`` otherwise.
    """

    _validate_columns(df, [open_col, high_col, low_col, close_col])
    _validate_positive_number(shadow_body_ratio, "shadow_body_ratio")
    _validate_positive_number(upper_shadow_ratio, "upper_shadow_ratio")

    body = _body(df, open_col, close_col)
    upper_shadow = _upper_shadow(df, open_col, high_col, close_col)
    lower_shadow = _lower_shadow(df, open_col, low_col, close_col)

    is_hammer_like = (
        (body > 0)
        & (lower_shadow >= shadow_body_ratio * body)
        & (upper_shadow <= upper_shadow_ratio * body)
    )

    result = pd.Series(0, index=df.index, dtype=np.int64)
    result = result.mask(is_hammer_like & (df[close_col] >= df[open_col]), 1)
    result = result.mask(is_hammer_like & (df[close_col] < df[open_col]), -1)

    return result.rename("hammer_hanging_man")


def dark_cloud_cover(
    df: pd.DataFrame,
    penetration_ratio: float = 0.5,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.Series:
    """Detect bearish dark cloud cover patterns.

    Args:
        df: Input OHLCV DataFrame.
        penetration_ratio: Required penetration into the previous candle body.
        open_col: Name of the open-price column.
        high_col: Name of the high-price column.
        low_col: Name of the low-price column.
        close_col: Name of the close-price column.

    Returns:
        A pandas Series containing ``-1`` for dark cloud cover and ``0``
        otherwise.
    """

    _validate_columns(df, [open_col, high_col, low_col, close_col])
    _validate_positive_number(penetration_ratio, "penetration_ratio")

    prev_open = df[open_col].shift(1)
    prev_close = df[close_col].shift(1)
    prev_body = prev_close.sub(prev_open).abs()

    is_pattern = (
        (prev_close > prev_open)
        & (df[close_col] < df[open_col])
        & (df[open_col] > prev_close)
        & (df[close_col] < prev_close - penetration_ratio * prev_body)
        & (df[close_col] > prev_open)
    )

    result = pd.Series(0, index=df.index, dtype=np.int64)
    result = result.mask(is_pattern, -1)

    return result.rename("dark_cloud_cover")


def piercing_line(
    df: pd.DataFrame,
    penetration_ratio: float = 0.5,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.Series:
    """Detect bullish piercing line patterns.

    Args:
        df: Input OHLCV DataFrame.
        penetration_ratio: Required penetration into the previous candle body.
        open_col: Name of the open-price column.
        high_col: Name of the high-price column.
        low_col: Name of the low-price column.
        close_col: Name of the close-price column.

    Returns:
        A pandas Series containing ``1`` for piercing line and ``0`` otherwise.
    """

    _validate_columns(df, [open_col, high_col, low_col, close_col])
    _validate_positive_number(penetration_ratio, "penetration_ratio")

    prev_open = df[open_col].shift(1)
    prev_close = df[close_col].shift(1)
    prev_body = prev_open.sub(prev_close).abs()

    is_pattern = (
        (prev_close < prev_open)
        & (df[close_col] > df[open_col])
        & (df[open_col] < prev_close)
        & (df[close_col] > prev_close + penetration_ratio * prev_body)
        & (df[close_col] < prev_open)
    )

    result = pd.Series(0, index=df.index, dtype=np.int64)
    result = result.mask(is_pattern, 1)

    return result.rename("piercing_line")


def engulfing_pattern(
    df: pd.DataFrame,
    min_body_ratio: float = 1.0,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.Series:
    """Detect bullish and bearish engulfing patterns.

    Args:
        df: Input OHLCV DataFrame.
        min_body_ratio: Minimum ratio of current body to previous body.
        open_col: Name of the open-price column.
        high_col: Name of the high-price column.
        low_col: Name of the low-price column.
        close_col: Name of the close-price column.

    Returns:
        A pandas Series containing ``1`` for bullish engulfing, ``-1`` for
        bearish engulfing, and ``0`` otherwise.
    """

    _validate_columns(df, [open_col, high_col, low_col, close_col])
    _validate_positive_number(min_body_ratio, "min_body_ratio")

    prev_open = df[open_col].shift(1)
    prev_close = df[close_col].shift(1)
    prev_body = prev_close.sub(prev_open).abs()
    curr_body = _body(df, open_col, close_col)

    bullish = (
        (prev_close < prev_open)
        & (df[close_col] > df[open_col])
        & (df[open_col] <= prev_close)
        & (df[close_col] >= prev_open)
        & (curr_body >= min_body_ratio * prev_body)
    )

    bearish = (
        (prev_close > prev_open)
        & (df[close_col] < df[open_col])
        & (df[open_col] >= prev_close)
        & (df[close_col] <= prev_open)
        & (curr_body >= min_body_ratio * prev_body)
    )

    result = pd.Series(0, index=df.index, dtype=np.int64)
    result = result.mask(bullish, 1)
    result = result.mask(bearish, -1)

    return result.rename("engulfing_pattern")


def candle_patterns(
    df: pd.DataFrame,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Return all paper-required candle pattern series in one DataFrame.

    Args:
        df: Input OHLCV DataFrame.
        open_col: Name of the open-price column.
        high_col: Name of the high-price column.
        low_col: Name of the low-price column.
        close_col: Name of the close-price column.

    Returns:
        A new DataFrame containing the required candle pattern occurrence
        series, indexed to match the input DataFrame.
    """

    patterns = [
        hammer_hanging_man(
            df,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
        ),
        dark_cloud_cover(
            df,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
        ),
        piercing_line(
            df,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
        ),
        engulfing_pattern(
            df,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
        ),
    ]

    return pd.concat(patterns, axis=1)
