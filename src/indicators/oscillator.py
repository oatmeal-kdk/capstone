"""Paper-based oscillator indicators for the ESN + GA project MVP.

The paper's oscillator coverage for this module is limited to the
Stochastic Oscillator system. Signal generation based on %K/%D crossover
conditions is intentionally excluded and will be implemented separately.
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


def _validate_window(window: int, name: str = "window") -> None:
    """Validate that a window parameter is a positive integer.

    Args:
        window: Window length to validate.
        name: Parameter name used in error messages.

    Raises:
        TypeError: If ``window`` is not an integer.
        ValueError: If ``window`` is not positive.
    """

    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError(f"{name} must be a positive integer.")

    if window <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def stochastic(
    df: pd.DataFrame,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> tuple[pd.Series, pd.Series]:
    """Compute the Stochastic Oscillator %K and %D series.

    Args:
        df: Input OHLCV DataFrame.
        window: Rolling lookback window for the highest high and lowest low.
        smooth_k: Smoothing window applied to raw %K values.
        smooth_d: Smoothing window applied to %K values to produce %D.
        high_col: Name of the high-price column.
        low_col: Name of the low-price column.
        close_col: Name of the close-price column.

    Returns:
        A tuple containing the smoothed %K series and the %D series.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame or window parameters
            are not positive integers.
        KeyError: If any required price column is missing.
        ValueError: If window parameters are not positive integers.
    """

    _validate_columns(df, [high_col, low_col, close_col])
    _validate_window(window, name="window")
    _validate_window(smooth_k, name="smooth_k")
    _validate_window(smooth_d, name="smooth_d")

    lowest_low = df[low_col].rolling(window=window, min_periods=window).min()
    highest_high = df[high_col].rolling(window=window, min_periods=window).max()

    denominator = highest_high - lowest_low
    safe_denominator = denominator.where(denominator != 0.0, np.nan)

    k_raw = df[close_col].sub(lowest_low).div(safe_denominator) * 100.0
    k = k_raw.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()

    return (
        k.rename(f"stoch_k_{window}_{smooth_k}"),
        d.rename(f"stoch_d_{window}_{smooth_k}_{smooth_d}"),
    )
