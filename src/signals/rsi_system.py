"""RSI signal system for the ESN + GA project MVP.

This module generates RSI-based buy, sell, and hold signals only. It does not
implement turning points, GA optimization, ESN logic, or backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.momentum import rsi


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


def rsi_signal(
    df: pd.DataFrame,
    window: int = 14,
    overbought: float = 80.0,
    oversold: float = 20.0,
    sell_slope: float = 0.0,
    buy_slope: float = 0.0,
    price_col: str = "Close",
) -> pd.Series:
    """Generate RSI signals using threshold-triggered reference lines.

    When RSI first moves above ``overbought`` or below ``oversold``, the
    function starts tracking a configurable linear reference line and emits a
    signal only when RSI crosses back through that reference.

    Args:
        df: Input price DataFrame.
        window: Lookback window for RSI.
        overbought: Upper threshold used to start sell tracking.
        oversold: Lower threshold used to start buy tracking.
        sell_slope: Per-step slope of the sell reference line.
        buy_slope: Per-step slope of the buy reference line.
        price_col: Price column used for RSI.

    Returns:
        A signal Series containing ``1`` for buy, ``-1`` for sell, and ``0``
        for hold.
    """

    _validate_columns(df, [price_col])
    _validate_positive_int(window, "window")
    _validate_numeric(overbought, "overbought")
    _validate_numeric(oversold, "oversold")
    _validate_numeric(sell_slope, "sell_slope")
    _validate_numeric(buy_slope, "buy_slope")

    if not 0 < oversold < overbought < 100:
        raise ValueError("Parameters must satisfy 0 < oversold < overbought < 100.")

    rsi_values = rsi(df, window=window, price_col=price_col)
    values = rsi_values.to_numpy(dtype=float)

    signal = pd.Series(0, index=df.index, dtype=np.int64)

    sell_active = False
    sell_start = 0.0
    sell_elapsed = 0

    buy_active = False
    buy_start = 0.0
    buy_elapsed = 0

    for i, current_value in enumerate(values):
        previous_value = values[i - 1] if i > 0 else np.nan

        if np.isnan(current_value):
            continue

        if (
            not sell_active
            and i > 0
            and not np.isnan(previous_value)
            and previous_value <= overbought
            and current_value > overbought
        ):
            sell_active = True
            sell_start = float(current_value)
            sell_elapsed = 0

        if (
            not buy_active
            and i > 0
            and not np.isnan(previous_value)
            and previous_value >= oversold
            and current_value < oversold
        ):
            buy_active = True
            buy_start = float(current_value)
            buy_elapsed = 0

        sell_trigger = False
        if sell_active:
            sell_reference = sell_start + sell_slope * sell_elapsed
            sell_trigger = current_value < sell_reference

        buy_trigger = False
        if buy_active:
            buy_reference = buy_start + buy_slope * buy_elapsed
            buy_trigger = current_value > buy_reference

        if buy_trigger and sell_trigger:
            signal.iat[i] = 0
            buy_active = False
            sell_active = False
            buy_elapsed = 0
            sell_elapsed = 0
            continue

        if sell_trigger:
            signal.iat[i] = -1
            sell_active = False
            sell_elapsed = 0

        if buy_trigger:
            signal.iat[i] = 1
            buy_active = False
            buy_elapsed = 0

        if sell_active:
            sell_elapsed += 1

        if buy_active:
            buy_elapsed += 1

    return signal.rename("rsi_signal")


__all__ = ["rsi_signal"]
