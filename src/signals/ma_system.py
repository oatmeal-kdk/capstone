"""Moving Average signal systems for the ESN + GA project MVP.

This module converts moving-average indicator values into rule-based buy, sell,
and hold signals. It does not implement turning points, GA optimization, ESN
logic, or backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.trend import sma


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


def ma_cross_signal(
    df: pd.DataFrame,
    short_window: int = 13,
    long_window: int = 26,
    a_buy: float = 2.0,
    b_buy: float = 1.0,
    c_buy: float = 0.01,
    a_sell: float = 2.0,
    b_sell: float = 1.0,
    c_sell: float = 0.01,
    price_col: str = "Close",
) -> pd.Series:
    """Generate paper-inspired moving average cross signals.

    The rule tracks positive and negative ``z = long_ma - short_ma`` regimes
    from their most recent sign crossing and emits signals using only values
    available up to the current index.

    Args:
        df: Input price DataFrame.
        short_window: Lookback window for the short SMA.
        long_window: Lookback window for the long SMA.
        a_buy: Buy-side retracement divisor.
        b_buy: Buy-side amplitude multiplier.
        c_buy: Buy-side ceiling parameter.
        a_sell: Sell-side retracement divisor.
        b_sell: Sell-side amplitude multiplier.
        c_sell: Sell-side ceiling parameter.
        price_col: Price column used for the moving averages.

    Returns:
        A signal Series containing ``1`` for buy, ``-1`` for sell, and ``0``
        for hold.
    """

    _validate_columns(df, [price_col])
    _validate_positive_int(short_window, "short_window")
    _validate_positive_int(long_window, "long_window")
    _validate_positive_number(a_buy, "a_buy")
    _validate_positive_number(b_buy, "b_buy")
    _validate_positive_number(c_buy, "c_buy")
    _validate_positive_number(a_sell, "a_sell")
    _validate_positive_number(b_sell, "b_sell")
    _validate_positive_number(c_sell, "c_sell")

    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window.")

    short_ma = sma(df, short_window, price_col=price_col)
    long_ma = sma(df, long_window, price_col=price_col)
    z = long_ma.sub(short_ma)

    signal = pd.Series(0, index=df.index, dtype=np.int64)

    positive_cross_active = False
    negative_cross_active = False
    max_positive = np.nan
    max_negative = np.nan

    for i in range(len(df)):
        current_z = z.iat[i]
        previous_z = z.iat[i - 1] if i > 0 else np.nan

        if pd.isna(current_z):
            positive_cross_active = False
            negative_cross_active = False
            max_positive = np.nan
            max_negative = np.nan
            continue

        if i > 0 and not pd.isna(previous_z):
            if previous_z <= 0 and current_z > 0:
                positive_cross_active = True
                negative_cross_active = False
                max_positive = float(current_z)
                max_negative = np.nan
            elif previous_z >= 0 and current_z < 0:
                negative_cross_active = True
                positive_cross_active = False
                max_negative = float(-current_z)
                max_positive = np.nan

        buy_signal = False
        if positive_cross_active and current_z >= 0:
            max_positive = max(float(max_positive), float(current_z))
            buy_limit = min(max_positive / a_buy, c_buy)
            buy_signal = max_positive > b_buy * c_buy and float(current_z) < buy_limit

        sell_signal = False
        if negative_cross_active and current_z < 0:
            current_w = float(-current_z)
            max_negative = max(float(max_negative), current_w)
            sell_limit = min(max_negative / a_sell, c_sell)
            sell_signal = max_negative > b_sell * c_sell and current_w < sell_limit

        if buy_signal and sell_signal:
            signal.iat[i] = 0
        elif buy_signal:
            signal.iat[i] = 1
        elif sell_signal:
            signal.iat[i] = -1

    return signal.rename("ma_cross_signal")


def ma_envelope_signal(
    df: pd.DataFrame,
    window: int = 20,
    upper_pct: float = 0.03,
    lower_pct: float = 0.03,
    price_col: str = "Close",
) -> pd.Series:
    """Generate moving-average envelope crossover signals.

    Args:
        df: Input price DataFrame.
        window: Lookback window for the envelope SMA.
        upper_pct: Percentage distance above the SMA for the upper band.
        lower_pct: Percentage distance below the SMA for the lower band.
        price_col: Price column used in the envelope calculation.

    Returns:
        A signal Series containing ``1`` for buy, ``-1`` for sell, and ``0``
        for hold.
    """

    _validate_columns(df, [price_col])
    _validate_positive_int(window, "window")
    _validate_positive_number(upper_pct, "upper_pct")
    _validate_positive_number(lower_pct, "lower_pct")

    ma = sma(df, window, price_col=price_col)
    upper = ma * (1.0 + upper_pct)
    lower = ma * (1.0 - lower_pct)

    price = df[price_col]
    previous_price = price.shift(1)
    previous_upper = upper.shift(1)
    previous_lower = lower.shift(1)

    buy_mask = (previous_price <= previous_upper) & (price > upper)
    sell_mask = (previous_price >= previous_lower) & (price < lower)

    signal = pd.Series(0, index=df.index, dtype=np.int64)
    signal = signal.mask(buy_mask & ~sell_mask, 1)
    signal = signal.mask(sell_mask & ~buy_mask, -1)

    return signal.rename("ma_envelope_signal")


__all__ = ["ma_cross_signal", "ma_envelope_signal"]
