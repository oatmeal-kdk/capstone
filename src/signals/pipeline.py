"""Signal aggregation pipeline for the ESN + GA project MVP.

This module combines the paper-based signal systems into one DataFrame for MVP
debugging and quick inspection. It does not implement turning points, GA
optimization, ESN logic, or backtesting.
"""

from __future__ import annotations

import pandas as pd

from .candle_system import candle_signal
from .ma_system import ma_cross_signal, ma_envelope_signal
from .roc_system import roc_signal
from .rsi_system import rsi_signal
from .stochastic_system import stochastic_signal


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input is a pandas DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")


def _validate_bool(value: bool, name: str) -> None:
    """Validate that a value is boolean."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def generate_signals(
    df: pd.DataFrame,
    include_ma_cross: bool = True,
    include_ma_envelope: bool = True,
    include_rsi: bool = True,
    include_roc: bool = True,
    include_stochastic: bool = True,
    include_candle: bool = True,
) -> pd.DataFrame:
    """Generate a DataFrame containing the selected signal systems.

    Args:
        df: Input DataFrame used by the signal generators.
        include_ma_cross: Whether to include moving-average cross signals.
        include_ma_envelope: Whether to include moving-average envelope signals.
        include_rsi: Whether to include RSI signals.
        include_roc: Whether to include ROC signals.
        include_stochastic: Whether to include stochastic signals.
        include_candle: Whether to include candle-pattern signals.

    Returns:
        A new DataFrame indexed like ``df`` containing only signal columns.
    """

    _validate_dataframe(df)
    _validate_bool(include_ma_cross, "include_ma_cross")
    _validate_bool(include_ma_envelope, "include_ma_envelope")
    _validate_bool(include_rsi, "include_rsi")
    _validate_bool(include_roc, "include_roc")
    _validate_bool(include_stochastic, "include_stochastic")
    _validate_bool(include_candle, "include_candle")

    signals = pd.DataFrame(index=df.index)

    if include_ma_cross:
        series = ma_cross_signal(df)
        signals[series.name] = series

    if include_ma_envelope:
        series = ma_envelope_signal(df)
        signals[series.name] = series

    if include_rsi:
        series = rsi_signal(df)
        signals[series.name] = series

    if include_roc:
        series = roc_signal(df)
        signals[series.name] = series

    if include_stochastic:
        series = stochastic_signal(df)
        signals[series.name] = series

    if include_candle:
        series = candle_signal(df)
        signals[series.name] = series

    return signals


__all__ = ["generate_signals"]
