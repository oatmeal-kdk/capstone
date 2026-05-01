"""Indicator aggregation pipeline for the ESN + GA project MVP.

This module creates the paper-required indicator value columns used by the
project's later stages. It combines outputs from the Moving Average, RSI, ROC,
Stochastic, and candle chart systems into a single DataFrame.

This pipeline does not create trading signals, turning point labels, GA
optimization outputs, or ESN features. The returned DataFrame is intended to be
consumed later by separate turning point, signal, GA, and ESN modules.
"""

from __future__ import annotations

import pandas as pd

from .candle import candle_patterns
from .momentum import roc, rsi
from .oscillator import stochastic
from .trend import sma

_REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


def _validate_ohlcv_columns(df: pd.DataFrame) -> None:
    """Validate that the input is a DataFrame with the required OHLCV columns.

    Args:
        df: Input market DataFrame expected to contain standard OHLCV columns.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        KeyError: If any required OHLCV column is missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing_columns = [column for column in _REQUIRED_OHLCV_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_str = ", ".join(f"'{column}'" for column in missing_columns)
        raise KeyError(f"Required OHLCV column(s) not found in the DataFrame: {missing_str}.")


def add_indicators(
    df: pd.DataFrame,
    sma_windows: tuple[int, ...] = (5, 20, 60),
    rsi_windows: tuple[int, ...] = (14,),
    roc_windows: tuple[int, ...] = (12,),
    stochastic_window: int = 14,
    stochastic_smooth_k: int = 3,
    stochastic_smooth_d: int = 3,
    include_candles: bool = True,
    drop_na: bool = False,
) -> pd.DataFrame:
    """Add paper-required indicator columns to a copied OHLCV DataFrame.

    The pipeline only computes and aggregates indicator values required by the
    paper's technical analysis systems. It does not create trading signals, it
    does not create turning point labels, and it does not run GA optimization
    or ESN logic. The returned DataFrame is intended for later use by separate
    turning point, signal, GA, and ESN modules.

    Indicator calculations are performed against the original input DataFrame
    to keep each computation independent and avoid accidental dependencies on
    previously added indicator columns.

    Args:
        df: Input OHLCV DataFrame.
        sma_windows: SMA lookback windows to compute.
        rsi_windows: RSI lookback windows to compute.
        roc_windows: ROC lookback windows to compute.
        stochastic_window: Lookback window for the Stochastic Oscillator.
        stochastic_smooth_k: Smoothing window for %K.
        stochastic_smooth_d: Smoothing window for %D.
        include_candles: Whether to include paper-required candle pattern
            columns.
        drop_na: Whether to drop rows containing NaN values after indicators
            are added.

    Returns:
        A copied DataFrame containing the original OHLCV data plus the
        requested indicator columns.
    """

    _validate_ohlcv_columns(df)

    result = df.copy()

    for window in sma_windows:
        sma_series = sma(df, window)
        result[sma_series.name] = sma_series

    for window in rsi_windows:
        rsi_series = rsi(df, window)
        result[rsi_series.name] = rsi_series

    for window in roc_windows:
        roc_series = roc(df, window)
        result[roc_series.name] = roc_series

    stoch_k, stoch_d = stochastic(
        df,
        window=stochastic_window,
        smooth_k=stochastic_smooth_k,
        smooth_d=stochastic_smooth_d,
    )
    result[stoch_k.name] = stoch_k
    result[stoch_d.name] = stoch_d

    if include_candles:
        result = result.join(candle_patterns(df))

    if drop_na:
        result = result.dropna()

    return result


__all__ = ["add_indicators"]
