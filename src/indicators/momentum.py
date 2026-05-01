"""Paper-based momentum indicators for the ESN + GA project MVP.

The paper uses RSI and ROC systems.
Additional momentum indicators are stored in
``src/indicators/experimental/momentum_extra.py`` for future ablation studies
or extended quant experiments.
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


def rsi(df: pd.DataFrame, window: int = 14, price_col: str = "Close") -> pd.Series:
    """Compute the Relative Strength Index for a price series.

    Args:
        df: Input OHLCV DataFrame.
        window: Rolling lookback window used for average gains and losses.
        price_col: Name of the price column to evaluate.

    Returns:
        A pandas Series containing the RSI values.
    """

    _validate_price_column(df, price_col)
    _validate_window(window)

    delta = df[price_col].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    rs = avg_gain.div(avg_loss.replace(0.0, np.nan))
    result = 100.0 - (100.0 / (1.0 + rs))

    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)

    return result.rename(f"rsi_{window}")


def roc(df: pd.DataFrame, window: int = 12, price_col: str = "Close") -> pd.Series:
    """Compute the Rate of Change for a price series.

    Args:
        df: Input OHLCV DataFrame.
        window: Lookback window used in the ROC ratio.
        price_col: Name of the price column to evaluate.

    Returns:
        A pandas Series containing the ROC values.
    """

    _validate_price_column(df, price_col)
    _validate_window(window)

    shifted = df[price_col].shift(window)
    result = df[price_col].div(shifted.replace(0.0, np.nan)) * 100.0
    return result.rename(f"roc_{window}")
