"""Turning-point label generation for the ESN + GA project MVP.

This module converts simplified offline turning points into label series aligned
with the original DataFrame index. These labels are intended for training data
preparation and GA fitness targets, not for live trading decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.turning_points.critical_points import find_turning_points


def _validate_dataframe_price_column(df: pd.DataFrame, price_col: str) -> None:
    """Validate that the input DataFrame contains a numeric price column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' was not found in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[price_col]):
        raise TypeError(f"Price column '{price_col}' must contain numeric values.")


def turning_point_labels(
    df: pd.DataFrame,
    price_col: str = "Close",
    min_interval: int = 5,
    min_change_pct: float = 0.05,
) -> pd.Series:
    """Generate turning-point labels aligned with the input DataFrame index.

    Labels follow the project convention:
    ``1`` for bottoms, ``-1`` for peaks, and ``0`` for normal points.

    Args:
        df: Input DataFrame containing the price series.
        price_col: Price column used for turning-point extraction.
        min_interval: Minimum positional distance between turning points.
        min_change_pct: Minimum normalized price change between turning points.

    Returns:
        A Series aligned with ``df.index`` and named ``"turning_label"``.
    """

    _validate_dataframe_price_column(df, price_col)

    points = find_turning_points(
        df[price_col],
        min_interval=min_interval,
        min_change_pct=min_change_pct,
    )

    labels = pd.Series(0, index=df.index, dtype=np.int64)
    for _, point in points.iterrows():
        if point["type"] == "bottom":
            labels.loc[point["index"]] = 1
        elif point["type"] == "peak":
            labels.loc[point["index"]] = -1

    return labels.rename("turning_label")


def turning_point_frame(
    df: pd.DataFrame,
    price_col: str = "Close",
    min_interval: int = 5,
    min_change_pct: float = 0.05,
) -> pd.DataFrame:
    """Return a new DataFrame with the price column and turning-point labels.

    Args:
        df: Input DataFrame containing the price series.
        price_col: Price column used for turning-point extraction.
        min_interval: Minimum positional distance between turning points.
        min_change_pct: Minimum normalized price change between turning points.

    Returns:
        A new DataFrame containing ``price_col`` and ``"turning_label"``.
    """

    _validate_dataframe_price_column(df, price_col)

    result = pd.DataFrame(index=df.index)
    result[price_col] = df[price_col].copy()
    result["turning_label"] = turning_point_labels(
        df,
        price_col=price_col,
        min_interval=min_interval,
        min_change_pct=min_change_pct,
    )

    return result


__all__ = ["turning_point_labels", "turning_point_frame"]
