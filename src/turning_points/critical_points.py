"""Critical point extraction and turning point filtering for the project MVP.

This module implements a simplified offline turning-point method inspired by
the paper's critical point idea. It is intended for training-label generation
and GA fitness targets, not for live trading, because the extracted labels can
depend on future observations around each turning point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_price_series(price: pd.Series) -> None:
    """Validate that the input is a usable numeric price series.

    Args:
        price: Price series used to extract local critical points.

    Raises:
        TypeError: If ``price`` is not a pandas Series or is not numeric.
        ValueError: If ``price`` is empty or has fewer than three non-NaN
            observations.
    """

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series.")

    if price.empty:
        raise ValueError("price must not be empty.")

    if not pd.api.types.is_numeric_dtype(price):
        raise TypeError("price must contain numeric values.")

    non_na_count = int(price.notna().sum())
    if non_na_count < 3:
        raise ValueError("price must contain at least 3 non-NaN values.")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a parameter is a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_positive_number(value: float, name: str) -> None:
    """Validate that a parameter is a strictly positive real number."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a positive number.")

    if value <= 0:
        raise ValueError(f"{name} must be a positive number.")


def _boundary_type(values: np.ndarray, at_start: bool) -> str:
    """Infer a conservative boundary type from the nearest unequal neighbor."""

    if at_start:
        base_value = float(values[0])
        for candidate in values[1:]:
            candidate_value = float(candidate)
            if candidate_value > base_value:
                return "bottom"
            if candidate_value < base_value:
                return "peak"
        return "bottom"

    base_value = float(values[-1])
    for candidate in values[-2::-1]:
        candidate_value = float(candidate)
        if candidate_value > base_value:
            return "bottom"
        if candidate_value < base_value:
            return "peak"
    return "peak"


def _more_extreme_point(left: pd.Series, right: pd.Series) -> pd.Series:
    """Return the more extreme critical point between two same-type points."""

    if left["type"] == "peak":
        return left if float(left["price"]) >= float(right["price"]) else right

    return left if float(left["price"]) <= float(right["price"]) else right


def _enforce_alternating_types(points: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive same-type points by keeping the more extreme one."""

    if points.empty:
        return points.copy()

    selected: list[pd.Series] = [points.iloc[0].copy()]
    for i in range(1, len(points)):
        current = points.iloc[i].copy()
        last_kept = selected[-1]

        if current["type"] == last_kept["type"]:
            selected[-1] = _more_extreme_point(last_kept, current).copy()
        else:
            selected.append(current)

    result = pd.DataFrame(selected)
    return result.sort_values("position").reset_index(drop=True)


def find_local_critical_points(price: pd.Series) -> pd.DataFrame:
    """Find local maxima and minima from a price series.

    The method drops NaN values, preserves the original index labels in the
    result, always includes the first and last valid observations as boundary
    critical points, and marks only strict interior extrema. Flat plateaus are
    handled conservatively by skipping interior points whose neighbors are not
    strictly above or below them.

    Args:
        price: Numeric price series.

    Returns:
        A DataFrame with columns ``position``, ``index``, ``price``, and
        ``type``, where ``type`` is either ``"peak"`` or ``"bottom"``.
    """

    _validate_price_series(price)

    clean_price = price.dropna()
    clean_values = clean_price.to_numpy(dtype=float, copy=True)
    clean_index = clean_price.index.to_list()

    points: list[dict[str, object]] = [
        {
            "position": 0,
            "index": clean_index[0],
            "price": float(clean_values[0]),
            "type": _boundary_type(clean_values, at_start=True),
        }
    ]

    for position in range(1, len(clean_values) - 1):
        previous_value = clean_values[position - 1]
        current_value = clean_values[position]
        next_value = clean_values[position + 1]

        if current_value > previous_value and current_value > next_value:
            point_type = "peak"
        elif current_value < previous_value and current_value < next_value:
            point_type = "bottom"
        else:
            continue

        points.append(
            {
                "position": position,
                "index": clean_index[position],
                "price": float(current_value),
                "type": point_type,
            }
        )

    points.append(
        {
            "position": len(clean_values) - 1,
            "index": clean_index[-1],
            "price": float(clean_values[-1]),
            "type": _boundary_type(clean_values, at_start=False),
        }
    )

    result = pd.DataFrame(points).sort_values("position").reset_index(drop=True)
    return _enforce_alternating_types(result)


def filter_turning_points(
    critical_points: pd.DataFrame,
    min_interval: int = 5,
    min_change_pct: float = 0.05,
) -> pd.DataFrame:
    """Filter local critical points into simplified turning points.

    This is an MVP-friendly sequential filter, not a full three-point CPM
    implementation. It keeps the first critical point, then accepts later
    points if they are sufficiently separated in time or normalized price
    change. Small oscillations are treated as noise.

    Args:
        critical_points: DataFrame returned by
            :func:`find_local_critical_points`.
        min_interval: Minimum positional distance between turning points.
        min_change_pct: Minimum normalized price change between turning points.

    Returns:
        A filtered DataFrame with columns ``position``, ``index``, ``price``,
        and ``type``.
    """

    if not isinstance(critical_points, pd.DataFrame):
        raise TypeError("critical_points must be a pandas DataFrame.")

    required_columns = ["position", "index", "price", "type"]
    missing_columns = [col for col in required_columns if col not in critical_points.columns]
    if missing_columns:
        missing_str = ", ".join(f"'{col}'" for col in missing_columns)
        raise KeyError(f"Required column(s) not found in critical_points: {missing_str}.")

    _validate_positive_int(min_interval, "min_interval")
    _validate_positive_number(min_change_pct, "min_change_pct")

    if len(critical_points) < 2:
        return critical_points.copy().sort_values("position").reset_index(drop=True)

    points = critical_points[required_columns].copy().sort_values("position").reset_index(drop=True)

    selected: list[pd.Series] = [points.iloc[0].copy()]

    for i in range(1, len(points)):
        current = points.iloc[i].copy()
        last_kept = selected[-1]

        interval = abs(int(current["position"]) - int(last_kept["position"]))
        current_price = float(current["price"])
        last_price = float(last_kept["price"])
        midpoint = (current_price + last_price) / 2.0

        if midpoint == 0.0:
            change_pct = 0.0 if current_price == last_price else np.inf
        else:
            change_pct = abs(current_price - last_price) / abs(midpoint)

        if interval >= min_interval or change_pct >= min_change_pct:
            selected.append(current)
            continue

        if current["type"] == last_kept["type"]:
            selected[-1] = _more_extreme_point(last_kept, current).copy()

    result = pd.DataFrame(selected)
    result = _enforce_alternating_types(result)
    return result.sort_values("position").reset_index(drop=True)


def find_turning_points(
    price: pd.Series,
    min_interval: int = 5,
    min_change_pct: float = 0.05,
) -> pd.DataFrame:
    """Find simplified turning points from a price series.

    This convenience function runs local critical-point extraction followed by
    sequential filtering. The resulting turning points are intended for offline
    label generation for training and GA fitness targets.

    Args:
        price: Numeric price series.
        min_interval: Minimum positional distance between turning points.
        min_change_pct: Minimum normalized price change between turning points.

    Returns:
        A DataFrame with columns ``position``, ``index``, ``price``, and
        ``type``.
    """

    critical_points = find_local_critical_points(price)
    return filter_turning_points(
        critical_points,
        min_interval=min_interval,
        min_change_pct=min_change_pct,
    )


__all__ = [
    "find_local_critical_points",
    "filter_turning_points",
    "find_turning_points",
]
