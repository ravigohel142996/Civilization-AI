"""
utils/helpers.py
-----------------
Shared utility functions used across the platform.
"""

from __future__ import annotations

import re
from typing import Any, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_population(value: Union[int, float]) -> str:
    """Format a population count into a human-readable string."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def fmt_gdp(value: Union[int, float]) -> str:
    """Format GDP in trillions / billions / millions."""
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def fmt_percent(value: float, decimals: int = 1) -> str:
    """Format a ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_index(value: float, decimals: int = 3) -> str:
    """Format a normalised [0, 1] index value."""
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalise_series(s: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s * 0.0
    return (s - lo) / (hi - lo)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def safe_divide(
    numerator: Union[pd.Series, np.ndarray, float],
    denominator: Union[pd.Series, np.ndarray, float],
    fill_value: float = 0.0,
) -> Union[pd.Series, np.ndarray, float]:
    """Division that replaces division-by-zero with *fill_value*."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.asarray(denominator) != 0,
            np.asarray(numerator) / np.asarray(denominator),
            fill_value,
        )
    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index)
    return result


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dictionary."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert a display string to a lowercase URL/key-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text
