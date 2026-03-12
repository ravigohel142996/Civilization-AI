"""
models/population_growth_model.py
----------------------------------
Ridge-regression-based population growth model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATURE_COLS: list[str] = [
    "gdp_per_capita",
    "technology_level",
    "climate_risk",
    "resource_reserves",
    "education_index",
    "stability_index",
]
TARGET_COL: str = "population"

# Approximate annual growth rate bounds
_MIN_GROWTH_RATE: float = -0.02
_MAX_GROWTH_RATE: float = 0.04


class PopulationGrowthModel:
    """Predicts annual population growth using macroeconomic indicators.

    The model estimates a *growth rate* per country, which is then applied
    multiplicatively during each simulation round.

    Attributes
    ----------
    model : Ridge
        Underlying sklearn estimator.
    scaler : StandardScaler
        Feature scaler fitted during training.
    is_fitted : bool
    """

    def __init__(self) -> None:
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PopulationGrowthModel":
        """Fit the growth-rate model.

        Growth rate is derived analytically from the base features rather
        than requiring labelled time-series data:

        * High GDP per capita → lower growth (demographic transition)
        * High technology level → lower growth
        * High climate risk → lower growth
        * High resource availability → slightly higher growth
        * High education → lower growth

        Parameters
        ----------
        df:
            World DataFrame with all :data:`FEATURE_COLS` and
            ``population``.

        Returns
        -------
        PopulationGrowthModel
        """
        X = df[FEATURE_COLS].values
        # Construct proxy growth-rate labels
        gdp_norm = (df["gdp_per_capita"] - df["gdp_per_capita"].min()) / (
            df["gdp_per_capita"].max() - df["gdp_per_capita"].min() + 1e-9
        )
        growth_rates = (
            0.035
            - 0.02 * gdp_norm.values
            - 0.005 * df["technology_level"].values
            - 0.008 * df["climate_risk"].values
            + 0.004 * df["resource_reserves"].values
            - 0.006 * df["education_index"].values
            + np.random.default_rng(1).normal(0, 0.002, size=len(df))
        )
        growth_rates = np.clip(growth_rates, _MIN_GROWTH_RATE, _MAX_GROWTH_RATE)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, growth_rates)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_growth_rate(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted annual population growth rate per country.

        Parameters
        ----------
        df:
            Must contain all :data:`FEATURE_COLS`.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` – annual growth rates (e.g. 0.012 = 1.2 %).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_growth_rate().")
        X = df[FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)
        rates = self.model.predict(X_scaled)
        return np.clip(rates, _MIN_GROWTH_RATE, _MAX_GROWTH_RATE)

    def apply_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with updated *population* values.

        Parameters
        ----------
        df:
            Current world state DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with ``population`` column updated.
        """
        rates = self.predict_growth_rate(df)
        updated = df.copy()
        updated["population"] = (updated["population"] * (1 + rates)).astype(int)
        return updated
