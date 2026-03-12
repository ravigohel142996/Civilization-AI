"""
models/gdp_predictor.py
-----------------------
RandomForestRegressor-based GDP prediction model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import config

FEATURE_COLS: list[str] = [
    "population",
    "technology_level",
    "resource_reserves",
    "trade_volume",
    "education_index",
]
TARGET_COL: str = "gdp"


class GDPPredictor:
    """Trains on historical world data and predicts future GDP values.

    Attributes
    ----------
    model : RandomForestRegressor
        Underlying sklearn estimator.
    scaler : StandardScaler
        Feature scaler fitted during training.
    is_fitted : bool
        Whether :py:meth:`fit` has been called successfully.
    """

    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=config.GDP_PREDICTOR_N_ESTIMATORS,
            random_state=config.GDP_PREDICTOR_RANDOM_STATE,
        )
        self.scaler = StandardScaler()
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "GDPPredictor":
        """Fit the model on a world DataFrame.

        Parameters
        ----------
        df:
            Must contain all :data:`FEATURE_COLS` and :data:`TARGET_COL`.

        Returns
        -------
        GDPPredictor
            Self, for method chaining.
        """
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted GDP for each row in *df*.

        Parameters
        ----------
        df:
            Must contain all :data:`FEATURE_COLS`.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` – predicted GDP values.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = df[FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Augmented training helper
    # ------------------------------------------------------------------

    @staticmethod
    def augment_training_data(
        df: pd.DataFrame, n_samples: int = 500, noise_scale: float = 0.05
    ) -> pd.DataFrame:
        """Generate synthetic rows via Gaussian perturbation for richer training.

        Parameters
        ----------
        df:
            Base world DataFrame.
        n_samples:
            Number of synthetic rows to add.
        noise_scale:
            Relative standard deviation of the Gaussian noise.

        Returns
        -------
        pd.DataFrame
            Combined original + synthetic data.
        """
        rng = np.random.default_rng(seed=0)
        cols = FEATURE_COLS + [TARGET_COL]
        base = df[cols].values
        idx = rng.integers(0, len(base), size=n_samples)
        noise = rng.normal(0, noise_scale, size=(n_samples, len(cols)))
        synthetic = base[idx] * (1 + noise)
        synthetic_df = pd.DataFrame(synthetic, columns=cols)
        return pd.concat([df[cols], synthetic_df], ignore_index=True)
