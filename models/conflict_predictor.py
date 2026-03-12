"""
models/conflict_predictor.py
-----------------------------
GradientBoosting classifier that predicts bilateral conflict probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import config

FEATURE_COLS: list[str] = [
    "military_strength_diff",
    "resource_shortage",
    "trade_dependency",
    "political_stability",
    "technology_gap",
    "climate_risk_avg",
]


class ConflictPredictor:
    """Predicts pairwise conflict probability between countries.

    Attributes
    ----------
    model : GradientBoostingClassifier
    scaler : StandardScaler
    is_fitted : bool
    """

    def __init__(self) -> None:
        self.model = GradientBoostingClassifier(
            n_estimators=config.CONFLICT_MODEL_N_ESTIMATORS,
            random_state=config.CONFLICT_MODEL_RANDOM_STATE,
        )
        self.scaler = StandardScaler()
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ConflictPredictor":
        """Train on the current world state using synthetically derived labels.

        Labels are derived analytically:
        conflict is *likely* (1) when military imbalance is high, resources
        are scarce, and political stability is low.

        Parameters
        ----------
        df:
            World DataFrame.

        Returns
        -------
        ConflictPredictor
        """
        pairs = self._build_pairwise_features(df)
        X = pairs[FEATURE_COLS].values
        # Analytical conflict probability → threshold to label
        conflict_prob = (
            0.4 * pairs["military_strength_diff"]
            + 0.3 * pairs["resource_shortage"]
            - 0.2 * pairs["trade_dependency"]
            - 0.3 * pairs["political_stability"]
            + 0.1 * pairs["technology_gap"]
            + 0.1 * pairs["climate_risk_avg"]
        )
        conflict_prob = (conflict_prob - conflict_prob.min()) / (
            conflict_prob.max() - conflict_prob.min() + 1e-9
        )
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.05, size=len(conflict_prob))
        labels = (conflict_prob + noise > 0.5).astype(int)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_conflict_risk(self, df: pd.DataFrame) -> pd.Series:
        """Return per-country aggregated conflict risk score [0, 1].

        Parameters
        ----------
        df:
            Current world state.

        Returns
        -------
        pd.Series
            Indexed by *country_name* with float conflict risk.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_conflict_risk().")
        pairs = self._build_pairwise_features(df)
        X = self.scaler.transform(pairs[FEATURE_COLS].values)
        probs = self.model.predict_proba(X)[:, 1]
        pairs = pairs.copy()
        pairs["conflict_prob"] = probs
        # Aggregate: max risk seen by each country across all its pairs
        risk_a = pairs.groupby("country_a")["conflict_prob"].mean()
        risk_b = pairs.groupby("country_b")["conflict_prob"].mean()
        combined = (risk_a.add(risk_b, fill_value=0) / 2).reindex(
            df["country_name"]
        )
        return combined.fillna(0).rename("conflict_risk")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pairwise_features(df: pd.DataFrame) -> pd.DataFrame:
        """Construct a pairwise feature matrix for all country combinations."""
        records = []
        rows = df.to_dict("records")
        for i, a in enumerate(rows):
            for j, b in enumerate(rows):
                if i >= j:
                    continue
                mil_diff = abs(a["military_strength"] - b["military_strength"])
                resource_shortage = 1.0 - (
                    a["resource_reserves"] + b["resource_reserves"]
                ) / 2
                # Trade dependency is estimated as proportional to combined
                # trade volume normalised to [0, 1]
                max_trade = df["trade_volume"].max() + 1e-9
                trade_dep = (a["trade_volume"] + b["trade_volume"]) / (2 * max_trade)
                pol_stability = (
                    a["stability_index"] + b["stability_index"]
                ) / 2
                tech_gap = abs(
                    a["technology_level"] - b["technology_level"]
                )
                climate_avg = (a["climate_risk"] + b["climate_risk"]) / 2
                records.append(
                    {
                        "country_a": a["country_name"],
                        "country_b": b["country_name"],
                        "military_strength_diff": mil_diff,
                        "resource_shortage": resource_shortage,
                        "trade_dependency": trade_dep,
                        "political_stability": pol_stability,
                        "technology_gap": tech_gap,
                        "climate_risk_avg": climate_avg,
                    }
                )
        return pd.DataFrame(records)
