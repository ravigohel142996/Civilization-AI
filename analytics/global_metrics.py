"""
analytics/global_metrics.py
-----------------------------
Compute aggregated global and per-country analytics from simulation results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.civilization_engine import SimulationResult


class GlobalMetrics:
    """Derives high-level metrics and trend frames from a :class:`SimulationResult`.

    Parameters
    ----------
    result:
        Completed simulation result object.
    """

    def __init__(self, result: SimulationResult) -> None:
        self.result = result

    # ------------------------------------------------------------------
    # Round-level series
    # ------------------------------------------------------------------

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Round-level aggregate metrics DataFrame."""
        return self.result.metrics_df

    # ------------------------------------------------------------------
    # Per-country time series
    # ------------------------------------------------------------------

    def gdp_time_series(self) -> pd.DataFrame:
        """Wide-format GDP by country across rounds.

        Returns
        -------
        pd.DataFrame
            Index = round numbers, columns = country names.
        """
        frames = []
        for rnd, df in enumerate(self.result.history):
            row = df.set_index("country_name")["gdp"].rename(rnd)
            frames.append(row)
        return pd.DataFrame(frames)

    def population_time_series(self) -> pd.DataFrame:
        """Wide-format population by country across rounds."""
        frames = []
        for rnd, df in enumerate(self.result.history):
            row = df.set_index("country_name")["population"].rename(rnd)
            frames.append(row)
        return pd.DataFrame(frames)

    def technology_time_series(self) -> pd.DataFrame:
        """Wide-format technology_level by country across rounds."""
        frames = []
        for rnd, df in enumerate(self.result.history):
            row = df.set_index("country_name")["technology_level"].rename(rnd)
            frames.append(row)
        return pd.DataFrame(frames)

    def conflict_risk_time_series(self) -> pd.DataFrame:
        """Wide-format conflict_risk by country across rounds."""
        frames = []
        for rnd, df in enumerate(self.result.history):
            row = df.set_index("country_name")["conflict_risk"].rename(rnd)
            frames.append(row)
        return pd.DataFrame(frames)

    # ------------------------------------------------------------------
    # Snapshot analytics
    # ------------------------------------------------------------------

    def top_economies(self, n: int = 10) -> pd.DataFrame:
        """Return the top *n* countries by GDP in the final round."""
        return (
            self.result.final_state.nlargest(n, "gdp")[
                ["country_name", "gdp", "gdp_per_capita", "population"]
            ]
            .reset_index(drop=True)
        )

    def highest_conflict_risk(self, n: int = 10) -> pd.DataFrame:
        """Return *n* countries with highest conflict risk in the final round."""
        return (
            self.result.final_state.nlargest(n, "conflict_risk")[
                ["country_name", "conflict_risk", "military_strength", "stability_index"]
            ]
            .reset_index(drop=True)
        )

    def technology_leaders(self, n: int = 10) -> pd.DataFrame:
        """Return *n* technology-leading countries in the final round."""
        return (
            self.result.final_state.nlargest(n, "technology_level")[
                ["country_name", "technology_level", "education_index", "gdp"]
            ]
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Summary KPIs (final round)
    # ------------------------------------------------------------------

    def summary_kpis(self) -> dict:
        """Return headline KPIs from the final simulation round."""
        final = self.result.final_state
        return {
            "world_population": int(final["population"].sum()),
            "global_gdp_trillion": float(final["gdp"].sum() / 1e12),
            "avg_stability_index": float(final["stability_index"].mean()),
            "avg_conflict_risk": float(final["conflict_risk"].mean()),
            "avg_technology_level": float(final["technology_level"].mean()),
            "avg_climate_risk": float(final["climate_risk"].mean()),
            "num_countries": len(final),
        }

    # ------------------------------------------------------------------
    # Growth rates
    # ------------------------------------------------------------------

    def gdp_growth_rates(self) -> pd.Series:
        """CAGR of GDP per country from round 0 to final round."""
        n_rounds = len(self.result.history) - 1
        if n_rounds <= 0:
            return pd.Series(dtype=float)
        initial = self.result.history[0].set_index("country_name")["gdp"]
        final = self.result.final_state.set_index("country_name")["gdp"]
        cagr = (final / initial.reindex(final.index).replace(0, np.nan)) ** (
            1 / n_rounds
        ) - 1
        return cagr.rename("gdp_cagr")
