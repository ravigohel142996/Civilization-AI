"""
simulation/agent_decision_system.py
-------------------------------------
Multi-agent decision system.  Each country acts as an autonomous AI agent
that selects the action most likely to maximise its expected utility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _utility(row: pd.Series) -> float:
    """Compute a scalar utility score for a country row."""
    return (
        0.35 * row["gdp"] / 1e12
        + 0.25 * row["stability_index"]
        + 0.20 * row["technology_level"]
        + 0.10 * row["resource_reserves"]
        + 0.10 * row["education_index"]
    )


def _apply_action(row: pd.Series, action: str, noise: float = 0.02) -> pd.Series:
    """Return an updated *row* after applying *action*.

    Each action perturbs attributes by deterministic deltas plus small noise
    so that agents do not all converge to the same strategy.
    """
    r = row.copy()
    rng = np.random.default_rng(seed=int(abs(hash(row["country_name"])) % 2**32))
    eps = lambda scale: rng.normal(0, scale)

    if action == "increase_technology_investment":
        r["technology_level"] = min(1.0, r["technology_level"] + 0.04 + eps(noise))
        r["gdp"] *= 1.02 + eps(0.005)
        r["stability_index"] = min(1.0, r["stability_index"] + 0.01)

    elif action == "expand_military":
        r["military_strength"] = min(1.0, r["military_strength"] + 0.05 + eps(noise))
        r["gdp"] *= 0.99 + eps(0.005)  # military spending crowds out investment
        r["stability_index"] = min(1.0, r["stability_index"] + 0.005)

    elif action == "increase_trade":
        r["trade_volume"] *= 1.08 + eps(0.01)
        r["gdp"] *= 1.03 + eps(0.005)
        r["stability_index"] = min(1.0, r["stability_index"] + 0.015)

    elif action == "invest_in_sustainability":
        r["climate_risk"] = max(0.0, r["climate_risk"] - 0.04 + eps(noise))
        r["stability_index"] = min(1.0, r["stability_index"] + 0.02)
        r["resource_reserves"] = min(1.0, r["resource_reserves"] + 0.02)

    elif action == "improve_infrastructure":
        r["education_index"] = min(1.0, r["education_index"] + 0.03 + eps(noise))
        r["gdp"] *= 1.015 + eps(0.005)
        r["stability_index"] = min(1.0, r["stability_index"] + 0.01)

    return r


class AgentDecisionSystem:
    """Orchestrates per-country action selection via utility maximisation.

    Each agent evaluates the expected utility of every available action and
    picks the one that produces the highest predicted payoff.
    """

    def __init__(self) -> None:
        self.actions: list[str] = config.AGENT_ACTIONS
        self.last_decisions: dict[str, str] = {}

    def decide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the best action for every country and return updated world.

        Parameters
        ----------
        df:
            Current world state DataFrame.

        Returns
        -------
        pd.DataFrame
            Updated world state after all agents have acted.
        """
        updated_rows: list[pd.Series] = []
        decisions: dict[str, str] = {}

        for _, row in df.iterrows():
            best_action = self._select_action(row)
            updated_row = _apply_action(row, best_action)
            updated_rows.append(updated_row)
            decisions[row["country_name"]] = best_action

        self.last_decisions = decisions
        return pd.DataFrame(updated_rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_action(self, row: pd.Series) -> str:
        """Return the action that maximises post-action utility for *row*."""
        best_action = self.actions[0]
        best_utility = -np.inf

        for action in self.actions:
            candidate = _apply_action(row, action)
            u = _utility(candidate)
            if u > best_utility:
                best_utility = u
                best_action = action

        return best_action

    def get_action_distribution(self) -> pd.Series:
        """Return a frequency count of chosen actions in the last round."""
        if not self.last_decisions:
            return pd.Series(dtype=int)
        return (
            pd.Series(self.last_decisions)
            .value_counts()
            .rename("count")
        )
