"""
simulation/civilization_engine.py
-----------------------------------
Core simulation engine that orchestrates all sub-systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from data.world_generator import generate_world
from models.conflict_predictor import ConflictPredictor
from models.gdp_predictor import GDPPredictor
from models.population_growth_model import PopulationGrowthModel
from simulation.agent_decision_system import AgentDecisionSystem
from simulation.trade_network import TradeNetwork


@dataclass
class SimulationConfig:
    """Immutable configuration for a single simulation run."""

    num_countries: int = config.DEFAULT_NUM_COUNTRIES
    simulation_rounds: int = config.DEFAULT_SIMULATION_ROUNDS
    resource_level: float = config.DEFAULT_RESOURCE_LEVEL
    technology_growth_rate: float = config.DEFAULT_TECHNOLOGY_GROWTH_RATE
    climate_stress_level: float = config.DEFAULT_CLIMATE_STRESS_LEVEL
    seed: Optional[int] = 42


@dataclass
class SimulationResult:
    """Accumulates metrics from every simulation round."""

    history: list[pd.DataFrame] = field(default_factory=list)
    round_metrics: list[dict] = field(default_factory=list)
    trade_history: list[pd.DataFrame] = field(default_factory=list)
    agent_actions: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def final_state(self) -> pd.DataFrame:
        """World state after the last round."""
        return self.history[-1] if self.history else pd.DataFrame()

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Round-level aggregate metrics as a tidy DataFrame."""
        return pd.DataFrame(self.round_metrics)


class CivilizationEngine:
    """Orchestrates world generation, ML models, agents, and trade.

    Usage
    -----
    >>> cfg = SimulationConfig(num_countries=20, simulation_rounds=5)
    >>> engine = CivilizationEngine(cfg)
    >>> result = engine.run()
    """

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self._gdp_model = GDPPredictor()
        self._pop_model = PopulationGrowthModel()
        self._conflict_model = ConflictPredictor()
        self._agent_system = AgentDecisionSystem()
        self._trade_network = TradeNetwork()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute the full simulation and return all results.

        Returns
        -------
        SimulationResult
        """
        result = SimulationResult()

        # 1. Generate initial world
        world = generate_world(
            num_countries=self.cfg.num_countries,
            resource_level=self.cfg.resource_level,
            technology_growth_rate=self.cfg.technology_growth_rate,
            climate_stress_level=self.cfg.climate_stress_level,
            seed=self.cfg.seed,
        )

        # 2. Train ML models on initial world state
        training_data = GDPPredictor.augment_training_data(world)
        self._gdp_model.fit(training_data)
        self._pop_model.fit(world)
        self._conflict_model.fit(world)

        # 3. Build trade network
        self._trade_network.build(world, seed=self.cfg.seed)

        # 4. Store round 0 (initial state)
        world = self._enrich(world)
        result.history.append(world.copy())
        result.round_metrics.append(self._aggregate_metrics(world, round_num=0))
        result.trade_history.append(self._trade_network.get_edge_dataframe())

        # 5. Simulation loop
        for rnd in range(1, self.cfg.simulation_rounds + 1):
            world = self._step(world, rnd)
            result.history.append(world.copy())
            result.round_metrics.append(self._aggregate_metrics(world, round_num=rnd))
            result.trade_history.append(self._trade_network.get_edge_dataframe())
            result.agent_actions.append(dict(self._agent_system.last_decisions))

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step(self, world: pd.DataFrame, rnd: int) -> pd.DataFrame:
        """Execute one simulation round."""
        # Agent decisions mutate world attributes
        world = self._agent_system.decide(world)

        # Update population
        world = self._pop_model.apply_growth(world)

        # Predict and update GDP
        world["gdp_per_capita"] = world["gdp"] / world["population"].clip(lower=1)
        predicted_gdp = self._gdp_model.predict(world)
        # Blend predicted with current to avoid abrupt jumps
        world["gdp"] = 0.6 * predicted_gdp + 0.4 * world["gdp"].values

        # Update trade network
        self._trade_network.update(world)

        # Re-derive GDP per capita after update
        world["gdp_per_capita"] = world["gdp"] / world["population"].clip(lower=1)

        # Apply technology growth trend
        world["technology_level"] = np.clip(
            world["technology_level"] * (1 + self.cfg.technology_growth_rate),
            0.0,
            1.0,
        )

        # Climate risk drifts upward unless agents invest in sustainability
        world["climate_risk"] = np.clip(
            world["climate_risk"] + 0.005 * self.cfg.climate_stress_level,
            0.0,
            1.0,
        )

        # Recalculate stability
        world["stability_index"] = self._compute_stability(world)

        # Enrich with derived columns
        world = self._enrich(world)

        return world

    def _enrich(self, world: pd.DataFrame) -> pd.DataFrame:
        """Add derived analytics columns to the world DataFrame."""
        world = world.copy()
        # Trade centrality
        centrality = self._trade_network.get_centrality()
        world["trade_centrality"] = world["country_name"].map(centrality).fillna(0)
        # Conflict risk
        conflict = self._conflict_model.predict_conflict_risk(world)
        world["conflict_risk"] = world["country_name"].map(conflict).fillna(0)
        return world

    @staticmethod
    def _compute_stability(world: pd.DataFrame) -> pd.Series:
        """Re-derive stability index from current world attributes."""
        weights = config.STABILITY_WEIGHTS
        max_gdp_pc = world["gdp_per_capita"].max() + 1e-9
        gdp_norm = world["gdp_per_capita"] / max_gdp_pc
        stability = (
            weights["technology_level"] * world["technology_level"]
            + weights["resource_reserves"] * world["resource_reserves"]
            + weights["gdp_per_capita_norm"] * gdp_norm
            + weights["climate_risk"] * world["climate_risk"]
            + weights["military_strength"] * world["military_strength"]
            + weights["education_index"] * world["education_index"]
        )
        return stability.clip(0.0, 1.0)

    @staticmethod
    def _aggregate_metrics(world: pd.DataFrame, round_num: int) -> dict:
        """Compute round-level aggregate KPIs."""
        return {
            "round": round_num,
            "world_population": int(world["population"].sum()),
            "global_gdp": float(world["gdp"].sum()),
            "avg_stability": float(world["stability_index"].mean()),
            "avg_conflict_risk": float(world["conflict_risk"].mean()),
            "avg_technology": float(world["technology_level"].mean()),
            "avg_climate_risk": float(world["climate_risk"].mean()),
            "total_trade_volume": float(world["trade_volume"].sum()),
        }
