"""
config.py
---------
Central configuration for the Civilization AI simulation platform.
All tuneable constants live here; no magic numbers elsewhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# World-generation defaults
# ---------------------------------------------------------------------------
DEFAULT_NUM_COUNTRIES: int = 25

POPULATION_RANGE: tuple[float, float] = (1_000_000, 1_500_000_000)
GDP_RANGE: tuple[float, float] = (5_000_000_000, 25_000_000_000_000)
TECHNOLOGY_LEVEL_RANGE: tuple[float, float] = (0.1, 1.0)
RESOURCE_RESERVES_RANGE: tuple[float, float] = (0.1, 1.0)
MILITARY_STRENGTH_RANGE: tuple[float, float] = (0.1, 1.0)
CLIMATE_RISK_RANGE: tuple[float, float] = (0.0, 1.0)
STABILITY_INDEX_RANGE: tuple[float, float] = (0.1, 1.0)
EDUCATION_INDEX_RANGE: tuple[float, float] = (0.2, 1.0)
TRADE_VOLUME_RANGE: tuple[float, float] = (1_000_000, 500_000_000_000)

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
DEFAULT_SIMULATION_ROUNDS: int = 10
DEFAULT_TECHNOLOGY_GROWTH_RATE: float = 0.03
DEFAULT_CLIMATE_STRESS_LEVEL: float = 0.3
DEFAULT_RESOURCE_LEVEL: float = 0.6

# ---------------------------------------------------------------------------
# ML model settings
# ---------------------------------------------------------------------------
GDP_PREDICTOR_N_ESTIMATORS: int = 100
GDP_PREDICTOR_RANDOM_STATE: int = 42
CONFLICT_MODEL_N_ESTIMATORS: int = 100
CONFLICT_MODEL_RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Trade-network settings
# ---------------------------------------------------------------------------
TRADE_EDGE_PROBABILITY: float = 0.25  # Erdős–Rényi p
MIN_TRADE_VOLUME_FRACTION: float = 0.01
MAX_TRADE_VOLUME_FRACTION: float = 0.10

# ---------------------------------------------------------------------------
# Agent decision system
# ---------------------------------------------------------------------------
AGENT_ACTIONS: list[str] = [
    "increase_technology_investment",
    "expand_military",
    "increase_trade",
    "invest_in_sustainability",
    "improve_infrastructure",
]

# ---------------------------------------------------------------------------
# Stability / conflict weights
# ---------------------------------------------------------------------------
STABILITY_WEIGHTS: dict[str, float] = {
    "technology_level": 0.20,
    "resource_reserves": 0.20,
    "gdp_per_capita_norm": 0.25,
    "climate_risk": -0.15,
    "military_strength": 0.10,
    "education_index": 0.10,
}

# ---------------------------------------------------------------------------
# Dashboard / UI
# ---------------------------------------------------------------------------
PAGE_TITLE: str = "Civilization AI — Autonomous World Simulation Engine"
PAGE_ICON: str = "🌍"
CHART_HEIGHT: int = 450
MAP_PROJECTION: str = "natural earth"
