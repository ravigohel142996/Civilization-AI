"""
data/world_generator.py
-----------------------
Generates a synthetic world composed of AI-controlled countries with
randomised but internally consistent initial attributes.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Country name corpus
# ---------------------------------------------------------------------------
_COUNTRY_NAMES: list[str] = [
    "Arendor", "Valdris", "Korethia", "Nimbara", "Solvane",
    "Draxion", "Thaloria", "Vesporia", "Quintas", "Melindra",
    "Orvanos", "Syrakka", "Ferholm", "Claratia", "Zandoria",
    "Ultrenos", "Batheron", "Cristalio", "Dunvale", "Estovea",
    "Galothia", "Hintara", "Indraxis", "Jorvesk", "Kelrantia",
    "Luxovon", "Myrenna", "Noctalis", "Opheris", "Pendura",
    "Quartos", "Rhelond", "Sanctuvia", "Telvoran", "Umbriva",
]


def _clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* clipped to [lo, hi]."""
    return max(lo, min(hi, value))


def generate_world(
    num_countries: int = config.DEFAULT_NUM_COUNTRIES,
    resource_level: float = config.DEFAULT_RESOURCE_LEVEL,
    technology_growth_rate: float = config.DEFAULT_TECHNOLOGY_GROWTH_RATE,
    climate_stress_level: float = config.DEFAULT_CLIMATE_STRESS_LEVEL,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate a :class:`~pandas.DataFrame` of synthetic countries.

    Parameters
    ----------
    num_countries:
        Number of countries to generate (capped at the name corpus size).
    resource_level:
        Baseline resource richness [0, 1] applied as an offset.
    technology_growth_rate:
        Used to skew initial *technology_level* distributions upward.
    climate_stress_level:
        Multiplier that raises *climate_risk* values.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per country with standardised column names.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = min(num_countries, len(_COUNTRY_NAMES))
    names = random.sample(_COUNTRY_NAMES, n)

    pop_lo, pop_hi = config.POPULATION_RANGE
    gdp_lo, gdp_hi = config.GDP_RANGE
    tech_lo, tech_hi = config.TECHNOLOGY_LEVEL_RANGE
    res_lo, res_hi = config.RESOURCE_RESERVES_RANGE
    mil_lo, mil_hi = config.MILITARY_STRENGTH_RANGE
    clim_lo, clim_hi = config.CLIMATE_RISK_RANGE
    stab_lo, stab_hi = config.STABILITY_INDEX_RANGE
    edu_lo, edu_hi = config.EDUCATION_INDEX_RANGE
    trade_lo, trade_hi = config.TRADE_VOLUME_RANGE

    populations = np.random.lognormal(mean=16.0, sigma=2.5, size=n)
    populations = np.clip(populations, pop_lo, pop_hi)

    gdps = np.random.lognormal(mean=24.0, sigma=2.0, size=n)
    gdps = np.clip(gdps, gdp_lo, gdp_hi)

    # resource_level biases resource reserves
    resource_base = np.random.beta(a=2, b=2, size=n)
    resource_reserves = np.clip(resource_base + (resource_level - 0.5) * 0.4, res_lo, res_hi)

    # technology_growth_rate biases initial technology
    tech_base = np.random.beta(a=2, b=3, size=n)
    technology_levels = np.clip(tech_base + technology_growth_rate * 5, tech_lo, tech_hi)

    military_strengths = np.random.beta(a=2, b=3, size=n)
    military_strengths = np.clip(military_strengths, mil_lo, mil_hi)

    # climate_stress_level biases climate risk
    climate_base = np.random.beta(a=2, b=3, size=n)
    climate_risks = np.clip(climate_base * (0.5 + climate_stress_level), clim_lo, clim_hi)

    education_indices = np.random.beta(a=3, b=2, size=n)
    education_indices = np.clip(education_indices, edu_lo, edu_hi)

    # Stability is partially derived from other attributes
    stability_indices = np.clip(
        0.3 * technology_levels
        + 0.2 * resource_reserves
        + 0.2 * education_indices
        - 0.3 * climate_risks
        + np.random.normal(0, 0.05, size=n),
        stab_lo,
        stab_hi,
    )

    trade_volumes = np.random.lognormal(mean=19.0, sigma=2.0, size=n)
    trade_volumes = np.clip(trade_volumes, trade_lo, trade_hi)

    gdp_per_capita = gdps / populations

    df = pd.DataFrame(
        {
            "country_name": names,
            "population": populations.astype(int),
            "gdp": gdps,
            "gdp_per_capita": gdp_per_capita,
            "technology_level": np.round(technology_levels, 4),
            "resource_reserves": np.round(resource_reserves, 4),
            "military_strength": np.round(military_strengths, 4),
            "climate_risk": np.round(climate_risks, 4),
            "stability_index": np.round(stability_indices, 4),
            "education_index": np.round(education_indices, 4),
            "trade_volume": trade_volumes,
        }
    )

    return df.reset_index(drop=True)
