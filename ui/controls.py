"""
ui/controls.py
--------------
Sidebar controls for the Streamlit dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

import config


@dataclass
class SimulationParams:
    """Container for sidebar parameter values."""

    num_countries: int
    resource_level: float
    technology_growth_rate: float
    climate_stress_level: float
    simulation_rounds: int
    run_simulation: bool


def render_sidebar() -> SimulationParams:
    """Render the sidebar and return the current parameter values.

    Returns
    -------
    SimulationParams
        The user-selected parameter values and the state of the run button.
    """
    st.sidebar.title("⚙️ Simulation Controls")
    st.sidebar.markdown("---")

    num_countries = st.sidebar.slider(
        "Number of Countries",
        min_value=5,
        max_value=35,
        value=config.DEFAULT_NUM_COUNTRIES,
        step=1,
        help="Total number of AI-controlled countries in the simulation.",
    )

    resource_level = st.sidebar.slider(
        "Resource Level",
        min_value=0.1,
        max_value=1.0,
        value=config.DEFAULT_RESOURCE_LEVEL,
        step=0.05,
        help="Global baseline resource richness (0 = scarce, 1 = abundant).",
    )

    technology_growth_rate = st.sidebar.slider(
        "Technology Growth Rate",
        min_value=0.00,
        max_value=0.10,
        value=config.DEFAULT_TECHNOLOGY_GROWTH_RATE,
        step=0.005,
        format="%.3f",
        help="Annual technology advancement rate applied each round.",
    )

    climate_stress_level = st.sidebar.slider(
        "Climate Stress Level",
        min_value=0.0,
        max_value=1.0,
        value=config.DEFAULT_CLIMATE_STRESS_LEVEL,
        step=0.05,
        help="Intensity of climate-change pressure on countries.",
    )

    simulation_rounds = st.sidebar.slider(
        "Simulation Rounds",
        min_value=2,
        max_value=30,
        value=config.DEFAULT_SIMULATION_ROUNDS,
        step=1,
        help="Number of time steps to simulate.",
    )

    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button(
        "🚀 Run Simulation",
        use_container_width=True,
        type="primary",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Civilization AI** — Autonomous World Simulation Engine\n\n"
        "Each country acts as an independent AI agent that selects the action "
        "maximising its economic utility each round."
    )

    return SimulationParams(
        num_countries=num_countries,
        resource_level=resource_level,
        technology_growth_rate=technology_growth_rate,
        climate_stress_level=climate_stress_level,
        simulation_rounds=simulation_rounds,
        run_simulation=run_simulation,
    )
