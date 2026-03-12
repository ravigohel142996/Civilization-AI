"""
app.py
------
Entry point for the Civilization AI Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config
from analytics.global_metrics import GlobalMetrics
from simulation.civilization_engine import CivilizationEngine, SimulationConfig
from ui.controls import render_sidebar
from ui.dashboard import (
    render_conflict_monitor,
    render_country_comparison,
    render_economy_analytics,
    render_global_overview,
    render_trade_network,
    render_world_map,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a more polished enterprise look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #f8fbff 0%, #f0f5ff 35%, #e9eefb 100%);
        color: #17223b;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #14213d;
        letter-spacing: 0.01em;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #17203f, #1f2b54);
        border: 1px solid #2f3f79;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 12px 24px rgba(20, 33, 61, 0.2);
    }

    div[data-testid="stMetricLabel"] {
        color: #c8d5ff !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.15;
    }

    div[data-testid="stMetricDelta"] {
        color: #93c5fd !important;
    }

    div[role="radiogroup"] {
        background: #ffffff;
        border: 1px solid #d7e1f5;
        border-radius: 12px;
        padding: 0.45rem;
        box-shadow: 0 4px 16px rgba(20, 33, 61, 0.08);
    }

    div[role="radiogroup"] label {
        border-radius: 8px;
        padding: 0.4rem 0.7rem;
        transition: all 0.2s ease;
    }

    div[role="radiogroup"] label:has(input:checked) {
        background: #e9f2ff;
        color: #0f3d8c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🌍 Civilization AI — Autonomous World Simulation Engine")
st.markdown(
    "*A production-quality multi-agent simulation platform that models "
    "AI-controlled nations evolving across economic, geopolitical, and "
    "technological dimensions.*"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

params = render_sidebar()

# ---------------------------------------------------------------------------
# Session state management
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state["result"] = None
    st.session_state["trade_network"] = None

if params.run_simulation:
    sim_cfg = SimulationConfig(
        num_countries=params.num_countries,
        simulation_rounds=params.simulation_rounds,
        resource_level=params.resource_level,
        technology_growth_rate=params.technology_growth_rate,
        climate_stress_level=params.climate_stress_level,
    )

    with st.spinner("⏳ Running simulation… This may take a few seconds."):
        engine = CivilizationEngine(sim_cfg)
        result = engine.run()
        st.session_state["result"] = result
        st.session_state["trade_network"] = engine._trade_network

    st.success(
        f"✅ Simulation complete — {params.simulation_rounds} rounds, "
        f"{params.num_countries} countries."
    )

# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

if st.session_state["result"] is not None:
    result = st.session_state["result"]
    trade_net = st.session_state["trade_network"]
    metrics = GlobalMetrics(result)

    sections = [
        "🌐 Global Overview",
        "📈 Economy Analytics",
        "🔗 Trade Network",
        "⚔️ Conflict Monitor",
        "🗺️ World Map",
        "📊 Country Comparison",
    ]

    selected_section = st.radio(
        "Navigate to section",
        sections,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    if selected_section == sections[0]:
        render_global_overview(metrics)
    elif selected_section == sections[1]:
        render_economy_analytics(metrics)
    elif selected_section == sections[2]:
        render_trade_network(trade_net, result.final_state)
    elif selected_section == sections[3]:
        render_conflict_monitor(metrics)
    elif selected_section == sections[4]:
        render_world_map(result.final_state)
    elif selected_section == sections[5]:
        render_country_comparison(result.final_state)

else:
    # Landing / welcome state
    st.info(
        "👈 **Configure the simulation parameters in the sidebar and click "
        "'Run Simulation' to start.**\n\n"
        "The engine will generate a synthetic world of AI-controlled nations, "
        "train ML models, run multi-agent decisions, and visualise the results."
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        """
        **🤖 Multi-Agent AI**
        Each country acts as an autonomous agent selecting actions
        that maximise its economic and stability utility.
        """
    )
    col2.markdown(
        """
        **📊 ML-Powered Predictions**
        GDP, population growth, and conflict probability are all
        predicted using trained scikit-learn models.
        """
    )
    col3.markdown(
        """
        **🔗 Trade Network**
        A NetworkX-based global trade graph tracks bilateral
        dependencies and commerce flows.
        """
    )
