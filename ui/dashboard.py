"""
ui/dashboard.py
---------------
Streamlit page layout helpers that compose the main dashboard sections.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from analytics.global_metrics import GlobalMetrics
from simulation.civilization_engine import SimulationResult
from simulation.trade_network import TradeNetwork
from ui import charts, world_map
from utils.helpers import fmt_gdp, fmt_percent, fmt_population


# ---------------------------------------------------------------------------
# Global overview
# ---------------------------------------------------------------------------


def render_global_overview(metrics: GlobalMetrics) -> None:
    """Render the Global Overview KPI section."""
    st.subheader("🌐 Global Overview")
    kpis = metrics.summary_kpis()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌍 World Population", fmt_population(kpis["world_population"]))
    col2.metric("💰 Global GDP", fmt_gdp(kpis["global_gdp_trillion"] * 1e12))
    col3.metric("⚖️ Avg Stability", f"{kpis['avg_stability_index']:.3f}")
    col4.metric("⚔️ Avg Conflict Risk", fmt_percent(kpis["avg_conflict_risk"]))

    col5, col6, col7 = st.columns(3)
    col5.metric("🔬 Avg Technology", f"{kpis['avg_technology_level']:.3f}")
    col6.metric("🌡️ Avg Climate Risk", fmt_percent(kpis["avg_climate_risk"]))
    col7.metric("🏳️ Countries", str(kpis["num_countries"]))

    st.plotly_chart(
        charts.global_kpi_trends(metrics.metrics_df),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Economy analytics
# ---------------------------------------------------------------------------


def render_economy_analytics(metrics: GlobalMetrics) -> None:
    """Render the Economy Analytics section."""
    st.subheader("📈 Economy Analytics")

    tab1, tab2, tab3 = st.tabs(["GDP Growth", "Population Trends", "Technology Index"])

    with tab1:
        st.plotly_chart(
            charts.gdp_growth_chart(metrics.gdp_time_series()),
            use_container_width=True,
        )
        st.plotly_chart(
            charts.top_gdp_bar(metrics.result.final_state),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            charts.population_trend_chart(metrics.population_time_series()),
            use_container_width=True,
        )

    with tab3:
        st.plotly_chart(
            charts.technology_growth_chart(metrics.technology_time_series()),
            use_container_width=True,
        )
        top_tech = metrics.technology_leaders()
        st.dataframe(top_tech, use_container_width=True)


# ---------------------------------------------------------------------------
# Trade network
# ---------------------------------------------------------------------------


def render_trade_network(trade_net: TradeNetwork, final_state: pd.DataFrame) -> None:
    """Render the Trade Network section."""
    st.subheader("🔗 Global Trade Network")
    st.plotly_chart(
        charts.trade_network_chart(trade_net.graph, final_state),
        use_container_width=True,
    )

    st.markdown("**Trade Volume by Edge (Top 20)**")
    edge_df = trade_net.get_edge_dataframe().nlargest(20, "weight")
    edge_df["weight"] = edge_df["weight"].apply(lambda v: fmt_gdp(v))
    st.dataframe(edge_df, use_container_width=True)

    centrality = (
        trade_net.get_centrality()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"index": "Country", "trade_centrality": "Centrality Score"})
    )
    st.markdown("**Top Trade Hubs (PageRank Centrality)**")
    st.dataframe(centrality, use_container_width=True)


# ---------------------------------------------------------------------------
# Conflict risk monitor
# ---------------------------------------------------------------------------


def render_conflict_monitor(metrics: GlobalMetrics) -> None:
    """Render the Conflict Risk Monitor section."""
    st.subheader("⚔️ Conflict Risk Monitor")

    st.plotly_chart(
        charts.conflict_risk_chart(metrics.conflict_risk_time_series()),
        use_container_width=True,
    )
    st.plotly_chart(
        charts.conflict_risk_bar(metrics.result.final_state),
        use_container_width=True,
    )

    st.markdown("**Highest Risk Countries — Final Round**")
    st.dataframe(metrics.highest_conflict_risk(), use_container_width=True)


# ---------------------------------------------------------------------------
# World map
# ---------------------------------------------------------------------------


def render_world_map(final_state: pd.DataFrame) -> None:
    """Render the World Map section."""
    st.subheader("🗺️ World Map")

    metric = st.selectbox(
        "Colour by metric",
        options=["gdp", "population", "stability_index", "conflict_risk", "technology_level"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    st.plotly_chart(
        world_map.world_map(final_state, metric=metric),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Country comparison
# ---------------------------------------------------------------------------


def render_country_comparison(final_state: pd.DataFrame) -> None:
    """Render a radar chart for comparing selected countries."""
    st.subheader("📊 Country Comparison")
    all_countries = sorted(final_state["country_name"].tolist())
    selected = st.multiselect(
        "Select countries to compare",
        options=all_countries,
        default=all_countries[:4],
    )
    if selected:
        st.plotly_chart(
            charts.stability_radar(final_state, selected),
            use_container_width=True,
        )

    st.markdown("**Full Country Table — Final Round**")
    display_cols = [
        "country_name", "population", "gdp", "gdp_per_capita",
        "technology_level", "stability_index", "conflict_risk",
        "military_strength", "resource_reserves", "climate_risk",
    ]
    st.dataframe(
        final_state[display_cols].sort_values("gdp", ascending=False),
        use_container_width=True,
    )
