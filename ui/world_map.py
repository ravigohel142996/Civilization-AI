"""
ui/world_map.py
---------------
Choropleth-style world-map visualisations using Plotly Express.

Note: Because countries are *fictional*, we use a scatter_geo plot with
country labels positioned at approximate lat/lon values generated
deterministically from the country index. This provides a map-like visual
without requiring real geographic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config


def _assign_pseudo_coords(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign deterministic pseudo lat/lon to fictional countries."""
    rng = np.random.default_rng(seed)
    n = len(df)
    lats = rng.uniform(-60, 75, size=n)
    lons = rng.uniform(-170, 170, size=n)
    df = df.copy()
    df["lat"] = np.round(lats, 2)
    df["lon"] = np.round(lons, 2)
    return df


def world_map(final_state: pd.DataFrame, metric: str = "gdp") -> go.Figure:
    """Render a pseudo world map coloured by *metric*.

    Parameters
    ----------
    final_state:
        World state DataFrame for the final simulation round.
    metric:
        Column name to use for colouring.  One of:
        ``'gdp'``, ``'population'``, ``'stability_index'``,
        ``'conflict_risk'``, ``'technology_level'``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = _assign_pseudo_coords(final_state)

    metric_labels: dict[str, str] = {
        "gdp": "GDP ($)",
        "population": "Population",
        "stability_index": "Stability Index",
        "conflict_risk": "Conflict Risk",
        "technology_level": "Technology Level",
    }

    colour_scales: dict[str, str] = {
        "gdp": "Viridis",
        "population": "Blues",
        "stability_index": "Greens",
        "conflict_risk": "Reds",
        "technology_level": "Plasma",
    }

    label = metric_labels.get(metric, metric.replace("_", " ").title())
    cscale = colour_scales.get(metric, "Viridis")

    display_col = metric
    if metric == "gdp":
        df = df.copy()
        df["gdp_display"] = df["gdp"] / 1e9
        display_col = "gdp_display"
        label = "GDP (Billion $)"

    hover_cols = {
        "country_name": True,
        "population": True,
        "stability_index": ":.3f",
        "conflict_risk": ":.3f",
        "technology_level": ":.3f",
    }

    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color=display_col,
        hover_name="country_name",
        hover_data=hover_cols,
        size=display_col,
        size_max=30,
        color_continuous_scale=cscale,
        projection=config.MAP_PROJECTION,
        title=f"World Map — {label}",
        template="plotly_dark",
        height=500,
    )
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="#1e1e2e",
            showocean=True,
            oceancolor="#0e0e1a",
            showlakes=False,
            showcountries=True,
            countrycolor="#444",
        ),
        coloraxis_colorbar=dict(title=label),
    )
    return fig
