"""
ui/charts.py
-------------
Plotly chart builders used throughout the dashboard.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config


# ---------------------------------------------------------------------------
# KPI / trend charts
# ---------------------------------------------------------------------------


def gdp_growth_chart(gdp_ts: pd.DataFrame) -> go.Figure:
    """Line chart of GDP over rounds for each country.

    Parameters
    ----------
    gdp_ts:
        Wide-format DataFrame (rows=rounds, columns=country names).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    long = gdp_ts.reset_index().melt(id_vars="index", var_name="Country", value_name="GDP")
    long.rename(columns={"index": "Round"}, inplace=True)
    long["GDP (Trillion $)"] = long["GDP"] / 1e12

    fig = px.line(
        long,
        x="Round",
        y="GDP (Trillion $)",
        color="Country",
        title="GDP Growth Over Simulation Rounds",
        height=config.CHART_HEIGHT,
        template="plotly_dark",
    )
    fig.update_layout(legend=dict(orientation="v", x=1.02))
    return fig


def population_trend_chart(pop_ts: pd.DataFrame) -> go.Figure:
    """Line chart of population over rounds."""
    long = pop_ts.reset_index().melt(id_vars="index", var_name="Country", value_name="Population")
    long.rename(columns={"index": "Round"}, inplace=True)
    long["Population (M)"] = long["Population"] / 1e6

    fig = px.line(
        long,
        x="Round",
        y="Population (M)",
        color="Country",
        title="Population Trends Over Simulation Rounds",
        height=config.CHART_HEIGHT,
        template="plotly_dark",
    )
    fig.update_layout(legend=dict(orientation="v", x=1.02))
    return fig


def technology_growth_chart(tech_ts: pd.DataFrame) -> go.Figure:
    """Line chart of technology index over rounds."""
    long = tech_ts.reset_index().melt(
        id_vars="index", var_name="Country", value_name="Technology Level"
    )
    long.rename(columns={"index": "Round"}, inplace=True)

    fig = px.line(
        long,
        x="Round",
        y="Technology Level",
        color="Country",
        title="Technology Index Growth Over Simulation Rounds",
        height=config.CHART_HEIGHT,
        template="plotly_dark",
    )
    fig.update_layout(yaxis=dict(range=[0, 1]), legend=dict(orientation="v", x=1.02))
    return fig


def conflict_risk_chart(conflict_ts: pd.DataFrame) -> go.Figure:
    """Area chart of conflict risk over rounds."""
    long = conflict_ts.reset_index().melt(
        id_vars="index", var_name="Country", value_name="Conflict Risk"
    )
    long.rename(columns={"index": "Round"}, inplace=True)

    fig = px.area(
        long,
        x="Round",
        y="Conflict Risk",
        color="Country",
        title="Geopolitical Conflict Risk Over Simulation Rounds",
        height=config.CHART_HEIGHT,
        template="plotly_dark",
    )
    fig.update_layout(yaxis=dict(range=[0, 1]), legend=dict(orientation="v", x=1.02))
    return fig


def global_kpi_trends(metrics_df: pd.DataFrame) -> go.Figure:
    """Multi-trace chart showing global aggregate KPIs per round."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=metrics_df["round"],
            y=metrics_df["global_gdp"] / 1e12,
            name="Global GDP (T$)",
            mode="lines+markers",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_df["round"],
            y=metrics_df["avg_stability"],
            name="Avg Stability",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_df["round"],
            y=metrics_df["avg_conflict_risk"],
            name="Avg Conflict Risk",
            mode="lines+markers",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Global KPI Trends",
        xaxis=dict(title="Simulation Round"),
        yaxis=dict(title="GDP (Trillion $)", side="left"),
        yaxis2=dict(title="Index [0–1]", side="right", overlaying="y", range=[0, 1]),
        template="plotly_dark",
        height=config.CHART_HEIGHT,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ---------------------------------------------------------------------------
# Trade network chart
# ---------------------------------------------------------------------------


def trade_network_chart(graph: "nx.DiGraph", final_state: pd.DataFrame) -> go.Figure:
    """Render the trade network as an interactive Plotly graph."""
    pos = nx.spring_layout(graph, seed=42, k=1.5)

    # Edges
    edge_x: list[float] = []
    edge_y: list[float] = []
    for src, dst in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="#888"),
        hoverinfo="none",
    )

    # Nodes
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    gdp_map = final_state.set_index("country_name")["gdp"].to_dict()

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        gdp_val = gdp_map.get(node, 0)
        node_text.append(f"{node}<br>GDP: ${gdp_val / 1e9:.1f}B")
        node_size.append(10 + (gdp_val / 1e12) * 3)
        node_color.append(gdp_val)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_text,
        text=[n[:4] for n in graph.nodes()],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            size=node_size,
            color=node_color,
            colorbar=dict(title="GDP ($)"),
            line_width=1,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Global Trade Network",
            showlegend=False,
            hovermode="closest",
            template="plotly_dark",
            height=config.CHART_HEIGHT,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Bar / ranking charts
# ---------------------------------------------------------------------------


def top_gdp_bar(final_state: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top *top_n* countries by GDP."""
    df = final_state.nlargest(top_n, "gdp")[["country_name", "gdp"]].copy()
    df["gdp_t"] = df["gdp"] / 1e12
    fig = px.bar(
        df,
        x="gdp_t",
        y="country_name",
        orientation="h",
        title=f"Top {top_n} Economies by GDP",
        labels={"gdp_t": "GDP (Trillion $)", "country_name": "Country"},
        color="gdp_t",
        color_continuous_scale="Teal",
        template="plotly_dark",
        height=config.CHART_HEIGHT,
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return fig


def conflict_risk_bar(final_state: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of highest conflict-risk countries."""
    df = final_state.nlargest(top_n, "conflict_risk")[
        ["country_name", "conflict_risk"]
    ].copy()
    fig = px.bar(
        df,
        x="conflict_risk",
        y="country_name",
        orientation="h",
        title=f"Top {top_n} Countries by Conflict Risk",
        labels={"conflict_risk": "Conflict Probability", "country_name": "Country"},
        color="conflict_risk",
        color_continuous_scale="Reds",
        template="plotly_dark",
        height=config.CHART_HEIGHT,
    )
    fig.update_layout(xaxis=dict(range=[0, 1]), yaxis=dict(categoryorder="total ascending"))
    return fig


def stability_radar(final_state: pd.DataFrame, countries: list[str]) -> go.Figure:
    """Radar chart comparing selected countries across key indices."""
    categories = [
        "stability_index",
        "technology_level",
        "resource_reserves",
        "education_index",
        "military_strength",
    ]
    labels = [c.replace("_", " ").title() for c in categories]

    fig = go.Figure()
    for country in countries:
        row = final_state[final_state["country_name"] == country]
        if row.empty:
            continue
        values = row[categories].values[0].tolist()
        values += [values[0]]  # close the polygon
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels + [labels[0]],
                fill="toself",
                name=country,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Country Comparison Radar",
        template="plotly_dark",
        height=config.CHART_HEIGHT,
    )
    return fig
