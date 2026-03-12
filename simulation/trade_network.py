"""
simulation/trade_network.py
----------------------------
NetworkX-based global trade network.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

import config


class TradeNetwork:
    """Models bilateral trade relationships as a weighted directed graph.

    Nodes represent countries; directed edges carry ``weight`` (trade volume)
    and ``resource_flow`` attributes.

    Attributes
    ----------
    graph : nx.DiGraph
        The underlying NetworkX directed graph.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame, seed: Optional[int] = 42) -> "TradeNetwork":
        """Initialise the trade network from a world DataFrame.

        Parameters
        ----------
        df:
            World state with ``country_name`` and ``trade_volume``.
        seed:
            Random seed for edge generation.

        Returns
        -------
        TradeNetwork
        """
        rng = np.random.default_rng(seed)
        self.graph = nx.DiGraph()

        # Add nodes
        for _, row in df.iterrows():
            self.graph.add_node(
                row["country_name"],
                gdp=row["gdp"],
                population=row["population"],
                trade_volume=row["trade_volume"],
                stability=row["stability_index"],
            )

        # Add directed edges (Erdős–Rényi with volume-proportional weights)
        names = df["country_name"].tolist()
        max_volume = df["trade_volume"].max()

        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                if i == j:
                    continue
                if rng.random() < config.TRADE_EDGE_PROBABILITY:
                    frac = rng.uniform(
                        config.MIN_TRADE_VOLUME_FRACTION,
                        config.MAX_TRADE_VOLUME_FRACTION,
                    )
                    weight = df.loc[df["country_name"] == src, "trade_volume"].values[0] * frac
                    resource_flow = rng.uniform(0.01, 0.30)
                    self.graph.add_edge(
                        src,
                        dst,
                        weight=weight,
                        resource_flow=resource_flow,
                    )

        return self

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(self, df: pd.DataFrame) -> None:
        """Update node attributes and re-weight edges from new world state.

        Parameters
        ----------
        df:
            Updated world state DataFrame.
        """
        for _, row in df.iterrows():
            name = row["country_name"]
            if name in self.graph.nodes:
                self.graph.nodes[name].update(
                    {
                        "gdp": row["gdp"],
                        "population": row["population"],
                        "trade_volume": row["trade_volume"],
                        "stability": row["stability_index"],
                    }
                )
        # Scale edge weights proportional to updated trade volumes
        for src, dst, data in self.graph.edges(data=True):
            src_vol = self.graph.nodes[src].get("trade_volume", 1)
            frac = data.get("weight", 0) / (
                self.graph.nodes[src].get("trade_volume", 1) + 1e-9
            )
            data["weight"] = src_vol * frac

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_centrality(self) -> pd.Series:
        """Return PageRank-based trade centrality per country."""
        pr = nx.pagerank(self.graph, weight="weight")
        return pd.Series(pr, name="trade_centrality")

    def get_total_trade_volume(self) -> float:
        """Sum of all edge weights in the graph."""
        return sum(d["weight"] for _, _, d in self.graph.edges(data=True))

    def get_edge_dataframe(self) -> pd.DataFrame:
        """Return edges as a tidy DataFrame for visualisation."""
        rows = []
        for src, dst, data in self.graph.edges(data=True):
            rows.append(
                {
                    "source": src,
                    "target": dst,
                    "weight": data.get("weight", 0),
                    "resource_flow": data.get("resource_flow", 0),
                }
            )
        return pd.DataFrame(rows)

    def get_trade_partners(self, country: str) -> list[str]:
        """Return the list of countries *country* exports to."""
        return list(self.graph.successors(country))
