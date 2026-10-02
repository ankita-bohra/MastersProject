
"""
RNQI Hierarchy Pillar
----------------------
Operational implementation of the three Hierarchy metrics named in the
RNQI documentation:

1. Betweenness Centrality
2. Road Class Distribution
3. Hierarchy Clarity

IMPORTANT:
The RNQI documentation names these metrics and describes their purpose, but
does not provide complete mathematical formulas or a sampling specification.
The operational definitions below are therefore stated explicitly rather than
being presented as formulas mandated by the RNQI documentation.

Operational definitions
-----------------------
A) Betweenness Centrality
   Approximate weighted edge betweenness is calculated on a simple undirected
   representation of the road graph using a reproducible random sample of
   source nodes. The raw city metric is the 90th percentile of normalized
   edge betweenness. This captures the prominence of high-flow structural
   conduits rather than the mean (which is not useful as a cross-network
   hierarchy measure).

B) Road Class Distribution
   OSM highway classes are mapped to an ordered functional hierarchy:
       motorway, trunk, primary, secondary, tertiary,
       unclassified, residential, living_street/service
   Length-weighted Shannon entropy is calculated across these classes and
   normalized by log(8), the maximum number of defined classes. This measures
   whether the network contains a differentiated distribution of road classes.

C) Hierarchy Clarity
   Spearman rank correlation between functional road-class order and edge
   betweenness. Higher positive values mean that higher-order roads tend to
   carry greater structural importance. The correlation is calculated over
   the same physical road links used for betweenness.

Scoring
-------
All three raw metrics are normalized across the five reference cities using
the RNQI min-max 0-100 convention. The Hierarchy pillar is the arithmetic
mean of the three normalized metrics.

The betweenness approximation is an operational computational choice because
exact betweenness on these city-scale graphs is prohibitively expensive.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CITIES = ["Chandigarh", "Jaipur", "Pune", "Kolkata", "Ranipet"]
RANDOM_SEED = 42

# Reproducible approximation: one source sample is used per city.
# This is a computational choice for scalability, not an RNQI-mandated sample.
BETWEENNESS_SOURCE_FRACTION = 0.005
BETWEENNESS_MIN_SOURCES = 100
BETWEENNESS_MAX_SOURCES = 500

# Road-class hierarchy: lower number represents a higher functional class.
ROAD_CLASS_ORDER = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 7,
    "living_street": 8,
    "service": 8,
}

ROAD_CLASS_ALIASES = {
    "motorway_link": "motorway",
    "trunk_link": "trunk",
    "primary_link": "primary",
    "secondary_link": "secondary",
    "tertiary_link": "tertiary",
}


def graph_path(city: str) -> Path:
    return Path("outputs") / "graphs" / f"{city.lower()}.graphml"


def load_graph(city: str) -> nx.MultiGraph:
    path = graph_path(city)
    if not path.exists():
        raise FileNotFoundError(f"Graph not found: {path}")

    G = nx.read_graphml(path)
    G = nx.MultiGraph(G)

    # GraphML may deserialize node IDs as strings and edge lengths as strings.
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            data["length"] = float(data.get("length", 1.0))
        except (TypeError, ValueError):
            data["length"] = 1.0

    return G


def highway_class(value) -> str | None:
    """Extract one normalized OSM highway class from a GraphML attribute."""
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        text = str(value)
        # GraphML often serializes a list as "['primary', 'secondary']".
        if text.startswith("[") and text.endswith("]"):
            text = text.strip("[]").replace("'", "").replace('"', "")
            values = [x.strip() for x in text.split(",") if x.strip()]
        else:
            values = [text]

    normalized = []
    for item in values:
        cls = str(item).strip().lower()
        cls = ROAD_CLASS_ALIASES.get(cls, cls)
        if cls in ROAD_CLASS_ORDER:
            normalized.append(cls)

    if not normalized:
        return None

    # If multiple functional tags exist, retain the highest-order class.
    return min(normalized, key=lambda x: ROAD_CLASS_ORDER[x])


def build_simple_graph(G_multi: nx.MultiGraph) -> nx.Graph:
    """
    Convert the MultiGraph to a simple undirected graph for betweenness.
    For parallel physical links, retain the minimum-length connection.
    """
    G = nx.Graph()

    for node, data in G_multi.nodes(data=True):
        G.add_node(node, **data)

    for u, v, data in G_multi.edges(data=True):
        length = float(data.get("length", 1.0))
        highway = highway_class(data.get("highway"))

        if G.has_edge(u, v):
            if length < G[u][v]["length"]:
                G[u][v]["length"] = length
                G[u][v]["highway"] = highway
        else:
            G.add_edge(u, v, length=length, highway=highway)

    return G


def choose_source_count(n: int) -> int:
    k = int(math.ceil(n * BETWEENNESS_SOURCE_FRACTION))
    return min(BETWEENNESS_MAX_SOURCES, max(BETWEENNESS_MIN_SOURCES, k))


def calculate_betweenness_metrics(G: nx.Graph, seed: int = RANDOM_SEED):
    n = len(G)
    k = min(n, choose_source_count(n))

    if k >= n:
        bc = nx.edge_betweenness_centrality(
            G, normalized=True, weight="length"
        )
    else:
        bc = nx.edge_betweenness_centrality(
            G,
            k=k,
            normalized=True,
            weight="length",
            seed=seed,
        )

    values = np.asarray(list(bc.values()), dtype=float)

    if values.size == 0:
        return {
            "betweenness_centrality": np.nan,
            "betweenness_source_count": k,
            "betweenness_p90": np.nan,
        }, bc

    p90 = float(np.quantile(values, 0.90))

    return {
        "betweenness_centrality": p90,
        "betweenness_source_count": k,
        "betweenness_p90": p90,
    }, bc


def calculate_road_class_distribution(G_multi: nx.MultiGraph):
    class_length = {}

    for _, _, data in G_multi.edges(data=True):
        cls = highway_class(data.get("highway"))
        if cls is None:
            continue

        try:
            length = float(data.get("length", 1.0))
        except (TypeError, ValueError):
            length = 1.0

        class_length[cls] = class_length.get(cls, 0.0) + max(length, 0.0)

    total = sum(class_length.values())

    if total <= 0 or not class_length:
        return np.nan, class_length

    proportions = np.asarray(
        [length / total for length in class_length.values()],
        dtype=float,
    )

    entropy = float(-np.sum(proportions * np.log(proportions)))
    max_entropy = math.log(len(ROAD_CLASS_ORDER))

    score = entropy / max_entropy if max_entropy > 0 else np.nan

    return float(score), class_length


def calculate_hierarchy_clarity(G: nx.Graph, edge_bc: dict):
    rows = []

    for (u, v), bc in edge_bc.items():
        data = G.get_edge_data(u, v, default={})
        cls = data.get("highway")
        if cls not in ROAD_CLASS_ORDER:
            continue

        rows.append(
            {
                "road_class_rank": ROAD_CLASS_ORDER[cls],
                "betweenness": float(bc),
            }
        )

    if len(rows) < 3:
        return np.nan, len(rows)

    df = pd.DataFrame(rows)

    # Road-class rank uses lower numbers for higher-order roads.
    # Reverse the rank so larger values represent higher-order roads
    # before calculating the Spearman correlation.
    class_importance = -df["road_class_rank"].to_numpy(dtype=float)
    bc_values = df["betweenness"].to_numpy(dtype=float)

    rho, _ = spearmanr(class_importance, bc_values)

    return float(rho), len(rows)


def calculate_city(city: str):
    start = time.perf_counter()

    print("\n" + "=" * 72)
    print(f"HIERARCHY: {city}")
    print("=" * 72)

    G_multi = load_graph(city)
    G = build_simple_graph(G_multi)

    print(f"MultiGraph nodes : {len(G_multi):,}")
    print(f"MultiGraph links : {G_multi.number_of_edges():,}")
    print(f"Simple graph nodes: {len(G):,}")
    print(f"Simple graph links: {G.number_of_edges():,}")

    print("\nCalculating approximate weighted edge betweenness...")
    bc_metrics, edge_bc = calculate_betweenness_metrics(G)
    print(
        f"Betweenness sources: "
        f"{bc_metrics['betweenness_source_count']:,}"
    )
    print(
        f"Betweenness P90    : "
        f"{bc_metrics['betweenness_centrality']:.10g}"
    )

    print("\nCalculating road-class distribution...")
    rcd, class_length = calculate_road_class_distribution(G_multi)

    print("\nLength-weighted road-class shares:")
    total_length = sum(class_length.values())
    if total_length > 0:
        for cls in sorted(class_length, key=lambda x: ROAD_CLASS_ORDER[x]):
            share = class_length[cls] / total_length
            print(f"  {cls:14s}: {share:.4%}")

    print(f"Road class distribution (entropy): {rcd:.10g}")

    print("\nCalculating hierarchy clarity...")
    clarity, clarity_n = calculate_hierarchy_clarity(G, edge_bc)
    print(f"Hierarchy clarity (Spearman rho): {clarity:.10g}")
    print(f"Edges used for clarity: {clarity_n:,}")

    elapsed = time.perf_counter() - start

    result = {
        "city": city,
        "number_of_nodes": len(G_multi),
        "number_of_links": G_multi.number_of_edges(),
        "hierarchy_betweenness_centrality": bc_metrics["betweenness_centrality"],
        "hierarchy_road_class_distribution": rcd,
        "hierarchy_clarity": clarity,
        "hierarchy_betweenness_sources": bc_metrics[
            "betweenness_source_count"
        ],
        "hierarchy_clarity_edges": clarity_n,
        "hierarchy_random_seed": RANDOM_SEED,
        "hierarchy_betweenness_source_fraction":
            BETWEENNESS_SOURCE_FRACTION,
        "hierarchy_method":
            "weighted approximate edge betweenness P90; "
            "length-weighted road-class entropy; "
            "Spearman road-class/betweenness correlation",
        "execution_time_seconds": elapsed,
    }

    print(f"\nCalculation time: {elapsed:.2f} seconds")

    return result


def minmax_normalize(values: pd.Series, higher_is_better=True):
    values = pd.to_numeric(values, errors="coerce")
    vmin = values.min()
    vmax = values.max()

    if math.isclose(vmax, vmin):
        return pd.Series(50.0, index=values.index)

    if higher_is_better:
        return 100.0 * (values - vmin) / (vmax - vmin)

    return 100.0 * (vmax - values) / (vmax - vmin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--city",
        default=None,
        choices=CITIES,
        help="Run one city. Omit to run all five.",
    )
    args = parser.parse_args()

    cities = [args.city] if args.city else CITIES

    results = []

    for city in cities:
        results.append(calculate_city(city))

    raw = pd.DataFrame(results)

    out_dir = Path("outputs") / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "city_wise_hierarchy_metrics.csv"
    raw.to_csv(raw_path, index=False)

    # All three raw metrics are oriented so that higher values represent
    # a stronger hierarchy signal before min-max normalization.
    raw["hierarchy_betweenness_score"] = minmax_normalize(
        raw["hierarchy_betweenness_centrality"], True
    )
    raw["hierarchy_road_class_distribution_score"] = minmax_normalize(
        raw["hierarchy_road_class_distribution"], True
    )
    raw["hierarchy_clarity_score"] = minmax_normalize(
        raw["hierarchy_clarity"], True
    )

    raw["hierarchy_score"] = raw[
        [
            "hierarchy_betweenness_score",
            "hierarchy_road_class_distribution_score",
            "hierarchy_clarity_score",
        ]
    ].mean(axis=1)

    score_cols = [
        "city",
        "hierarchy_betweenness_score",
        "hierarchy_road_class_distribution_score",
        "hierarchy_clarity_score",
        "hierarchy_score",
    ]

    scores = raw[score_cols].copy()

    scores_path = out_dir / "city_wise_hierarchy_scores.csv"
    scores.to_csv(scores_path, index=False)

    reference_cols = [
        "city",
        "hierarchy_betweenness_centrality",
        "hierarchy_road_class_distribution",
        "hierarchy_clarity",
    ]
    reference = raw[reference_cols].copy()

    reference_path = out_dir / "hierarchy_normalization_reference.csv"
    reference.to_csv(reference_path, index=False)

    print("\n" + "=" * 72)
    print("HIERARCHY PIPELINE SUMMARY")
    print("=" * 72)
    print(f"Cities processed: {len(cities)}/{len(CITIES)}")
    print(f"Raw metrics saved to: {raw_path}")
    print(f"Hierarchy scores saved to: {scores_path}")
    print(f"Normalization reference saved to: {reference_path}")

    print("\nHIERARCHY SCORES")
    print(scores.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
