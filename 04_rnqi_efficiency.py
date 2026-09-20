from pathlib import Path
import argparse
import json
import math
import time

import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import t


# ============================================================
# RNQI EFFICIENCY PILLAR
# ============================================================
#
# RNQI documentation names the following Efficiency metrics:
#   - Global Efficiency
#   - ASP Length
#   - Circuity
#   - RCRC
#
# The RNQI document describes these conceptually but does not give
# full computational formulas or a sampling scheme. Therefore this
# script makes the operational definitions explicit below rather
# than silently inventing them.
#
# Source-grounded definitions:
#   Global Efficiency: mean inverse shortest-path distance.
#   ASP Length: mean shortest-path length over reachable pairs.
#   Circuity: network shortest-path distance / straight-line distance.
#   RCRC: average road connectivity / average road circuity.
#
# For RCRC, the external methodological source used here defines
# RCRC as gamma connectivity divided by average road circuity.
# See: Dingil et al. (2019), DOI 10.2495/TDI-V3-N4-331-343.
#
# Computational design used here:
#   - undirected MultiGraph, preserving parallel road segments
#   - edge 'length' as routing distance in metres
#   - exact weighted Dijkstra for sampled source nodes
#   - simple random sampling without replacement
#   - two-stage adaptive sample sizing
#   - finite-population correction
#   - 95% t confidence intervals
#
# This sampling design is an implementation choice for scalability;
# it is NOT prescribed by the RNQI document.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
BOUNDARY_DIR = OUTPUT_DIR / "boundaries"
GRAPH_DIR = OUTPUT_DIR / "graphs"
METADATA_DIR = OUTPUT_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "Chandigarh": "chandigarh",
    "Jaipur": "jaipur",
    "Pune": "pune",
    "Kolkata": "kolkata",
    "Ranipet": "ranipet",
}

# Performance / sampling configuration
EFFICIENCY_BATCH_SIZE = 32
EFFICIENCY_RANDOM_SEED = 42
EFFICIENCY_PILOT_FRACTION = 0.02
EFFICIENCY_CONFIDENCE_LEVEL = 0.95
EFFICIENCY_RELATIVE_PRECISION = 0.02


# ============================================================
# DATA LOADING / GRAPH PREPARATION
# ============================================================

def load_city_boundary(city_name):
    path = BOUNDARY_DIR / f"{CITIES[city_name]}_boundary.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")
    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"Boundary file is empty: {path}")
    return boundary, path


def load_city_graph(city_name):
    path = GRAPH_DIR / f"{CITIES[city_name]}.graphml"
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    return ox.load_graphml(path), path


def prepare_undirected_graph(graph, city_name):
    print(f"Preparing undirected graph for {city_name}...")
    start = time.perf_counter()
    graph = nx.MultiGraph(graph)
    elapsed = time.perf_counter() - start
    components = nx.number_connected_components(graph)
    print(
        f"Undirected graph: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} links, {components:,} connected components"
    )
    return graph, elapsed


def calculate_city_area(boundary, graph):
    start = time.perf_counter()
    projected_boundary = boundary.to_crs(graph.graph["crs"])
    area_km2 = projected_boundary.geometry.area.sum() / 1_000_000
    return float(area_km2), time.perf_counter() - start


# ============================================================
# NODE COORDINATES
# ============================================================

def get_node_coordinates(graph, nodes):
    """Return node coordinates in metres when the graph CRS is projected."""
    xs = np.asarray([float(graph.nodes[n]["x"]) for n in nodes], dtype=np.float64)
    ys = np.asarray([float(graph.nodes[n]["y"]) for n in nodes], dtype=np.float64)

    crs = graph.graph.get("crs")
    is_geographic = bool(getattr(crs, "is_geographic", False))

    if not is_geographic:
        return xs, ys, False

    # Geographic graph: convert lon/lat degrees to approximate metres
    # using a local equirectangular projection around the graph centroid.
    lon0 = np.deg2rad(float(np.mean(xs)))
    lat0 = np.deg2rad(float(np.mean(ys)))
    earth_radius = 6_371_000.0

    x_m = np.deg2rad(xs) * earth_radius * np.cos(lat0)
    y_m = np.deg2rad(ys) * earth_radius
    return x_m, y_m, True


# ============================================================
# WEIGHTED SPARSE ADJACENCY
# ============================================================

def build_weighted_adjacency(graph, nodes):
    """Build an undirected weighted sparse matrix using the shortest parallel-edge length."""
    node_to_index = {node: i for i, node in enumerate(nodes)}
    best = {}

    for u, v, data in graph.edges(data=True):
        try:
            length = float(data.get("length"))
        except (TypeError, ValueError):
            raise ValueError(
                "Efficiency requires an edge 'length' attribute in metres. "
                f"Missing/invalid length on edge {u}-{v}."
            )

        if not np.isfinite(length) or length <= 0:
            raise ValueError(
                f"Invalid edge length {length!r} on edge {u}-{v}."
            )

        i = node_to_index[u]
        j = node_to_index[v]
        key = (min(i, j), max(i, j))

        if key not in best or length < best[key]:
            best[key] = length

    rows = []
    cols = []
    data = []

    for (i, j), length in best.items():
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([length, length])

    adjacency = coo_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(len(nodes), len(nodes)),
    ).tocsr()

    return adjacency


# ============================================================
# RCRC SUPPORT METRICS
# ============================================================

def calculate_average_edge_circuity(graph):
    """Average road circuity = edge length / straight-line endpoint distance."""
    nodes = list(graph.nodes())
    x, y, _ = get_node_coordinates(graph, nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    ratios = []

    # Each parallel road segment is retained as a link and contributes
    # separately, matching the MultiGraph representation used by RNQI.
    for u, v, data in graph.edges(data=True):
        length = float(data.get("length"))
        i = idx[u]
        j = idx[v]
        straight = math.hypot(x[i] - x[j], y[i] - y[j])
        if straight > 0:
            ratios.append(length / straight)

    if not ratios:
        raise ValueError("Could not calculate road circuity from graph edges.")

    return float(np.mean(ratios))


def calculate_gamma_connectivity(graph):
    """Planar gamma connectivity: E / [3(V-2)]."""
    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    if n <= 2:
        return 0.0
    return float(e / (3.0 * (n - 2)))


# ============================================================
# BATCHED EFFICIENCY METRICS
# ============================================================

def calculate_efficiency_for_sources(
    adjacency,
    source_indices,
    x_coords,
    y_coords,
    batch_size=32,
):
    """Exact weighted shortest paths for selected sources.

    Returns one per-source observation for each metric:
      global efficiency contribution
      average shortest path length over reachable destinations
      average pairwise circuity over reachable destinations
    """
    n = adjacency.shape[0]
    source_indices = np.asarray(source_indices, dtype=int)

    global_values = []
    asp_values = []
    circuity_values = []

    total = len(source_indices)

    for start in range(0, total, batch_size):
        batch = source_indices[start:start + batch_size]

        distances = dijkstra(
            adjacency,
            directed=False,
            indices=batch,
            unweighted=False,
        )

        for row, source in enumerate(batch):
            d = distances[row]
            finite = np.isfinite(d)
            finite[source] = False

            reachable = d[finite]

            if reachable.size == 0:
                global_values.append(0.0)
                asp_values.append(np.nan)
                circuity_values.append(np.nan)
                continue

            # Global efficiency uses the full node population as the denominator;
    # unreachable destinations therefore contribute zero.
            global_values.append(
                float(np.sum(1.0 / reachable) / (n - 1))
            )

            asp_values.append(float(np.mean(reachable)))

            destination_indices = np.flatnonzero(finite)
            dx = x_coords[destination_indices] - x_coords[source]
            dy = y_coords[destination_indices] - y_coords[source]
            euclidean = np.hypot(dx, dy)
            valid = euclidean > 0

            if np.any(valid):
                circuity_values.append(
                    float(np.mean(reachable[valid] / euclidean[valid]))
                )
            else:
                circuity_values.append(np.nan)

        completed = min(start + batch_size, total)
        print(
            f"      Efficiency sources: {completed:,}/{total:,}",
            end="\r",
        )

    print()

    return {
        "global_efficiency": np.asarray(global_values, dtype=np.float64),
        "asp_length": np.asarray(asp_values, dtype=np.float64),
        "circuity": np.asarray(circuity_values, dtype=np.float64),
    }


# ============================================================
# STATISTICS / SAMPLE SIZE
# ============================================================

def clean_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def required_sample_size(N, pilot_values, relative_precision, confidence_level, pilot_n):
    values = clean_values(pilot_values)
    if len(values) < 2:
        return N

    mean = abs(float(np.mean(values)))
    std = float(np.std(values, ddof=1))

    if mean <= 0 or std <= 0:
        return pilot_n

    margin = mean * relative_precision
    alpha = 1.0 - confidence_level
    critical = t.ppf(1.0 - alpha / 2.0, df=max(pilot_n - 1, 1))

    for _ in range(20):
        numerator = N * critical**2 * std**2
        denominator = (N - 1) * margin**2 + critical**2 * std**2
        n = math.ceil(numerator / denominator)
        n = min(max(n, pilot_n), N)
        new_critical = t.ppf(1.0 - alpha / 2.0, df=max(n - 1, 1))
        if abs(new_critical - critical) < 1e-10:
            break
        critical = new_critical

    return int(n)


def metric_statistics(values, N, confidence_level):
    values = clean_values(values)
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    if n <= 1:
        return {
            "mean": mean,
            "std": std,
            "standard_error": None,
            "ci_lower": None,
            "ci_upper": None,
            "relative_margin": None,
        }

    fpc = math.sqrt((N - n) / (N - 1))
    se = std / math.sqrt(n) * fpc
    critical = t.ppf(1.0 - (1.0 - confidence_level) / 2.0, df=n - 1)
    margin = critical * se

    return {
        "mean": mean,
        "std": std,
        "standard_error": float(se),
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
        "relative_margin": float(margin / abs(mean)) if mean else None,
    }


# ============================================================
# ADAPTIVE EFFICIENCY ESTIMATION
# ============================================================

def calculate_efficiency_adaptive(graph, random_seed=42):
    start_time = time.perf_counter()

    nodes = list(graph.nodes())
    N = len(nodes)
    if N < 2:
        raise ValueError("Graph must contain at least two nodes.")

    print("Building weighted sparse adjacency matrix...")
    adjacency = build_weighted_adjacency(graph, nodes)
    x, y, _ = get_node_coordinates(graph, nodes)

    rng = np.random.default_rng(random_seed)
    permutation = rng.permutation(N)

    pilot_n = min(max(2, math.ceil(N * EFFICIENCY_PILOT_FRACTION)), N)
    pilot_indices = permutation[:pilot_n]

    print()
    print("=" * 70)
    print("ADAPTIVE EFFICIENCY")
    print("=" * 70)
    print(f"Population nodes       : {N:,}")
    print(f"Pilot sample size      : {pilot_n:,}")
    print(f"Pilot sample fraction  : {pilot_n / N:.4%}")
    print(f"Confidence level       : {EFFICIENCY_CONFIDENCE_LEVEL:.0%}")
    print(f"Target relative margin : {EFFICIENCY_RELATIVE_PRECISION:.2%}")
    print()
    print("Stage 1: calculating pilot shortest paths...")

    pilot = calculate_efficiency_for_sources(
        adjacency, pilot_indices, x, y, EFFICIENCY_BATCH_SIZE
    )

    required = {}
    for metric, values in pilot.items():
        required[metric] = required_sample_size(
            N,
            values,
            EFFICIENCY_RELATIVE_PRECISION,
            EFFICIENCY_CONFIDENCE_LEVEL,
            pilot_n,
        )

    final_n = max(required.values())
    final_n = min(max(final_n, pilot_n), N)

    print()
    for metric, n_req in required.items():
        print(f"Required sample - {metric:18s}: {n_req:,}")
    print(f"Final common sample size : {final_n:,}")
    print(f"Final sample fraction    : {final_n / N:.4%}")

    if final_n > pilot_n:
        additional_indices = permutation[pilot_n:final_n]
        print()
        print(f"Stage 2: calculating additional {len(additional_indices):,} nodes...")
        additional = calculate_efficiency_for_sources(
            adjacency, additional_indices, x, y, EFFICIENCY_BATCH_SIZE
        )
        final = {
            metric: np.concatenate([pilot[metric], additional[metric]])
            for metric in pilot
        }
    else:
        print("Pilot sample already satisfies all estimated sample-size requirements.")
        final = pilot

    stats = {
        metric: metric_statistics(values, N, EFFICIENCY_CONFIDENCE_LEVEL)
        for metric, values in final.items()
    }

    print()
    print("=" * 70)
    print("FINAL EFFICIENCY RESULT")
    print("=" * 70)

    for metric in ["global_efficiency", "asp_length", "circuity"]:
        s = stats[metric]
        achieved = s["relative_margin"] is not None and s["relative_margin"] <= EFFICIENCY_RELATIVE_PRECISION
        print(f"{metric:20s}: {s['mean']:.12f}")
        print(f"  95% CI             : [{s['ci_lower']:.12f}, {s['ci_upper']:.12f}]")
        print(f"  Relative margin    : {s['relative_margin']:.4%}")
        print(f"  Target achieved    : {'YES' if achieved else 'NO'}")

    arc = calculate_average_edge_circuity(graph)
    gamma = calculate_gamma_connectivity(graph)
    rcrc = gamma / arc if arc > 0 else 0.0

    print(f"Average road circuity: {arc:.12f}")
    print(f"Gamma connectivity   : {gamma:.12f}")
    print(f"RCRC                 : {rcrc:.12f}")

    elapsed = time.perf_counter() - start_time
    print(f"Calculation time     : {elapsed:.2f} seconds")
    print("=" * 70)

    metadata = {
        "efficiency_global_efficiency": stats["global_efficiency"]["mean"],
        "efficiency_asp_length": stats["asp_length"]["mean"],
        "efficiency_circuity": stats["circuity"]["mean"],
        "efficiency_average_road_circuity": arc,
        "efficiency_gamma_connectivity": gamma,
        "efficiency_rcrc": rcrc,
        "efficiency_method": "Weighted shortest-path distances using OSM road length attributes",
        "efficiency_sampling": "Simple random sampling without replacement with adaptive common sample size",
        "efficiency_random_seed": random_seed,
        "efficiency_population_nodes": N,
        "efficiency_pilot_sample_size": pilot_n,
        "efficiency_final_sample_size": final_n,
        "efficiency_sample_fraction": final_n / N,
        "efficiency_confidence_level": EFFICIENCY_CONFIDENCE_LEVEL,
        "efficiency_relative_precision_target": EFFICIENCY_RELATIVE_PRECISION,
        "efficiency_global_efficiency_std": stats["global_efficiency"]["std"],
        "efficiency_global_efficiency_ci_lower": stats["global_efficiency"]["ci_lower"],
        "efficiency_global_efficiency_ci_upper": stats["global_efficiency"]["ci_upper"],
        "efficiency_global_efficiency_relative_margin": stats["global_efficiency"]["relative_margin"],
        "efficiency_asp_length_std": stats["asp_length"]["std"],
        "efficiency_asp_length_ci_lower": stats["asp_length"]["ci_lower"],
        "efficiency_asp_length_ci_upper": stats["asp_length"]["ci_upper"],
        "efficiency_asp_length_relative_margin": stats["asp_length"]["relative_margin"],
        "efficiency_circuity_std": stats["circuity"]["std"],
        "efficiency_circuity_ci_lower": stats["circuity"]["ci_lower"],
        "efficiency_circuity_ci_upper": stats["circuity"]["ci_upper"],
        "efficiency_circuity_relative_margin": stats["circuity"]["relative_margin"],
        "efficiency_calculation_seconds": elapsed,
    }
    return metadata


# ============================================================
# CITY PIPELINE
# ============================================================

def process_city(city_name):
    print("\n" + "=" * 70)
    print(f"Efficiency metrics: {city_name}")
    print("=" * 70)

    city_start = time.perf_counter()
    boundary, boundary_path = load_city_boundary(city_name)
    graph, graph_path = load_city_graph(city_name)
    graph, undirected_seconds = prepare_undirected_graph(graph, city_name)
    area_km2, area_seconds = calculate_city_area(boundary, graph)

    result = {
        "city": city_name,
        "number_of_nodes": graph.number_of_nodes(),
        "number_of_links": graph.number_of_edges(),
        "connected_components": nx.number_connected_components(graph),
        "city_area_km2": area_km2,
        "boundary_path": str(boundary_path),
        "graph_path": str(graph_path),
        "area_calculation_seconds": round(area_seconds, 2),
        "undirected_graph_seconds": round(undirected_seconds, 2),
    }

    result.update(calculate_efficiency_adaptive(graph, EFFICIENCY_RANDOM_SEED))
    result["execution_time_seconds"] = round(time.perf_counter() - city_start, 2)

    print(f"\nCompleted: {city_name}")
    print(json.dumps(result, indent=2))
    return result


# ============================================================
# NORMALIZATION / PILLAR SCORE
# ============================================================

def min_max(values, higher_is_better=True):
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        return np.full_like(values, 50.0), lo, hi
    if higher_is_better:
        scores = (values - lo) / (hi - lo) * 100.0
    else:
        scores = (hi - values) / (hi - lo) * 100.0
    return scores, lo, hi


def calculate_efficiency_scores(results):
    df = pd.DataFrame(results).copy()

    # Higher is better for Global Efficiency and RCRC.
    # Lower is better for ASP Length and Circuity.
    definitions = {
        "efficiency_global_efficiency": ("efficiency_global_efficiency_score", True),
        "efficiency_asp_length": ("efficiency_asp_length_score", False),
        "efficiency_circuity": ("efficiency_circuity_score", False),
        "efficiency_rcrc": ("efficiency_rcrc_score", True),
    }

    refs = []
    for raw, (score_col, higher) in definitions.items():
        scores, lo, hi = min_max(df[raw].to_numpy(), higher)
        df[score_col] = scores
        refs.append({
            "metric": raw,
            "minimum_reference_value": lo,
            "maximum_reference_value": hi,
            "direction": "higher-is-better" if higher else "lower-is-better",
            "normalization": "Min-max 0-100",
        })

    score_cols = [x[0] for x in definitions.values()]
    df["efficiency_score"] = df[score_cols].mean(axis=1)
    return df, pd.DataFrame(refs)


def save_results(results):
    raw_path = METADATA_DIR / "city_wise_efficiency_metrics.csv"
    score_path = METADATA_DIR / "city_wise_efficiency_scores.csv"
    ref_path = METADATA_DIR / "efficiency_normalization_reference.csv"

    raw_df = pd.DataFrame(results)
    raw_df.to_csv(raw_path, index=False)

    if set(CITIES).issubset(set(raw_df["city"])):
        scored, refs = calculate_efficiency_scores(results)
        scored.to_csv(score_path, index=False)
        refs.to_csv(ref_path, index=False)
        return raw_path, score_path, ref_path, scored

    return raw_path, None, None, None


def run_pipeline(selected_city=None):
    city_list = [selected_city] if selected_city else list(CITIES)
    results = []
    start = time.perf_counter()

    print("=" * 70)
    print("RNQI EFFICIENCY METRIC CALCULATION")
    print("=" * 70)

    for i, city in enumerate(city_list, 1):
        print(f"\nCity {i}/{len(city_list)}: {city}")
        try:
            results.append(process_city(city))
        except Exception as exc:
            print(f"Failed to process {city}: {type(exc).__name__}: {exc}")

    raw_path, score_path, ref_path, scored = save_results(results)
    total = time.perf_counter() - start

    print("\n" + "=" * 70)
    print("EFFICIENCY PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total execution time: {total:.2f}s")
    print(f"Cities processed: {len(results)}/{len(city_list)}")
    print(f"Raw metrics saved to: {raw_path}")

    if scored is not None:
        print(f"Efficiency scores saved to: {score_path}")
        print(f"Normalization reference saved to: {ref_path}")
        print("\nEFFICIENCY SCORES:")
        print(scored[[
            "city",
            "efficiency_global_efficiency_score",
            "efficiency_asp_length_score",
            "efficiency_circuity_score",
            "efficiency_rcrc_score",
            "efficiency_score",
        ]])

    return scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate RNQI Efficiency metrics.")
    parser.add_argument("--city", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(args.city)
