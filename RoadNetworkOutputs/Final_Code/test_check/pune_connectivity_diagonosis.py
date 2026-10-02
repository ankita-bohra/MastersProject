# from pathlib import Path
# import networkx as nx
# import osmnx as ox
# import pandas as pd


# # ============================================================
# # 1. PATH
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent.parent

# GRAPH_PATH = (
#     BASE_DIR
#     / "outputs"
#     / "graphs"
#     / "pune.graphml"
# )


# # ============================================================
# # 2. LOAD GRAPH
# # ============================================================

# print("=" * 90)
# print("PUNE ROAD NETWORK CONNECTIVITY DIAGNOSTIC")
# print("=" * 90)

# print("\nLoading graph...")
# print(GRAPH_PATH)

# if not GRAPH_PATH.exists():
#     raise FileNotFoundError(
#         f"\nGraph not found:\n{GRAPH_PATH}"
#     )

# G = ox.load_graphml(GRAPH_PATH)

# print("\nGraph loaded successfully.")


# # ============================================================
# # 3. BASIC NETWORK STATISTICS
# # ============================================================

# n_nodes = G.number_of_nodes()
# n_edges = G.number_of_edges()

# components = list(nx.weakly_connected_components(G))
# components_sorted = sorted(
#     components,
#     key=len,
#     reverse=True
# )

# largest_component = components_sorted[0]

# largest_nodes = len(largest_component)

# largest_pct = (
#     largest_nodes / n_nodes * 100
# )

# print("\n" + "=" * 90)
# print("1. BASIC CONNECTIVITY")
# print("=" * 90)

# print(f"Total nodes              : {n_nodes:,}")
# print(f"Total edges              : {n_edges:,}")
# print(f"Connected components     : {len(components):,}")
# print(
#     f"Largest component nodes  : "
#     f"{largest_nodes:,}"
# )
# print(
#     f"Largest component share  : "
#     f"{largest_pct:.2f}%"
# )
# print(
#     f"Nodes outside largest    : "
#     f"{n_nodes - largest_nodes:,}"
# )


# # ============================================================
# # 4. DEGREE DISTRIBUTION
# # ============================================================

# print("\n" + "=" * 90)
# print("2. DEGREE DISTRIBUTION")
# print("=" * 90)

# degree_values = [
#     degree
#     for _, degree in G.degree()
# ]

# degree_series = pd.Series(
#     degree_values
# )

# degree_counts = (
#     degree_series
#     .value_counts()
#     .sort_index()
# )

# degree_table = pd.DataFrame({
#     "degree": degree_counts.index,
#     "nodes": degree_counts.values
# })

# degree_table["percentage"] = (
#     degree_table["nodes"]
#     / n_nodes
#     * 100
# )

# print(
#     degree_table.to_string(
#         index=False,
#         formatters={
#             "percentage": "{:.2f}%".format
#         }
#     )
# )


# # ============================================================
# # 5. KEY DEGREE GROUPS
# # ============================================================

# degree_1 = sum(
#     d == 1
#     for d in degree_values
# )

# degree_2 = sum(
#     d == 2
#     for d in degree_values
# )

# degree_3_plus = sum(
#     d >= 3
#     for d in degree_values
# )

# degree_4_plus = sum(
#     d >= 4
#     for d in degree_values
# )

# print("\n" + "=" * 90)
# print("3. NETWORK STRUCTURE")
# print("=" * 90)

# print(
#     f"Degree-1 nodes (dead ends) : "
#     f"{degree_1:,} "
#     f"({degree_1 / n_nodes * 100:.2f}%)"
# )

# print(
#     f"Degree-2 nodes             : "
#     f"{degree_2:,} "
#     f"({degree_2 / n_nodes * 100:.2f}%)"
# )

# print(
#     f"Degree-3+ nodes            : "
#     f"{degree_3_plus:,} "
#     f"({degree_3_plus / n_nodes * 100:.2f}%)"
# )

# print(
#     f"Degree-4+ nodes            : "
#     f"{degree_4_plus:,} "
#     f"({degree_4_plus / n_nodes * 100:.2f}%)"
# )

# print(
#     f"Average degree             : "
#     f"{sum(degree_values) / n_nodes:.4f}"
# )


# # ============================================================
# # 6. LARGEST COMPONENT DEGREE
# # ============================================================

# print("\n" + "=" * 90)
# print("4. LARGEST COMPONENT STRUCTURE")
# print("=" * 90)

# largest_G = G.subgraph(
#     largest_component
# ).copy()

# largest_degree_values = [
#     d
#     for _, d in largest_G.degree()
# ]

# largest_n = (
#     largest_G.number_of_nodes()
# )

# largest_edges = (
#     largest_G.number_of_edges()
# )

# largest_degree_1 = sum(
#     d == 1
#     for d in largest_degree_values
# )

# largest_degree_2 = sum(
#     d == 2
#     for d in largest_degree_values
# )

# largest_degree_3_plus = sum(
#     d >= 3
#     for d in largest_degree_values
# )

# largest_degree_4_plus = sum(
#     d >= 4
#     for d in largest_degree_values
# )

# largest_avg_degree = (
#     sum(largest_degree_values)
#     / largest_n
# )

# print(
#     f"Largest component nodes : "
#     f"{largest_n:,}"
# )

# print(
#     f"Largest component edges : "
#     f"{largest_edges:,}"
# )

# print(
#     f"Average degree          : "
#     f"{largest_avg_degree:.4f}"
# )

# print(
#     f"Degree-1 nodes          : "
#     f"{largest_degree_1:,} "
#     f"({largest_degree_1 / largest_n * 100:.2f}%)"
# )

# print(
#     f"Degree-2 nodes          : "
#     f"{largest_degree_2:,} "
#     f"({largest_degree_2 / largest_n * 100:.2f}%)"
# )

# print(
#     f"Degree-3+ nodes         : "
#     f"{largest_degree_3_plus:,} "
#     f"({largest_degree_3_plus / largest_n * 100:.2f}%)"
# )

# print(
#     f"Degree-4+ nodes         : "
#     f"{largest_degree_4_plus:,} "
#     f"({largest_degree_4_plus / largest_n * 100:.2f}%)"
# )


# # ============================================================
# # 7. ARTICULATION / BRIDGE CHECK
# # ============================================================

# print("\n" + "=" * 90)
# print("5. CRITICAL CONNECTIVITY STRUCTURE")
# print("=" * 90)

# # Convert to undirected graph for articulation analysis.
# UG = nx.Graph(largest_G)

# print("Calculating articulation points...")

# articulation_points = list(
#     nx.articulation_points(UG)
# )

# print(
#     f"Articulation points      : "
#     f"{len(articulation_points):,}"
# )

# if largest_n > 0:
#     print(
#         f"Articulation point share : "
#         f"{len(articulation_points) / largest_n * 100:.2f}%"
#     )

# print("\nCalculating bridges...")

# bridges = list(
#     nx.bridges(UG)
# )

# print(
#     f"Bridge edges             : "
#     f"{len(bridges):,}"
# )

# if largest_edges > 0:
#     print(
#         f"Bridge edge share        : "
#         f"{len(bridges) / largest_edges * 100:.2f}%"
#     )


# # ============================================================
# # 8. COMPONENT SIZE DISTRIBUTION
# # ============================================================

# print("\n" + "=" * 90)
# print("6. DISCONNECTED COMPONENTS")
# print("=" * 90)

# component_sizes = sorted(
#     [len(c) for c in components],
#     reverse=True
# )

# component_summary = pd.DataFrame({
#     "rank": range(
#         1,
#         min(20, len(component_sizes)) + 1
#     ),
#     "nodes": component_sizes[
#         :20
#     ]
# })

# component_summary["percentage"] = (
#     component_summary["nodes"]
#     / n_nodes
#     * 100
# )

# print(
#     component_summary.to_string(
#         index=False,
#         formatters={
#             "percentage": "{:.3f}%".format
#         }
#     )
# )


# # ============================================================
# # 9. INTERPRETATION FLAGS
# # ============================================================

# print("\n" + "=" * 90)
# print("7. DIAGNOSTIC FLAGS")
# print("=" * 90)

# dead_end_pct = (
#     degree_1 / n_nodes * 100
# )

# intersection_pct = (
#     degree_3_plus / n_nodes * 100
# )

# print(
#     f"Dead-end proportion       : "
#     f"{dead_end_pct:.2f}%"
# )

# print(
#     f"Intersection proportion   : "
#     f"{intersection_pct:.2f}%"
# )

# print(
#     f"Largest component         : "
#     f"{largest_pct:.2f}%"
# )

# print(
#     f"Average degree            : "
#     f"{largest_avg_degree:.2f}"
# )

# print(
#     f"Articulation points       : "
#     f"{len(articulation_points):,}"
# )

# print(
#     f"Bridge edges              : "
#     f"{len(bridges):,}"
# )


# # ============================================================
# # 10. FINAL DIAGNOSTIC SUMMARY
# # ============================================================

# print("\n" + "=" * 90)
# print("8. PUNE CONNECTIVITY DIAGNOSTIC SUMMARY")
# print("=" * 90)

# if largest_pct >= 95:
#     print(
#         "✓ The network is largely connected."
#     )
# else:
#     print(
#         "⚠ A substantial share of the network "
#         "is outside the largest component."
#     )

# if largest_avg_degree < 2.5:
#     print(
#         "⚠ Very low average degree in the "
#         "largest component."
#     )
# elif largest_avg_degree < 3:
#     print(
#         "⚠ Average degree is relatively low."
#     )
# else:
#     print(
#         "✓ Average degree is relatively high."
#     )

# if dead_end_pct > 25:
#     print(
#         "⚠ High proportion of dead-end nodes."
#     )
# else:
#     print(
#         "✓ Dead-end proportion is not dominant."
#     )

# if intersection_pct < 30:
#     print(
#         "⚠ Relatively small share of nodes "
#         "have degree >= 3."
#     )
# else:
#     print(
#         "✓ Reasonable share of nodes have "
#         "degree >= 3."
#     )

# print("\nDiagnostic complete.")
# print("=" * 90)


from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import random
import time


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GRAPH_DIR = BASE_DIR / "..outputs" / "graphs"
BOUNDARY_DIR = BASE_DIR / "..outputs" / "boundaries"
OUTPUT_DIR = BASE_DIR / "outputs" / "diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CITIES = [
    "ranipet",
    "chandigarh",
    "kolkata",
    "pune",
    "jaipur",
]

RANDOM_SEED = 42
CLOSENESS_SAMPLE_SIZE = 1000


# ============================================================
# HELPERS
# ============================================================

def load_graph(city):
    graph_path = GRAPH_DIR / f"{city}.graphml"

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    print(f"\nLoading {city}:")
    print(graph_path)

    G = nx.read_graphml(graph_path)

    # Convert to undirected graph
    G = nx.Graph(G)

    # Remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def calculate_closeness_sample(G, sample_size=1000, seed=42):
    """
    Estimate average closeness using sampled source nodes.

    For disconnected graphs, networkx.closeness_centrality()
    computes closeness within each reachable component.

    We calculate the mean over sampled nodes.
    """

    nodes = list(G.nodes())

    if len(nodes) == 0:
        return np.nan

    sample_size = min(sample_size, len(nodes))

    rng = random.Random(seed)
    sampled_nodes = rng.sample(nodes, sample_size)

    values = []

    for node in sampled_nodes:
        try:
            value = nx.closeness_centrality(G, node)
            values.append(value)
        except Exception:
            continue

    if not values:
        return np.nan

    return float(np.mean(values))


def calculate_metrics(G, city_area_km2, label):

    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0:
        return {
            "network_type": label,
            "nodes": 0,
            "edges": 0,
            "node_density": np.nan,
            "link_density": np.nan,
            "average_degree": np.nan,
            "closeness": np.nan,
        }

    node_density = n / city_area_km2
    link_density = e / city_area_km2
    average_degree = (2 * e) / n

    closeness = calculate_closeness_sample(
        G,
        sample_size=CLOSENESS_SAMPLE_SIZE,
        seed=RANDOM_SEED
    )

    return {
        "network_type": label,
        "nodes": n,
        "edges": e,
        "node_density": node_density,
        "link_density": link_density,
        "average_degree": average_degree,
        "closeness": closeness,
    }


# ============================================================
# CITY AREA
# ============================================================
#
# IMPORTANT:
# Use the SAME city areas that were used in your current RNQI
# connectivity calculation.
#
# These are taken from your diagnostic output.
# ============================================================

CITY_AREAS = {
    "ranipet": 5.5,
    "chandigarh": 163.25,
    "kolkata": 697.0,
    "pune": 926.75,
    "jaipur": 133.0,
}


# ============================================================
# MAIN ANALYSIS
# ============================================================

all_results = []
summary_results = []


for city in CITIES:

    print("\n")
    print("=" * 90)
    print(f"{city.upper()} — FULL GRAPH VS LARGEST COMPONENT")
    print("=" * 90)

    start_time = time.time()

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    G = load_graph(city)

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    components = list(nx.connected_components(G))
    components = sorted(components, key=len, reverse=True)

    number_components = len(components)

    largest_nodes = components[0]

    G_largest = G.subgraph(largest_nodes).copy()

    largest_nodes_count = G_largest.number_of_nodes()
    largest_edges_count = G_largest.number_of_edges()

    largest_share = (
        largest_nodes_count / total_nodes * 100
        if total_nodes > 0 else np.nan
    )

    city_area = CITY_AREAS[city]

    # --------------------------------------------------------
    # Print basic information
    # --------------------------------------------------------

    print("\nBASIC STRUCTURE")
    print("-" * 90)

    print(f"Total nodes              : {total_nodes:,}")
    print(f"Total edges              : {total_edges:,}")
    print(f"Connected components     : {number_components:,}")
    print(f"Largest component nodes  : {largest_nodes_count:,}")
    print(f"Largest component share  : {largest_share:.2f}%")

    # --------------------------------------------------------
    # FULL GRAPH
    # --------------------------------------------------------

    full_metrics = calculate_metrics(
        G,
        city_area,
        "full_graph"
    )

    # --------------------------------------------------------
    # LARGEST COMPONENT
    # --------------------------------------------------------

    largest_metrics = calculate_metrics(
        G_largest,
        city_area,
        "largest_component"
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    for metrics in [full_metrics, largest_metrics]:

        row = {
            "city": city,
            "city_area_km2": city_area,
            "connected_components": number_components,
            "largest_component_share_pct": largest_share,
            **metrics,
        }

        all_results.append(row)

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    def pct_difference(full, largest):

        if full == 0 or pd.isna(full) or pd.isna(largest):
            return np.nan

        return abs(largest - full) / abs(full) * 100

    node_density_diff = pct_difference(
        full_metrics["node_density"],
        largest_metrics["node_density"]
    )

    link_density_diff = pct_difference(
        full_metrics["link_density"],
        largest_metrics["link_density"]
    )

    degree_diff = pct_difference(
        full_metrics["average_degree"],
        largest_metrics["average_degree"]
    )

    closeness_diff = pct_difference(
        full_metrics["closeness"],
        largest_metrics["closeness"]
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print("\nMETRIC COMPARISON")
    print("-" * 90)

    comparison = pd.DataFrame([
        {
            "metric": "Node density",
            "full_graph": full_metrics["node_density"],
            "largest_component": largest_metrics["node_density"],
            "difference_pct": node_density_diff,
        },
        {
            "metric": "Link density",
            "full_graph": full_metrics["link_density"],
            "largest_component": largest_metrics["link_density"],
            "difference_pct": link_density_diff,
        },
        {
            "metric": "Average degree",
            "full_graph": full_metrics["average_degree"],
            "largest_component": largest_metrics["average_degree"],
            "difference_pct": degree_diff,
        },
        {
            "metric": "Closeness",
            "full_graph": full_metrics["closeness"],
            "largest_component": largest_metrics["closeness"],
            "difference_pct": closeness_diff,
        },
    ])

    print(comparison.to_string(index=False))

    # --------------------------------------------------------
    # Summary row
    # --------------------------------------------------------

    summary_results.append({
        "city": city,
        "city_area_km2": city_area,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "connected_components": number_components,
        "largest_component_nodes": largest_nodes_count,
        "largest_component_share_pct": largest_share,

        "full_node_density":
            full_metrics["node_density"],

        "largest_node_density":
            largest_metrics["node_density"],

        "node_density_difference_pct":
            node_density_diff,

        "full_link_density":
            full_metrics["link_density"],

        "largest_link_density":
            largest_metrics["link_density"],

        "link_density_difference_pct":
            link_density_diff,

        "full_average_degree":
            full_metrics["average_degree"],

        "largest_average_degree":
            largest_metrics["average_degree"],

        "average_degree_difference_pct":
            degree_diff,

        "full_closeness":
            full_metrics["closeness"],

        "largest_component_closeness":
            largest_metrics["closeness"],

        "closeness_difference_pct":
            closeness_diff,

        "runtime_seconds":
            time.time() - start_time,
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

summary_df = pd.DataFrame(summary_results)


results_path = (
    OUTPUT_DIR /
    "full_vs_largest_component_metrics.csv"
)

summary_path = (
    OUTPUT_DIR /
    "full_vs_largest_component_summary.csv"
)


results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("FINAL FIVE-CITY COMPARISON")
print("=" * 100)

display_columns = [
    "city",
    "largest_component_share_pct",
    "full_node_density",
    "largest_node_density",
    "node_density_difference_pct",
    "full_link_density",
    "largest_link_density",
    "link_density_difference_pct",
    "full_average_degree",
    "largest_average_degree",
    "average_degree_difference_pct",
    "full_closeness",
    "largest_component_closeness",
    "closeness_difference_pct",
]

print(
    summary_df[display_columns]
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print(results_path)
print(summary_path)
