from pathlib import Path
import argparse
import json
import time

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np

from scipy.sparse.csgraph import dijkstra

import math
from scipy.stats import t


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
BOUNDARY_DIR = OUTPUT_DIR / "boundaries"
GRAPH_DIR = OUTPUT_DIR / "graphs"
METADATA_DIR = OUTPUT_DIR / "metadata"

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# RNQI benchmark cities used throughout the project
CITIES = {
    "Chandigarh": "chandigarh",
    "Jaipur": "jaipur",
    "Pune": "pune",
    "Kolkata": "kolkata",
    "Ranipet": "ranipet",
}


# ============================================================
# 2. PERFORMANCE CONFIGURATION
# ============================================================

# ============================================================
# CLOSENESS CENTRALITY SAMPLING CONFIGURATION
# ============================================================

CLOSENESS_BATCH_SIZE = 128

# Reproducibility controls
CLOSENESS_VALIDATION_RUNS = 5
CLOSENESS_VALIDATION_MASTER_SEED = 42

# Initial pilot sample used to estimate variability.
# This is NOT the final sample size.
CLOSENESS_PILOT_FRACTION = 0.02

# Desired statistical precision for the adaptive estimate.
CLOSENESS_CONFIDENCE_LEVEL = 0.95
CLOSENESS_RELATIVE_PRECISION = 0.02


# ============================================================
# 3. TIMING HELPER
# ============================================================

def print_stage(stage_name, start_time):

    elapsed = (
        time.perf_counter() -
        start_time
    )

    print(
        f"[{stage_name}] "
        f"Completed in {elapsed:.2f}s"
    )

    return elapsed


# ============================================================
# 4. LOAD CITY DATA
# ============================================================

def load_city_boundary(city_name):

    city_id = CITIES[city_name]

    boundary_path = (
        BOUNDARY_DIR /
        f"{city_id}_boundary.geojson"
    )

    if not boundary_path.exists():

        raise FileNotFoundError(
            f"Boundary file not found: "
            f"{boundary_path}"
        )

    boundary = gpd.read_file(
        boundary_path
    )

    if boundary.empty:

        raise ValueError(
            f"Boundary file is empty: "
            f"{boundary_path}"
        )

    return boundary, boundary_path


def load_city_graph(city_name):

    city_id = CITIES[city_name]

    graph_path = (
        GRAPH_DIR /
        f"{city_id}.graphml"
    )

    if not graph_path.exists():

        raise FileNotFoundError(
            f"Graph file not found: "
            f"{graph_path}"
        )

    graph = ox.load_graphml(
        graph_path
    )

    return graph, graph_path


# ============================================================
# 5. PREPARE UNDIRECTED GRAPH
# ============================================================

def prepare_undirected_graph(
    graph,
    city_name
):

    print(
        f"Preparing undirected graph "
        f"for {city_name}..."
    )

    start = time.perf_counter()

    # IMPORTANT:
    #
    # MultiGraph removes direction but
    # preserves parallel road-segment edges.
    #
    # This matches our primal graph interpretation:
    # nodes = intersections
    # edges = road segments

    undirected_graph = nx.MultiGraph(
        graph
    )

    elapsed = print_stage(
        "Prepare undirected graph",
        start
    )

    component_count = (
        nx.number_connected_components(
            undirected_graph
        )
    )

    print(
        f"Undirected graph: "
        f"{undirected_graph.number_of_nodes()} "
        f"nodes, "
        f"{undirected_graph.number_of_edges()} "
        f"links, "
        f"{component_count} "
        f"connected components"
    )

    return (
        undirected_graph,
        elapsed
    )


# ============================================================
# 6. CALCULATE CITY AREA
# ============================================================

def calculate_city_area(
    boundary,
    graph
):

    start = time.perf_counter()

    graph_crs = graph.graph["crs"]

    projected_boundary = (
        boundary.to_crs(graph_crs)
    )

    city_area_m2 = (
        projected_boundary
        .geometry
        .area
        .sum()
    )

    city_area_km2 = (
        city_area_m2 /
        1_000_000
    )

    elapsed = print_stage(
        "Calculate city area",
        start
    )

    return (
        city_area_m2,
        city_area_km2,
        elapsed
    )


# ============================================================
# 7. FAST EXACT CLOSENESS CENTRALITY
# ============================================================

# ============================================================
# CLOSENESS CENTRALITY
# ADAPTIVE TWO-STAGE SAMPLING
# ============================================================

def calculate_closeness_for_sources(
    adjacency,
    source_indices,
    number_of_nodes,
    batch_size=128
):
    """
    Calculate exact unweighted closeness centrality for a selected
    set of source nodes.

    The shortest paths themselves are exact.

    Only the number of source nodes evaluated is sampled.

    This uses the same Wasserman-Faust normalization as
    NetworkX closeness_centrality() for disconnected graphs.
    """

    source_indices = np.asarray(source_indices, dtype=int)

    closeness_values = []

    total_sources = len(source_indices)

    for start in range(0, total_sources, batch_size):

        batch_indices = source_indices[
            start:start + batch_size
        ]

        # Exact shortest-path distances from the selected
        # source nodes to every reachable node.
        distances = dijkstra(
            adjacency,
            directed=False,
            indices=batch_indices,
            unweighted=True
        )

        for row in range(len(batch_indices)):

            distance_row = distances[row]

            # Only finite distances are reachable.
            finite_mask = np.isfinite(distance_row)

            reachable_count = int(
                np.sum(finite_mask)
            )

            # Need at least one other reachable node.
            if reachable_count <= 1:

                closeness_values.append(0.0)

                continue

            reachable_distances = distance_row[
                finite_mask
            ]

            distance_sum = float(
                np.sum(reachable_distances)
            )

            if distance_sum <= 0:

                closeness_values.append(0.0)

                continue

            # ------------------------------------------------
            # Wasserman-Faust normalized closeness
            #
            # C(u) =
            #
            # ((reachable - 1) / sum(distance))
            #
            # ×
            #
            # ((reachable - 1) / (N - 1))
            #
            # ------------------------------------------------

            reachable_minus_one = reachable_count - 1

            closeness = (
                (reachable_minus_one / distance_sum)
                *
                (
                    reachable_minus_one
                    /
                    (number_of_nodes - 1)
                )
            )

            closeness_values.append(
                float(closeness)
            )

        completed = min(
            start + batch_size,
            total_sources
        )

        print(
            f"      Closeness sources: "
            f"{completed:,}/{total_sources:,}",
            end="\r"
        )

    print()

    return np.asarray(
        closeness_values,
        dtype=np.float64
    )


# ============================================================
# SAMPLE SIZE CALCULATION
# ============================================================

def calculate_required_sample_size(
    population_size,
    pilot_mean,
    pilot_std,
    relative_precision,
    confidence_level,
    pilot_sample_size
):
    """
    Estimate the required final sample size for estimating a
    finite-population mean using a desired relative margin of error.

    Finite population correction is included.

    The critical t-value is refined iteratively because the
    required sample size determines the degrees of freedom.
    """

    N = population_size
    mean = abs(float(pilot_mean))
    std = float(pilot_std)

    # Relative precision requires a non-zero mean.
    if mean <= 0:

        return N

    # No observed variation means the pilot indicates no
    # sampling variability.
    if std <= 0:

        return pilot_sample_size

    # Desired absolute half-width.
    margin = mean * relative_precision

    alpha = 1.0 - confidence_level

    # Initial critical value using pilot degrees of freedom.
    critical = t.ppf(
        1.0 - alpha / 2.0,
        df=max(pilot_sample_size - 1, 1)
    )

    # Initial estimate.
    numerator = (
        N
        * (critical ** 2)
        * (std ** 2)
    )

    denominator = (
        (N - 1)
        * (margin ** 2)
        +
        (critical ** 2)
        * (std ** 2)
    )

    n = math.ceil(
        numerator / denominator
    )

    # Refine using the t critical value corresponding
    # to the estimated final sample size.
    for _ in range(20):

        n = min(
            max(n, pilot_sample_size),
            N
        )

        critical = t.ppf(
            1.0 - alpha / 2.0,
            df=max(n - 1, 1)
        )

        numerator = (
            N
            * (critical ** 2)
            * (std ** 2)
        )

        denominator = (
            (N - 1)
            * (margin ** 2)
            +
            (critical ** 2)
            * (std ** 2)
        )

        new_n = math.ceil(
            numerator / denominator
        )

        new_n = min(
            max(new_n, pilot_sample_size),
            N
        )

        if new_n == n:

            break

        n = new_n

    return int(n)


# ============================================================
# FINAL CONFIDENCE INTERVAL
# ============================================================

def calculate_closeness_statistics(
    closeness_values,
    population_size,
    confidence_level
):
    """
    Calculate sample mean, sample standard deviation,
    finite-population-corrected standard error and
    t-based confidence interval.
    """

    values = np.asarray(
        closeness_values,
        dtype=np.float64
    )

    n = len(values)
    N = population_size

    if n == 0:

        raise ValueError(
            "No closeness values were calculated."
        )

    mean = float(
        np.mean(values)
    )

    if n > 1:

        std = float(
            np.std(
                values,
                ddof=1
            )
        )

    else:

        std = 0.0

    # Cannot calculate a meaningful t-based CI
    # with a single observation.
    if n <= 1:

        return {
            "mean": mean,
            "std": std,
            "standard_error": None,
            "confidence_interval_lower": None,
            "confidence_interval_upper": None,
            "relative_margin": None
        }

    # Finite Population Correction.
    fpc = math.sqrt(
        (N - n) / (N - 1)
    )

    standard_error = (
        std
        /
        math.sqrt(n)
        *
        fpc
    )

    alpha = 1.0 - confidence_level

    critical = t.ppf(
        1.0 - alpha / 2.0,
        df=n - 1
    )

    margin_of_error = (
        critical
        *
        standard_error
    )

    lower = mean - margin_of_error
    upper = mean + margin_of_error

    if mean != 0:

        relative_margin = (
            margin_of_error
            /
            abs(mean)
        )

    else:

        relative_margin = None

    return {
        "mean": mean,
        "std": std,
        "standard_error": float(standard_error),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "relative_margin": (
            float(relative_margin)
            if relative_margin is not None
            else None
        )
    }


# ============================================================
# ADAPTIVE CLOSENESS ESTIMATION
# ============================================================

def calculate_mean_closeness_adaptive(
    graph,
    pilot_fraction=0.02,
    confidence_level=0.95,
    relative_precision=0.02,
    batch_size=128,
    random_seed=None
):
    """
    Estimate the network-wide mean closeness centrality using
    two-stage adaptive simple random sampling without replacement.

    Stage 1:
        Draw a pilot sample.

    Stage 2:
        Estimate variability from the pilot and calculate the
        required final sample size.

    The pilot nodes are reused in the final sample.

    Shortest paths are exact and unweighted.

    Returns:
        mean closeness
        and complete sampling/statistical metadata.
    """

    start_time = time.time()

    nodes = list(
        graph.nodes()
    )

    number_of_nodes = len(nodes)

    if number_of_nodes < 2:

        raise ValueError(
            "Graph must contain at least two nodes."
        )

    # --------------------------------------------------------
    # Build sparse adjacency matrix.
    # --------------------------------------------------------

    print(
        "Building sparse adjacency matrix..."
    )

    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodes,
        weight=None,
        format="csr",
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Important:
    #
    # MultiGraph may contain parallel road-segment edges.
    #
    # For unweighted topological shortest paths, multiple
    # parallel edges should count as ONE adjacency.
    #
    # Therefore convert every non-zero matrix entry to 1.
    # --------------------------------------------------------

    adjacency.data[:] = 1.0

    # --------------------------------------------------------
    # Create one reproducible random permutation.
    #
    # The first part becomes the pilot.
    # Additional nodes are taken from the same permutation.
    #
    # Therefore the final sample is still a simple random
    # sample without replacement.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_seed
    )

    permutation = rng.permutation(
        number_of_nodes
    )

    # --------------------------------------------------------
    # PILOT SAMPLE
    # --------------------------------------------------------

    pilot_sample_size = max(
        2,
        math.ceil(
            number_of_nodes
            * pilot_fraction
        )
    )

    pilot_sample_size = min(
        pilot_sample_size,
        number_of_nodes
    )

    pilot_indices = permutation[
        :pilot_sample_size
    ]

    print()
    print("=" * 70)
    print("ADAPTIVE CLOSENESS CENTRALITY")
    print("=" * 70)

    print(
        f"Population nodes       : "
        f"{number_of_nodes:,}"
    )

    print(
        f"Pilot sample size      : "
        f"{pilot_sample_size:,}"
    )

    print(
        f"Pilot sample fraction  : "
        f"{pilot_sample_size / number_of_nodes:.4%}"
    )

    print(
        f"Confidence level       : "
        f"{confidence_level:.0%}"
    )

    print(
        f"Target relative margin : "
        f"{relative_precision:.2%}"
    )

    print()
    print(
        "Stage 1: calculating pilot closeness..."
    )

    pilot_closeness = calculate_closeness_for_sources(
        adjacency=adjacency,
        source_indices=pilot_indices,
        number_of_nodes=number_of_nodes,
        batch_size=batch_size
    )

    pilot_stats = calculate_closeness_statistics(
        closeness_values=pilot_closeness,
        population_size=number_of_nodes,
        confidence_level=confidence_level
    )

    print(
        f"Pilot mean             : "
        f"{pilot_stats['mean']:.12f}"
    )

    print(
        f"Pilot standard deviation: "
        f"{pilot_stats['std']:.12f}"
    )

    # --------------------------------------------------------
    # REQUIRED FINAL SAMPLE SIZE
    # --------------------------------------------------------

    final_sample_size = calculate_required_sample_size(
        population_size=number_of_nodes,
        pilot_mean=pilot_stats["mean"],
        pilot_std=pilot_stats["std"],
        relative_precision=relative_precision,
        confidence_level=confidence_level,
        pilot_sample_size=pilot_sample_size
    )

    print()
    print(
        f"Calculated final sample : "
        f"{final_sample_size:,}"
    )

    print(
        f"Final sample fraction   : "
        f"{final_sample_size / number_of_nodes:.4%}"
    )

    # --------------------------------------------------------
    # FINAL SAMPLE
    # --------------------------------------------------------

    if final_sample_size > pilot_sample_size:

        additional_indices = permutation[
            pilot_sample_size:final_sample_size
        ]

        print()
        print(
            "Stage 2: calculating additional "
            f"{len(additional_indices):,} nodes..."
        )

        additional_closeness = (
            calculate_closeness_for_sources(
                adjacency=adjacency,
                source_indices=additional_indices,
                number_of_nodes=number_of_nodes,
                batch_size=batch_size
            )
        )

        final_closeness = np.concatenate(
            [
                pilot_closeness,
                additional_closeness
            ]
        )

    else:

        print()
        print(
            "Pilot sample already satisfies "
            "the estimated sample-size requirement."
        )

        final_closeness = pilot_closeness

    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    final_stats = calculate_closeness_statistics(
        closeness_values=final_closeness,
        population_size=number_of_nodes,
        confidence_level=confidence_level
    )

    elapsed_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL CLOSENESS RESULT")
    print("=" * 70)

    print(
        f"Mean closeness        : "
        f"{final_stats['mean']:.12f}"
    )

    print(
        f"Sample standard dev.  : "
        f"{final_stats['std']:.12f}"
    )

    print(
        f"Standard error        : "
        f"{final_stats['standard_error']:.12f}"
    )

    print(
        f"95% CI                : "
        f"["
        f"{final_stats['confidence_interval_lower']:.12f}, "
        f"{final_stats['confidence_interval_upper']:.12f}"
        f"]"
    )

    print(
        f"Relative margin       : "
        f"{final_stats['relative_margin']:.4%}"
    )

    print(
        f"Final sample size     : "
        f"{len(final_closeness):,}"
    )

    print(
        f"Final sample fraction : "
        f"{len(final_closeness) / number_of_nodes:.4%}"
    )

    print(
        f"Calculation time      : "
        f"{elapsed_time:.2f} seconds"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Return both the estimate and metadata.
    # --------------------------------------------------------

    metadata = {

        "closeness_centrality":
            final_stats["mean"],

        "closeness_method":
            "Adaptive sampled exact unweighted shortest paths",

        "closeness_sampling":
            "Simple random sampling without replacement",

        "closeness_random_seed":
            random_seed,

        "closeness_confidence_level":
            confidence_level,

        "closeness_relative_precision_target":
            relative_precision,

        "closeness_population_nodes":
            number_of_nodes,

        "closeness_pilot_sample_size":
            pilot_sample_size,

        "closeness_final_sample_size":
            len(final_closeness),

        "closeness_sample_fraction":
            len(final_closeness)
            / number_of_nodes,

        "closeness_pilot_mean":
            pilot_stats["mean"],

        "closeness_pilot_std":
            pilot_stats["std"],

        "closeness_standard_error":
            final_stats["standard_error"],

        "closeness_confidence_interval_lower":
            final_stats[
                "confidence_interval_lower"
            ],

        "closeness_confidence_interval_upper":
            final_stats[
                "confidence_interval_upper"
            ],

        "closeness_relative_margin":
            final_stats["relative_margin"],

        "closeness_batch_size":
            batch_size,

        "metric_calculation_seconds":
            elapsed_time
    }

    return (
        final_stats["mean"],
        metadata
    )


# ============================================================
# 8. DYNAMIC MULTI-SEED CLOSENESS VALIDATION
# ============================================================

# Exact all-node values previously calculated with the same graph/methodology.
# These reference values are retained only for validation of the sampled method.
# These are used ONLY for validation, not for RNQI production calculation.
CLOSENESS_EXACT_VALIDATION_REFERENCE = {
    "Chandigarh": 0.021794012194634804,
    "Jaipur": 0.011480352310,
    "Pune": 0.00755618450753064013,
    "Kolkata": 0.012093208669,
    "Ranipet": 0.012010212816,
}


def generate_validation_seeds(number_of_runs, master_seed=None):
    """Generate independent child seeds dynamically from one master seed."""
    if number_of_runs < 1:
        raise ValueError("number_of_runs must be at least 1.")

    seed_sequence = np.random.SeedSequence(master_seed)
    child_sequences = seed_sequence.spawn(number_of_runs)

    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in child_sequences
    ]


def run_dynamic_multiseed_validation(
    city_name,
    number_of_runs=5,
    master_seed=42
):
    """Run adaptive closeness repeatedly with dynamically generated seeds.

    This is a validation experiment only. It does not replace the single
    adaptive estimate used by the RNQI production pipeline.
    """
    import contextlib
    import io

    graph, _ = load_city_graph(city_name)
    undirected_graph, _ = prepare_undirected_graph(graph, city_name)

    seeds = generate_validation_seeds(number_of_runs, master_seed)
    exact_reference = CLOSENESS_EXACT_VALIDATION_REFERENCE.get(city_name)
    rows = []

    print()
    print("=" * 70)
    print("DYNAMIC MULTI-SEED CLOSENESS VALIDATION")
    print("=" * 70)
    print(f"City                  : {city_name}")
    print(f"Validation runs       : {number_of_runs}")
    print(f"Master seed           : {master_seed}")
    print("Seeds generated       : dynamically")
    print()

    validation_start = time.perf_counter()

    for run_number, seed in enumerate(seeds, start=1):
        # Keep the validation output concise while retaining all metadata.
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            estimate, metadata = calculate_mean_closeness_adaptive(
                graph=undirected_graph,
                pilot_fraction=CLOSENESS_PILOT_FRACTION,
                confidence_level=CLOSENESS_CONFIDENCE_LEVEL,
                relative_precision=CLOSENESS_RELATIVE_PRECISION,
                batch_size=CLOSENESS_BATCH_SIZE,
                random_seed=seed
            )

        row = {
            "city": city_name,
            "run": run_number,
            "seed": seed,
            "estimate": estimate,
            "sample_size": metadata["closeness_final_sample_size"],
            "sample_fraction": metadata["closeness_sample_fraction"],
            "relative_margin": metadata["closeness_relative_margin"],
            "ci_lower": metadata["closeness_confidence_interval_lower"],
            "ci_upper": metadata["closeness_confidence_interval_upper"],
            "calculation_seconds": metadata["metric_calculation_seconds"],
        }

        if exact_reference is not None:
            row["exact_reference"] = exact_reference
            row["absolute_error"] = abs(estimate - exact_reference)
            row["relative_error"] = (
                abs(estimate - exact_reference) / abs(exact_reference)
            )
            row["exact_inside_ci"] = (
                metadata["closeness_confidence_interval_lower"]
                <= exact_reference
                <= metadata["closeness_confidence_interval_upper"]
            )

        rows.append(row)

        print(
            f"Run {run_number:>2}/{number_of_runs} | "
            f"seed={seed} | estimate={estimate:.12f} | "
            f"n={row['sample_size']:,} | "
            f"margin={row['relative_margin']:.3%}"
        )

    validation_seconds = time.perf_counter() - validation_start
    df = pd.DataFrame(rows)
    output_path = METADATA_DIR / "closeness_multiseed_validation.csv"
    df.to_csv(output_path, index=False)

    print()
    print("MULTI-SEED VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Mean of sampled estimates: {df['estimate'].mean():.12f}")
    if len(df) > 1:
        print(f"Std. dev. of estimates   : {df['estimate'].std(ddof=1):.12f}")
    else:
        print("Std. dev. of estimates   : n/a")

    if exact_reference is not None:
        print(f"Exact reference          : {exact_reference:.12f}")
        print(f"Mean absolute error      : {df['absolute_error'].mean():.12f}")
        print(f"Mean relative error      : {df['relative_error'].mean():.4%}")
        print(
            f"95% CI coverage          : "
            f"{df['exact_inside_ci'].mean():.1%} "
            f"({int(df['exact_inside_ci'].sum())}/{len(df)})"
        )

    print(f"Total validation time    : {validation_seconds:.2f}s")
    print(f"Validation results saved : {output_path}")
    print("=" * 70)

    return df


# ============================================================
# 8. CONNECTIVITY METRICS
# ============================================================

def calculate_connectivity_metrics(
    city_name,
    graph,
    boundary
):

    print(
        f"\nConnectivity metrics: "
        f"{city_name}"
    )

    print("=" * 60)

    total_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # City area
    # --------------------------------------------------------

    (
        city_area_m2,
        city_area_km2,
        area_seconds
    ) = calculate_city_area(
        boundary,
        graph
    )

    # --------------------------------------------------------
    # Basic graph statistics
    # --------------------------------------------------------

    number_of_nodes = (
        graph.number_of_nodes()
    )

    number_of_links = (
        graph.number_of_edges()
    )

    connected_components = (
        nx.number_connected_components(
            graph
        )
    )

    # --------------------------------------------------------
    # 1. Node Density
    # --------------------------------------------------------

    node_density = (
        number_of_nodes /
        city_area_km2
    )

    # --------------------------------------------------------
    # 2. Link Density
    # --------------------------------------------------------

    link_density = (
        number_of_links /
        city_area_km2
    )

    # --------------------------------------------------------
    # 3. Average Degree
    # --------------------------------------------------------

    degree_values = [
        degree
        for _, degree
        in graph.degree()
    ]

    average_degree = (
        sum(degree_values) /
        number_of_nodes
    )

    # --------------------------------------------------------
    # 4. Closeness Centrality
    # --------------------------------------------------------

    closeness, closeness_metadata = calculate_mean_closeness_adaptive(
        graph=graph,
        pilot_fraction=CLOSENESS_PILOT_FRACTION,
        confidence_level=CLOSENESS_CONFIDENCE_LEVEL,
        relative_precision=CLOSENESS_RELATIVE_PRECISION,
        batch_size=CLOSENESS_BATCH_SIZE,
        random_seed=CLOSENESS_VALIDATION_MASTER_SEED
    )

    # The adaptive closeness function records its own calculation time.
    closeness_seconds = closeness_metadata[
        "metric_calculation_seconds"
    ]

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    metric_calculation_seconds = (
        time.perf_counter() -
        total_start
    )

    # --------------------------------------------------------
    # Validation checks
    # --------------------------------------------------------

    degree_sum = sum(
        degree_values
    )

    degree_edge_check = (
        degree_sum ==
        2 * number_of_links
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "city": city_name,

        "number_of_nodes": int(
            number_of_nodes
        ),

        "number_of_links": int(
            number_of_links
        ),

        "connected_components": int(
            connected_components
        ),

        "city_area_km2": float(
            city_area_km2
        ),

        "node_density": float(
            node_density
        ),

        "link_density": float(
            link_density
        ),

        "average_degree": float(
            average_degree
        ),

        "closeness_centrality": float(
            closeness,
        ),

        **closeness_metadata,

        "degree_sum": int(
            degree_sum
        ),

        "degree_edge_check": bool(
            degree_edge_check
        ),

        # "closeness_method": (
        #     "SciPy sparse exact "
        #     "unweighted shortest paths"
        # ),

        # "closeness_batch_size": int(
        #     CLOSENESS_BATCH_SIZE
        # ),

        "boundary_path": str(
            BOUNDARY_DIR /
            f"{CITIES[city_name]}_boundary.geojson"
        ),

        "graph_path": str(
            GRAPH_DIR /
            f"{CITIES[city_name]}.graphml"
        ),

        "area_calculation_seconds": round(
            area_seconds,
            2
        ),

        "closeness_calculation_seconds": round(
            closeness_seconds,
            2
        ),

        "metric_calculation_seconds": round(
            metric_calculation_seconds,
            2
        ),
    }

    return result


# ============================================================
# 9. PROCESS ONE CITY
# ============================================================

def process_city(city_name):

    print("\n" + "=" * 60)

    print(
        f"Connectivity metrics: "
        f"{city_name}"
    )

    print("=" * 60)

    city_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    start = time.perf_counter()

    boundary, boundary_path = (
        load_city_boundary(
            city_name
        )
    )

    boundary_load_seconds = (
        print_stage(
            "Load boundary",
            start
        )
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    start = time.perf_counter()

    graph, graph_path = (
        load_city_graph(
            city_name
        )
    )

    graph_load_seconds = (
        print_stage(
            "Load projected graph",
            start
        )
    )

    # --------------------------------------------------------
    # Undirected graph
    # --------------------------------------------------------

    (
        undirected_graph,
        undirected_seconds
    ) = prepare_undirected_graph(
        graph,
        city_name
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    result = (
        calculate_connectivity_metrics(
            city_name,
            undirected_graph,
            boundary
        )
    )

    # --------------------------------------------------------
    # Timing information
    # --------------------------------------------------------

    result[
        "load_boundary_seconds"
    ] = round(
        boundary_load_seconds,
        2
    )

    result[
        "load_graph_seconds"
    ] = round(
        graph_load_seconds,
        2
    )

    result[
        "undirected_graph_seconds"
    ] = round(
        undirected_seconds,
        2
    )

    result[
        "execution_time_seconds"
    ] = round(
        time.perf_counter()
        - city_start,
        2
    )

    result[
        "execution_time"
    ] = (
        f"{result['execution_time_seconds']:.2f}s"
    )

    print(
        f"\nCompleted: "
        f"{city_name}"
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    return result


# ============================================================
# 10. CONNECTIVITY NORMALIZATION AND PILLAR SCORE
# ============================================================


def min_max_normalize(values):
    """Normalize reference-group values to the RNQI 0-100 scale."""

    values = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if np.isclose(maximum, minimum):
        # No differentiation exists within the reference group.
        # Assign the neutral midpoint rather than divide by zero.
        scores = np.full_like(values, 50.0, dtype=np.float64)
    else:
        scores = (values - minimum) / (maximum - minimum) * 100.0

    return scores, minimum, maximum


def calculate_connectivity_scores(results):
    """
    Calculate normalized Connectivity metrics and the Connectivity
    pillar score for the five RNQI reference cities.

    All four Connectivity metrics are higher-is-better. Each is
    normalized to 0-100 using the full five-city reference group,
    then the Connectivity pillar score is their arithmetic mean.
    """

    if not results:
        raise ValueError(
            "No Connectivity results available for normalization."
        )

    df = pd.DataFrame(results).copy()

    metric_columns = {
        "node_density": "connectivity_node_density_score",
        "link_density": "connectivity_link_density_score",
        "average_degree": "connectivity_average_degree_score",
        "closeness_centrality": "connectivity_closeness_score",
    }

    reference_rows = []

    for raw_column, score_column in metric_columns.items():
        if raw_column not in df.columns:
            raise KeyError(
                f"Required Connectivity metric missing: {raw_column}"
            )

        scores, minimum, maximum = min_max_normalize(
            df[raw_column].to_numpy(dtype=np.float64)
        )

        df[score_column] = scores

        reference_rows.append({
            "metric": raw_column,
            "minimum_reference_value": minimum,
            "maximum_reference_value": maximum,
            "normalization": "Min-max 0-100, higher-is-better",
        })

    score_columns = list(metric_columns.values())
    df["connectivity_score"] = df[score_columns].mean(axis=1)

    reference_df = pd.DataFrame(reference_rows)

    return df, reference_df


# ============================================================
# 11. SAVE RESULTS
# ============================================================

def save_results(results):

    raw_output_path = (
        METADATA_DIR /
        "city_wise_connectivity_metrics.csv"
    )

    scored_output_path = (
        METADATA_DIR /
        "city_wise_connectivity_scores.csv"
    )

    normalization_output_path = (
        METADATA_DIR /
        "connectivity_normalization_reference.csv"
    )

    raw_df = pd.DataFrame(results)

    raw_df.to_csv(
        raw_output_path,
        index=False
    )

    # Connectivity normalization is defined across the complete
    # five-city reference group. Do not calculate a reference score
    # from a single city's own minimum and maximum.
    reference_cities = list(CITIES.keys())
    processed_cities = set(raw_df["city"].tolist())
    full_reference_group = all(
        city in processed_cities
        for city in reference_cities
    )

    if not full_reference_group:
        print()
        print(
            "Connectivity normalization skipped: the full "
            "five-city reference group is required."
        )
        print(
            "Run the pipeline without --city to calculate "
            "reference-normalized Connectivity scores."
        )

        return (
            raw_output_path,
            None,
            None,
            None
        )

    scored_df, normalization_df = calculate_connectivity_scores(
        results
    )

    scored_df.to_csv(
        scored_output_path,
        index=False
    )

    normalization_df.to_csv(
        normalization_output_path,
        index=False
    )

    return (
        raw_output_path,
        scored_output_path,
        normalization_output_path,
        scored_df
    )


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def run_pipeline(
    selected_city=None
):

    print("=" * 60)

    print(
        "RNQI CONNECTIVITY "
        "METRIC CALCULATION"
    )

    print("=" * 60)

    if selected_city:

        if selected_city not in CITIES:

            raise ValueError(
                f"Unknown city: "
                f"{selected_city}\n"
                f"Available cities: "
                f"{', '.join(CITIES.keys())}"
            )

        city_list = [
            selected_city
        ]

    else:

        city_list = list(
            CITIES.keys()
        )

    pipeline_start = (
        time.perf_counter()
    )

    results = []

    for index, city_name in enumerate(
        city_list,
        start=1
    ):

        print(
            f"\nCity "
            f"{index}/{len(city_list)}: "
            f"{city_name}"
        )

        try:

            result = process_city(
                city_name
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"\nFailed to process "
                f"{city_name}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    (
        raw_output_path,
        scored_output_path,
        normalization_output_path,
        scored_df
    ) = save_results(
        results
    )

    total_seconds = (
        time.perf_counter()
        - pipeline_start
    )

    print("\n" + "=" * 60)

    print(
        "CONNECTIVITY PIPELINE "
        "SUMMARY"
    )

    print("=" * 60)

    print(
        f"Total execution time: "
        f"{total_seconds:.2f}s"
    )

    print(
        f"Cities processed: "
        f"{len(results)}/"
        f"{len(city_list)}"
    )

    print(
        f"Raw metrics saved to: "
        f"{raw_output_path}"
    )

    if scored_df is not None:
        print(
            f"Connectivity scores saved to: "
            f"{scored_output_path}"
        )

        print(
            f"Normalization reference saved to: "
            f"{normalization_output_path}"
        )

        print(
            "\nCONNECTIVITY SCORES:"
        )

        print(
            scored_df[[
                "city",
                "connectivity_node_density_score",
                "connectivity_link_density_score",
                "connectivity_average_degree_score",
                "connectivity_closeness_score",
                "connectivity_score",
            ]]
        )

    return scored_df


# ============================================================
# 13. COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Calculate RNQI "
            "Connectivity metrics."
        )
    )

    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help=(
            "Calculate metrics for "
            "one city. "
            "Example: "
            "--city Chandigarh"
        )
    )

    parser.add_argument(
        "--validate-seeds",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run dynamic multi-seed validation for one city using N "
            "independent child seeds. This is validation only."
        )
    )

    parser.add_argument(
        "--validation-master-seed",
        type=int,
        default=CLOSENESS_VALIDATION_MASTER_SEED,
        help=(
            "Master seed used to generate validation seeds dynamically."
        )
    )

    args = parser.parse_args()

    if args.validate_seeds > 0:
        if not args.city:
            parser.error("--validate-seeds requires --city.")
        if args.city not in CITIES:
            parser.error(
                f"Unknown city: {args.city}. "
                f"Available cities: {', '.join(CITIES.keys())}"
            )

        run_dynamic_multiseed_validation(
            city_name=args.city,
            number_of_runs=args.validate_seeds,
            master_seed=args.validation_master_seed
        )
    else:
        run_pipeline(
            selected_city=args.city
        )