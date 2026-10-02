"""
RNQI City Evaluator
===================

Evaluate one city with the same graph-preparation and metric implementations
used by the RNQI production pipeline.

The evaluator is intentionally thin:
    local OSM XML
        -> production data preparation
        -> production Connectivity / Efficiency / Hierarchy functions
        -> fixed five-city RNQI normalization references
        -> equal-weight RNQI
        -> reference-relative interpretation

It does not maintain separate copies of the metric algorithms.
It does not compare the evaluated city against a saved production score.
Its interpretation uses only the five-city reference statistics.

Prerequisite:
    The reusable `rnqi/` package folder must be placed in the project root.

Usage:
    python 07_city_rnqi_evaluator_package.py Kolkata
    python 07_city_rnqi_evaluator_package.py Pune
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from rnqi.data import prepare_graph_from_local_osm
from rnqi.connectivity import (
    calculate_connectivity_metrics,
    prepare_undirected_graph,
)
from rnqi.efficiency import calculate_efficiency_adaptive
from rnqi.hierarchy import (
    build_simple_graph,
    calculate_betweenness_metrics,
    calculate_hierarchy_clarity,
    calculate_road_class_distribution,
)
from rnqi.final import calculate_rnqi


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
BOUNDARY_DIR = OUTPUT_DIR / "boundaries"
METADATA_DIR = OUTPUT_DIR / "metadata"
REPORT_DIR = OUTPUT_DIR / "rnqi_city_reports"

CITIES = {
    "Chandigarh": "chandigarh",
    "Jaipur": "jaipur",
    "Pune": "pune",
    "Kolkata": "kolkata",
    "Ranipet": "ranipet",
}

REFERENCE_FILES = {
    "connectivity": METADATA_DIR / "connectivity_normalization_reference.csv",
    "efficiency": METADATA_DIR / "efficiency_normalization_reference.csv",
    "hierarchy": METADATA_DIR / "hierarchy_normalization_reference.csv",
    "final_reference": METADATA_DIR / "final_rnqi_reference_range.csv",
    "final_scores": METADATA_DIR / "final_rnqi_scores.csv",
}


def load_boundary(city: str):
    """Load the current data-derived city extent."""
    path = BOUNDARY_DIR / f"{CITIES[city]}_city_extent.geojson"

    if not path.exists():
        raise FileNotFoundError(f"City extent file not found: {path}")

    boundary = gpd.read_file(path)

    if boundary.empty:
        raise ValueError(f"City extent file is empty: {path}")
    if boundary.crs is None:
        raise ValueError(f"City extent has no CRS: {path}")

    return boundary, path


def normalize_from_reference(value: float, minimum: float, maximum: float, higher_is_better: bool = True) -> float:
    """Apply the production RNQI min-max 0-100 formula to one value."""
    if np.isclose(maximum, minimum):
        return 50.0

    if higher_is_better:
        score = (value - minimum) / (maximum - minimum) * 100.0
    else:
        score = (maximum - value) / (maximum - minimum) * 100.0

    return float(score)


def reference_min_max(path: Path, metric: str):
    """Read a production normalization reference, supporting both output schemas.

    The production stages do not all write normalization references in the same
    orientation:

    1. Row-oriented schema (used by connectivity/efficiency):
       metric, minimum_reference_value, maximum_reference_value

    2. City-oriented schema (used by hierarchy):
       city, <metric_1>, <metric_2>, ...

    This function detects the schema from the columns and returns the same
    ``(minimum, maximum)`` pair to the evaluator in either case.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization reference not found: {path}\n"
            "Run the five-city production pipeline first."
        )

    df = pd.read_csv(path)
    columns = set(df.columns)

    # Schema A: one row per metric.
    row_schema = {
        "metric",
        "minimum_reference_value",
        "maximum_reference_value",
    }
    if row_schema.issubset(columns):
        rows = df.loc[df["metric"].astype(str).str.strip() == metric]
        if rows.empty:
            available = df["metric"].astype(str).str.strip().tolist()
            raise KeyError(
                f"Metric '{metric}' not found in reference: {path}. "
                f"Available metrics: {available}"
            )

        row = rows.iloc[0]
        minimum = pd.to_numeric(
            row["minimum_reference_value"], errors="coerce"
        )
        maximum = pd.to_numeric(
            row["maximum_reference_value"], errors="coerce"
        )

    # Schema B: one row per city, one column per metric.
    elif metric in columns:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            raise ValueError(
                f"No usable reference values for '{metric}' in {path}"
            )

        minimum = values.min()
        maximum = values.max()

    else:
        raise KeyError(
            f"Metric '{metric}' not found in normalization reference: {path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    if pd.isna(minimum) or pd.isna(maximum):
        raise ValueError(
            f"No usable reference min/max for '{metric}' in {path}"
        )

    return float(minimum), float(maximum)


def evaluate_connectivity(city: str, graph: nx.MultiGraph, boundary: gpd.GeoDataFrame):
    """Calculate Connectivity using the production implementation."""
    raw = calculate_connectivity_metrics(
        city_name=city,
        graph=graph,
        boundary=boundary,
    )

    definitions = {
        "node_density": "connectivity_node_density_score",
        "link_density": "connectivity_link_density_score",
        "average_degree": "connectivity_average_degree_score",
        "closeness_centrality": "connectivity_closeness_score",
    }

    scores = {}
    for metric, score_name in definitions.items():
        minimum, maximum = reference_min_max(
            REFERENCE_FILES["connectivity"], metric
        )
        scores[score_name] = normalize_from_reference(
            float(raw[metric]), minimum, maximum, higher_is_better=True
        )

    scores["connectivity_score"] = float(np.mean(list(scores.values())))

    return raw, scores


def evaluate_efficiency(city: str, graph: nx.MultiGraph):
    """Calculate Efficiency using the production implementation."""
    raw = calculate_efficiency_adaptive(
        graph,
        random_seed=42,
    )

    definitions = {
        "efficiency_global_efficiency": (
            "efficiency_global_efficiency_score",
            True,
        ),
        "efficiency_asp_length": (
            "efficiency_asp_length_score",
            False,
        ),
        "efficiency_circuity": (
            "efficiency_circuity_score",
            False,
        ),
        "efficiency_rcrc": (
            "efficiency_rcrc_score",
            True,
        ),
    }

    scores = {}
    for metric, (score_name, higher_is_better) in definitions.items():
        value = float(raw[metric])
        minimum, maximum = reference_min_max(
            REFERENCE_FILES["efficiency"], metric
        )
        scores[score_name] = normalize_from_reference(
            value, minimum, maximum, higher_is_better
        )

    scores["efficiency_score"] = float(np.mean(list(scores.values())))

    return raw, scores


def evaluate_hierarchy(graph: nx.MultiGraph):
    """Calculate Hierarchy using the production metric implementations."""
    simple_graph = build_simple_graph(graph)

    bc_metrics, edge_bc = calculate_betweenness_metrics(simple_graph)
    rcd, class_length = calculate_road_class_distribution(graph)
    clarity, clarity_n = calculate_hierarchy_clarity(simple_graph, edge_bc)

    raw = {
        "hierarchy_betweenness_centrality": float(
            bc_metrics["betweenness_centrality"]
        ),
        "hierarchy_road_class_distribution": float(rcd),
        "hierarchy_clarity": float(clarity),
        "hierarchy_betweenness_sources": int(
            bc_metrics["betweenness_source_count"]
        ),
        "hierarchy_clarity_edges": int(clarity_n),
    }

    scores = {}
    for metric, score_name in {
        "hierarchy_betweenness_centrality": "hierarchy_betweenness_score",
        "hierarchy_road_class_distribution": "hierarchy_road_class_distribution_score",
        "hierarchy_clarity": "hierarchy_clarity_score",
    }.items():
        minimum, maximum = reference_min_max(
            REFERENCE_FILES["hierarchy"], metric
        )
        scores[score_name] = normalize_from_reference(
            float(raw[metric]), minimum, maximum, higher_is_better=True
        )

    scores["hierarchy_score"] = float(np.mean(list(scores.values())))

    return raw, scores



def load_interpretation_reference():
    """Load the five-city reference statistics used for interpretation.

    Prefer the dedicated final RNQI reference-range file when available.
    Otherwise derive the same reference mean, population SD, and pillar means
    from the saved five-city final RNQI scores.

    This is reference information only; it does not compare the evaluated
    city against a saved production result.
    """
    path = REFERENCE_FILES["final_reference"]

    if path.exists():
        df = pd.read_csv(path)
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        if df.empty:
            raise ValueError(f"Reference file is empty: {path}")

        required = [
            "rnqi_mean",
            "rnqi_sd",
            "rnqi_reference_lower",
            "rnqi_reference_upper",
            "connectivity_mean",
            "efficiency_mean",
            "hierarchy_mean",
            "reference_sd_basis",
        ]
        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(
                f"Final reference file missing columns: {missing}"
            )

        row = df.iloc[0]
        return {
            key: (
                str(row[key])
                if key == "reference_sd_basis"
                else float(row[key])
            )
            for key in required
        }
    path = REFERENCE_FILES["final_scores"]

    if not path.exists():
        raise FileNotFoundError(
            "No final RNQI reference statistics found. Expected either:\n"
            f"  {REFERENCE_FILES['final_reference']}\n"
            f"  {REFERENCE_FILES['final_scores']}"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required = [
        "city",
        "connectivity_score",
        "efficiency_score",
        "hierarchy_score",
        "rnqi_score",
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Final RNQI scores file missing columns: {missing}"
        )

    for column in required[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required[1:])

    if df.empty:
        raise ValueError(
            f"No usable RNQI reference rows found in {path}"
        )

    rnqi = df["rnqi_score"]
    mean = float(rnqi.mean())
    # The five benchmark areas are treated as the complete reference population.
    sd = float(rnqi.std(ddof=0))

    return {
        "rnqi_mean": mean,
        "rnqi_sd": sd,
        "rnqi_reference_lower": mean - sd,
        "rnqi_reference_upper": mean + sd,
        "connectivity_mean": float(df["connectivity_score"].mean()),
        "efficiency_mean": float(df["efficiency_score"].mean()),
        "hierarchy_mean": float(df["hierarchy_score"].mean()),
    }


def interpret_pillar(score, reference_mean):
    """Interpret a pillar relative to the five-city reference mean."""
    difference = score - reference_mean
    tolerance = 0.05 * max(abs(reference_mean), 1.0)

    if difference > tolerance:
        return "above the five-city reference mean"
    if difference < -tolerance:
        return "below the five-city reference mean"

    return "close to the five-city reference mean"


def interpret_rnqi(score, reference):
    """Position RNQI relative to the reference mean ± one SD."""
    lower = reference["rnqi_reference_lower"]
    upper = reference["rnqi_reference_upper"]
    mean = reference["rnqi_mean"]

    if score < lower:
        position = "below the reference range"
    elif score > upper:
        position = "above the reference range"
    else:
        position = "within the reference range"

    if score > mean:
        mean_position = "above"
    elif score < mean:
        mean_position = "below"
    else:
        mean_position = "equal to"

    return position, mean_position


def build_interpretation(city, scores, reference):
    """Build the overall and pillar-level textual interpretation."""
    rnqi = scores["rnqi_score"]

    range_position, mean_position = interpret_rnqi(
        rnqi,
        reference,
    )

    pillar_text = {}

    for pillar in ["connectivity", "efficiency", "hierarchy"]:
        pillar_text[pillar] = interpret_pillar(
            scores[f"{pillar}_score"],
            reference[f"{pillar}_mean"],
        )

    overall = (
        f"{city} has an RNQI score of {rnqi:.2f}, "
        f"which is {range_position} of the five-city reference "
        f"range ({reference['rnqi_reference_lower']:.2f}–"
        f"{reference['rnqi_reference_upper']:.2f}) and "
        f"{mean_position} the reference RNQI mean "
        f"({reference['rnqi_mean']:.2f})."
    )

    details = {
        "Connectivity": (
            f"Connectivity = {scores['connectivity_score']:.2f}; "
            f"{pillar_text['connectivity']} "
            f"(reference mean {reference['connectivity_mean']:.2f})."
        ),
        "Efficiency": (
            f"Efficiency = {scores['efficiency_score']:.2f}; "
            f"{pillar_text['efficiency']} "
            f"(reference mean {reference['efficiency_mean']:.2f})."
        ),
        "Hierarchy": (
            f"Hierarchy = {scores['hierarchy_score']:.2f}; "
            f"{pillar_text['hierarchy']} "
            f"(reference mean {reference['hierarchy_mean']:.2f})."
        ),
    }

    return overall, details

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one city using the production RNQI implementations."
    )
    parser.add_argument(
        "city",
        choices=sorted(CITIES),
        help="City to evaluate.",
    )
    args = parser.parse_args()

    city = args.city

    print("=" * 72)
    print(f"RNQI EVALUATION — {city}")
    print("=" * 72)
    print()

    boundary, boundary_path = load_boundary(city)

    print(f"Boundary: {boundary_path}")
    print()

    # This is the critical parity step. The package calls the production
    # data-stage functions: local OSM XML -> original boundary -> cleaning
    # -> projection.
    print("Preparing graph using production data-stage functions...")
    graph = prepare_graph_from_local_osm(city, boundary)

    print(
        f"Projected directed graph: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges"
    )

    graph, _ = prepare_undirected_graph(graph, city)

    print(
        f"Production analysis graph: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} links"
    )
    print()

    print("Calculating Connectivity...")
    connectivity_raw, connectivity_scores = evaluate_connectivity(
        city, graph, boundary
    )
    print(
        f"Connectivity: {connectivity_scores['connectivity_score']:.4f}"
    )
    print()

    print("Calculating Efficiency...")
    efficiency_raw, efficiency_scores = evaluate_efficiency(city, graph)
    print(
        f"Efficiency: {efficiency_scores['efficiency_score']:.4f}"
    )
    print()

    print("Calculating Hierarchy...")
    hierarchy_raw, hierarchy_scores = evaluate_hierarchy(graph)
    print(
        f"Hierarchy: {hierarchy_scores['hierarchy_score']:.4f}"
    )
    print()

    rnqi = calculate_rnqi(
        connectivity_scores["connectivity_score"],
        efficiency_scores["efficiency_score"],
        hierarchy_scores["hierarchy_score"],
    )

    scores = {
        "connectivity_score": float(
            connectivity_scores["connectivity_score"]
        ),
        "efficiency_score": float(
            efficiency_scores["efficiency_score"]
        ),
        "hierarchy_score": float(
            hierarchy_scores["hierarchy_score"]
        ),
        "rnqi_score": float(rnqi),
    }

    reference = load_interpretation_reference()
    overall, pillar_interpretations = build_interpretation(
        city,
        scores,
        reference,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "city": city,
        **scores,
        "reference_rnqi_mean": reference["rnqi_mean"],
        "reference_rnqi_sd": reference["rnqi_sd"],
        "reference_rnqi_lower": reference["rnqi_reference_lower"],
        "reference_rnqi_upper": reference["rnqi_reference_upper"],
        "reference_connectivity_mean": reference["connectivity_mean"],
        "reference_efficiency_mean": reference["efficiency_mean"],
        "reference_hierarchy_mean": reference["hierarchy_mean"],
        "overall_interpretation": overall,
        "connectivity_interpretation":
            pillar_interpretations["Connectivity"],
        "efficiency_interpretation":
            pillar_interpretations["Efficiency"],
        "hierarchy_interpretation":
            pillar_interpretations["Hierarchy"],
    }

    result_path = (
        REPORT_DIR / f"{CITIES[city]}_rnqi_evaluation.csv"
    )
    pd.DataFrame([result]).to_csv(result_path, index=False)

    report_path = REPORT_DIR / f"{CITIES[city]}_rnqi_report.txt"
    report_lines = [
        "=" * 72,
        f"RNQI CITY EVALUATION — {city}",
        "=" * 72,
        "",
        "PILLAR SCORES",
        "-" * 72,
        f"Connectivity : {scores['connectivity_score']:.4f} "
        f"(reference mean: {reference['connectivity_mean']:.4f})",
        f"Efficiency   : {scores['efficiency_score']:.4f} "
        f"(reference mean: {reference['efficiency_mean']:.4f})",
        f"Hierarchy    : {scores['hierarchy_score']:.4f} "
        f"(reference mean: {reference['hierarchy_mean']:.4f})",
        "",
        f"RNQI         : {scores['rnqi_score']:.4f}",
        "",
        "REFERENCE RNQI",
        "-" * 72,
        f"Reference mean      : {reference['rnqi_mean']:.4f}",
        f"Reference SD        : {reference['rnqi_sd']:.4f}",
        f"Reference range     : "
        f"{reference['rnqi_reference_lower']:.4f} – "
        f"{reference['rnqi_reference_upper']:.4f}",
        "",
        "OVERALL INTERPRETATION",
        "-" * 72,
        overall,
        "",
        "PILLAR INTERPRETATION",
        "-" * 72,
        pillar_interpretations["Connectivity"],
        pillar_interpretations["Efficiency"],
        pillar_interpretations["Hierarchy"],
        "",
        "NOTE",
        "-" * 72,
        "The interpretation is relative to the five-city RNQI reference "
        "group. It is not an absolute road-network quality classification.",
        "",
    ]
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 72)
    print("FINAL RNQI RESULT")
    print("=" * 72)
    print(f"Connectivity : {scores['connectivity_score']:.4f}")
    print(f"Efficiency   : {scores['efficiency_score']:.4f}")
    print(f"Hierarchy    : {scores['hierarchy_score']:.4f}")
    print(f"RNQI         : {scores['rnqi_score']:.4f}")

    print()
    print("REFERENCE")
    print("-" * 72)
    print(f"RNQI mean    : {reference['rnqi_mean']:.4f}")
    print(f"RNQI SD      : {reference['rnqi_sd']:.4f}")
    print(
        f"RNQI range   : {reference['rnqi_reference_lower']:.4f} – "
        f"{reference['rnqi_reference_upper']:.4f}"
    )

    print()
    print("OVERALL INTERPRETATION")
    print("-" * 72)
    print(overall)

    print()
    print("PILLAR INTERPRETATION")
    print("-" * 72)
    for interpretation in pillar_interpretations.values():
        print(interpretation)

    print()
    print(f"Detailed CSV saved : {result_path}")
    print(f"Report saved       : {report_path}")

    print()
    print("RAW METRICS")
    print("-" * 72)
    print(f"Connectivity node density       : {connectivity_raw['node_density']:.10g}")
    print(f"Connectivity link density       : {connectivity_raw['link_density']:.10g}")
    print(f"Connectivity average degree     : {connectivity_raw['average_degree']:.10g}")
    print(f"Connectivity closeness           : {connectivity_raw['closeness_centrality']:.10g}")
    print(f"Efficiency global efficiency      : {efficiency_raw['efficiency_global_efficiency']:.10g}")
    print(f"Efficiency ASP length             : {efficiency_raw['efficiency_asp_length']:.10g}")
    print(f"Efficiency circuity               : {efficiency_raw['efficiency_circuity']:.10g}")
    print(f"Efficiency RCRC                   : {efficiency_raw['efficiency_rcrc']:.10g}")
    print(f"Hierarchy betweenness P90         : {hierarchy_raw['hierarchy_betweenness_centrality']:.10g}")
    print(f"Hierarchy road-class entropy      : {hierarchy_raw['hierarchy_road_class_distribution']:.10g}")
    print(f"Hierarchy clarity                 : {hierarchy_raw['hierarchy_clarity']:.10g}")


if __name__ == "__main__":
    main()
