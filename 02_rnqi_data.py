
from pathlib import Path
import argparse
import json
import time

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd


# ============================================================
# 1. OSMnx CONFIGURATION
# ============================================================

# ox.settings.requests_timeout = 300
# ox.settings.use_cache = True
# ox.settings.overpass_url = "https://overpass-api.de/api"


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OSM_DIR = BASE_DIR / "data" / "osm"

OUTPUT_DIR = BASE_DIR / "outputs"
BOUNDARY_DIR = OUTPUT_DIR / "boundaries"
GRAPH_DIR = OUTPUT_DIR / "graphs"
NODE_DIR = OUTPUT_DIR / "nodes"
EDGE_DIR = OUTPUT_DIR / "edges"
METADATA_DIR = OUTPUT_DIR / "metadata"


for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    OSM_DIR,
    OUTPUT_DIR,
    BOUNDARY_DIR,
    GRAPH_DIR,
    NODE_DIR,
    EDGE_DIR,
    METADATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# The five reference cities from the RNQI documentation.
CITIES = {
    "Chandigarh": "Chandigarh, India",
    "Jaipur": "Jaipur, Rajasthan, India",
    "Pune": "Pune, Maharashtra, India",
    "Kolkata": "Kolkata, West Bengal, India",
    "Ranipet": "Ranipet, Tamil Nadu, India",
}

CITY_OSM_FILES = {
    "Chandigarh": OSM_DIR / "chandigarh.osm",
    "Jaipur": OSM_DIR / "jaipur.osm",
    "Pune": OSM_DIR / "pune.osm",
    "Kolkata": OSM_DIR / "kolkata.osm",
    "Ranipet": OSM_DIR / "ranipet.osm",
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def city_id_from_name(city_name):
    """Convert a city name into a consistent file identifier."""
    return city_name.lower().replace(" ", "_")


def format_seconds(seconds):
    """Format elapsed time for readable terminal output."""
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, remaining_seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

    hours, minutes = divmod(int(minutes), 60)

    return f"{hours}h {minutes}m {remaining_seconds:.2f}s"


def stage_timer():
    """Return a timer reference."""
    return time.perf_counter()


def log_stage(stage_name, start_time):
    """Print the elapsed time for a completed stage."""
    elapsed = time.perf_counter() - start_time
    print(f"[{stage_name}] Completed in {format_seconds(elapsed)}")
    return elapsed


# ============================================================
# 4. CITY BOUNDARY LOADING
# ============================================================

def get_city_boundary(city_name, place_query, boundary_path, force=False):
    """
    Load the saved city boundary when available.

    If no saved boundary exists, retain the existing OSMnx geocoding fallback
    so the original boundary-generation workflow remains available.
    """

    if boundary_path.exists() and not force:
        print(f"Reusing existing boundary: {boundary_path}")

        boundary = gpd.read_file(boundary_path)

        if boundary.empty:
            raise ValueError(
                f"Existing boundary file is empty for {city_name}."
            )

        boundary = boundary.to_crs(epsg=4326)

        return boundary

    print(f"Getting boundary for {city_name}...")

    boundary = ox.geocode_to_gdf(place_query)

    if boundary.empty:
        raise ValueError(
            f"No boundary found for {city_name}. "
            f"Check the place query: {place_query}"
        )

    # Keep the first returned place, preserving the existing boundary-selection behavior.
    boundary = boundary.iloc[[0]].copy()

    # Standard geographic CRS.
    boundary = boundary.to_crs(epsg=4326)

    boundary["city"] = city_name

    boundary.to_file(boundary_path, driver="GeoJSON")

    print(f"Saved boundary: {boundary_path}")

    return boundary


# ============================================================
# 5. ROAD NETWORK LOADING
# ============================================================

def download_road_network(city_name, boundary):
    """
    Load the locally prepared OSM road network for a city.

    The OSM file is generated separately from a regional OSM PBF.
    The actual city boundary is then applied here so that the
    analysis area remains consistent with the original pipeline.
    """

    print(f"Loading local road network for {city_name}...")

    # --------------------------------------------------------
    # Locate the prepared city-specific OSM XML file.
    # These files are created separately from regional OSM PBF extracts.
    # --------------------------------------------------------

    if city_name not in CITY_OSM_FILES:
        raise ValueError(
            f"No local OSM file configured for {city_name}."
        )

    osm_path = CITY_OSM_FILES[city_name]

    if not osm_path.exists():
        raise FileNotFoundError(
            f"Local OSM file not found for {city_name}:\n"
            f"{osm_path}\n\n"
            f"Prepare the city OSM file before running the "
            f"RNQI pipeline."
        )

    print(f"Using OSM file: {osm_path}")

    # --------------------------------------------------------
    # Load the prepared OSM XML into OSMnx.
    # --------------------------------------------------------

    graph = ox.graph_from_xml(
        osm_path,
        simplify=True,
        retain_all=True,
        bidirectional=False,
    )

    print(
        f"Loaded local graph: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    # --------------------------------------------------------
    # Apply the saved city boundary used for the analysis area.
    # --------------------------------------------------------

    polygon = (
        boundary
        .to_crs(graph.graph["crs"])
        .geometry
        .iloc[0]
    )

    graph = ox.truncate.truncate_graph_polygon(
        graph,
        polygon,
        truncate_by_edge=True,
    )

    print(
        f"Graph after city boundary: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    print(
        f"Successfully loaded road network for {city_name}"
    )

    return graph

# ============================================================
# 6. BASIC GRAPH PREPROCESSING
# ============================================================

def preprocess_graph(graph, city_name):
    """
    Perform initial graph cleaning and validation.

    Preserve disconnected components because network
    disconnectedness is relevant to RNQI.
    """

    print(f"Preprocessing graph for {city_name}...")

    graph = graph.copy()

    # Remove self-loop edges.
    self_loops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(self_loops)

    # Validate the graph.
    if graph.number_of_nodes() == 0:
        raise ValueError(
            f"No nodes found in the graph for {city_name}."
        )

    if graph.number_of_edges() == 0:
        raise ValueError(
            f"No edges found in the graph for {city_name}."
        )

    # Validate node coordinates.
    for node, attributes in graph.nodes(data=True):
        if "x" not in attributes or "y" not in attributes:
            raise ValueError(
                f"Node {node} in {city_name} has no coordinates."
            )

    # Report connectivity without deleting components.
    component_count = nx.number_weakly_connected_components(graph)

    print(
        f"{city_name}: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, "
        f"{component_count} weakly connected components"
    )

    return graph


# ============================================================
# 7. GRAPH PROJECTION
# ============================================================

def project_graph(graph):
    """
    Project the graph to a suitable local metric CRS.

    Required for metric distances, lengths, areas, and densities.
    """

    return ox.project_graph(graph)


# ============================================================
# 8. GRAPH EXPORT
# ============================================================

def export_graph_data(graph, city_name, force=False):
    """
    Export projected graph, nodes, and edges.

    Existing files are reused unless force=True.
    """

    city_id = city_id_from_name(city_name)

    graph_path = GRAPH_DIR / f"{city_id}.graphml"
    nodes_path = NODE_DIR / f"{city_id}_nodes.geojson"
    edges_path = EDGE_DIR / f"{city_id}_edges.geojson"

    # Save projected GraphML only when needed.
    if not graph_path.exists() or force:
        ox.save_graphml(graph, filepath=graph_path)
        print(f"Saved projected graph: {graph_path}")
    else:
        print(f"Reusing projected graph: {graph_path}")

    # Convert once and reuse for both exports.
    nodes, edges = ox.graph_to_gdfs(
        graph,
        nodes=True,
        edges=True,
        node_geometry=True,
        fill_edge_geometry=True,
    )

    if not nodes_path.exists() or force:
        nodes.to_file(nodes_path, driver="GeoJSON")
        print(f"Saved nodes: {nodes_path}")
    else:
        print(f"Reusing nodes: {nodes_path}")

    if not edges_path.exists() or force:
        edges.to_file(edges_path, driver="GeoJSON")
        print(f"Saved edges: {edges_path}")
    else:
        print(f"Reusing edges: {edges_path}")

    return {
        "graph_path": str(graph_path),
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
    }, nodes, edges


# ============================================================
# 9. CITY-WISE SUMMARY
# ============================================================

def create_city_summary(city_name, boundary, graph, nodes, edges):
    """
    Create descriptive network statistics.

    These are not RNQI scores.
    """

    # The graph is projected, so area and lengths are metric.
    city_area_m2 = (
        boundary.to_crs(graph.graph["crs"]).geometry.area.iloc[0]
    )

    total_road_length_m = edges["length"].sum()

    summary = {
        "city": city_name,
        "crs": str(graph.graph["crs"]),
        "number_of_nodes": int(graph.number_of_nodes()),
        "number_of_edges": int(graph.number_of_edges()),
        "city_area_m2": float(city_area_m2),
        "city_area_km2": float(city_area_m2 / 1_000_000),
        "total_road_length_m": float(total_road_length_m),
        "total_road_length_km": float(total_road_length_m / 1000),
        "average_node_degree": float(
            sum(dict(graph.degree()).values())
            / graph.number_of_nodes()
        ),
    }

    return summary


# ============================================================
# 10. LOAD EXISTING GRAPH FILES
# ============================================================

def load_existing_graph(graph_path, city_name, graph_type):
    """
    Load a previously saved GraphML file.

    graph_type is used only for clear terminal logging.
    """

    print(f"Loading existing {graph_type} graph for {city_name}...")
    graph = ox.load_graphml(graph_path)

    if graph.number_of_nodes() == 0:
        raise ValueError(
            f"Existing {graph_type} graph has no nodes: {graph_path}"
        )

    if graph.number_of_edges() == 0:
        raise ValueError(
            f"Existing {graph_type} graph has no edges: {graph_path}"
        )

    print(
        f"Loaded {graph_type} graph: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    return graph


# ============================================================
# 11. CITY-WISE PIPELINE
# ============================================================

def process_city(city_name, place_query, force=False):
    """
    Execute the complete initial pipeline for one city.

    Automatic reuse order:
        1. Existing boundary
        2. Existing projected graph
        3. Existing raw graph
        4. Prepared local OSM XML when a fresh raw graph is required
    """

    city_start = stage_timer()
    city_id = city_id_from_name(city_name)

    print("\n" + "=" * 60)
    print(f"Processing: {city_name}")
    print("=" * 60)

    boundary_path = BOUNDARY_DIR / f"{city_id}_boundary.geojson"
    raw_graph_path = RAW_DIR / f"{city_id}_raw.graphml"
    projected_graph_path = GRAPH_DIR / f"{city_id}.graphml"

    # --------------------------------------------------------
    # Stage 1: Boundary
    # --------------------------------------------------------

    stage_start = stage_timer()

    boundary = get_city_boundary(
        city_name,
        place_query,
        boundary_path,
        force=force,
    )

    boundary_time = log_stage("Boundary", stage_start)

    # --------------------------------------------------------
    # Stage 2: Raw graph
    # --------------------------------------------------------

    stage_start = stage_timer()

    if raw_graph_path.exists() and not force:
        print(f"Reusing raw graph: {raw_graph_path}")

        raw_graph = load_existing_graph(
            raw_graph_path,
            city_name,
            "raw",
        )

    else:
        raw_graph = download_road_network(
            city_name,
            boundary,
        )

        ox.save_graphml(
            raw_graph,
            filepath=raw_graph_path,
        )

        print(f"Saved raw graph: {raw_graph_path}")

    raw_graph_time = log_stage("Raw graph", stage_start)

    # --------------------------------------------------------
    # Stage 3: Preprocessing and projection
    # --------------------------------------------------------

    stage_start = stage_timer()

    if projected_graph_path.exists() and not force:
        print(
            f"Reusing existing projected graph: "
            f"{projected_graph_path}"
        )

        projected_graph = load_existing_graph(
            projected_graph_path,
            city_name,
            "projected",
        )

        preprocessing_time = 0.0
        projection_time = 0.0

        print(
            "Preprocessing and projection skipped "
            "(existing projected graph reused)."
        )

    else:
        cleaned_graph = preprocess_graph(
            raw_graph,
            city_name,
        )

        preprocessing_time = time.perf_counter() - stage_start

        print(
            f"[Preprocessing] Completed in "
            f"{format_seconds(preprocessing_time)}"
        )

        projection_start = stage_timer()

        projected_graph = project_graph(cleaned_graph)

        projection_time = log_stage(
            "Projection",
            projection_start,
        )

    # --------------------------------------------------------
    # Stage 4: Export
    # --------------------------------------------------------

    stage_start = stage_timer()

    exported_files, nodes, edges = export_graph_data(
        projected_graph,
        city_name,
        force=force,
    )

    export_time = log_stage("Export", stage_start)

    # --------------------------------------------------------
    # Stage 5: Summary
    # --------------------------------------------------------

    stage_start = stage_timer()

    summary = create_city_summary(
        city_name,
        boundary,
        projected_graph,
        nodes,
        edges,
    )

    summary_time = log_stage("Summary", stage_start)

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    city_execution_time = time.perf_counter() - city_start

    summary["boundary_path"] = str(boundary_path)
    summary["raw_graph_path"] = str(raw_graph_path)
    summary.update(exported_files)

    summary["boundary_execution_time_seconds"] = round(
        boundary_time,
        2,
    )

    summary["raw_graph_execution_time_seconds"] = round(
        raw_graph_time,
        2,
    )

    summary["preprocessing_execution_time_seconds"] = round(
        preprocessing_time,
        2,
    )

    summary["projection_execution_time_seconds"] = round(
        projection_time,
        2,
    )

    summary["export_execution_time_seconds"] = round(
        export_time,
        2,
    )

    summary["summary_execution_time_seconds"] = round(
        summary_time,
        2,
    )

    summary["execution_time_seconds"] = round(
        city_execution_time,
        2,
    )

    summary["execution_time"] = format_seconds(
        city_execution_time
    )

    print(
        f"\nCompleted: {city_name} "
        f"(Execution time: {format_seconds(city_execution_time)})"
    )

    print(json.dumps(summary, indent=2))

    return summary


# ============================================================
# 12. PIPELINE EXECUTION
# ============================================================

def run_pipeline(selected_cities=None, force=False):
    """
    Process all five reference cities or selected cities.

    Existing files are reused automatically.
    """

    pipeline_start = stage_timer()

    all_summaries = []

    cities_to_process = CITIES

    if selected_cities:
        cities_to_process = {
            city: CITIES[city]
            for city in selected_cities
            if city in CITIES
        }

        if not cities_to_process:
            raise ValueError(
                "None of the selected cities match the configured cities."
            )

    total_cities = len(cities_to_process)

    for city_index, (city_name, place_query) in enumerate(
        cities_to_process.items(),
        start=1,
    ):

        print(
            f"\nCity {city_index}/{total_cities}: {city_name}"
        )

        try:
            summary = process_city(
                city_name,
                place_query,
                force=force,
            )

            all_summaries.append(summary)

        except Exception as error:
            print(
                f"\nFailed to process {city_name}: "
                f"{type(error).__name__}: {error}"
            )

        # Keep a short pause between cities. This is only relevant when
        # processing multiple cities and is retained for the boundary fallback.
        if city_index < total_cities:
            time.sleep(2)

    # --------------------------------------------------------
    # Combined city-wise summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(all_summaries)

    summary_path = (
        METADATA_DIR / "city_wise_network_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    pipeline_execution_time = time.perf_counter() - pipeline_start

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)

    print(
        f"Total execution time: "
        f"{format_seconds(pipeline_execution_time)}"
    )

    print(
        f"Cities processed: "
        f"{len(all_summaries)}/{total_cities}"
    )

    print(f"Summary saved to: {summary_path}")

    print("\nAll available city summaries:")
    print(summary_df)

    return summary_df


# ============================================================
# 13. COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "RNQI initial city-wise data extraction and "
            "preprocessing pipeline."
        )
    )

    parser.add_argument(
        "--city",
        nargs="+",
        choices=list(CITIES.keys()),
        help=(
            "Process only selected cities. "
            "If omitted, process all five cities."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore existing boundaries and graphs and "
            "download/process fresh data."
        ),
    )

    return parser.parse_args()


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    run_pipeline(
        selected_cities=args.city,
        force=args.force,
    )