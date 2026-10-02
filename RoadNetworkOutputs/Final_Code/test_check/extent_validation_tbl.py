from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
BOUNDARY_DIR = OUTPUT_DIR / "boundaries"
GRAPH_DIR = OUTPUT_DIR / "graphs"

DIAGNOSTIC_DIR = OUTPUT_DIR / "extent_network_diagnostic"
PLOT_DIR = DIAGNOSTIC_DIR / "plots"

DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


CITIES = [
    "ranipet",
    "chandigarh",
    "kolkata",
    "pune",
    "jaipur",
]


# Road-buffer distances in metres
BUFFER_DISTANCES = [50, 100, 250]


# Distances used for built-up cell accessibility
ROAD_DISTANCE_THRESHOLDS = [100, 250, 500]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_column(gdf, candidates):
    """
    Find the first matching column from a list of possible names.
    """
    lower_map = {str(c).lower(): c for c in gdf.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


# =============================================================================
# LOAD CITY EXTENT
# =============================================================================

def load_city_extent(city):
    """
    Load the existing city extent.
    """

    path = BOUNDARY_DIR / f"{city}_city_extent.geojson"

    if not path.exists():
        raise FileNotFoundError(
            f"City extent not found:\n{path}"
        )

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"{city}: city extent is empty.")

    if gdf.crs is None:
        raise ValueError(
            f"{city}: city extent has no CRS."
        )

    # Convert to metric CRS
    if gdf.crs.to_epsg() != 3857:
        gdf = gdf.to_crs(epsg=3857)

    # Combine all geometries
    extent_geometry = gdf.geometry.union_all()

    extent = gpd.GeoDataFrame(
        {"city": [city]},
        geometry=[extent_geometry],
        crs="EPSG:3857"
    )

    return extent


# =============================================================================
# LOAD GRAPH
# =============================================================================
def load_graph(city):
    """
    Load the projected OSMnx GraphML and preserve its original CRS.

    The graph was originally projected using ox.project_graph(),
    so its node coordinates are already in a local metric CRS
    such as UTM. Do NOT assume projected coordinates are EPSG:3857.
    """

    path = GRAPH_DIR / f"{city}.graphml"

    if not path.exists():
        raise FileNotFoundError(
            f"Graph not found:\n{path}"
        )

    print("Loading graph:")
    print(path)

    # -------------------------------------------------------------
    # Load graph through OSMnx so graph CRS metadata is preserved
    # -------------------------------------------------------------
    G = ox.load_graphml(path)

    if len(G.nodes) == 0:
        raise ValueError(
            f"{city}: graph contains no nodes."
        )

    if len(G.edges) == 0:
        raise ValueError(
            f"{city}: graph contains no edges."
        )

    # -------------------------------------------------------------
    # Read the CRS stored in the projected GraphML
    # -------------------------------------------------------------
    graph_crs = G.graph.get("crs")

    if graph_crs is None:
        raise ValueError(
            f"{city}: GraphML does not contain graph CRS metadata."
        )

    print(f"Graph CRS from GraphML: {graph_crs}")

    # -------------------------------------------------------------
    # Convert graph nodes to GeoDataFrame
    # -------------------------------------------------------------
    nodes = ox.graph_to_gdfs(
        G,
        nodes=True,
        edges=False,
        node_geometry=True
    )

    if nodes.empty:
        raise ValueError(
            f"{city}: graph contains no node geometries."
        )

    # -------------------------------------------------------------
    # Make sure node CRS matches graph CRS
    # -------------------------------------------------------------
    if nodes.crs is None:
        nodes = nodes.set_crs(graph_crs)

    else:
        nodes = nodes.to_crs(graph_crs)

    return G, nodes

# =============================================================================
# CREATE ROAD-LIKE NETWORK GEOMETRY
# =============================================================================

def create_network_geometry(G, node_gdf):
    """
    Create LineStrings from graph edges using node coordinates.
    """

    from shapely.geometry import LineString

    node_lookup = node_gdf.geometry.to_dict()

    line_records = []

    for u, v in G.edges():

        if u not in node_lookup or v not in node_lookup:
            continue

        p1 = node_lookup[u]
        p2 = node_lookup[v]

        if p1.equals(p2):
            continue

        line_records.append({
            "u": u,
            "v": v,
            "geometry": LineString([
                (p1.x, p1.y),
                (p2.x, p2.y)
            ])
        })

    if not line_records:
        raise ValueError(
            "Could not create road-network geometries."
        )

    return gpd.GeoDataFrame(
        line_records,
        geometry="geometry",
        crs=node_gdf.crs
    )

# =============================================================================
# BUILT-UP CELLS
# =============================================================================

def derive_builtup_area_from_extent(extent):
    """
    Since your current city_extent is constructed from selected built-up cells,
    the extent itself represents the selected built-up area.

    This function keeps the terminology explicit rather than pretending that
    an independent building footprint layer is available.
    """

    return extent.geometry.iloc[0]


# =============================================================================
# ROAD BUFFER ANALYSIS
# =============================================================================

def calculate_buffer_metrics(
    city,
    extent_geometry,
    roads
):

    extent_area_m2 = extent_geometry.area

    results = {}

    for buffer_distance in BUFFER_DISTANCES:

        buffered_roads = roads.geometry.buffer(
            buffer_distance
        )

        # Combine all buffers
        buffer_union = buffered_roads.union_all()

        # Intersection with city extent
        intersection = buffer_union.intersection(
            extent_geometry
        )

        covered_area_m2 = intersection.area

        coverage_pct = (
            covered_area_m2 / extent_area_m2 * 100
            if extent_area_m2 > 0
            else np.nan
        )

        results[f"road_buffer_{buffer_distance}m_area_km2"] = (
            covered_area_m2 / 1_000_000
        )

        results[f"road_buffer_{buffer_distance}m_coverage_pct"] = (
            coverage_pct
        )

    return results


# =============================================================================
# DISTANCE OF EXTENT TO ROAD
# =============================================================================

def calculate_extent_road_distance(
    extent_geometry,
    roads
):
    """
    Approximate distance from representative points across the city extent
    to the nearest road.

    We use the centroid of each grid-like polygon component when possible.
    """

    road_union = roads.geometry.union_all()

    # Generate points from a regular grid over the extent.
    minx, miny, maxx, maxy = extent_geometry.bounds

    spacing = 500  # metres

    xs = np.arange(minx, maxx + spacing, spacing)
    ys = np.arange(miny, maxy + spacing, spacing)

    points = []

    for x in xs:
        for y in ys:

            p = Point(x, y)

            if extent_geometry.contains(p):
                points.append(p)

    if not points:
        points = [extent_geometry.centroid]

    distances = np.array([
        p.distance(road_union)
        for p in points
    ])

    return distances


# =============================================================================
# PLOT 1 — EXTENT + ROAD NETWORK
# =============================================================================

def plot_extent_network(
    city,
    extent,
    roads_in_extent,
    nodes_in_extent
):

    fig, ax = plt.subplots(figsize=(10, 10))

    # ---------------------------------------------------------
    # City extent
    # ---------------------------------------------------------
    extent.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=2,
        label="Selected city extent"
    )

    # ---------------------------------------------------------
    # Roads clipped to extent
    # ---------------------------------------------------------
    roads_in_extent.plot(
        ax=ax,
        linewidth=0.35,
        alpha=0.7,
        label="Road network"
    )

    # ---------------------------------------------------------
    # Nodes inside extent
    # ---------------------------------------------------------
    if len(nodes_in_extent) > 10000:

        nodes_plot = nodes_in_extent.sample(
            10000,
            random_state=42
        )

    else:

        nodes_plot = nodes_in_extent

    nodes_plot.plot(
        ax=ax,
        markersize=0.8,
        alpha=0.5,
        label="Network nodes"
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # force map to city extent
    # ---------------------------------------------------------

    minx, miny, maxx, maxy = extent.total_bounds

    margin_x = (maxx - minx) * 0.03
    margin_y = (maxy - miny) * 0.03

    ax.set_xlim(
        minx - margin_x,
        maxx + margin_x
    )

    ax.set_ylim(
        miny - margin_y,
        maxy + margin_y
    )

    ax.set_title(
        f"{city.title()} — Selected Extent and Road Network"
    )

    ax.set_axis_off()

    ax.legend()

    plt.tight_layout()

    output = (
        PLOT_DIR /
        f"{city}_extent_network.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output

# =============================================================================
# PLOT 2 — ROAD BUFFER COVERAGE
# =============================================================================

def plot_buffer_coverage(summary):

    df = pd.DataFrame(summary)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for buffer_distance in BUFFER_DISTANCES:

        column = (
            f"road_buffer_{buffer_distance}m_coverage_pct"
        )

        ax.plot(
            df["city"],
            df[column],
            marker="o",
            label=f"{buffer_distance} m"
        )

    ax.set_ylabel(
        "Extent covered by road buffer (%)"
    )

    ax.set_xlabel(
        "City"
    )

    ax.set_title(
        "Road-Network Buffer Coverage of Selected City Extent"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        PLOT_DIR /
        "all_cities_road_buffer_coverage.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output


# =============================================================================
# PLOT 3 — NODE / EDGE DENSITY
# =============================================================================

def plot_network_density(summary):

    df = pd.DataFrame(summary)

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        df["nodes_per_extent_km2"],
        width,
        label="Nodes / km²"
    )

    ax.bar(
        x + width / 2,
        df["edges_per_extent_km2"],
        width,
        label="Edges / km²"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["city"]
    )

    ax.set_ylabel(
        "Network density"
    )

    ax.set_title(
        "Road-Network Density Within Selected Extent"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        PLOT_DIR /
        "all_cities_network_density.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output


# =============================================================================
# PLOT 4 — DISTANCE TO NEAREST ROAD
# =============================================================================

def plot_road_distance(summary):

    df = pd.DataFrame(summary)

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - width,
        df["road_distance_median"],
        width,
        label="Median"
    )

    ax.bar(
        x,
        df["road_distance_p90"],
        width,
        label="90th percentile"
    )

    ax.bar(
        x + width,
        df["road_distance_max"],
        width,
        label="Maximum"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        df["city"]
    )

    ax.set_ylabel(
        "Distance to nearest road (m)"
    )

    ax.set_title(
        "Distance of Selected Urban Extent to Road Network"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        PLOT_DIR /
        "all_cities_road_distance.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output


# =============================================================================
# MAIN CITY ANALYSIS
# =============================================================================

def analyze_city(city):

    print()
    print("=" * 100)
    print(city.upper())
    print("=" * 100)

    # -------------------------------------------------------------
    # Load extent
    # -------------------------------------------------------------

    extent = load_city_extent(city)

    extent_geometry = extent.geometry.iloc[0]

    extent_area_km2 = (
        extent_geometry.area / 1_000_000
    )

    print(
        f"City extent area : {extent_area_km2:.2f} km²"
    )

    # -------------------------------------------------------------
    # Load graph
    # -------------------------------------------------------------

    G, nodes = load_graph(city)

    print(
        f"Nodes            : {len(G.nodes):,}"
    )

    print(
        f"Edges            : {len(G.edges):,}"
    )

    # -------------------------------------------------------------
    # Align city extent to graph CRS
    # -------------------------------------------------------------
    if extent.crs != nodes.crs:
        extent = extent.to_crs(nodes.crs)

    extent_geometry = extent.geometry.iloc[0]

    extent_area_km2 = (
        extent_geometry.area / 1_000_000
    )
    
    # -------------------------------------------------------------
    # Network geometry
    # -------------------------------------------------------------

    print("Creating road-network geometries...")

    roads = create_network_geometry(
        G,
        nodes
    )

    # Clip network to a reasonable analysis envelope
    roads_in_extent = roads[
    roads.geometry.intersects(extent_geometry)
    ].copy()

    roads_in_extent["geometry"] = (
        roads_in_extent.geometry.intersection(extent_geometry)
    )

    roads_in_extent = roads_in_extent[
        ~roads_in_extent.geometry.is_empty
    ].copy()

    nodes_in_extent = nodes[
        nodes.geometry.within(extent_geometry)
    ].copy()


    print(
        f"Road segments intersecting extent : "
        f"{len(roads_in_extent):,}"
    )

    # -------------------------------------------------------------
    # Buffer metrics
    # -------------------------------------------------------------

    buffer_metrics = calculate_buffer_metrics(
        city,
        extent_geometry,
        roads_in_extent
    )

    # -------------------------------------------------------------
    # Distance to roads
    # -------------------------------------------------------------

    print(
        "Calculating representative "
        "distance-to-road statistics..."
    )

    distances = calculate_extent_road_distance(
        extent_geometry,
        roads_in_extent
    )

    # -------------------------------------------------------------
    # Distance threshold percentages
    # -------------------------------------------------------------

    threshold_metrics = {}

    for threshold in ROAD_DISTANCE_THRESHOLDS:

        pct = (
            np.mean(
                distances <= threshold
            ) * 100
        )

        threshold_metrics[
            f"extent_points_within_{threshold}m_pct"
        ] = pct



    print("\nCRS / SPATIAL CHECK")
    print("--------------------")

    print("Extent CRS:")
    print(extent.crs)

    print("\nExtent bounds:")
    print(extent.total_bounds)

    print("\nNode CRS:")
    print(nodes.crs)

    print("\nNode bounds:")
    print(nodes.total_bounds)

    print("\nNodes inside extent:")
    print(len(nodes_in_extent))

    print("\nRoads intersecting extent:")
    print(len(roads_in_extent))
    # -------------------------------------------------------------
    # Density
    # -------------------------------------------------------------
    nodes_per_km2 = (
    len(nodes_in_extent) / extent_area_km2
)

    edges_per_km2 = (
        len(roads_in_extent) / extent_area_km2
    )
    

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    print("Creating city plot...")

    plot_path = plot_extent_network(
        city,
        extent,
        roads_in_extent,
        nodes_in_extent
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    result = {
        "city": city,
        "city_extent_area_km2": extent_area_km2,

        "nodes": len(G.nodes),
        "edges": len(G.edges),

        "nodes_per_extent_km2":
            nodes_per_km2,

        "edges_per_extent_km2":
            edges_per_km2,

        "road_distance_median":
            np.median(distances),

        "road_distance_p90":
            np.percentile(distances, 90),

        "road_distance_max":
            np.max(distances),

        "diagnostic_plot":
            str(plot_path),
    }

    result.update(
        buffer_metrics
    )

    result.update(
        threshold_metrics
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 100)
    print("EXTENT VS NETWORK DENSITY + ROAD ACCESSIBILITY DIAGNOSTIC")
    print("=" * 100)

    results = []

    for city in CITIES:

        try:

            result = analyze_city(
                city
            )

            results.append(
                result
            )

        except Exception as e:

            print()
            print(
                f"ERROR processing {city}:"
            )

            print(
                repr(e)
            )

    # -----------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------

    if not results:

        print(
            "No cities were successfully processed."
        )

        return

    summary = pd.DataFrame(
        results
    )

    summary_path = (
        DIAGNOSTIC_DIR /
        "extent_network_diagnostic.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # -----------------------------------------------------------------
    # Print important results
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL EXTENT VS NETWORK DIAGNOSTIC")
    print("=" * 100)

    display_columns = [
        "city",
        "city_extent_area_km2",
        "nodes",
        "edges",
        "nodes_per_extent_km2",
        "edges_per_extent_km2",
        "road_distance_median",
        "road_distance_p90",
        "road_distance_max",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Buffer coverage
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("ROAD BUFFER COVERAGE")
    print("=" * 100)

    coverage_columns = [
        "city"
    ]

    for distance in BUFFER_DISTANCES:

        coverage_columns.append(
            f"road_buffer_{distance}m_coverage_pct"
        )

    print(
        summary[
            coverage_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Distance thresholds
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("EXTENT POINTS WITHIN DISTANCE OF ROAD")
    print("=" * 100)

    distance_columns = [
        "city"
    ]

    for distance in ROAD_DISTANCE_THRESHOLDS:

        distance_columns.append(
            f"extent_points_within_{distance}m_pct"
        )

    print(
        summary[
            distance_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Comparison plots
    # -----------------------------------------------------------------

    print()
    print("Creating comparison plots...")

    plot_buffer_coverage(
        results
    )

    plot_network_density(
        results
    )

    plot_road_distance(
        results
    )

    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)

    print()
    print(
        f"CSV saved to:\n{summary_path}"
    )

    print()
    print(
        f"Plots saved to:\n{PLOT_DIR}"
    )


if __name__ == "__main__":
    main()