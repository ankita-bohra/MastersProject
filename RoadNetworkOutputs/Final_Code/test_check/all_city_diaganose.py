from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
BOUNDARY_DIR = OUTPUT_DIR / "boundaries"
GRAPH_DIR = OUTPUT_DIR / "graphs"
URBAN_CORE_DIR = OUTPUT_DIR / "urban_core"


# ============================================================
# CITIES
# ============================================================

CITIES = [
    "ranipet",
    "chandigarh",
    "kolkata",
    "pune",
    "jaipur",
]


# ============================================================
# FILE FINDER
# ============================================================

def find_file(folder, possible_names):

    for name in possible_names:

        path = folder / name

        if path.exists():
            return path

    return None


# ============================================================
# LOAD CITY EXTENT
# ============================================================

def load_city_extent(city):

    possible_names = [
        f"{city}_city_extent.geojson",
        f"{city}_extent.geojson",
        f"{city}_boundary.geojson",
    ]

    path = find_file(
        BOUNDARY_DIR,
        possible_names
    )

    if path is None:

        print(
            f"WARNING: City extent not found for {city}"
        )

        return None

    gdf = gpd.read_file(path)

    if gdf.empty:

        print(
            f"WARNING: Empty city extent for {city}"
        )

        return None

    return gdf


# ============================================================
# LOAD BUILT-UP CELLS
# ============================================================

def load_builtup_cells(city):

    possible_names = [

        f"{city}_built_up_cells.geojson",
        f"{city}_builtup_cells.geojson",
        f"{city}_built_up.geojson",
        f"{city}_grid.geojson",

    ]

    path = find_file(
        URBAN_CORE_DIR,
        possible_names
    )

    if path is None:

        # Search recursively
        matches = list(
            URBAN_CORE_DIR.rglob(
                f"*{city}*built*"
            )
        )

        if matches:
            path = matches[0]

    if path is None:

        print(
            f"WARNING: Built-up cells not found for {city}"
        )

        return None

    gdf = gpd.read_file(path)

    if gdf.empty:

        print(
            f"WARNING: Empty built-up file for {city}"
        )

        return None

    return gdf


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(city):

    path = GRAPH_DIR / f"{city}.graphml"

    if not path.exists():

        print(
            f"WARNING: Graph not found for {city}"
        )

        return None

    try:

        G = nx.read_graphml(path)

        return G

    except Exception as e:

        print(
            f"ERROR loading graph for {city}: {e}"
        )

        return None


# ============================================================
# NETWORK EXTENT
# ============================================================

def calculate_network_extent(G, city_extent):

    """
    Calculates the spatial extent occupied by the road network.

    Graph nodes are expected to have x/y coordinates.
    """

    coordinates = []

    for node, data in G.nodes(data=True):

        x = data.get("x")
        y = data.get("y")

        if x is None or y is None:
            continue

        try:

            x = float(x)
            y = float(y)

            coordinates.append((x, y))

        except Exception:
            continue

    if len(coordinates) == 0:

        return None

    network_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [p[0] for p in coordinates],
            [p[1] for p in coordinates],
        ),
        crs=city_extent.crs,
    )

    # If graph coordinates appear to be lon/lat,
    # convert them to the city extent CRS.
    if (
        network_points.total_bounds[0] >= -180
        and network_points.total_bounds[2] <= 180
        and network_points.total_bounds[1] >= -90
        and network_points.total_bounds[3] <= 90
    ):

        network_points = network_points.to_crs(
            city_extent.crs
        )

    return network_points


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

results = []


print()
print("=" * 100)
print("EXTENT VS NETWORK DENSITY DIAGNOSTIC")
print("=" * 100)


for city in CITIES:

    print()
    print("=" * 100)
    print(city.upper())
    print("=" * 100)

    # --------------------------------------------------------
    # CITY EXTENT
    # --------------------------------------------------------

    city_extent = load_city_extent(city)

    if city_extent is None:
        continue

    # Ensure projected CRS
    if city_extent.crs is None:

        print(
            "WARNING: City extent has no CRS."
        )

        continue

    if city_extent.crs.is_geographic:

        city_extent = city_extent.to_crs(
            "EPSG:3857"
        )

    city_extent_area_km2 = (
        city_extent.geometry.area.sum()
        / 1_000_000
    )


    # --------------------------------------------------------
    # BUILT-UP CELLS
    # --------------------------------------------------------

    builtup = load_builtup_cells(city)

    builtup_area_km2 = np.nan
    builtup_percentage = np.nan

    if builtup is not None:

        if builtup.crs != city_extent.crs:

            builtup = builtup.to_crs(
                city_extent.crs
            )

        # Clip to selected city extent
        builtup_clipped = gpd.clip(
            builtup,
            city_extent
        )

        if not builtup_clipped.empty:

            builtup_area_km2 = (
                builtup_clipped.geometry.area.sum()
                / 1_000_000
            )

            builtup_percentage = (
                builtup_area_km2
                / city_extent_area_km2
                * 100
            )


    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    G = load_graph(city)

    if G is None:
        continue

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()


    # --------------------------------------------------------
    # NETWORK DENSITY — FULL EXTENT
    # --------------------------------------------------------

    nodes_per_extent_km2 = (
        total_nodes /
        city_extent_area_km2
    )

    edges_per_extent_km2 = (
        total_edges /
        city_extent_area_km2
    )


    # --------------------------------------------------------
    # NETWORK DENSITY — BUILT-UP AREA
    # --------------------------------------------------------

    if (
        not np.isnan(builtup_area_km2)
        and builtup_area_km2 > 0
    ):

        nodes_per_builtup_km2 = (
            total_nodes /
            builtup_area_km2
        )

        edges_per_builtup_km2 = (
            total_edges /
            builtup_area_km2
        )

    else:

        nodes_per_builtup_km2 = np.nan
        edges_per_builtup_km2 = np.nan


    # --------------------------------------------------------
    # ROAD NETWORK SPATIAL EXTENT
    # --------------------------------------------------------

    network_points = calculate_network_extent(
        G,
        city_extent
    )

    road_network_area_km2 = np.nan
    road_network_coverage_pct = np.nan

    if network_points is not None:

        # Convex hull of road-network nodes
        network_hull = (
            network_points
            .unary_union
            .convex_hull
        )

        road_network_area_km2 = (
            network_hull.area /
            1_000_000
        )

        road_network_coverage_pct = (
            road_network_area_km2 /
            city_extent_area_km2
            * 100
        )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"City extent area          : "
        f"{city_extent_area_km2:.2f} km²"
    )

    print(
        f"Built-up area             : "
        f"{builtup_area_km2:.2f} km²"
    )

    print(
        f"Built-up % of extent      : "
        f"{builtup_percentage:.2f}%"
    )

    print(
        f"Road-network hull area    : "
        f"{road_network_area_km2:.2f} km²"
    )

    print(
        f"Network coverage of extent: "
        f"{road_network_coverage_pct:.2f}%"
    )

    print(
        f"Nodes                     : "
        f"{total_nodes:,}"
    )

    print(
        f"Edges                     : "
        f"{total_edges:,}"
    )

    print(
        f"Nodes / extent km²        : "
        f"{nodes_per_extent_km2:.2f}"
    )

    print(
        f"Edges / extent km²        : "
        f"{edges_per_extent_km2:.2f}"
    )

    print(
        f"Nodes / built-up km²      : "
        f"{nodes_per_builtup_km2:.2f}"
    )

    print(
        f"Edges / built-up km²      : "
        f"{edges_per_builtup_km2:.2f}"
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({

        "city": city,

        "city_extent_area_km2":
            city_extent_area_km2,

        "builtup_area_km2":
            builtup_area_km2,

        "builtup_percentage_of_extent":
            builtup_percentage,

        "road_network_hull_area_km2":
            road_network_area_km2,

        "road_network_coverage_pct":
            road_network_coverage_pct,

        "nodes":
            total_nodes,

        "edges":
            total_edges,

        "nodes_per_extent_km2":
            nodes_per_extent_km2,

        "edges_per_extent_km2":
            edges_per_extent_km2,

        "nodes_per_builtup_km2":
            nodes_per_builtup_km2,

        "edges_per_builtup_km2":
            edges_per_builtup_km2,

    })


# ============================================================
# FINAL TABLE
# ============================================================

if results:

    df = pd.DataFrame(results)

    print()
    print("=" * 100)
    print("FINAL EXTENT VS NETWORK DENSITY TABLE")
    print("=" * 100)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    output_path = (
        OUTPUT_DIR /
        "extent_vs_network_density_diagnostic.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved diagnostic to:\n{output_path}"
    )

else:

    print(
        "No results generated."
    )