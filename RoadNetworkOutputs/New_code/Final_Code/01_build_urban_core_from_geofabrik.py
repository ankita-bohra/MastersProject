# ============================================================
# 01_build_urban_core_from_geofabrik.py
#
# Purpose:
#   1. Use city-specific OSM road data as the source for
#      building extraction.
#   2. Extract OSM building footprints from regional Geofabrik PBF.
#   3. Create a complete regular 500 m grid.
#   4. Calculate building coverage for every grid cell.
#   5. Identify built-up cells.
#   6. Find connected built-up components.
#   7. Select the component associated with the target city.
#   8. Save the complete data-derived city extent.
#   9. Derive a denser urban core as a secondary layer.
#
# IMPORTANT:
#   - NO fixed study-area radius is used.
#   - NO municipal/administrative boundary is used as the
#     final city extent.
#   - The city centre is NOT used to create a radius.
#   - The city centre is used only to identify/validate the
#     target urban component.
#   - The final primary study area is city_extent.
#   - urban_core is only a secondary dense subset.
#
# ============================================================

from pathlib import Path

import argparse
import math
import json
import warnings
import shutil
import subprocess
import networkx as nx

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox

from shapely.geometry import box, Point
from shapely.ops import unary_union
from shapely.validation import make_valid

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

GRID_SIZE_M = 500

# 0.005 = 0.5% of a 500m x 500m grid cell.
#
# This is ONLY a built-up classification threshold.
# It does NOT define the city boundary.
BUILTUP_CELL_COVERAGE_THRESHOLD = 0.005

# Diagnostic only.
# This value does NOT determine the selected city extent.
MIN_COMPONENT_CELLS_DIAGNOSTIC = 20

USE_8_NEIGHBOR_CONNECTIVITY = True

# Always rebuild initially so stale/wrong building datasets
# cannot silently enter the analysis.
REBUILD_WHOLE_CITY_BUILDINGS = True

# Geographic sanity check only.
#
# This is NOT a study-area radius.
# It only catches obviously unrelated datasets such as
# Ranipet data accidentally containing Bengaluru buildings.
MAX_DATASET_CENTROID_DISTANCE_KM = 100


# ============================================================
# 2. DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GEOFABRIK_DIR = (
    BASE_DIR
    / "data"
    / "osm"
    / "geofabrik"
)

OUTPUT_DIR = BASE_DIR / "outputs"

# Final data-derived city extents are stored here.
BOUNDARY_DIR = (
    OUTPUT_DIR
    / "boundaries"
)

URBAN_DIR = (
    OUTPUT_DIR
    / "urban_core"
)

CITY_PBF_DIR = (
    URBAN_DIR
    / "city_pbf"
)

BUILDING_DIR = (
    URBAN_DIR
    / "buildings"
)

ROAD_OSM_DIR = (
    BASE_DIR
    / "data"
    / "osm"
)

GRID_DIR = (
    URBAN_DIR
    / "grid"
)

CORE_DIR = (
    URBAN_DIR
    / "core"
)

REPORT_DIR = (
    URBAN_DIR
    / "reports"
)


for directory in [
    BOUNDARY_DIR,
    CITY_PBF_DIR,
    BUILDING_DIR,
    GRID_DIR,
    CORE_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 3. CITY CONFIGURATION
# ============================================================
CITY_UTM_CRS = {
    "ranipet": "EPSG:32644",
    "chandigarh": "EPSG:32643",
    "kolkata": "EPSG:32645",
    "pune": "EPSG:32643",
    "jaipur": "EPSG:32643",
}


CITIES = {

    "Chandigarh": {
        "lat": 30.7333,
        "lon": 76.7794,
        "region_pbf":
            GEOFABRIK_DIR
            / "northern-zone-260924.osm.pbf",
    },

    "Jaipur": {
        "lat": 26.9124,
        "lon": 75.7873,
        "region_pbf":
            GEOFABRIK_DIR
            / "northern-zone-260924.osm.pbf",
    },

    "Pune": {
        "lat": 18.5204,
        "lon": 73.8567,
        "region_pbf":
            GEOFABRIK_DIR
            / "western-zone-260924.osm.pbf",
    },

    "Kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "region_pbf":
            GEOFABRIK_DIR
            / "eastern-zone-260924.osm.pbf",
    },

    "Ranipet": {
        "lat": 12.9249,
        "lon": 79.3333,
        "region_pbf":
            GEOFABRIK_DIR
            / "southern-zone-260924.osm.pbf",
    },
}


# ============================================================
# 4. CITY-SPECIFIC ROAD OSM VALIDATION
# ============================================================

def load_city_road_data(
    city,
    road_osm_path,
):
    """
    Load the existing city-specific road OSM.

    IMPORTANT:
        This dataset is NOT itself the final city boundary.

        It is only the source used to derive the building
        extraction envelope.

    """

    if not road_osm_path.exists():
        raise FileNotFoundError(
            f"{city}: road OSM file not found:\n"
            f"{road_osm_path}"
        )

    print(
        "\nLoading city-specific road OSM:"
    )
    print(
        road_osm_path
    )

    graph = ox.graph_from_xml(
        road_osm_path,
        simplify=False,
        retain_all=True,
    )

    if graph.number_of_nodes() == 0:
        raise RuntimeError(
            f"{city}: road OSM contains no nodes."
        )

    if graph.number_of_edges() == 0:
        raise RuntimeError(
            f"{city}: road OSM contains no edges."
        )

    print(
        f"{city} road nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"{city} road edges: "
        f"{graph.number_of_edges():,}"
    )

    nodes = ox.graph_to_gdfs(
        graph,
        nodes=True,
        edges=False,
    )

    if nodes.empty:
        raise RuntimeError(
            f"{city}: road OSM node layer is empty."
        )

    if nodes.crs is None:
        nodes = nodes.set_crs(
            "EPSG:4326"
        )

    nodes = nodes.to_crs(
        "EPSG:4326"
    )

    return graph, nodes



# ============================================================
# 5. CREATE DATA-DRIVEN BUILDING EXTRACTION ENVELOPE
# ============================================================
def create_building_extraction_envelope_from_road_osm(
    city,
    road_osm_path,
    config,
):
    """
    Create a temporary building-extraction envelope directly
    from the regional Geofabrik PBF.

    IMPORTANT
    ---------
    This envelope is ONLY used to limit the amount of
    building data extracted from the regional Geofabrik PBF.

    It is NOT:
        - the final RNQI study boundary
        - the administrative city boundary
        - the final city_extent
        - the urban_core
        - the final analysis area

    The final city_extent is determined later using:

        Geofabrik buildings
              ↓
        500 m grid
              ↓
        building coverage
              ↓
        built-up cells
              ↓
        connected components
              ↓
        component containing target city centre
              ↓
        final city_extent

    The road OSM is intentionally NOT loaded here.

    This avoids the previous extremely expensive operation:

        road graph
             ↓
        hundreds of thousands of edges
             ↓
        geometry union
             ↓
        large extraction polygon

    Instead, the regional Geofabrik PBF is queried directly
    around the target city.

    The extraction window is deliberately larger than the
    expected urban footprint so that the building-density
    analysis, rather than an arbitrary radius, determines
    the final city extent.
    """

    # ========================================================
    # 1. GET REGIONAL GEOFABRIK PBF
    # ========================================================

    region_pbf = config.get(
        "region_pbf"
    )

    if region_pbf is None:
        raise RuntimeError(
            f"{city}: 'region_pbf' is missing from city "
            "configuration."
        )

    region_pbf = Path(
        region_pbf
    )

    if not region_pbf.exists():
        raise FileNotFoundError(
            f"{city}: Geofabrik regional PBF was not found:\n"
            f"{region_pbf}"
        )

    print(
        "\nUsing regional Geofabrik PBF directly:"
    )

    print(
        region_pbf
    )

    # ========================================================
    # 2. CITY CENTRE
    # ========================================================

    lat = float(
        config["lat"]
    )

    lon = float(
        config["lon"]
    )

    print(
        f"\n{city} target centre:"
    )

    print(
        f"  latitude  = {lat:.6f}"
    )

    print(
        f"  longitude = {lon:.6f}"
    )

    # ========================================================
    # 3. EXTRACTION WINDOW
    # ========================================================
    #
    # IMPORTANT:
    #
    # This is NOT the final study radius.
    #
    # It is only a computational extraction window used
    # to obtain enough buildings from the regional PBF.
    #
    # The final city extent is still determined from the
    # connected built-up component later.
    #
    # 25 km half-width gives a 50 km x 50 km candidate
    # window.
    #
    # This is deliberately generous for the five cities
    # and avoids the previous enormous regional road graph.
    #
    # ========================================================

    EXTRACTION_HALF_WIDTH_KM = float(
        config.get(
            "building_extraction_half_width_km",
            50.0,
        )
    )

    if EXTRACTION_HALF_WIDTH_KM <= 0:
        raise ValueError(
            f"{city}: building extraction half-width "
            "must be greater than zero."
        )

    print(
        "\nTemporary building extraction window:"
    )

    print(
        f"  half-width = "
        f"{EXTRACTION_HALF_WIDTH_KM:.1f} km"
    )

    print(
        "  IMPORTANT: this is NOT the final RNQI extent."
    )

    # ========================================================
    # 4. APPROXIMATE DEGREE CONVERSION
    # ========================================================
    #
    # 1 degree latitude is approximately 111.32 km.
    #
    # Longitude degree length depends on latitude.
    #
    # This conversion is only used to create the temporary
    # PBF extraction polygon.
    #
    # ========================================================

    km_per_degree_lat = 111.32

    km_per_degree_lon = (
        111.32
        * math.cos(
            math.radians(lat)
        )
    )

    if km_per_degree_lon <= 0:
        raise RuntimeError(
            f"{city}: invalid longitude conversion."
        )

    delta_lat = (
        EXTRACTION_HALF_WIDTH_KM
        / km_per_degree_lat
    )

    delta_lon = (
        EXTRACTION_HALF_WIDTH_KM
        / km_per_degree_lon
    )

    min_lat = lat - delta_lat
    max_lat = lat + delta_lat

    min_lon = lon - delta_lon
    max_lon = lon + delta_lon

    print(
        "\nTemporary extraction bounds:"
    )

    print(
        f"  min_lon = {min_lon:.8f}"
    )

    print(
        f"  min_lat = {min_lat:.8f}"
    )

    print(
        f"  max_lon = {max_lon:.8f}"
    )

    print(
        f"  max_lat = {max_lat:.8f}"
    )

    # ========================================================
    # 5. CREATE EXTRACTION POLYGON
    # ========================================================

    envelope = gpd.GeoDataFrame(
        {
            "city": [city],
            "method": [
                "geofabrik-direct-temporary-extraction-window"
            ],
            "temporary_half_width_km": [
                EXTRACTION_HALF_WIDTH_KM
            ],
        },
        geometry=[
            box(
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            )
        ],
        crs="EPSG:4326",
    )

    # ========================================================
    # 6. CALCULATE TEMPORARY WINDOW AREA
    # ========================================================

    metric_crs = CITY_UTM_CRS[city.lower()]
    
    envelope_metric = envelope.to_crs(
        metric_crs
    )

    envelope_area_km2 = (
        envelope_metric.geometry.area.sum()
        / 1e6
    )

    print(
        f"\n{city} temporary extraction window area:"
    )

    print(
        f"  {envelope_area_km2:.2f} km²"
    )

    # ========================================================
    # 7. CHECK CENTRE IS INSIDE WINDOW
    # ========================================================

    center_point = gpd.GeoSeries(
        [
            Point(
                lon,
                lat,
            )
        ],
        crs="EPSG:4326",
    )

    if not envelope.geometry.iloc[0].contains(
        center_point.iloc[0]
    ):
        raise RuntimeError(
            f"{city}: target city centre is not inside "
            "the temporary extraction window."
        )

    # ========================================================
    # 8. SAVE EXTRACTION ENVELOPE
    # ========================================================

    candidate_path = (
        BOUNDARY_DIR
        / (
            f"{city.lower()}_"
            f"building_extraction_envelope.geojson"
        )
    )

    envelope.to_file(
        candidate_path,
        driver="GeoJSON",
    )

    print(
        "\nSaved temporary building extraction envelope:"
    )

    print(
        candidate_path
    )

    # ========================================================
    # 9. DIAGNOSTIC INFORMATION
    # ========================================================

    envelope_centroid = (
        envelope_metric
        .geometry
        .iloc[0]
        .centroid
    )

    
    center_metric = (
        center_point
        .to_crs(metric_crs)
        .iloc[0]
    )

    centroid_distance_km = (
        envelope_centroid.distance(
            center_metric
        )
        / 1000
    )

    print(
        f"\n{city} extraction-envelope centroid "
        f"distance from target centre: "
        f"{centroid_distance_km:.2f} km"
    )

    # ========================================================
    # 10. RETURN ONLY THE TEMPORARY EXTRACTION ENVELOPE
    # ========================================================
    #
    # The actual building extraction function will use this
    # polygon against the regional Geofabrik PBF.
    #
    # No road graph is constructed.
    #
    # No road geometry union is performed.
    #
    # ========================================================

    return envelope

# ============================================================
# 6. BUILDING DATASET VALIDATION
# ============================================================

def validate_city_building_dataset(
    city,
    buildings,
    config,
):
    """
    Validate building data geographically.

    The distance check is ONLY a data-quality safeguard.

    It does NOT define the city extent.
    """

    if buildings.empty:
        raise RuntimeError(
            f"{city}: building dataset is empty."
        )

    if buildings.crs is None:
       buildings = buildings.set_crs("EPSG:4326")

    buildings = buildings.to_crs(
        CITY_UTM_CRS[city.lower()]
    )

    bounds = buildings.total_bounds

    if not np.all(
        np.isfinite(bounds)
    ):
        raise RuntimeError(
            f"{city}: building dataset has "
            "invalid bounds."
        )

    valid_geometry_mask = (
        buildings.geometry.notna()
        &
        ~buildings.geometry.is_empty
    )

    if not valid_geometry_mask.any():
        raise RuntimeError(
            f"{city}: building dataset contains "
            "no valid geometries."
        )

    geometry_union = (
        buildings.loc[
            valid_geometry_mask,
            "geometry",
        ].union_all()
    )

    if geometry_union.is_empty:
        raise RuntimeError(
            f"{city}: building geometry union is empty."
        )

    centroid = geometry_union.centroid

    center = Point(
        config["lon"],
        config["lat"],
    )

    center_gs = gpd.GeoSeries(
        [center],
        crs="EPSG:4326",
    )

    centroid_gs = gpd.GeoSeries(
        [centroid],
        crs=buildings.crs,
    )

    metric_crs = CITY_UTM_CRS[city.lower()]
    center_metric = center_gs.to_crs(metric_crs)
    centroid_metric = centroid_gs.to_crs(metric_crs)

    distance_km = (
        centroid_metric
        .distance(center_metric)
        .iloc[0]
        / 1000
    )

    print(
        f"\n{city} building dataset validation:"
    )

    print(
        f"  CRS: {buildings.crs}"
    )

    print(
        f"  Building count: "
        f"{len(buildings):,}"
    )

    print(
        f"  Bounds: "
        f"{bounds.tolist()}"
    )

    print(
        f"  Target centre: "
        f"{config['lat']}, "
        f"{config['lon']}"
    )

    print(
        f"  Dataset centroid: "
        f"{centroid.y:.8f}, "
        f"{centroid.x:.8f}"
    )

    print(
        f"  Centroid distance from target centre: "
        f"{distance_km:.2f} km"
    )

    # --------------------------------------------------------
    # Geographic sanity check
    # --------------------------------------------------------

    if (
        distance_km
        > MAX_DATASET_CENTROID_DISTANCE_KM
    ):
        raise RuntimeError(
            f"{city}: building dataset does not "
            f"appear to correspond to the target city. "
            f"Dataset centroid is "
            f"{distance_km:.2f} km away."
        )

    print(
        f"{city}: building dataset passed "
        "geographic validation."
    )

    return buildings


# ============================================================
# 7. CITY PATHS
# ============================================================

def get_city_paths(city):
    """
    Return all output paths for a city.
    """

    city_slug = (
        city.lower()
        .replace(" ", "_")
    )

    return {

        "city_pbf":
            CITY_PBF_DIR
            / f"{city_slug}.osm.pbf",

        "building_pbf":
            BUILDING_DIR
            / f"{city_slug}_buildings.osm.pbf",

        "buildings_geojson":
            BUILDING_DIR
            / f"{city_slug}_buildings.geojson",

        "grid_geojson":
            GRID_DIR
            / f"{city_slug}_grid.geojson",

        "built_cells_geojson":
            GRID_DIR
            / f"{city_slug}_built_cells.geojson",

        "core_geojson":
            CORE_DIR
            / f"{city_slug}_core.geojson",

        "city_extent_geojson":
            BOUNDARY_DIR
            / f"{city_slug}_city_extent.geojson",

        "components_csv":
            CORE_DIR
            / f"{city_slug}_components.csv",

        "report_json":
            REPORT_DIR
            / f"{city_slug}_report.json",
    }


# ============================================================
# 8. EXTRACT BUILDINGS FROM REGIONAL GEOFABRIK PBF
# ============================================================

def extract_whole_city_buildings(
    osmium_path,
    city,
    region_pbf,
    candidate_polygon,
    paths,
):
    """
    Extract building footprints from the regional PBF.

    The candidate polygon is only an extraction envelope.
    """

    if candidate_polygon.empty:
        raise RuntimeError(
            f"{city}: extraction envelope is empty."
        )

    candidate_polygon = (
        candidate_polygon
        .to_crs("EPSG:4326")
    )

    bounds = candidate_polygon.total_bounds

    print(
        f"\n{city} building extraction bounds:"
    )

    print(
        f"  min_lon = {bounds[0]:.8f}"
    )

    print(
        f"  min_lat = {bounds[1]:.8f}"
    )

    print(
        f"  max_lon = {bounds[2]:.8f}"
    )

    print(
        f"  max_lat = {bounds[3]:.8f}"
    )

    candidate_path = (
        BOUNDARY_DIR
        / (
            f"{city.lower()}_"
            f"building_extraction_envelope.geojson"
        )
    )

    candidate_polygon.to_file(
        candidate_path,
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # Delete stale outputs when rebuilding
    # --------------------------------------------------------

    if REBUILD_WHOLE_CITY_BUILDINGS:

        print(
            "\nRemoving stale building outputs..."
        )

        for key in (
            "city_pbf",
            "building_pbf",
            "buildings_geojson",
        ):

            paths[key].unlink(
                missing_ok=True
            )

    # --------------------------------------------------------
    # Extract city-area PBF
    # --------------------------------------------------------

    if not paths["city_pbf"].exists():

        print(
            "\nExtracting target area from "
            "regional Geofabrik PBF..."
        )

        subprocess.run(
            [
                osmium_path,
                "extract",
                "-p",
                str(candidate_path),
                "-o",
                str(paths["city_pbf"]),
                str(region_pbf),
            ],
            check=True,
        )

    # --------------------------------------------------------
    # Extract buildings
    # --------------------------------------------------------

    if not paths["building_pbf"].exists():

        print(
            "\nExtracting building features..."
        )

        subprocess.run(
            [
                osmium_path,
                "tags-filter",
                str(paths["city_pbf"]),
                "w/building",
                "-o",
                str(paths["building_pbf"]),
            ],
            check=True,
        )

    # --------------------------------------------------------
    # Export building polygons
    # --------------------------------------------------------

    if not paths["buildings_geojson"].exists():

        print(
            "\nExporting building polygons..."
        )

        subprocess.run(
            [
                osmium_path,
                "export",
                str(paths["building_pbf"]),
                "-o",
                str(paths["buildings_geojson"]),
                "--geometry-types=polygon,multipolygon",
            ],
            check=True,
        )

    # --------------------------------------------------------
    # Immediate validation
    # --------------------------------------------------------

    print(
        "\nValidating freshly extracted "
        "building dataset..."
    )

    buildings_check = gpd.read_file(
        paths["buildings_geojson"]
    )

    buildings_check = (
        validate_city_building_dataset(
            city,
            buildings_check,
            CITIES[city],
        )
    )

    print(
        f"{city}: extracted building dataset "
        "passed geographic validation."
    )

    print(
        "Building data ready:"
    )

    print(
        paths["buildings_geojson"]
    )


# ============================================================
# 9. CLEAN BUILDINGS
# ============================================================

def clean_buildings(
    buildings,
):
    """
    Clean OSM building geometries.
    """

    buildings = buildings.copy()

    print(
        "\nCleaning building geometries..."
    )

    # --------------------------------------------------------
    # Remove missing geometries
    # --------------------------------------------------------

    buildings = buildings[
        buildings.geometry.notna()
    ].copy()

    # --------------------------------------------------------
    # Remove empty geometries
    # --------------------------------------------------------

    buildings = buildings[
        ~buildings.geometry.is_empty
    ].copy()

    print(
        "Buildings after removing empty/missing "
        f"geometries: {len(buildings):,}"
    )

    # --------------------------------------------------------
    # Repair invalid geometries
    # --------------------------------------------------------

    invalid_mask = (
        ~buildings.geometry.is_valid
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    print(
        "Invalid geometries before repair: "
        f"{invalid_count:,}"
    )

    if invalid_count > 0:

        print(
            "Repairing invalid geometries..."
        )

        buildings.loc[
            invalid_mask,
            "geometry",
        ] = (
            buildings.loc[
                invalid_mask,
                "geometry",
            ]
            .apply(make_valid)
        )

    # --------------------------------------------------------
    # Remove remaining invalid geometries
    # --------------------------------------------------------

    invalid_after = (
        ~buildings.geometry.is_valid
    )

    invalid_after_count = int(
        invalid_after.sum()
    )

    print(
        "Invalid geometries after repair: "
        f"{invalid_after_count:,}"
    )

    if invalid_after_count > 0:

        buildings = buildings[
            ~invalid_after
        ].copy()

    # --------------------------------------------------------
    # Keep polygonal geometries
    # --------------------------------------------------------

    buildings = buildings[
        buildings.geometry.geom_type.isin(
            [
                "Polygon",
                "MultiPolygon",
            ]
        )
    ].copy()

    buildings.reset_index(
        drop=True,
        inplace=True,
    )

    if buildings.empty:
        raise RuntimeError(
            "No valid polygonal buildings remain "
            "after cleaning."
        )

    print(
        f"Valid polygonal buildings: "
        f"{len(buildings):,}"
    )

    return buildings


# ============================================================
# 10. LOAD BUILDINGS
# ============================================================

def load_buildings(
    city,
    paths,
):
    """
    Load validated buildings and project them to a metric CRS.
    """

    building_file = (
        paths["buildings_geojson"]
    )

    if not building_file.exists():

        raise FileNotFoundError(
            f"{city}: building file not found:\n"
            f"{building_file}"
        )

    print(
        "\nLoading buildings:"
    )

    print(
        building_file
    )

    buildings = gpd.read_file(
        building_file
    )

    if buildings.empty:
        raise RuntimeError(
            f"{city}: building dataset is empty."
        )

    if buildings.crs is None:
        buildings = buildings.set_crs("EPSG:4326")  

    # --------------------------------------------------------
    # Validate geographic location BEFORE projection
    # --------------------------------------------------------

    buildings = (
        validate_city_building_dataset(
            city,
            buildings,
            CITIES[city],
        )
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    buildings = clean_buildings(
        buildings
    )

    # --------------------------------------------------------
    # Validate again
    # --------------------------------------------------------

    buildings = (
        validate_city_building_dataset(
            city,
            buildings,
            CITIES[city],
        )
    )

    # --------------------------------------------------------
    # Project to metric CRS
    #
    # EPSG:3857 is retained for consistency with the
    # current project grid/RNQI workflow.
    # --------------------------------------------------------

    buildings = buildings.to_crs(CITY_UTM_CRS[city.lower()])

    if buildings.empty:
        raise RuntimeError(
            f"{city}: no valid buildings remain."
        )

    total_building_area = (
        buildings.geometry.area.sum()
    )

    print(
        f"Buildings loaded: "
        f"{len(buildings):,}"
    )

    print(
        f"Total building footprint: "
        f"{total_building_area / 1e6:.2f} km²"
    )

    return buildings


# ============================================================
# 11. CREATE COMPLETE 500m GRID
# ============================================================

def create_grid(
    buildings,
):
    """
    Create a complete 500m grid over the extracted
    building-data extent.

    Empty cells are retained.

    No radius is used.

    No administrative boundary is used.
    """

    print(
        "\nCreating complete 500m spatial grid..."
    )

    if buildings.empty:
        raise RuntimeError(
            "Cannot create grid from empty buildings."
        )

    # --------------------------------------------------------
    # Use entire building-data bounding extent
    # --------------------------------------------------------

    minx, miny, maxx, maxy = (
        buildings.total_bounds
    )

    if not all(
        np.isfinite(
            [
                minx,
                miny,
                maxx,
                maxy,
            ]
        )
    ):
        raise RuntimeError(
            "Building extent contains invalid bounds."
        )

    size = GRID_SIZE_M

    # --------------------------------------------------------
    # Snap to 500m grid
    # --------------------------------------------------------

    minx = (
        math.floor(minx / size)
        * size
    )

    maxx = (
        math.ceil(maxx / size)
        * size
    )

    miny = (
        math.floor(miny / size)
        * size
    )

    maxy = (
        math.ceil(maxy / size)
        * size
    )

    n_cols = int(
        round(
            (maxx - minx)
            / size
        )
    )

    n_rows = int(
        round(
            (maxy - miny)
            / size
        )
    )

    total_cells = (
        n_rows
        * n_cols
    )

    print(
        f"Grid dimensions: "
        f"{n_rows} rows × "
        f"{n_cols} columns"
    )

    print(
        f"Total grid cells: "
        f"{total_cells:,}"
    )

    grid_area_km2 = (total_cells * size * size) / 1e6
    print(f"Grid area: {grid_area_km2:.2f} km²")    

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if total_cells > 2_000_000:

        raise RuntimeError(
            f"{total_cells:,} grid cells would be created.\n\n"
            "The extracted building dataset is unexpectedly "
            "large.\n\n"
            "This usually means the city-specific OSM "
            "extraction envelope is too broad.\n\n"
            "Check data/osm/<city>.osm before continuing."
        )

    # --------------------------------------------------------
    # Create every grid cell
    # --------------------------------------------------------

    cells = []

    grid_id = 0

    for row in range(
        n_rows
    ):

        y = (
            miny
            + row * size
        )

        for col in range(
            n_cols
        ):

            x = (
                minx
                + col * size
            )

            cells.append(
                {
                    "grid_id":
                        grid_id,

                    "row":
                        row,

                    "col":
                        col,

                    "geometry":
                        box(
                            x,
                            y,
                            x + size,
                            y + size,
                        ),
                }
            )

            grid_id += 1

    grid = gpd.GeoDataFrame(
        cells,
        crs=buildings.crs,
    )

    grid["cell_area_m2"] = (
        grid.geometry.area
    )

    print(
        f"Grid cells created: "
        f"{len(grid):,}"
    )

    print(
        "Median grid-cell area: "
        f"{grid['cell_area_m2'].median():.2f} m²"
    )

    return grid


# ============================================================
# 12. CALCULATE BUILDING COVERAGE
# ============================================================

def calculate_building_coverage(
    grid,
    buildings,
    threshold,
):
    """
    Calculate building footprint coverage for every grid cell.
    """

    print(
        "\nCalculating building coverage..."
    )

    intersections = gpd.overlay(
        grid[
            [
                "grid_id",
                "geometry",
            ]
        ],
        buildings[
            [
                "geometry",
            ]
        ],
        how="intersection",
        keep_geom_type=False,
    )

    print(
        f"Building-grid intersections: "
        f"{len(intersections):,}"
    )

    grid = grid.copy()

    if intersections.empty:

        grid[
            "building_area_m2"
        ] = 0.0

    else:

        intersections[
            "intersection_area_m2"
        ] = (
            intersections.geometry.area
        )

        building_area_by_grid = (
            intersections
            .groupby(
                "grid_id"
            )[
                "intersection_area_m2"
            ]
            .sum()
        )

        grid[
            "building_area_m2"
        ] = (
            grid["grid_id"]
            .map(
                building_area_by_grid
            )
            .fillna(0.0)
        )

    # --------------------------------------------------------
    # Coverage ratio
    # --------------------------------------------------------

    grid[
        "building_coverage"
    ] = (
        grid["building_area_m2"]
        / grid["cell_area_m2"]
    ).clip(
        lower=0,
        upper=1,
    )

    # --------------------------------------------------------
    # Built-up classification
    # --------------------------------------------------------

    grid[
        "built_up"
    ] = (
        grid["building_coverage"]
        >= threshold
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    print(
        "\nBuilding-density statistics:"
    )

    print(
        grid["building_coverage"].describe()
    )

    built_count = int(
        grid["built_up"].sum()
    )

    print(
        "\nBuilt-up cells: "
        f"{built_count:,} / "
        f"{len(grid):,}"
    )

    built_percentage = (
        built_count / len(grid) * 100
    )

    print(
        f"Built-up percentage: {built_percentage:.2f}%"
    )
    return grid


# ============================================================
# 13. FIND CONNECTED COMPONENTS
# ============================================================

def find_connected_components(
    grid,
):
    """
    Find connected components among built-up cells.

    Connectivity is based on grid row/column coordinates.
    """

    print(
        "\nFinding connected built-up components..."
    )

    built = grid[
        grid["built_up"]
    ].copy()

    built.reset_index(
        drop=True,
        inplace=True,
    )

    if built.empty:
        raise RuntimeError(
            "No built-up cells found."
        )

    # --------------------------------------------------------
    # Coordinate lookup
    # --------------------------------------------------------

    coordinate_to_id = {}

    id_to_coordinate = {}

    for _, row in built.iterrows():

        coord = (
            int(row["row"]),
            int(row["col"]),
        )

        grid_id = int(
            row["grid_id"]
        )

        coordinate_to_id[
            coord
        ] = grid_id

        id_to_coordinate[
            grid_id
        ] = coord

    # --------------------------------------------------------
    # Neighbours
    # --------------------------------------------------------

    if USE_8_NEIGHBOR_CONNECTIVITY:

        neighbours = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    else:

        neighbours = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

    # --------------------------------------------------------
    # DFS
    # --------------------------------------------------------

    unvisited = set(
        id_to_coordinate.keys()
    )

    components = []

    while unvisited:

        start_id = next(
            iter(unvisited)
        )

        stack = [
            start_id
        ]

        component = []

        while stack:

            current_id = stack.pop()

            if current_id not in unvisited:
                continue

            unvisited.remove(
                current_id
            )

            component.append(
                current_id
            )

            row, col = (
                id_to_coordinate[
                    current_id
                ]
            )

            for dr, dc in neighbours:

                neighbour_coord = (
                    row + dr,
                    col + dc,
                )

                neighbour_id = (
                    coordinate_to_id.get(
                        neighbour_coord
                    )
                )

                if (
                    neighbour_id is not None
                    and neighbour_id in unvisited
                ):

                    stack.append(
                        neighbour_id
                    )

        components.append(
            component
        )

    # --------------------------------------------------------
    # Component IDs
    # --------------------------------------------------------

    grid_id_to_component = {}

    for (
        component_id,
        component
    ) in enumerate(
        components
    ):

        for grid_id in component:

            grid_id_to_component[
                grid_id
            ] = component_id

    built[
        "component_id"
    ] = (
        built["grid_id"]
        .map(
            grid_id_to_component
        )
    )

    # --------------------------------------------------------
    # Component statistics
    # --------------------------------------------------------

    component_records = []

    for (
        component_id,
        component
    ) in enumerate(
        components
    ):

        component_cells = built[
            built["component_id"]
            == component_id
        ]

        component_area = (
            component_cells[
                "cell_area_m2"
            ].sum()
        )

        building_area = (
            component_cells[
                "building_area_m2"
            ].sum()
        )

        mean_coverage = (
            component_cells[
                "building_coverage"
            ].mean()
        )

        median_coverage = (
            component_cells[
                "building_coverage"
            ].median()
        )

        component_records.append(
            {
                "component_id":
                    component_id,

                "cell_count":
                    len(component_cells),

                "component_area_m2":
                    component_area,

                "component_area_km2":
                    component_area / 1e6,

                "building_area_m2":
                    building_area,

                "mean_building_coverage":
                    mean_coverage,

                "median_building_coverage":
                    median_coverage,
            }
        )

    component_df = pd.DataFrame(
        component_records
    )

    component_df = (
        component_df
        .sort_values(
            "cell_count",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nConnected built-up components: "
        f"{len(component_df):,}"
    )

    print(
        "\nLargest components:"
    )

    print(
        component_df.head(10).to_string(
            index=False
        )
    )

    return (
        built,
        component_df,
    )


# ============================================================
# 14. SELECT TARGET CITY COMPONENT
# ============================================================

def select_city_extent(
    city,
    config,
    built,
    component_df,
    grid,
):
    """
    Select the data-derived urban extent for a city.

    SIMPLE SELECTION RULE
    ---------------------

    1. Calculate the distance between the configured city centre
       and every connected built-up component.

    2. Select the component closest to the city centre.

    3. If components are effectively tied in distance, select
       the component with the larger number of built-up cells.

    4. After selecting the main component, include every other
       built-up component whose geometry is within 1 km of the
       selected component.

    5. The final city extent is the union of all selected
       components.

    IMPORTANT
    ---------

    The city centre is ONLY used to identify the primary
    component.

    The 1 km rule is measured BETWEEN COMPONENTS, not as a
    1 km radius around the city centre.

    No fixed city-radius is used.
    No administrative boundary is used.
    """

    # ============================================================
    # 1. BASIC VALIDATION
    # ============================================================

    if component_df.empty:
        raise RuntimeError(
            f"{city}: no connected components found."
        )

    if built.empty:
        raise RuntimeError(
            f"{city}: built-up grid is empty."
        )

    # ============================================================
    # 2. CITY CENTRE
    # ============================================================

    center = gpd.GeoSeries(
        [
            Point(
                config["lon"],
                config["lat"]
            )
        ],
        crs="EPSG:4326",
    ).to_crs(
        grid.crs
    ).iloc[0]

    print("\n" + "=" * 80)
    print("CITY EXTENT SELECTION")
    print("=" * 80)

    print(f"City: {city}")

    print("\nConfigured city centre:")
    print(f"  Latitude  : {config['lat']:.6f}")
    print(f"  Longitude : {config['lon']:.6f}")

    # ============================================================
    # 3. CREATE GEOMETRY FOR EVERY COMPONENT
    # ============================================================

    component_geometries = (
        built
        .dissolve(
            by="component_id"
        )
    )

    # Make sure component_id is available as a column
    component_geometries = (
        component_geometries
        .reset_index()
    )

    # ============================================================
    # 4. COMPONENT CENTROIDS
    # ============================================================

    component_geometries[
        "component_centroid"
    ] = (
        component_geometries.geometry.centroid
    )

    # ============================================================
    # 5. DISTANCE FROM CITY CENTRE
    # ============================================================

    component_geometries[
        "distance_to_center_km"
    ] = (
        component_geometries[
            "component_centroid"
        ]
        .distance(center)
        / 1000
    )

    # ============================================================
    # 6. MERGE COMPONENT STATISTICS
    # ============================================================

    stats_columns = [
        "component_id",
        "cell_count",
        "component_area_km2",
        "mean_building_coverage",
        "median_building_coverage",
    ]

    available_columns = [
        c
        for c in stats_columns
        if c in component_df.columns
    ]

    components = component_geometries.merge(
        component_df[
            available_columns
        ],
        on="component_id",
        how="left",
    )

    # ============================================================
    # 7. PRINT ALL COMPONENTS
    # ============================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ALL BUILT-UP COMPONENTS — "
        "DISTANCE FROM CITY CENTRE"
    )

    print(
        "=" * 80
    )

    print(
        components[
            [
                "component_id",
                "cell_count",
                "component_area_km2",
                "mean_building_coverage",
                "median_building_coverage",
                "distance_to_center_km",
            ]
        ]
        .sort_values(
            [
                "distance_to_center_km",
                "cell_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )

    # ============================================================
    # 8. SELECT PRIMARY COMPONENT
    # ============================================================
    #
    # PRIMARY RULE:
    #
    #   Minimum distance from city centre.
    #
    # TIE-BREAKER:
    #
    #   Larger number of cells.
    #
    # ============================================================

    primary_component = (
        components
        .sort_values(
            [
                "distance_to_center_km",
                "cell_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .iloc[0]
    )

    selected_id = int(
        primary_component[
            "component_id"
        ]
    )

    primary_distance = float(
        primary_component[
            "distance_to_center_km"
        ]
    )

    primary_cells = int(
        primary_component[
            "cell_count"
        ]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PRIMARY COMPONENT SELECTED"
    )

    print(
        "=" * 80
    )

    print(
        f"Component ID       : {selected_id}"
    )

    print(
        f"Distance to centre : "
        f"{primary_distance:.3f} km"
    )

    print(
        f"Built-up cells     : "
        f"{primary_cells}"
    )

    print(
        "\nSelection rule:"
    )

    print(
        "1. Minimum distance to city centre"
    )

    print(
        "2. Larger cell count used as tie-breaker"
    )

    # ============================================================
    # 9. FIND COMPONENTS WITHIN 1 KM OF PRIMARY COMPONENT
    # ============================================================
    #
    # IMPORTANT:
    #
    # This is NOT distance from the city centre.
    #
    # It is the shortest geometric distance between the
    # selected component and every other component.
    #
    # ============================================================

    primary_geometry = (
        components.loc[
            components["component_id"]
            == selected_id,
            "geometry",
        ]
        .iloc[0]
    )

    components[
        "distance_to_selected_component_km"
    ] = (
        components.geometry
        .distance(primary_geometry)
        / 1000
    )

    # ============================================================
    # 10. SELECT COMPONENTS WITHIN 1 KM
    # ============================================================

    COMBINE_DISTANCE_KM = 1.0

    selected_components = components[
        components[
            "distance_to_selected_component_km"
        ]
        <= COMBINE_DISTANCE_KM
    ].copy()

    # Make absolutely sure primary component is included
    if selected_id not in set(
        selected_components["component_id"]
    ):
        selected_components = pd.concat(
            [
                selected_components,
                components[
                    components["component_id"]
                    == selected_id
                ],
            ],
            ignore_index=True,
        )

    selected_component_ids = (
        selected_components[
            "component_id"
        ]
        .astype(int)
        .tolist()
    )

    # ============================================================
    # 11. PRINT COMBINATION RESULT
    # ============================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "COMPONENTS WITHIN 1 KM OF PRIMARY COMPONENT"
    )

    print(
        "=" * 80
    )

    print(
        selected_components[
            [
                "component_id",
                "cell_count",
                "component_area_km2",
                "distance_to_center_km",
                "distance_to_selected_component_km",
                "mean_building_coverage",
            ]
        ]
        .sort_values(
            "distance_to_selected_component_km"
        )
        .to_string(
            index=False
        )
    )

    # ============================================================
    # 12. SELECT ALL CELLS FROM SELECTED COMPONENTS
    # ============================================================

    extent_cells = built[
        built["component_id"].isin(
            selected_component_ids
        )
    ].copy()

    if extent_cells.empty:
        raise RuntimeError(
            f"{city}: selected components contain no cells."
        )

    # ============================================================
    # 13. FINAL EXTENT GEOMETRY
    # ============================================================

    extent_geometry = unary_union(
        extent_cells.geometry
    )

    if extent_geometry.is_empty:
        raise RuntimeError(
            f"{city}: final extent geometry is empty."
        )

    # ============================================================
    # 14. FINAL EXTENT STATISTICS
    # ============================================================

    extent_centroid = (
        extent_geometry.centroid
    )

    distance_to_city_center_km = (
        extent_centroid.distance(center)
        / 1000
    )

    extent_area_km2 = (
        extent_cells[
            "cell_area_m2"
        ].sum()
        / 1e6
    )

    extent_cell_count = len(
        extent_cells
    )

    # ============================================================
    # 15. SELECTION METHOD
    # ============================================================

    selection_method = (
        "nearest city-centre component "
        "+ components within 1 km"
    )

    # ============================================================
    # 16. CREATE FINAL EXTENT DATASET
    # ============================================================

    extent = gpd.GeoDataFrame(
        [
            {
                "city":
                    city,

                "component_id":
                    selected_id,

                "selected_component_ids":
                    ",".join(
                        map(
                            str,
                            selected_component_ids
                        )
                    ),

                "selection_method":
                    selection_method,

                "combine_distance_km":
                    COMBINE_DISTANCE_KM,

                "extent_cells":
                    extent_cell_count,

                "extent_area_km2":
                    extent_area_km2,

                "distance_to_city_center_km":
                    float(
                        distance_to_city_center_km
                    ),

                "geometry":
                    extent_geometry,
            }
        ],
        crs=grid.crs,
    )

    # ============================================================
    # 17. CANDIDATE TABLE
    # ============================================================

    candidates = components.copy()

    candidates[
        "selected"
    ] = (
        candidates[
            "component_id"
        ]
        .isin(
            selected_component_ids
        )
    )

    # ============================================================
    # 18. FINAL OUTPUT
    # ============================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL DATA-DERIVED CITY EXTENT"
    )

    print(
        "=" * 80
    )

    print(
        f"City                         : {city}"
    )

    print(
        f"Primary component            : {selected_id}"
    )

    print(
        f"Primary component distance   : "
        f"{primary_distance:.3f} km"
    )

    print(
        f"Primary component cells      : "
        f"{primary_cells}"
    )

    print(
        f"Components combined          : "
        f"{len(selected_component_ids)}"
    )

    print(
        f"Selected component IDs       : "
        f"{selected_component_ids}"
    )

    print(
        f"Combination distance        : "
        f"{COMBINE_DISTANCE_KM:.1f} km"
    )

    print(
        f"Final extent cells           : "
        f"{extent_cell_count:,}"
    )

    print(
        f"Final extent area            : "
        f"{extent_area_km2:.2f} km²"
    )

    print(
        f"Final centroid distance      : "
        f"{distance_to_city_center_km:.2f} km"
    )

    print(
        "Fixed radius used            : NO"
    )

    print(
        "Administrative boundary used : NO"
    )

    print(
        "Complete selected components: YES"
    )

    print(
        "=" * 80
    )

    return (
        extent,
        extent_cells,
        candidates,
    )


# ============================================================
# 15. DERIVE SECONDARY URBAN CORE
# ============================================================

def derive_urban_core(
    city,
    extent_cells,
    grid_crs,
):
    """
    Derive a denser secondary urban core.

    IMPORTANT:

        city_extent remains the PRIMARY study extent.

        urban_core is secondary.
    """

    if extent_cells.empty:

        raise RuntimeError(
            f"{city}: cannot derive core "
            "from empty extent."
        )

    core_threshold = (
        extent_cells[
            "building_coverage"
        ].quantile(
            0.75
        )
    )

    core_cells = extent_cells[
        extent_cells[
            "building_coverage"
        ]
        >= core_threshold
    ].copy()

    if core_cells.empty:

        core_cells = (
            extent_cells.copy()
        )

    core_geometry = unary_union(
        core_cells.geometry
    )

    component_id = int(
        extent_cells[
            "component_id"
        ].iloc[0]
    )

    core = gpd.GeoDataFrame(
        [
            {
                "city":
                    city,

                "component_id":
                    component_id,

                "core_cells":
                    len(core_cells),

                "core_area_km2":
                    (
                        core_cells[
                            "cell_area_m2"
                        ].sum()
                        / 1e6
                    ),

                "core_threshold":
                    float(
                        core_threshold
                    ),

                "mean_building_coverage":
                    float(
                        core_cells[
                            "building_coverage"
                        ].mean()
                    ),

                "median_building_coverage":
                    float(
                        core_cells[
                            "building_coverage"
                        ].median()
                    ),

                "geometry":
                    core_geometry,
            }
        ],
        crs=grid_crs,
    )

    print(
        "\nSECONDARY URBAN CORE"
    )

    print(
        f"Core cells: "
        f"{len(core_cells):,}"
    )

    print(
        f"Core area: "
        f"{core_cells['cell_area_m2'].sum() / 1e6:.2f} km²"
    )

    print(
        f"Core coverage threshold: "
        f"{core_threshold:.4f}"
    )

    print(
        f"Core mean building coverage: "
        f"{core_cells['building_coverage'].mean():.4f}"
    )

    print(
        f"Core median building coverage: "
        f"{core_cells['building_coverage'].median():.4f}"
    )

    return (
        core_cells,
        core,
    )


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

def save_outputs(
    city,
    paths,
    grid,
    built,
    extent,
    core_cells,
    core,
    component_df,
):
    """
    Save all outputs.
    """

    print(
        "\nSaving outputs..."
    )

    # --------------------------------------------------------
    # Complete grid
    # --------------------------------------------------------

    grid.to_file(
        paths["grid_geojson"],
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # Built-up cells
    # --------------------------------------------------------

    built.to_file(
        paths["built_cells_geojson"],
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # Primary city extent
    # --------------------------------------------------------

    extent.to_file(
        paths["city_extent_geojson"],
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # Secondary urban core
    # --------------------------------------------------------

    core.to_file(
        paths["core_geojson"],
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # Component table
    # --------------------------------------------------------

    component_df.to_csv(
        paths["components_csv"],
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_grid_cells = (
        len(grid)
    )

    grid_area_km2 = (
        grid.geometry.area.sum()
        / 1e6
    )

    city_extent_area_km2 = float(
        extent[
            "extent_area_km2"
        ].iloc[0]
    )

    built_up_cells = (
        len(built)
    )

    city_extent_cells = int(
       extent["extent_cells"].iloc[0]
    )

    selected_core_cells = (
        len(core_cells)
    )

    core_area_km2 = (
        core_cells[
            "cell_area_m2"
        ].sum()
        / 1e6
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "city":
            city,

        "method":
            "data-derived built-up urban extent",

        "fixed_radius_used":
            False,

        "administrative_boundary_used_as_final_extent":
            False,

        "primary_rnqi_extent":
            "city_extent",

        "building_extraction_extent":
            "city-specific road-data extraction envelope",

        "final_city_extent":
            "target-city connected built-up component",

        "urban_core":
            "upper-25-percent building-density subset "
            "inside city_extent",

        "grid_size_m":
            GRID_SIZE_M,

        "builtup_cell_coverage_threshold":
            BUILTUP_CELL_COVERAGE_THRESHOLD,

        "grid_area_km2":
            grid_area_km2,

        "total_grid_cells":
            int(
                total_grid_cells
            ),

        "built_up_cells":
            int(
                built_up_cells
            ),

        "built_up_cell_percentage":
            float(
                built_up_cells
                / total_grid_cells
                * 100
            ),

        "connected_components":
            int(
                len(component_df)
            ),

        "components_ge_min_size":
            int(
                (
                    component_df[
                        "cell_count"
                    ]
                    >= MIN_COMPONENT_CELLS_DIAGNOSTIC
                ).sum()
            ),

        "selected_city_extent_cells":
            int(
                extent[
                    "extent_cells"
                ].iloc[0]
            ),

        "selected_city_extent_area_km2":
            city_extent_area_km2,

        "selected_city_extent_centroid_distance_km":
            float(
                extent[
                    "distance_to_city_center_km"
                ].iloc[0]
            ),

        "selected_core_cells":
            int(
                selected_core_cells
            ),

        "selected_core_cell_percentage_of_total_grid":
            float(
                selected_core_cells
                / total_grid_cells
                * 100
            ),

        "selected_core_area_km2":
            float(
                core_area_km2
            ),

        "selected_core_mean_building_coverage":
            float(
                core_cells[
                    "building_coverage"
                ].mean()
            ),

        "selected_core_median_building_coverage":
            float(
                core_cells[
                    "building_coverage"
                ].median()
            ),

        "use_8_neighbor_connectivity":
            USE_8_NEIGHBOR_CONNECTIVITY,
    }

    with open(
        paths["report_json"],
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    print(
        "\nSaved:"
    )

    print(
        f"  Grid: "
        f"{paths['grid_geojson']}"
    )

    print(
        f"  Built cells: "
        f"{paths['built_cells_geojson']}"
    )

    print(
        f"  City extent: "
        f"{paths['city_extent_geojson']}"
    )

    print(
        f"  Urban core: "
        f"{paths['core_geojson']}"
    )

    print(
        f"  Components: "
        f"{paths['components_csv']}"
    )

    print(
        f"  Report: "
        f"{paths['report_json']}"
    )

    return report


# ============================================================
# 17. PROCESS ONE CITY
# ============================================================

def process_city(
    city,
    config,
):
    """
    Complete processing pipeline for one city.
    """

    print(
        "\n\n"
        + "#" * 80
    )

    print(
        f"# PROCESSING {city.upper()}"
    )

    print(
        "# BUILDING EXTRACTION = "
        "CITY-SPECIFIC ROAD-DATA ENVELOPE"
    )

    print(
        "# PRIMARY STUDY EXTENT = "
        "DATA-DERIVED BUILT-UP CITY EXTENT"
    )

    print(
        "# SECONDARY LAYER = "
        "DENSER URBAN CORE"
    )

    print(
        "#" * 80
    )

    paths = get_city_paths(
        city
    )

    # --------------------------------------------------------
    # Check regional PBF
    # --------------------------------------------------------

    region_pbf = config[
        "region_pbf"
    ]

    if not region_pbf.exists():

        raise FileNotFoundError(
            f"Regional Geofabrik PBF does not exist:\n"
            f"{region_pbf}"
        )

    # --------------------------------------------------------
    # Check osmium
    # --------------------------------------------------------

    osmium_path = shutil.which(
        "osmium"
    )

    if osmium_path is None:

        raise FileNotFoundError(
            "osmium-tool is required.\n"
            "Install with:\n"
            "brew install osmium-tool"
        )

    # --------------------------------------------------------
    # City-specific road OSM
    # --------------------------------------------------------

    road_osm_path = (
        ROAD_OSM_DIR
        / f"{city.lower()}.osm"
    )

    # --------------------------------------------------------
    # Create extraction envelope
    # --------------------------------------------------------

    candidate_polygon = (
        create_building_extraction_envelope_from_road_osm(
            city,
            road_osm_path,
            config,
        )
    )

    # --------------------------------------------------------
    # BUILDING DATASET
    # --------------------------------------------------------

    if (
        paths["buildings_geojson"].exists()
        and not REBUILD_WHOLE_CITY_BUILDINGS
    ):

        print(
            "\nExisting building dataset found."
        )

        print(
            paths["buildings_geojson"]
        )

        cached_buildings = (
            gpd.read_file(
                paths["buildings_geojson"]
            )
        )

        cached_buildings = (
            validate_city_building_dataset(
                city,
                cached_buildings,
                config,
            )
        )

        print(
            "Cached building dataset passed "
            "geographic validation."
        )

    else:

        print(
            "\nExtracting building dataset "
            "from regional Geofabrik PBF..."
        )

        extract_whole_city_buildings(
            osmium_path,
            city,
            region_pbf,
            candidate_polygon,
            paths,
        )

    # --------------------------------------------------------
    # Load buildings
    # --------------------------------------------------------

    buildings = load_buildings(
        city,
        paths,
    )

    # --------------------------------------------------------
    # Create complete grid
    # --------------------------------------------------------

    grid = create_grid(
        buildings
    )

    # --------------------------------------------------------
    # Calculate coverage
    # --------------------------------------------------------

    grid = calculate_building_coverage(
        grid,
        buildings,
        BUILTUP_CELL_COVERAGE_THRESHOLD,
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    (
        built,
        component_df,
    ) = find_connected_components(
        grid
    )

    # --------------------------------------------------------
    # Select city extent
    # --------------------------------------------------------

    (
        extent,
        extent_cells,
        candidates,
    ) = select_city_extent(
        city,
        config,
        built,
        component_df,
        grid,
    )

    # --------------------------------------------------------
    # Derive secondary urban core
    # --------------------------------------------------------

    (
        core_cells,
        core,
    ) = derive_urban_core(
        city,
        extent_cells,
        grid.crs,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    report = save_outputs(
        city,
        paths,
        grid,
        built,
        extent,
        core_cells,
        core,
        candidates,
    )

    # --------------------------------------------------------
    # Final concise report
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"{city} COMPLETED"
    )

    print(
        "-" * 70
    )

    for key, value in report.items():

        print(
            f"{key}: {value}"
        )

    return report


# ============================================================
# 18. MAIN
# ============================================================

def main(
    selected_cities=None,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DATA-DERIVED WHOLE URBAN CITY EXTENT EXTRACTION"
    )

    print(
        "=" * 80
    )

    print(
        "\nMethod:"
    )

    print(
        "1. City-specific road OSM "
        "-> building extraction envelope"
    )

    print(
        "2. OSM building footprints"
    )

    print(
        "3. Complete 500m grid"
    )

    print(
        "4. Building coverage"
    )

    print(
        "5. Built-up cells"
    )

    print(
        "6. Connected components"
    )

    print(
        "7. Target-city component"
    )

    print(
        "8. Final city_extent"
    )

    print(
        "9. Secondary urban_core"
    )

    print(
        "\nNo fixed study-area radius is used."
    )

    print(
        "No administrative/municipal boundary "
        "is used as the final extent."
    )

    print(
        "City centre is used only as a target-city "
        "seed/validation point."
    )

    print(
        f"\nGRID_SIZE_M = "
        f"{GRID_SIZE_M}"
    )

    print(
        f"BUILTUP_CELL_COVERAGE_THRESHOLD = "
        f"{BUILTUP_CELL_COVERAGE_THRESHOLD}"
    )

    print(
        f"MIN_COMPONENT_CELLS_DIAGNOSTIC = "
        f"{MIN_COMPONENT_CELLS_DIAGNOSTIC}"
    )

    if selected_cities:

        cities_to_process = {
            city: CITIES[city]
            for city in selected_cities
        }

    else:

        cities_to_process = CITIES

    results = []

    for city, config in (
        cities_to_process.items()
    ):

        try:

            result = process_city(
                city,
                config,
            )

            results.append(
                result
            )

        except Exception as e:

            print(
                "\n"
                + "!" * 80
            )

            print(
                f"FAILED: {city}"
            )

            print(
                f"ERROR: "
                f"{type(e).__name__}: {e}"
            )

            print(
                "!" * 80
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    if results:

        summary_df = pd.DataFrame(
            results
        )

        summary_file = (
            REPORT_DIR
            / "urban_extent_summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False,
        )

        print(
            "\n\n"
            + "=" * 80
        )

        print(
            "FINAL SUMMARY - "
            "DATA-DERIVED WHOLE CITY EXTENTS"
        )

        print(
            "=" * 80
        )

        print(
            summary_df.to_string(
                index=False
            )
        )

        print(
            "\nSummary saved to:"
        )

        print(
            summary_file
        )

    else:

        raise RuntimeError(
            "No city results were produced."
        )


# ============================================================
# 19. COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Build data-derived whole-city urban "
            "extents from OSM building footprints "
            "and regional Geofabrik PBF data."
        )
    )

    parser.add_argument(
        "--city",
        nargs="+",
        choices=list(
            CITIES.keys()
        ),
        help=(
            "Process only the selected city "
            "or cities."
        ),
    )

    args = parser.parse_args()

    main(
        selected_cities=args.city
    )