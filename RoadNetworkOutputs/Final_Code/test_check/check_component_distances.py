import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ============================================================
# CITY CENTRES
# Use the SAME coordinates as your RNQI configuration.
# ============================================================

CITY_CENTRES = {

    "ranipet": {
        "lat": 12.9249,
        "lon": 79.3333,
    },

    "chandigarh": {
        "lat": 30.7333,
        "lon": 76.7794,
    },

    "kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
    },

    "pune": {
        "lat": 18.5204,
        "lon": 73.8567,
    },

    "jaipur": {
        "lat": 26.9124,
        "lon": 75.7873,
    },
}


for city, coords in CITY_CENTRES.items():

    print("\n" + "=" * 90)
    print(f"{city.upper()} — COMPONENT DISTANCE CHECK")
    print("=" * 90)

    components_path = (
        f"../outputs/urban_core/core/"
        f"{city}_components.csv"
    )

    built_path = (
        f"../outputs/urban_core/grid/"
        f"{city}_built_cells.geojson"
    )

    if not os.path.exists(components_path):
        print("Missing:", components_path)
        continue

    if not os.path.exists(built_path):
        print("Missing:", built_path)
        continue

    components = pd.read_csv(
        components_path
    )

    built = gpd.read_file(
        built_path
    )

    # --------------------------------------------------------
    # Centre
    # --------------------------------------------------------

    centre = gpd.GeoSeries(
        [
            Point(
                coords["lon"],
                coords["lat"]
            )
        ],
        crs="EPSG:4326"
    ).to_crs(
        built.crs
    ).iloc[0]

    # --------------------------------------------------------
    # Component geometries
    # --------------------------------------------------------

    geometries = (
        built
        .dissolve(
            by="component_id"
        )
    )

    geometries["component_centroid"] = (
        geometries.geometry.centroid
    )

    geometries["distance_km"] = (
        geometries
        ["component_centroid"]
        .distance(centre)
        / 1000.0
    )

    # --------------------------------------------------------
    # Merge statistics
    # --------------------------------------------------------

    result = components.merge(
        geometries[
            ["distance_km"]
        ],
        left_on="component_id",
        right_index=True,
        how="left"
    )

    result = result.sort_values(
        "distance_km"
    )

    print("\nNearest components:")

    print(
        result[
            [
                "component_id",
                "cell_count",
                "component_area_km2",
                "mean_building_coverage",
                "distance_km",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )