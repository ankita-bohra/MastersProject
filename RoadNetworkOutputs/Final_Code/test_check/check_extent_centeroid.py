import os
import geopandas as gpd
from shapely.geometry import Point


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

    print("\n" + "=" * 80)
    print(city.upper())
    print("=" * 80)

    path = (
        f"../outputs/urban_core/grid/"
        f"{city}_built_cells.geojson"
    )

    built = gpd.read_file(path)

    selected = built[
        built["selected"] == True
    ].copy()

    if selected.empty:
        print("No selected cells.")
        continue

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

    extent = selected.geometry.union_all()

    centroid = extent.centroid

    distance_km = (
        centroid.distance(centre)
        / 1000
    )

    area_km2 = (
        extent.area
        / 1_000_000
    )

    print(
        "Selected cells :",
        len(selected)
    )

    print(
        "Selected area  :",
        round(area_km2, 3),
        "km²"
    )

    print(
        "Centroid dist. :",
        round(distance_km, 3),
        "km"
    )