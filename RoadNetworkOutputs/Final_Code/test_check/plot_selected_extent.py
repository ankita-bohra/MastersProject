import os
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd


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


OUTPUT_DIR = (
    "outputs/urban_extent_validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


for city, coords in CITY_CENTRES.items():

    print(
        f"\nPlotting {city}..."
    )

    path = (
        f"outputs/urban_core/grid/"
        f"{city}_built_cells.geojson"
    )

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    built = gpd.read_file(path)

    

    extent_path = (
        f"outputs/boundaries/"
        f"{city}_city_extent.geojson"
    )

    if not os.path.exists(extent_path):
        print("Missing extent:", extent_path)
        continue

    extent_gdf = gpd.read_file(extent_path)
    selected_ids_value = extent_gdf.iloc[0].get(
        "selected_component_ids"
    )

    if selected_ids_value is None or pd.isna(selected_ids_value):
        print("No selected component IDs in:", extent_path)
        continue

    selected_ids = [
        int(value.strip())
        for value in str(selected_ids_value).split(",")
        if value.strip()
    ]

    selected = built[
        built["component_id"].isin(selected_ids)
    ].copy()

    if selected.empty:
        print("No built cells match selected components:", city)
        continue

    # --------------------------------------------------------
    # City centre
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
    )

    # --------------------------------------------------------
    # Selected extent
    # --------------------------------------------------------

    extent = selected.geometry.union_all()

    centroid = gpd.GeoSeries(
        [extent.centroid],
        crs=built.crs
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    # All built cells
    built.plot(
        ax=ax,
        alpha=0.25,
        edgecolor="grey",
        linewidth=0.2,
    )

    # Selected cells
    selected.plot(
        ax=ax,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.4,
    )

    # City centre
    centre.plot(
        ax=ax,
        marker="*",
        markersize=180,
        label="Configured city centre",
    )

    # Extent centroid
    centroid.plot(
        ax=ax,
        marker="x",
        markersize=120,
        linewidth=3,
        label="Selected extent centroid",
    )

    ax.set_title(
        f"{city.title()} — Selected Urban Extent"
    )

    ax.legend()

    ax.set_axis_off()

    output = os.path.join(
        OUTPUT_DIR,
        f"{city}_selected_extent.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )