import os
import pandas as pd
import geopandas as gpd


CITIES = [
    "ranipet",
    "chandigarh",
    "kolkata",
    "pune",
    "jaipur",
]


for city in CITIES:

    print("\n" + "=" * 90)
    print(f"{city.upper()} — SELECTED EXTENT CHECK")
    print("=" * 90)

    path = (
        f"../outputs/urban_core/grid/"
        f"{city}_built_cells.geojson"
    )

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    built = gpd.read_file(path)

    if "selected" not in built.columns:

        print(
            "\nWARNING:"
            " 'selected' column does not exist."
        )

        print(
            "Columns:",
            built.columns.tolist()
        )

        continue

    selected = built[
        built["selected"] == True
    ].copy()

    print(
        "\nTotal built cells:",
        len(built)
    )

    print(
        "Selected cells:",
        len(selected)
    )

    if selected.empty:

        print(
            "No selected cells found."
        )

        continue

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    selected_area_km2 = (
        selected.geometry.area.sum()
        / 1_000_000
    )

    # --------------------------------------------------------
    # Extent centroid
    # --------------------------------------------------------

    extent_geometry = (
        selected.geometry
        .union_all()
    )

    centroid = (
        extent_geometry.centroid
    )

    # --------------------------------------------------------
    # Component breakdown
    # --------------------------------------------------------

    component_summary = (
        selected
        .groupby("component_id")
        .agg(
            cells=("grid_id", "count"),
            area_m2=(
                "cell_area_m2",
                "sum"
            ),
            mean_coverage=(
                "building_coverage",
                "mean"
            ),
        )
        .reset_index()
    )

    component_summary[
        "area_km2"
    ] = (
        component_summary["area_m2"]
        / 1_000_000
    )

    print(
        "\nSelected components:"
    )

    print(
        component_summary[
            [
                "component_id",
                "cells",
                "area_km2",
                "mean_coverage",
            ]
        ]
        .sort_values(
            "cells",
            ascending=False
        )
        .to_string(index=False)
    )

    print(
        "\nFinal selected extent:"
    )

    print(
        "Cells       :",
        len(selected)
    )

    print(
        "Area km²    :",
        round(
            selected_area_km2,
            3
        )
    )

    print(
        "Components  :",
        len(component_summary)
    )

    print(
        "Centroid    :",
        centroid
    )