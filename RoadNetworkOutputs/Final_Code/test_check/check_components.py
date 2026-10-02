import os
import pandas as pd

CITIES = [
    "ranipet",
    "chandigarh",
    "kolkata",
    "pune",
    "jaipur",
]

BASE = "../outputs/urban_core/core"

for city in CITIES:

    path = os.path.join(
        BASE,
        f"{city}_components.csv"
    )

    print("\n" + "=" * 90)
    print(f"{city.upper()} — COMPONENT SUMMARY")
    print("=" * 90)

    if not os.path.exists(path):
        print("FILE NOT FOUND:", path)
        continue

    df = pd.read_csv(path)

    print("Total components:", len(df))

    print("\nLargest components:")
    print(
        df[
            [
                "component_id",
                "cell_count",
                "component_area_km2",
                "mean_building_coverage",
                "median_building_coverage",
                "selected",
            ]
        ]
        .sort_values(
            "cell_count",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nComponents >= 20 cells:")
    print(
        df[df["cell_count"] >= 20]
        [
            [
                "component_id",
                "cell_count",
                "component_area_km2",
                "mean_building_coverage",
                "median_building_coverage",
            ]
        ]
        .sort_values(
            "cell_count",
            ascending=False
        )
        .to_string(index=False)
    )