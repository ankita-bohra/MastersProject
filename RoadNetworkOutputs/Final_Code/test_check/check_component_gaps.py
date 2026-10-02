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

MAX_GAP_KM = 1.0


for city in CITIES:

    print("\n" + "=" * 90)
    print(f"{city.upper()} — COMPONENT GAP CHECK")
    print("=" * 90)

    path = (
        f"../outputs/urban_core/grid/"
        f"{city}_built_cells.geojson"
    )

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    built = gpd.read_file(path)

    # --------------------------------------------------------
    # Dissolve each component
    # --------------------------------------------------------

    components = (
        built
        .dissolve(
            by="component_id"
        )
        .reset_index()
    )

    print(
        "Components:",
        len(components)
    )

    # --------------------------------------------------------
    # Calculate pairwise gaps
    # --------------------------------------------------------

    rows = []

    for i in range(len(components)):

        geom_a = components.iloc[i].geometry
        id_a = components.iloc[i]["component_id"]

        for j in range(i + 1, len(components)):

            geom_b = components.iloc[j].geometry
            id_b = components.iloc[j]["component_id"]

            gap_km = (
                geom_a.distance(geom_b)
                / 1000.0
            )

            if gap_km <= MAX_GAP_KM:

                rows.append(
                    {
                        "component_a": id_a,
                        "component_b": id_b,
                        "gap_km": gap_km,
                    }
                )

    gaps = pd.DataFrame(rows)

    if gaps.empty:

        print(
            "\nNo component pairs within",
            MAX_GAP_KM,
            "km."
        )

        continue

    gaps = gaps.sort_values(
        "gap_km"
    )

    print(
        f"\nComponent pairs with gap <= "
        f"{MAX_GAP_KM} km:"
    )

    print(
        gaps.to_string(
            index=False
        )
    )