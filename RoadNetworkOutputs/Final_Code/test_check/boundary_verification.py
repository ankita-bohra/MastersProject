from pathlib import Path
import time

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.validation import make_valid


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "boundary_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Add a real contact address to identify this script's requests.
HEADERS = {
    "User-Agent": "MastersProject-RNQI-boundary-diagnostic/1.0 (contact: your-email@example.com)"
}

CANDIDATES = {
    "Pune": [1986140],
    "Chandigarh": [1942809, 12773165, 9552697],
    "Kolkata": [9381363, 10371838],
    "Jaipur": [1950062],
    "Ranipet": [10330032],
}

CSV_COLUMNS = [
    "city",
    "osm_id",
    "name",
    "official_name",
    "short_name",
    "boundary",
    "admin_level",
    "type",
    "place",
    "government",
    "wikidata",
    "wikipedia",
    "ref",
    "geometry_type",
    "area_km2",
    "classification",
    # Helpful audit fields in addition to the requested columns.
    "bbox_wgs84",
    "classification_reason",
    "geometry_file",
    "error",
]


def overpass_relation(session, relation_id):
    """Fetch one relation's tags and member geometry from Overpass."""
    query = (
        "[out:json][timeout:180];\n"
        f"rel({int(relation_id)});\n"
        "out tags geom;"
    )

    errors = []
    headers = {
        **HEADERS,
        "Accept": "application/json",
    }

    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(1, 4):
            try:
                response = session.post(
                    endpoint,
                    data={"data": query},
                    headers=headers,
                    timeout=240,
                )

                if not response.ok:
                    errors.append(
                        f"{endpoint}, attempt {attempt}, "
                        f"HTTP {response.status_code}: "
                        f"{response.text[:1000]}"
                    )
                    time.sleep(5 * attempt)
                    continue

                payload = response.json()

                for element in payload.get("elements", []):
                    if (
                        element.get("type") == "relation"
                        and int(element.get("id", -1)) == int(relation_id)
                    ):
                        return element

                errors.append(
                    f"{endpoint}, attempt {attempt}: "
                    f"response contained no relation {relation_id}"
                )

            except Exception as error:
                errors.append(
                    f"{endpoint}, attempt {attempt}: "
                    f"{type(error).__name__}: {error}"
                )

            time.sleep(5 * attempt)

    raise RuntimeError(" | ".join(errors))


def coord_key(point):
    """Stable endpoint key for joining member-way segments into rings."""
    return round(point[0], 7), round(point[1], 7)


def stitch_rings(segments):
    """
    Join Overpass relation-member way geometries into closed rings.

    Overpass returns relation member ways individually. Their order and
    direction are not guaranteed, so join matching endpoints before building
    Polygon geometry.
    """
    remaining = [
        list(segment)
        for segment in segments
        if len(segment) >= 2
    ]
    rings = []

    while remaining:
        ring = remaining.pop()

        while coord_key(ring[0]) != coord_key(ring[-1]):
            joined = False
            ring_start = coord_key(ring[0])
            ring_end = coord_key(ring[-1])

            for index, segment in enumerate(remaining):
                seg_start = coord_key(segment[0])
                seg_end = coord_key(segment[-1])

                if ring_end == seg_start:
                    ring.extend(segment[1:])
                elif ring_end == seg_end:
                    ring.extend(reversed(segment[:-1]))
                elif ring_start == seg_end:
                    ring = segment[:-1] + ring
                elif ring_start == seg_start:
                    ring = list(reversed(segment[1:])) + ring
                else:
                    continue

                remaining.pop(index)
                joined = True
                break

            if not joined:
                raise ValueError(
                    "Could not join all relation member ways into closed rings."
                )

        if len(ring) < 4:
            raise ValueError("Closed ring has fewer than four coordinates.")

        rings.append(ring)

    return rings


def relation_geometry(element):
    """Build Polygon/MultiPolygon geometry from Overpass relation members."""
    # Support a direct geometry array if an endpoint supplies one.
    direct_geometry = element.get("geometry")
    if direct_geometry:
        coords = [
            (float(point["lon"]), float(point["lat"]))
            for point in direct_geometry
            if "lon" in point and "lat" in point
        ]
        if len(coords) >= 4 and coord_key(coords[0]) == coord_key(coords[-1]):
            return Polygon(coords)

    outer_segments = []
    inner_segments = []

    for member in element.get("members", []):
        if member.get("type") != "way":
            continue

        segment = [
            (float(point["lon"]), float(point["lat"]))
            for point in member.get("geometry", [])
            if "lon" in point and "lat" in point
        ]
        if len(segment) < 2:
            continue

        role = (member.get("role") or "").lower()
        if role == "inner":
            inner_segments.append(segment)
        else:
            # Empty and "outer" roles are treated as outer rings.
            outer_segments.append(segment)

    if not outer_segments:
        raise ValueError("Relation response contained no outer way geometry.")

    outer_rings = stitch_rings(outer_segments)
    inner_rings = stitch_rings(inner_segments) if inner_segments else []

    outer_polygons = [Polygon(ring) for ring in outer_rings]
    holes_by_outer = [[] for _ in outer_polygons]

    for inner in inner_rings:
        inner_polygon = Polygon(inner)
        representative = inner_polygon.representative_point()

        containing = [
            index
            for index, outer in enumerate(outer_polygons)
            if outer.covers(representative)
        ]
        if not containing:
            raise ValueError("An inner ring could not be assigned to an outer ring.")

        holes_by_outer[containing[0]].append(inner)

    polygons = [
        Polygon(outer.exterior.coords, holes_by_outer[index])
        for index, outer in enumerate(outer_polygons)
    ]

    geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    return make_valid(geometry)


def classify_indian_relation(tags):
    """
    Apply the documented India OSM hierarchy, using tags rather than name/area.

    An admin_level=8 alone is insufficient for MUNICIPAL_OR_CITY_BOUNDARY:
    corroborating government or place tagging is also required.
    """
    if tags.get("boundary", "").lower() != "administrative":
        return "UNKNOWN", "Relation is not tagged boundary=administrative."

    try:
        level = int(tags.get("admin_level", ""))
    except (TypeError, ValueError):
        return "UNKNOWN", "Missing or non-numeric admin_level."

    if level == 4:
        return "STATE_OR_UT", "India OSM hierarchy: admin_level=4 is State/UT."
    if level == 5:
        return "DISTRICT", "India OSM hierarchy: admin_level=5 is District."
    if level == 6:
        return "SUBDISTRICT", "India OSM hierarchy: admin_level=6 is Subdistrict."
    if level == 7:
        return "METROPOLITAN_AREA", (
            "India OSM hierarchy: admin_level=7 is Metropolitan Area."
        )
    if level == 8:
        government = (tags.get("government") or "").strip().lower()
        place = (tags.get("place") or "").strip().lower()

        municipal_government_values = {
            "municipality",
            "municipal corporation",
            "municipal_corporation",
            "city council",
            "city_council",
            "municipal council",
            "municipal_council",
        }
        municipal_place_values = {
            "city",
            "town",
            "municipality",
        }

        if (
            government in municipal_government_values
            or place in municipal_place_values
        ):
            return (
                "MUNICIPAL_OR_CITY_BOUNDARY",
                "India OSM hierarchy: admin_level=8, corroborated by government/place tag.",
            )

        return (
            "UNKNOWN",
            "admin_level=8 is the Indian municipal/city level, but this relation lacks "
            "corroborating government or place tags; inspect relation details.",
        )

    return (
        "OTHER_ADMINISTRATIVE",
        f"Administrative relation uses India OSM admin_level={level}, outside levels 4–8.",
    )


def geometry_area_and_bbox(geometry):
    """Return projected area and geographic bounds for a relation geometry."""
    gdf = gpd.GeoDataFrame(
        [{"geometry": geometry}],
        crs="EPSG:4326",
    )

    metric_crs = gdf.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("Could not estimate a local projected CRS.")

    area_km2 = gdf.to_crs(metric_crs).geometry.area.iloc[0] / 1_000_000
    west, south, east, north = geometry.bounds
    bbox = f"{west:.8f},{south:.8f},{east:.8f},{north:.8f}"

    return float(area_km2), bbox


def save_relation_geojson(city, relation_id, tags, geometry):
    """Save one candidate polygon without selecting it as the final boundary."""
    output_path = OUTPUT_DIR / (
        f"{city.lower()}_relation_{relation_id}.geojson"
    )

    properties = {
        "city": city,
        "osm_id": int(relation_id),
        **{
            key: tags.get(key)
            for key in [
                "name",
                "official_name",
                "short_name",
                "boundary",
                "admin_level",
                "type",
                "place",
                "government",
                "wikidata",
                "wikipedia",
                "ref",
            ]
        },
    }

    gdf = gpd.GeoDataFrame(
        [properties],
        geometry=[geometry],
        crs="EPSG:4326",
    )
    gdf.to_file(output_path, driver="GeoJSON")
    return output_path


def main():
    session = requests.Session()
    rows = []

    for city, relation_ids in CANDIDATES.items():
        for relation_id in relation_ids:
            print(f"\nQuerying {city}, relation {relation_id}")

            row = {
                "city": city,
                "osm_id": relation_id,
                "name": None,
                "official_name": None,
                "short_name": None,
                "boundary": None,
                "admin_level": None,
                "type": None,
                "place": None,
                "government": None,
                "wikidata": None,
                "wikipedia": None,
                "ref": None,
                "geometry_type": None,
                "area_km2": None,
                "classification": "UNKNOWN",
                "bbox_wgs84": None,
                "classification_reason": None,
                "geometry_file": None,
                "error": None,
            }

            try:
                element = overpass_relation(session, relation_id)
                tags = element.get("tags") or {}

                for key in [
                    "name",
                    "official_name",
                    "short_name",
                    "boundary",
                    "admin_level",
                    "type",
                    "place",
                    "government",
                    "wikidata",
                    "wikipedia",
                    "ref",
                ]:
                    row[key] = tags.get(key)

                row["classification"], row["classification_reason"] = (
                    classify_indian_relation(tags)
                )

                geometry = relation_geometry(element)
                row["geometry_type"] = geometry.geom_type

                if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                    raise ValueError(
                        f"Unexpected geometry type: {geometry.geom_type}"
                    )

                row["area_km2"], row["bbox_wgs84"] = (
                    geometry_area_and_bbox(geometry)
                )
                row["geometry_file"] = str(
                    save_relation_geojson(city, relation_id, tags, geometry)
                )

            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                row["classification_reason"] = (
                    "Could not classify because the relation query/geometry failed."
                )

            print(
                f"CITY: {city}\n"
                f"Candidate relation ID: {relation_id}\n"
                f"Name: {row['name']}\n"
                f"Boundary: {row['boundary']}\n"
                f"Admin level: {row['admin_level']}\n"
                f"Type: {row['type']}\n"
                f"Area: {row['area_km2']} km²\n"
                f"Classification: {row['classification']}\n"
                f"Reason: {row['classification_reason']}"
            )
            if row["error"]:
                print(f"Error: {row['error']}")

            rows.append(row)
            time.sleep(1.0)

    result = pd.DataFrame(rows, columns=CSV_COLUMNS)
    csv_path = OUTPUT_DIR / "osm_relation_details.csv"
    result.to_csv(csv_path, index=False)

    print("\nCANDIDATE SUMMARY")
    summary_columns = [
        "city",
        "osm_id",
        "admin_level",
        "boundary",
        "type",
        "area_km2",
        "classification",
    ]
    print(
        result[summary_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )
    print(f"\nFull CSV saved to: {csv_path}")
    print(f"Candidate GeoJSON files saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()