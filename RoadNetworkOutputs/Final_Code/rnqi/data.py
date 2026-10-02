"""Reusable data-preparation functions from the RNQI data stage."""

from ._legacy import load_stage_module

_stage = load_stage_module("02_rnqi_data")

# Reuse the exact production implementations; do not duplicate them here.
CITY_OSM_FILES = _stage.CITY_OSM_FILES
CITIES = _stage.CITIES
get_city_boundary = _stage.get_city_boundary
download_road_network = _stage.download_road_network
preprocess_graph = _stage.preprocess_graph
project_graph = _stage.project_graph
load_existing_graph = _stage.load_existing_graph


def prepare_graph_from_local_osm(city_name, boundary):
    """Prepare a projected graph using the same data-stage operations as production.

    This calls the existing production functions in their production order:
    local OSM XML -> original city boundary -> preprocessing -> projection.
    """
    graph = download_road_network(city_name, boundary)
    graph = preprocess_graph(graph, city_name)
    graph = project_graph(graph)
    return graph
