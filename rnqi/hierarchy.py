"""Reusable Hierarchy pillar functions from the production stage."""

from ._legacy import load_stage_module

_stage = load_stage_module("05_rnqi_hierarchy")

CITIES = _stage.CITIES
RANDOM_SEED = _stage.RANDOM_SEED
ROAD_CLASS_ORDER = _stage.ROAD_CLASS_ORDER
ROAD_CLASS_ALIASES = _stage.ROAD_CLASS_ALIASES

load_graph = _stage.load_graph
highway_class = _stage.highway_class
build_simple_graph = _stage.build_simple_graph
choose_source_count = _stage.choose_source_count
calculate_betweenness_metrics = _stage.calculate_betweenness_metrics
calculate_road_class_distribution = _stage.calculate_road_class_distribution
calculate_hierarchy_clarity = _stage.calculate_hierarchy_clarity
minmax_normalize = _stage.minmax_normalize
