"""Reusable Efficiency pillar functions from the production stage."""

from ._legacy import load_stage_module

_stage = load_stage_module("04_rnqi_efficiency")

calculate_city_area = _stage.calculate_city_area
prepare_undirected_graph = _stage.prepare_undirected_graph
calculate_average_edge_circuity = _stage.calculate_average_edge_circuity
calculate_gamma_connectivity = _stage.calculate_gamma_connectivity
calculate_efficiency_adaptive = _stage.calculate_efficiency_adaptive
calculate_efficiency_scores = _stage.calculate_efficiency_scores
min_max = _stage.min_max

EFFICIENCY_RANDOM_SEED = _stage.EFFICIENCY_RANDOM_SEED
