"""Reusable Connectivity pillar functions from the production stage."""

from ._legacy import load_stage_module

_stage = load_stage_module("03_rnqi_connectivity")

CITIES = _stage.CITIES
calculate_city_area = _stage.calculate_city_area
prepare_undirected_graph = _stage.prepare_undirected_graph
calculate_mean_closeness_adaptive = _stage.calculate_mean_closeness_adaptive
calculate_connectivity_metrics = _stage.calculate_connectivity_metrics
calculate_connectivity_scores = _stage.calculate_connectivity_scores
min_max_normalize = _stage.min_max_normalize

# Production constants used by calculate_mean_closeness_adaptive.
CLOSENESS_PILOT_FRACTION = _stage.CLOSENESS_PILOT_FRACTION
CLOSENESS_CONFIDENCE_LEVEL = _stage.CLOSENESS_CONFIDENCE_LEVEL
CLOSENESS_RELATIVE_PRECISION = _stage.CLOSENESS_RELATIVE_PRECISION
CLOSENESS_BATCH_SIZE = _stage.CLOSENESS_BATCH_SIZE
CLOSENESS_VALIDATION_MASTER_SEED = _stage.CLOSENESS_VALIDATION_MASTER_SEED
