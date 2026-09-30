from occupancy_model_tasks.tasks._checks import check_target_event_types
from occupancy_model_tasks.tasks._covariates import (
    label_grid_with_distance_covariates,
    label_grid_with_gee_covariates,
    set_distance_covariates,
    set_gee_covariates,
)
from occupancy_model_tasks.tasks._design import (
    build_occupancy_design,
    count_events_and_effort,
    missing_trajectories_or_grid,
)
from occupancy_model_tasks.tasks._model import (
    extract_fit_metric,
    fit_occupancy_glm,
    get_occupancy_fit_summary,
    get_posterior_draws,
    predict_occupancy_surface,
    summarize_occupancy_run,
)
from occupancy_model_tasks.tasks._plot import draw_coefficient_plot
from occupancy_model_tasks.tasks._raster import write_grid_cog
from occupancy_model_tasks.tasks._zones import classify_occupancy_zones
from occupancy_model_tasks.tasks._skip import all_dependencies_skipped

__all__ = [
    "all_dependencies_skipped",
    "build_occupancy_design",
    "check_target_event_types",
    "classify_occupancy_zones",
    "count_events_and_effort",
    "draw_coefficient_plot",
    "extract_fit_metric",
    "fit_occupancy_glm",
    "get_occupancy_fit_summary",
    "get_posterior_draws",
    "label_grid_with_distance_covariates",
    "label_grid_with_gee_covariates",
    "missing_trajectories_or_grid",
    "predict_occupancy_surface",
    "set_distance_covariates",
    "set_gee_covariates",
    "summarize_occupancy_run",
    "write_grid_cog",
]
