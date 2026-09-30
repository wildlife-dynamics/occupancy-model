"""Join per-cell detections, patrol effort and covariates into the model's design frame."""

from typing import Annotated, Any, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from ecoscope.platform.annotations import AnyGeoDataFrame
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from wt_task import SkippedDependencyFallback

from occupancy_model_tasks.tasks._grid import KEY_COLUMNS, with_cell_keys

SURVEY_PRESENCE = "Presence"
SURVEY_ABSENCE = "Absence"
SURVEY_UNSURVEYED = "Not surveyed"


@register()
def build_occupancy_design(
    cells: Annotated[
        AnyGeoDataFrame,
        Field(
            description="Grid with per-cell 'event_count' and 'patrol_effort_km' (from calculate_encounter_rate_grid).",
            exclude=True,
        ),
    ],
    covariate_frames: Annotated[
        list[AnyGeoDataFrame],
        Field(
            description="Labelled grids (from the covariate labelling tasks), joined on cell keys.",
            exclude=True,
        ),
    ],
    min_effort_km: Annotated[
        float,
        Field(
            title="Minimum Patrol Effort per Cell (km)",
            description="A cell counts as surveyed once patrols covered at least this distance in it; surveyed cells without a target event are absences.",
            ge=0,
        ),
    ] = 0.5,
) -> AnyGeoDataFrame:
    """One row per grid cell: ``y`` (target event recorded), ``effort_km``, ``surveyed``,
    ``surveyed_y`` (``y``, NaN where unsurveyed), ``survey_status`` and every covariate column. Unsurveyed cells are kept: they are
    predicted into, never fitted.
    """
    base = with_cell_keys(cast(gpd.GeoDataFrame, cells))
    events = pd.to_numeric(base.get("event_count", 0), errors="coerce")
    effort = pd.to_numeric(base.get("patrol_effort_km", 0), errors="coerce")
    design = base[KEY_COLUMNS + ["geometry"]].copy()
    design["event_count"] = np.nan_to_num(np.asarray(events, dtype=float), nan=0.0).astype(int)
    design["effort_km"] = np.nan_to_num(np.asarray(effort, dtype=float), nan=0.0)
    design["surveyed"] = design["effort_km"] >= float(min_effort_km)
    design["y"] = ((design["event_count"] > 0) & design["surveyed"]).astype(int)
    design["surveyed_y"] = design["y"].where(design["surveyed"]).astype(float)
    design["survey_status"] = np.where(
        ~design["surveyed"],
        SURVEY_UNSURVEYED,
        np.where(design["y"] == 1, SURVEY_PRESENCE, SURVEY_ABSENCE),
    )

    for frame in covariate_frames:
        if frame is None or len(frame) == 0:
            continue
        keyed = with_cell_keys(cast(gpd.GeoDataFrame, frame))
        value_cols = [c for c in keyed.columns if c not in KEY_COLUMNS + ["geometry"]]
        clash = sorted(set(value_cols) & set(design.columns))
        if clash:
            raise ValueError(f"Covariate columns clash with design columns: {clash}")
        design = design.merge(
            pd.DataFrame(keyed[KEY_COLUMNS + value_cols]), on=KEY_COLUMNS, how="left", validate="one_to_one"
        )
    return cast(AnyGeoDataFrame, gpd.GeoDataFrame(design, geometry="geometry", crs=base.crs))


def _none_if_skipped(obj: Any) -> Any:
    from wt_task.skip import SkipSentinel

    return None if isinstance(obj, SkipSentinel) else obj


def _is_nonempty_gdf_of(obj: Any, geom_types: set[str]) -> bool:
    return (
        isinstance(obj, gpd.GeoDataFrame)
        and len(obj) > 0
        and bool(obj.geometry.geom_type.isin(geom_types).any())
    )


@register()
def missing_trajectories_or_grid(*args: Any) -> bool:
    """skipif condition for ``count_events_and_effort``: skip only when the patrol track
    lines or the grid polygons are missing. A skipped or empty events frame does not skip
    the task, because cells with effort and no events are absences, not missing data.
    """
    has_lines = any(_is_nonempty_gdf_of(a, {"LineString", "MultiLineString"}) for a in args)
    has_polys = any(_is_nonempty_gdf_of(a, {"Polygon", "MultiPolygon"}) for a in args)
    return not (has_lines and has_polys)


@register()
def count_events_and_effort(
    trajectories: Annotated[
        AnyGeoDataFrame,
        Field(description="Patrol track segments (from relocations_to_trajectory).", exclude=True),
    ],
    meshgrid: Annotated[
        AnyGeoDataFrame,
        Field(description="Grid cells (from create_meshgrid).", exclude=True),
    ],
    events: Annotated[
        SkipJsonSchema[None] | AnyGeoDataFrame,
        Field(description="Target events; skipped or empty means no events.", exclude=True),
        SkippedDependencyFallback(_none_if_skipped),
    ] = None,
) -> AnyGeoDataFrame:
    """Per grid cell: ``event_count`` (target events inside the cell) and
    ``patrol_effort_km`` (length of patrol track inside the cell, measured in the local UTM
    zone). Every grid cell is returned. With no events, all counts are 0 and effort still
    defines the surveyed cells.
    """
    grid = with_cell_keys(cast(gpd.GeoDataFrame, meshgrid))[KEY_COLUMNS + ["geometry"]]
    out = grid.copy()

    traj = cast(gpd.GeoDataFrame, trajectories)
    traj = traj[traj.geometry.notna() & ~traj.geometry.is_empty][["geometry"]].to_crs(grid.crs)
    pieces = gpd.overlay(traj, grid, how="intersection", keep_geom_type=True)
    if len(pieces):
        utm = grid.estimate_utm_crs()
        pieces["km"] = pieces.to_crs(utm).length / 1000.0
        effort = pieces.groupby(KEY_COLUMNS, as_index=False)["km"].sum()
        out = out.merge(effort, on=KEY_COLUMNS, how="left")
    else:
        out["km"] = 0.0
    out["patrol_effort_km"] = out.pop("km").fillna(0.0).round(4)

    out["event_count"] = 0
    ev = events if isinstance(events, gpd.GeoDataFrame) else None
    if ev is not None and len(ev) and "geometry" in ev:
        ev = ev[ev.geometry.notna() & ~ev.geometry.is_empty][["geometry"]].to_crs(grid.crs)
        hits = gpd.sjoin(ev, grid, how="inner", predicate="within")
        counts = hits.groupby(KEY_COLUMNS).size().rename("n").reset_index()
        out = out.drop(columns="event_count").merge(counts, on=KEY_COLUMNS, how="left")
        out["event_count"] = out.pop("n").fillna(0).astype(int)
    return cast(AnyGeoDataFrame, gpd.GeoDataFrame(out, geometry="geometry", crs=grid.crs))
