"""Classify a prediction surface into smooth Low / Medium / High occupancy-zone polygons.

Port of the ``raster_to_risk_vector_contour`` approach: filled contour bands are computed
from the continuous surface (marching squares on cell centres, via contourpy) so neighbouring
bands share identical, smooth edges; small patches are dropped, small holes filled, the result
is clipped to the study area and written as one EPSG:4326 GeoJSON per class, each feature with
a unique ``name`` (EarthRanger styles one colour per Feature Type, hence one file per class).
"""


from typing import Annotated, Literal, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from ecoscope.platform.annotations import AnyGeoDataFrame
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from occupancy_model_tasks.tasks._grid import GRID_CRS, grid_spec, with_cell_keys

ZONE_LABELS = ["Low occupancy", "Medium occupancy", "High occupancy"]
ZONE_FILE_SUFFIXES = ["low", "medium", "high"]
ZONE_COLUMNS = ["zone_class", "zone_label", "name", "area_km2", "geometry"]


def _dilate_once(z: np.ndarray) -> np.ndarray:
    """Fill NaN cells that touch data with the mean of their finite neighbours."""
    padded = np.pad(z, 1, constant_values=np.nan)
    stack = np.stack(
        [padded[1 + dy : 1 + dy + z.shape[0], 1 + dx : 1 + dx + z.shape[1]]
         for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    )
    with np.errstate(invalid="ignore"):
        fill = np.nanmean(np.where(np.isfinite(stack), stack, np.nan), axis=0)
    out = z.copy()
    mask = ~np.isfinite(z) & np.isfinite(fill)
    out[mask] = fill[mask]
    return out


def _band_polygons(cg, lower: float, upper: float):
    from shapely.geometry import Polygon

    points_list, offsets_list = cg.filled(lower, upper)
    polys = []
    for pts, offs in zip(points_list, offsets_list):
        rings = [pts[offs[i] : offs[i + 1]] for i in range(len(offs) - 1)]
        rings = [r for r in rings if len(r) >= 4]
        if rings:
            polys.append(Polygon(rings[0], rings[1:]))
    return polys


def _clean(geom, min_area_m2: float, max_hole_m2: float):
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import make_valid

    geom = make_valid(geom)
    parts = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, Polygon)]
    kept = []
    for p in parts:
        if p.area < min_area_m2:
            continue
        holes = [h for h in p.interiors if Polygon(h).area >= max_hole_m2]
        kept.append(Polygon(p.exterior, holes))
    if not kept:
        return None
    return kept[0] if len(kept) == 1 else MultiPolygon(kept)


@register()
def classify_occupancy_zones(
    gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="Predicted grid (from predict_occupancy_surface).", exclude=True),
    ],
    root_path: Annotated[str, Field(description="Results directory for the GeoJSON files.", exclude=True)],
    boundary: Annotated[
        SkipJsonSchema[None] | AnyGeoDataFrame,
        Field(description="Study area to clip the zones to.", exclude=True),
    ] = None,
    value_column: Annotated[str, Field(description="Column to classify.")] = "p_mean",
    class_method: Annotated[
        Literal["tertiles"],
        Field(
            title="Zone Class Breaks",
            description="Tertiles: the lowest, middle and highest third of cells with a non-zero value.",
        ),
    ] = "tertiles",
    min_patch_area_km2: Annotated[
        float,
        Field(title="Minimum Zone Patch Area (km²)", description="Smaller patches are dropped.", ge=0),
    ] = 2.0,
    max_hole_area_km2: Annotated[
        float,
        Field(title="Maximum Hole Area to Fill (km²)", description="Smaller holes inside a zone are filled.", ge=0),
    ] = 2.0,
    filename_prefix: Annotated[str, Field(description="Prefix for the GeoJSON filenames.")] = "occupancy_zones",
) -> AnyGeoDataFrame:
    """Low / Medium / High occupancy zones as EPSG:4326 polygons (one row per class), written as
    ``<prefix>_low|medium|high.geojson``. Zero-valued cells fold into Low.
    """
    from contourpy import FillType, contour_generator
    from ecoscope.platform.serde import _persist_bytes

    keyed = with_cell_keys(cast(gpd.GeoDataFrame, gdf))
    keyed = keyed[np.isfinite(keyed[value_column].astype(float))]
    empty = gpd.GeoDataFrame(columns=ZONE_COLUMNS, geometry="geometry", crs="EPSG:4326")
    if keyed.empty:
        return cast(AnyGeoDataFrame, empty)

    spec = grid_spec(keyed)
    values = keyed[value_column].to_numpy(dtype=float)
    nonzero = values[values > 0]
    if class_method != "tertiles" or nonzero.size == 0:
        raise ValueError(f"Unsupported class method or no non-zero values: {class_method}")
    t1, t2 = np.quantile(nonzero, [1 / 3, 2 / 3])
    eps = 1e-9
    lowers = [-eps, t1, t2]
    uppers = [t1, t2, float(values.max()) + eps]

    z = _dilate_once(spec.to_array(keyed, value_column))
    xs, ys = spec.cell_centres()
    z, ys = z[::-1], ys[::-1]  # contourpy wants increasing y
    cg = contour_generator(x=xs, y=ys, z=np.ma.masked_invalid(z), fill_type=FillType.OuterOffset)

    utm = keyed.estimate_utm_crs()
    clip = None
    if boundary is not None and len(boundary) > 0:
        clip = cast(gpd.GeoDataFrame, boundary).to_crs(utm).union_all()

    rows = []
    for i, (lo, hi) in enumerate(zip(lowers, uppers)):
        if hi <= lo:
            continue
        polys = _band_polygons(cg, lo, hi)
        if not polys:
            continue
        band = gpd.GeoSeries(polys, crs=GRID_CRS).to_crs(utm).union_all()
        if clip is not None:
            band = band.intersection(clip)
        band = _clean(band, min_patch_area_km2 * 1e6, max_hole_area_km2 * 1e6)
        if band is None:
            continue
        rows.append({"zone_class": i + 1, "zone_label": ZONE_LABELS[i], "area_km2": band.area / 1e6, "geometry": band})
    if not rows:
        return cast(AnyGeoDataFrame, empty)

    zones = gpd.GeoDataFrame(rows, geometry="geometry", crs=utm).to_crs("EPSG:4326")
    zones["name"] = zones["zone_label"] + "_" + (zones.groupby("zone_label").cumcount() + 1).astype(str)
    zones["area_km2"] = zones["area_km2"].round(3)
    zones = zones[ZONE_COLUMNS]
    for i, suffix in enumerate(ZONE_FILE_SUFFIXES):
        cls = zones[zones["zone_class"] == i + 1]
        if len(cls):
            _persist_bytes(cls.to_json().encode("utf-8"), root_path, f"{filename_prefix}_{suffix}.geojson")
    return cast(AnyGeoDataFrame, zones.reset_index(drop=True))
