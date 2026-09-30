"""Covariate configuration and grid-cell labelling.

Two kinds of covariate, each configured as a list on its own form card:

- **Earth Engine covariates** (``set_gee_covariates``): an image band, a terrain derivative
  of a DEM, a temporal composite of an image collection, or distance to the features of a GEE
  FeatureCollection. ``label_grid_with_gee_covariates`` reduces each one over every grid cell.
- **Distance covariates** (``set_distance_covariates``): distance from each cell centre to the
  nearest feature of an EarthRanger spatial feature group or a vector file/URL, computed locally
  by ``label_grid_with_distance_covariates``.

Both labelling tasks take the grid from ``create_meshgrid`` and return it with one column per
covariate plus the ``cell_x`` / ``cell_y`` join keys. The same configuration lists are wired
into ``fit_occupancy_glm`` so the model knows its terms and which ones get a quadratic.
Heavy imports (``ee``) happen inside task bodies so the registry scan stays light.
"""


import logging
from typing import Annotated, Literal, Union, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope.platform.connections import EarthEngineClient, EarthRangerClient
from ecoscope.platform.tasks.filter._filter import TimeRange
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from occupancy_model_tasks.tasks._grid import KEY_COLUMNS, with_cell_keys

logger = logging.getLogger(__name__)

_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,39}$"
_DATE_PATTERN = r"^(\d{4}-\d{2}-\d{2})?$"

CovariateName = Annotated[
    str,
    Field(
        title="Name",
        description="Column name for this covariate (letters, digits and underscores, e.g. ndvi).",
        pattern=_NAME_PATTERN,
    ),
]
Quadratic = Annotated[
    bool,
    Field(
        title="Add Quadratic Term",
        description="Also fit this covariate squared, for a hump-shaped (or U-shaped) response.",
    ),
]
ScaleM = Annotated[
    float,
    Field(
        title="Sampling Scale (m)",
        description="Pixel size Earth Engine samples at. Cells are averaged at no finer than a tenth of the cell size.",
        gt=0,
    ),
]
Multiplier = Annotated[
    float,
    Field(title="Multiplier", description="Scale factor applied to the values (e.g. 0.0001 for MODIS NDVI)."),
]


class ImageBandCovariate(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "Image band"})

    kind: Annotated[Literal["image_band"], Field(title="Covariate Type")] = "image_band"
    name: CovariateName
    image_id: Annotated[
        str,
        Field(title="Image ID", description="Earth Engine image asset ID, e.g. USGS/SRTMGL1_003."),
    ]
    band: Annotated[str, Field(title="Band", description="Band to sample, e.g. elevation.")]
    reducer: Annotated[
        Literal["mean", "median", "min", "max", "mode"],
        Field(title="Cell Summary", description="How pixel values are summarised within each cell."),
    ] = "mean"
    scale_m: ScaleM = 30.0
    multiplier: Multiplier = 1.0
    quadratic: Quadratic = False


class TerrainCovariate(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "Terrain derivative"})

    kind: Annotated[Literal["terrain"], Field(title="Covariate Type")] = "terrain"
    name: CovariateName
    derivative: Annotated[
        Literal["slope", "aspect", "tri"],
        Field(
            title="Derivative",
            description="slope (degrees), aspect (degrees) or tri (terrain ruggedness index, m).",
        ),
    ] = "slope"
    dem_image_id: Annotated[
        str, Field(title="DEM Image ID", description="Earth Engine DEM image asset ID.")
    ] = "USGS/SRTMGL1_003"
    band: Annotated[str, Field(title="Elevation Band")] = "elevation"
    scale_m: ScaleM = 30.0
    quadratic: Quadratic = False


class CollectionCompositeCovariate(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "Collection composite"})

    kind: Annotated[Literal["collection_composite"], Field(title="Covariate Type")] = "collection_composite"
    name: CovariateName
    collection_id: Annotated[
        str,
        Field(title="Image Collection ID", description="Earth Engine ImageCollection ID, e.g. MODIS/061/MOD13Q1."),
    ]
    band: Annotated[str, Field(title="Band", description="Band to composite, e.g. NDVI.")]
    composite: Annotated[
        Literal["mean", "median", "min", "max"],
        Field(title="Composite", description="How images in the window are combined."),
    ] = "mean"
    start_date: Annotated[
        str,
        Field(
            title="Start Date",
            description="YYYY-MM-DD. Leave both dates empty to use the workflow time range.",
            pattern=_DATE_PATTERN,
        ),
    ] = ""
    end_date: Annotated[
        str,
        Field(title="End Date", description="YYYY-MM-DD (exclusive).", pattern=_DATE_PATTERN),
    ] = ""
    scale_m: ScaleM = 250.0
    multiplier: Multiplier = 1.0
    quadratic: Quadratic = False


class GeeDistanceCovariate(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "Distance to GEE features"})

    kind: Annotated[Literal["gee_distance"], Field(title="Covariate Type")] = "gee_distance"
    name: CovariateName
    feature_collection_id: Annotated[
        str,
        Field(title="Feature Collection ID", description="Earth Engine FeatureCollection ID."),
    ]
    filter_property: Annotated[
        str,
        Field(title="Filter Property", description="Optional property to filter features on."),
    ] = ""
    filter_value: Annotated[
        str,
        Field(title="Filter Value", description="Keep features whose Filter Property equals this."),
    ] = ""
    max_distance_m: Annotated[
        float,
        Field(title="Maximum Distance (m)", description="Distances beyond this are capped at it.", gt=0),
    ] = 50000.0
    scale_m: ScaleM = 100.0
    quadratic: Quadratic = False


GeeCovariate = Annotated[
    Union[ImageBandCovariate, TerrainCovariate, CollectionCompositeCovariate, GeeDistanceCovariate],
    Field(discriminator="kind"),
]


class EarthRangerFeatureSource(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "EarthRanger feature group"})

    source: Annotated[Literal["earthranger"], Field(title="Source")] = "earthranger"
    spatial_features_group_name: Annotated[
        str,
        Field(title="Feature Group", description="Name of the spatial feature group in EarthRanger."),
    ]


class FileFeatureSource(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "File or URL"})

    source: Annotated[Literal["file"], Field(title="Source")] = "file"
    path_or_url: Annotated[
        str,
        Field(title="Path or URL", description="Vector file (GeoPackage, GeoJSON, zipped shapefile, …)."),
    ]
    layer: Annotated[
        str,
        Field(title="Layer", description="Layer name, for multi-layer files (optional)."),
    ] = ""


class DistanceCovariate(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "Distance to features"})

    name: CovariateName
    features: Annotated[
        Union[EarthRangerFeatureSource, FileFeatureSource],
        Field(title="Features", discriminator="source"),
    ]
    quadratic: Quadratic = False


DEFAULT_GEE_COVARIATES: list = [
    ImageBandCovariate(name="elevation", image_id="USGS/SRTMGL1_003", band="elevation", scale_m=30.0),
    TerrainCovariate(name="slope", derivative="slope"),
    CollectionCompositeCovariate(
        name="ndvi",
        collection_id="MODIS/061/MOD13Q1",
        band="NDVI",
        composite="mean",
        scale_m=250.0,
        multiplier=0.0001,
    ),
]


def covariate_terms(
    gee_covariates: list | None, distance_covariates: list | None
) -> list[tuple[str, bool]]:
    """(column name, quadratic?) for every configured covariate, GEE first then distance."""
    terms: list[tuple[str, bool]] = []
    for c in list(gee_covariates or []) + list(distance_covariates or []):
        terms.append((c.name, bool(c.quadratic)))
    names = [t[0] for t in terms]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Covariate names must be unique; duplicated: {dupes}")
    return terms


@register()
def set_gee_covariates(
    covariates: Annotated[
        list[GeeCovariate],
        Field(
            title="Earth Engine Covariates",
            description="Covariates sampled from Google Earth Engine for every grid cell.",
        ),
    ] = DEFAULT_GEE_COVARIATES,  # noqa: B006
) -> list[GeeCovariate]:
    """Configure the Earth Engine covariates; passthrough for the labelling and fit tasks."""
    covariate_terms(covariates, None)
    return covariates


@register()
def set_distance_covariates(
    covariates: Annotated[
        list[DistanceCovariate],
        Field(
            title="Distance Covariates",
            description="Distance from each cell centre to the nearest feature of your own layers (e.g. ranger stations, roads).",
        ),
    ] = [],  # noqa: B006
) -> list[DistanceCovariate]:
    """Configure the distance-to-feature covariates; passthrough for the labelling and fit tasks."""
    covariate_terms(None, covariates)
    return covariates


# ---------------------------------------------------------------- Earth Engine labelling


def _date_window(cov: CollectionCompositeCovariate, time_range: TimeRange | None) -> tuple[str, str]:
    if cov.start_date and cov.end_date:
        return cov.start_date, cov.end_date
    if time_range is None:
        raise ValueError(f"Covariate '{cov.name}' needs start/end dates or a workflow time range.")
    since = pd.Timestamp(time_range.since)
    until = pd.Timestamp(time_range.until) + pd.Timedelta(days=1)
    return since.strftime("%Y-%m-%d"), until.strftime("%Y-%m-%d")


def _ee_image(cov, time_range: TimeRange | None):
    import ee  # type: ignore[import-untyped]

    if isinstance(cov, ImageBandCovariate):
        img = ee.Image(cov.image_id).select([cov.band])
        mult = cov.multiplier
    elif isinstance(cov, TerrainCovariate):
        dem = ee.Image(cov.dem_image_id).select([cov.band])
        if cov.derivative == "slope":
            img = ee.Terrain.slope(dem)
        elif cov.derivative == "aspect":
            img = ee.Terrain.aspect(dem)
        else:  # Riley et al. (1999) terrain ruggedness index over the 3x3 neighbourhood
            neighbours = dem.neighborhoodToBands(ee.Kernel.square(1))
            img = neighbours.subtract(dem).pow(2).reduce(ee.Reducer.sum()).sqrt()
        mult = 1.0
    elif isinstance(cov, CollectionCompositeCovariate):
        start, end = _date_window(cov, time_range)
        coll = ee.ImageCollection(cov.collection_id).filterDate(start, end).select([cov.band])
        if coll.size().getInfo() == 0:
            raise ValueError(
                f"Covariate '{cov.name}': no images in {cov.collection_id} between {start} and {end}."
            )
        img = getattr(coll, cov.composite)()
        mult = cov.multiplier
    elif isinstance(cov, GeeDistanceCovariate):
        fc = ee.FeatureCollection(cov.feature_collection_id)
        if cov.filter_property:
            fc = fc.filter(ee.Filter.eq(cov.filter_property, cov.filter_value))
        img = fc.distance(searchRadius=cov.max_distance_m, maxError=cov.scale_m).unmask(cov.max_distance_m)
        mult = 1.0
    else:  # pragma: no cover - guarded by the discriminated union
        raise TypeError(f"Unsupported covariate type: {type(cov).__name__}")
    if mult != 1.0:
        img = img.multiply(mult)
    return img.rename([cov.name])


def _ee_reducer(cov):
    import ee  # type: ignore[import-untyped]

    name = getattr(cov, "reducer", "mean")
    return getattr(ee.Reducer, name)().setOutputs([cov.name])


def _reduce_cells(img, reducer, cells_4326: gpd.GeoDataFrame, name: str, scale: float, chunk: int) -> pd.DataFrame:
    import ee  # type: ignore[import-untyped]

    rows: list[dict] = []
    for start in range(0, len(cells_4326), chunk):
        sub = cells_4326.iloc[start : start + chunk][KEY_COLUMNS + ["geometry"]]
        fc = ee.FeatureCollection(sub.__geo_interface__)
        reduced = img.reduceRegions(collection=fc, reducer=reducer, scale=scale, tileScale=4)
        info = reduced.select(KEY_COLUMNS + [name], retainGeometry=False).getInfo()
        for feat in info.get("features", []):
            props = feat.get("properties", {})
            rows.append({"cell_x": props["cell_x"], "cell_y": props["cell_y"], name: props.get(name)})
    out = pd.DataFrame(rows, columns=KEY_COLUMNS + [name])
    out[name] = pd.to_numeric(out[name], errors="coerce")
    return out


@register(tags=["io"])
def label_grid_with_gee_covariates(
    client: EarthEngineClient,
    grid: Annotated[
        AnyGeoDataFrame,
        Field(description="Regular grid of cells (from create_meshgrid).", exclude=True),
    ],
    covariates: Annotated[
        list[GeeCovariate],
        Field(description="Earth Engine covariates (from set_gee_covariates).", exclude=True),
    ],
    time_range: Annotated[
        TimeRange | SkipJsonSchema[None],
        Field(description="Workflow time range; default window for collection composites.", exclude=True),
    ] = None,
    df_chunk_size: Annotated[
        int,
        Field(description="Cells per Earth Engine request. Lower it if you hit GEE limits.", gt=0),
    ] = 2000,
) -> AnyGeoDataFrame:
    """Reduce each Earth Engine covariate over every grid cell.

    Returns the grid (EPSG:3857) with ``cell_x`` / ``cell_y`` keys and one column per
    covariate. Cells without data come back as NaN. Each covariate is sampled at
    ``max(scale_m, cell_size / 10)`` so large cells don't pull millions of pixels.
    """
    cells = with_cell_keys(cast(gpd.GeoDataFrame, grid))
    out = cells[KEY_COLUMNS + ["geometry"]].copy()
    if len(cells) == 0 or not covariates:
        return cast(AnyGeoDataFrame, out)
    cell_size = float(np.median(cells.geometry.bounds.eval("maxx - minx")))
    cells_4326 = out.to_crs("EPSG:4326")
    for cov in covariates:
        scale = max(float(cov.scale_m), cell_size / 10.0)
        logger.info("Labelling %d cells with '%s' at %.0f m", len(cells), cov.name, scale)
        values = _reduce_cells(
            _ee_image(cov, time_range), _ee_reducer(cov), cells_4326, cov.name, scale, df_chunk_size
        )
        out = out.merge(values, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return cast(AnyGeoDataFrame, out)


# ---------------------------------------------------------------- distance labelling


def _load_features(client, source) -> gpd.GeoDataFrame:
    if isinstance(source, EarthRangerFeatureSource):
        from ecoscope.platform.tasks.io import get_spatial_features_group

        gdf = get_spatial_features_group(client=client, spatial_features_group_name=source.spatial_features_group_name)
    else:
        path = source.path_or_url
        if path.startswith("file://"):
            path = path[len("file://") :]
        gdf = gpd.read_file(path, layer=source.layer) if source.layer else gpd.read_file(path)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


@register(tags=["io"])
def label_grid_with_distance_covariates(
    client: EarthRangerClient,
    grid: Annotated[
        AnyGeoDataFrame,
        Field(description="Regular grid of cells (from create_meshgrid).", exclude=True),
    ],
    covariates: Annotated[
        list[DistanceCovariate],
        Field(description="Distance covariates (from set_distance_covariates).", exclude=True),
    ],
) -> AnyGeoDataFrame:
    """Distance (m) from each cell centre to the nearest feature of each configured layer.

    Points, lines and polygons are all used (a cell inside a polygon is at distance 0).
    Distances are measured in the local UTM zone, not in Web Mercator.
    """
    cells = with_cell_keys(cast(gpd.GeoDataFrame, grid))
    out = cells[KEY_COLUMNS + ["geometry"]].copy()
    if len(cells) == 0 or not covariates:
        return cast(AnyGeoDataFrame, out)
    utm = cells.estimate_utm_crs()
    centres = gpd.GeoDataFrame(cells[KEY_COLUMNS], geometry=cells.geometry.centroid, crs=cells.crs).to_crs(utm)
    for cov in covariates:
        feats = _load_features(client, cov.features).to_crs(utm)
        if feats.empty:
            raise ValueError(f"Distance covariate '{cov.name}': the feature source has no geometries.")
        joined = gpd.sjoin_nearest(centres, feats[["geometry"]], how="left", distance_col=cov.name)
        dist = joined.groupby(KEY_COLUMNS, as_index=False)[cov.name].min()
        out = out.merge(dist, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return cast(AnyGeoDataFrame, out)
