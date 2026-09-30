"""Write a column of a regular grid as a Cloud-Optimised GeoTIFF."""


import os
import tempfile
from typing import Annotated, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from ecoscope.platform.annotations import AnyGeoDataFrame
from pydantic import Field
from wt_registry import register

from occupancy_model_tasks.tasks._grid import GRID_CRS, grid_spec, with_cell_keys


@register()
def write_grid_cog(
    gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="Regular grid with the value column (e.g. from predict_occupancy_surface).", exclude=True),
    ],
    value_column: Annotated[str, Field(description="Column to write as the raster band.")],
    root_path: Annotated[str, Field(description="Results directory.", exclude=True)],
    filename: Annotated[str, Field(description="Output filename, without extension.")],
    nodata: Annotated[float, Field(description="Nodata value for cells without a value.")] = -9999.0,
) -> Annotated[str, Field(description="Path to the written COG.")]:
    """Rasterise one column of a ``create_meshgrid`` grid onto its own cell lattice
    (EPSG:3857, one pixel per cell) and write it as a deflate-compressed COG with overviews.
    """
    import rasterio  # type: ignore[import-untyped]
    from ecoscope.platform.serde import _persist_bytes

    keyed = with_cell_keys(cast(gpd.GeoDataFrame, gdf))
    spec = grid_spec(keyed)
    arr = spec.to_array(keyed, value_column)
    arr = np.where(np.isfinite(arr), arr, nodata).astype("float32")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{filename}.tif")
        with rasterio.open(
            path,
            "w",
            driver="COG",
            width=spec.width,
            height=spec.height,
            count=1,
            dtype="float32",
            crs=GRID_CRS,
            transform=spec.transform,
            nodata=nodata,
            compress="DEFLATE",
            predictor=3,
            overview_resampling="average",
        ) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, value_column)
        with open(path, "rb") as f:
            data = f.read()
    return _persist_bytes(data, root_path, f"{filename}.tif")
