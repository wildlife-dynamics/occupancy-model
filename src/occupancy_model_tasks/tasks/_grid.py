"""Helpers for the regular square grids produced by ``create_meshgrid``.

Every grid-shaped frame in this package is joined on ``cell_x`` / ``cell_y``: the cell
centroid in EPSG:3857, rounded to the metre. Keys survive round-trips through parquet
fixtures and mock-io, where row order and index are not guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

GRID_CRS = "EPSG:3857"
KEY_COLUMNS = ["cell_x", "cell_y"]


def with_cell_keys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return ``gdf`` in EPSG:3857 with ``cell_x`` / ``cell_y`` centroid keys added."""
    out = gdf.to_crs(GRID_CRS) if gdf.crs is not None and gdf.crs != GRID_CRS else gdf.copy()
    if out.crs is None:
        out = out.set_crs(GRID_CRS)
    centroids = out.geometry.centroid
    out["cell_x"] = np.round(centroids.x.to_numpy()).astype("int64")
    out["cell_y"] = np.round(centroids.y.to_numpy()).astype("int64")
    return out


@dataclass(frozen=True)
class GridSpec:
    """Raster geometry of a regular grid: origin (top-left), cell size and shape."""

    xmin: float
    ymax: float
    cell_size: float
    width: int
    height: int

    @property
    def transform(self):
        from rasterio.transform import from_origin  # type: ignore[import-untyped]

        return from_origin(self.xmin, self.ymax, self.cell_size, self.cell_size)

    def rows_cols(self, gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
        cx = gdf["cell_x"].to_numpy(dtype=float)
        cy = gdf["cell_y"].to_numpy(dtype=float)
        cols = np.floor((cx - self.xmin) / self.cell_size).astype(int)
        rows = np.floor((self.ymax - cy) / self.cell_size).astype(int)
        return rows, cols

    def to_array(self, gdf: gpd.GeoDataFrame, column: str, fill: float = np.nan) -> np.ndarray:
        arr = np.full((self.height, self.width), fill, dtype="float64")
        rows, cols = self.rows_cols(gdf)
        values = pd.to_numeric(gdf[column], errors="coerce").to_numpy(dtype="float64")
        arr[rows, cols] = values
        return arr

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.xmin + (np.arange(self.width) + 0.5) * self.cell_size
        ys = self.ymax - (np.arange(self.height) + 0.5) * self.cell_size
        return xs, ys


def grid_spec(gdf: gpd.GeoDataFrame) -> GridSpec:
    """Infer the raster geometry of a keyed, regular square grid (EPSG:3857)."""
    if len(gdf) == 0:
        raise ValueError("Cannot infer a grid from an empty frame.")
    b = gdf.geometry.bounds
    sizes = (b["maxx"] - b["minx"]).to_numpy()
    cell_size = float(np.median(sizes))
    if not np.allclose(sizes, cell_size, rtol=1e-6) or not np.allclose(
        (b["maxy"] - b["miny"]).to_numpy(), cell_size, rtol=1e-6
    ):
        raise ValueError("Grid cells are not uniform squares; expected output of create_meshgrid.")
    xmin, ymax = float(b["minx"].min()), float(b["maxy"].max())
    width = int(round((float(b["maxx"].max()) - xmin) / cell_size))
    height = int(round((ymax - float(b["miny"].min())) / cell_size))
    return GridSpec(xmin=xmin, ymax=ymax, cell_size=cell_size, width=width, height=height)
