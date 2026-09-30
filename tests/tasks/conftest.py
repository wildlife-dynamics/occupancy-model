"""Synthetic grids with known coefficients for the task unit tests (no real data)."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

CELL = 1000.0
X0, Y0 = 4_200_000.0, -300_000.0  # EPSG:3857, inside the Tsavo-ish test extent
TRUE_ALPHA = -2.0  # log encounter intensity per km at covariate means
TRUE_BETA = {"x1": 1.0, "x2": -0.8}


def make_grid(nx: int = 40, ny: int = 40) -> gpd.GeoDataFrame:
    cells = [box(X0 + i * CELL, Y0 + j * CELL, X0 + (i + 1) * CELL, Y0 + (j + 1) * CELL) for j in range(ny) for i in range(nx)]
    return gpd.GeoDataFrame(geometry=cells, crs="EPSG:3857")


def smooth_field(rng, nx, ny, scale=8):
    noise = rng.normal(size=(ny + 2 * scale, nx + 2 * scale))
    k = np.ones(scale) / scale
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, noise)
    sm = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, sm)
    f = sm[scale:-scale, scale:-scale]
    return ((f - f.mean()) / f.std()).reshape(-1)


@pytest.fixture(scope="session")
def synthetic():
    """Grid + encounter columns + covariates, with detections drawn from the cloglog model."""
    rng = np.random.default_rng(7)
    nx = ny = 40
    grid = make_grid(nx, ny)
    x1 = smooth_field(rng, nx, ny)
    x2 = smooth_field(rng, nx, ny)
    effort = np.where(rng.random(len(grid)) < 0.7, rng.gamma(2.0, 1.5, len(grid)), 0.0)
    eta = TRUE_ALPHA + TRUE_BETA["x1"] * x1 + TRUE_BETA["x2"] * x2
    p = 1 - np.exp(-np.exp(eta) * effort)
    detected = rng.random(len(grid)) < p
    cells = grid.copy()
    cells["event_count"] = np.where(detected, rng.integers(1, 4, len(grid)), 0)
    cells["patrol_effort_km"] = np.where(effort > 0, effort, np.nan)
    cov = grid.copy()
    cov["x1"] = x1 * 100 + 1200  # raw units, so standardisation matters
    cov["x2"] = x2 * 0.1 + 0.5
    return {"grid": grid, "cells": cells, "covariates": cov}
