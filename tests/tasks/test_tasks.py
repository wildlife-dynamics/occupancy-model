import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import LineString, Point

from occupancy_model_tasks.tasks import (
    build_occupancy_design,
    classify_occupancy_zones,
    count_events_and_effort,
    draw_coefficient_plot,
    extract_fit_metric,
    fit_occupancy_glm,
    get_occupancy_fit_summary,
    get_posterior_draws,
    label_grid_with_distance_covariates,
    missing_trajectories_or_grid,
    predict_occupancy_surface,
    set_distance_covariates,
    set_gee_covariates,
    summarize_occupancy_run,
    write_grid_cog,
)
from occupancy_model_tasks.tasks._covariates import (
    DEFAULT_GEE_COVARIATES,
    DistanceCovariate,
    FileFeatureSource,
    ImageBandCovariate,
)
from occupancy_model_tasks.tasks._grid import grid_spec, with_cell_keys
from occupancy_model_tasks.tasks._model import McmcSettings

from .conftest import CELL, TRUE_BETA, X0, Y0

FAST = McmcSettings(draws=300, tune=300, chains=2, random_seed=1)


def _cov(name, quadratic=False):
    return ImageBandCovariate(name=name, image_id="synthetic/asset", band="b", quadratic=quadratic)


@pytest.fixture(scope="module")
def design(synthetic):
    return build_occupancy_design(
        cells=synthetic["cells"], covariate_frames=[synthetic["covariates"]], min_effort_km=0.5
    )


@pytest.fixture(scope="module")
def fit(design, tmp_path_factory):
    return fit_occupancy_glm(
        design=design,
        gee_covariates=[_cov("x1"), _cov("x2")],
        distance_covariates=[],
        model_type="cloglog_effort",
        prior_sd=1.0,
        holdout_fraction=0.2,
        mcmc=FAST,
    )


# ---------------------------------------------------------------- config + grid


def test_default_gee_covariates_validate():
    out = set_gee_covariates(covariates=DEFAULT_GEE_COVARIATES)
    assert [c.name for c in out] == ["elevation", "slope", "ndvi"]


def test_duplicate_covariate_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        set_gee_covariates(covariates=[_cov("a"), _cov("a")])


def test_gee_covariate_union_parses_from_form_json():
    from pydantic import TypeAdapter

    from occupancy_model_tasks.tasks._covariates import GeeCovariate

    parsed = TypeAdapter(list[GeeCovariate]).validate_python(
        [
            {"kind": "terrain", "name": "tri", "derivative": "tri"},
            {"kind": "collection_composite", "name": "ndvi", "collection_id": "MODIS/061/MOD13Q1", "band": "NDVI"},
        ]
    )
    assert [type(p).__name__ for p in parsed] == ["TerrainCovariate", "CollectionCompositeCovariate"]


def test_grid_spec_roundtrip(synthetic):
    keyed = with_cell_keys(synthetic["grid"])
    spec = grid_spec(keyed)
    assert (spec.width, spec.height, spec.cell_size) == (40, 40, CELL)
    rows, cols = spec.rows_cols(keyed)
    assert rows.min() == 0 and rows.max() == 39 and cols.min() == 0 and cols.max() == 39
    assert len(set(zip(rows, cols))) == len(keyed)


# ---------------------------------------------------------------- design


def test_design_labels(design, synthetic):
    assert len(design) == len(synthetic["grid"])
    assert set(["y", "effort_km", "surveyed", "survey_status", "x1", "x2"]) <= set(design.columns)
    # a presence needs a surveyed cell and an event
    assert (design.loc[design.y == 1, "surveyed"]).all()
    assert (design.loc[design.y == 1, "event_count"] > 0).all()
    assert set(design.survey_status) == {"Presence", "Absence", "Not surveyed"}
    assert design["x1"].notna().all()
    assert design["surveyed_y"].isna().sum() == (~design["surveyed"]).sum()


def test_design_joins_by_keys_not_row_order(synthetic):
    shuffled = synthetic["covariates"].sample(frac=1.0, random_state=0)
    d1 = build_occupancy_design(cells=synthetic["cells"], covariate_frames=[synthetic["covariates"]])
    d2 = build_occupancy_design(cells=synthetic["cells"], covariate_frames=[shuffled])
    pd.testing.assert_series_equal(d1["x1"], d2["x1"])


# ---------------------------------------------------------------- fit / predict


def test_fit_recovers_planted_coefficients(fit):
    assert fit["status"] == "ok", fit.get("message")
    rows = {r["Term"]: r for r in fit["summary"]}
    for name, true in TRUE_BETA.items():
        lo, hi = rows[name]["3%"], rows[name]["97%"]
        assert np.sign(lo) == np.sign(hi) == np.sign(true), (name, lo, hi)
    assert fit["metrics"]["max_r_hat"] < 1.05
    assert fit["metrics"]["auc"] > 0.6
    assert fit["metrics"]["auc_basis"] == "holdout"
    json.dumps(fit)  # must be JSON-serialisable
    draws = get_posterior_draws(fit=fit)
    assert list(draws.columns) == ["chain", "draw", "Intercept", "x1", "x2"]
    assert len(draws) == 600 and set(draws.chain) == {0, 1}
    assert draws["x1"].mean() == pytest.approx(next(r["Mean"] for r in fit["summary"] if r["Term"] == "x1"))


def test_predict_surface(fit, design):
    pred = predict_occupancy_surface(fit=fit, design=design, reference_effort_km=1.0, n_draws=200)
    assert len(pred) == len(design)
    p = pred["p_mean"].to_numpy()
    assert np.all((p >= 0) & (p <= 1))
    assert (pred["p_q05"] <= pred["p_mean"]).all() and (pred["p_mean"] <= pred["p_q95"]).all()
    # higher x1 (positive effect) -> higher predicted probability, on average
    hi = pred["x1"] > pred["x1"].median()
    assert pred.loc[hi, "p_mean"].mean() > pred.loc[~hi, "p_mean"].mean()
    # more reference effort -> higher detection probability (cloglog)
    pred5 = predict_occupancy_surface(fit=fit, design=design, reference_effort_km=5.0, n_draws=200)
    assert (pred5["p_mean"] >= pred["p_mean"] - 1e-12).all()


def test_fit_logit_with_quadratic(design, tmp_path):
    fit = fit_occupancy_glm(
        design=design,
        gee_covariates=[_cov("x1", quadratic=True), _cov("x2")],
        distance_covariates=[],
        model_type="logit",
        holdout_fraction=0.0,
        mcmc=McmcSettings(draws=200, tune=200, chains=2, random_seed=3),
    )
    assert fit["status"] == "ok"
    assert [t["label"] for t in fit["terms"]] == ["x1", "x1^2", "x2"]
    assert fit["metrics"]["auc_basis"] == "in-sample"
    summary = get_occupancy_fit_summary(fit=fit)
    assert list(summary["Term"]) == ["Intercept", "x1", "x1^2", "x2"]


def test_not_fitted_without_presences(design, tmp_path):
    no_events = design.copy()
    no_events["y"] = 0
    fit = fit_occupancy_glm(
        design=no_events,
        gee_covariates=[_cov("x1")],
        distance_covariates=[],
        mcmc=FAST,
    )
    assert fit["status"] == "not_fitted"
    assert "Not enough data" in fit["message"]
    assert len(predict_occupancy_surface(fit=fit, design=design)) == 0
    assert len(get_occupancy_fit_summary(fit=fit)) == 0
    assert extract_fit_metric(fit=fit, metric="auc") is None
    assert extract_fit_metric(fit=fit, metric="n_surveyed") == float(design.surveyed.sum())
    assert "No model was fitted" in summarize_occupancy_run(fit=fit)


def test_missing_covariate_column_is_an_error(design, tmp_path):
    with pytest.raises(ValueError, match="missing"):
        fit_occupancy_glm(
            design=design, gee_covariates=[_cov("nope")], distance_covariates=[], mcmc=FAST
        )


def test_summary_text_and_metrics(fit):
    txt = summarize_occupancy_run(fit=fit)
    assert "AUC" in txt and "Effort-adjusted" in txt
    assert extract_fit_metric(fit=fit, metric="n_presence") == fit["n_presence"]


# ---------------------------------------------------------------- outputs


def test_write_grid_cog(fit, design, tmp_path):
    pred = predict_occupancy_surface(fit=fit, design=design, n_draws=100)
    path = write_grid_cog(gdf=pred, value_column="p_mean", root_path=str(tmp_path), filename="occupancy_mean")
    local = path.replace("file://", "")
    with rasterio.open(local) as src:
        assert src.crs.to_epsg() == 3857
        assert (src.width, src.height) == (40, 40)
        assert src.res == (CELL, CELL)
        assert src.bounds.left == X0 and src.bounds.bottom == Y0
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        arr = src.read(1)
    # top-left pixel is the cell whose centre is (X0 + CELL/2, Y0 + 40*CELL - CELL/2)
    keyed = with_cell_keys(pred)
    tl = keyed[(keyed.cell_x == X0 + CELL / 2) & (keyed.cell_y == Y0 + 39.5 * CELL)]
    assert arr[0, 0] == pytest.approx(float(tl["p_mean"].iloc[0]), rel=1e-6)


def test_classify_occupancy_zones(fit, design, tmp_path):
    pred = predict_occupancy_surface(fit=fit, design=design, n_draws=100)
    zones = classify_occupancy_zones(gdf=pred, root_path=str(tmp_path), min_patch_area_km2=1.0, max_hole_area_km2=1.0)
    assert zones.crs.to_epsg() == 4326
    assert list(zones["zone_label"]) == ["Low occupancy", "Medium occupancy", "High occupancy"]
    assert zones["name"].is_unique and zones["name"].notna().all()
    assert zones.is_valid.all()
    # bands should not overlap and together roughly cover the grid (40 x 40 km)
    utm = zones.to_crs(zones.estimate_utm_crs())
    assert utm.union_all().area / 1e6 == pytest.approx(1600, rel=0.1)
    assert sum(g.area for g in utm.geometry) / 1e6 == pytest.approx(utm.union_all().area / 1e6, rel=0.01)
    for suffix in ("low", "medium", "high"):
        f = tmp_path / f"occupancy_zones_{suffix}.geojson"
        assert f.exists()
        assert gpd.read_file(f).crs.to_epsg() == 4326


def test_classify_occupancy_zones_clips_to_boundary(fit, design, tmp_path):
    pred = predict_occupancy_surface(fit=fit, design=design, n_draws=100)
    boundary = gpd.GeoDataFrame(
        geometry=[Point(X0 + 20 * CELL, Y0 + 20 * CELL).buffer(10 * CELL)], crs="EPSG:3857"
    )
    zones = classify_occupancy_zones(gdf=pred, root_path=str(tmp_path), boundary=boundary, min_patch_area_km2=0, max_hole_area_km2=0)
    b = boundary.to_crs(4326).union_all().buffer(1e-6)
    assert all(b.contains(g) for g in zones.geometry)


def test_draw_coefficient_plot(fit):
    html = draw_coefficient_plot(summary=get_occupancy_fit_summary(fit=fit), title="Effects")
    assert "<html" in html and "x1" in html and "Intercept" not in html


# ---------------------------------------------------------------- distance covariates


def test_distance_covariates_from_file(synthetic, tmp_path):
    # a vertical line along x = X0 + 10 km, and a point at the grid's top-right corner cell centre
    line = LineString([(X0 + 10 * CELL, Y0), (X0 + 10 * CELL, Y0 + 40 * CELL)])
    feats = gpd.GeoDataFrame({"name": ["road"]}, geometry=[line], crs="EPSG:3857").to_crs(4326)
    path = tmp_path / "roads.geojson"
    feats.to_file(path, driver="GeoJSON")
    covs = set_distance_covariates(
        covariates=[DistanceCovariate(name="dist_road", features=FileFeatureSource(path_or_url=str(path)))]
    )
    out = label_grid_with_distance_covariates(client=None, grid=synthetic["grid"], covariates=covs)
    keyed = out.set_index(["cell_x", "cell_y"])
    near = keyed.loc[(int(X0 + 9.5 * CELL), int(Y0 + 20.5 * CELL)), "dist_road"]
    far = keyed.loc[(int(X0 + 39.5 * CELL), int(Y0 + 20.5 * CELL)), "dist_road"]
    # 500 m and 29.5 km in Web Mercator; UTM ground distance is ~cos(lat) smaller (lat ~ -2.7)
    assert near == pytest.approx(500 * np.cos(np.radians(2.7)), rel=0.02)
    assert far == pytest.approx(29500 * np.cos(np.radians(2.7)), rel=0.02)


# ---------------------------------------------------------------- events and effort


def _track_and_events():
    # a straight 3 km track along the row of cells at y in [Y0, Y0+CELL], crossing 3 cells
    line = LineString([(X0 + 0.5 * CELL, Y0 + 0.5 * CELL), (X0 + 3.5 * CELL, Y0 + 0.5 * CELL)])
    traj = gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857").to_crs(4326)
    events = gpd.GeoDataFrame(
        {"event_type": ["t", "t", "t"]},
        geometry=[Point(X0 + 1.2 * CELL, Y0 + 0.5 * CELL), Point(X0 + 1.7 * CELL, Y0 + 0.6 * CELL), Point(X0 + 3.1 * CELL, Y0 + 0.4 * CELL)],
        crs="EPSG:3857",
    ).to_crs(4326)
    return traj, events


def test_count_events_and_effort(synthetic):
    traj, events = _track_and_events()
    out = count_events_and_effort(trajectories=traj, meshgrid=synthetic["grid"], events=events).set_index(["cell_x", "cell_y"])
    row = lambda i: out.loc[(int(X0 + (i + 0.5) * CELL), int(Y0 + 0.5 * CELL))]  # noqa: E731
    k = np.cos(np.radians(2.7))  # Web Mercator metres -> ground metres at this latitude
    assert row(0)["patrol_effort_km"] == pytest.approx(0.5 * k, rel=0.02)
    assert row(1)["patrol_effort_km"] == pytest.approx(1.0 * k, rel=0.02)
    assert row(3)["patrol_effort_km"] == pytest.approx(0.5 * k, rel=0.02)
    assert (row(1)["event_count"], row(3)["event_count"], row(0)["event_count"]) == (2, 1, 0)
    assert len(out) == len(synthetic["grid"]) and out["event_count"].sum() == 3


def test_count_effort_without_events(synthetic):
    traj, _ = _track_and_events()
    out = count_events_and_effort(trajectories=traj, meshgrid=synthetic["grid"], events=None)
    assert out["event_count"].sum() == 0 and out["patrol_effort_km"].sum() > 2.5
    empty_events = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    out2 = count_events_and_effort(trajectories=traj, meshgrid=synthetic["grid"], events=empty_events)
    assert out2["event_count"].sum() == 0


def test_missing_trajectories_or_grid(synthetic):
    traj, events = _track_and_events()
    grid = synthetic["grid"]
    assert missing_trajectories_or_grid(traj, grid, events) is False
    assert missing_trajectories_or_grid(traj, grid, None) is False
    assert missing_trajectories_or_grid(traj.iloc[0:0], grid, events) is True
    assert missing_trajectories_or_grid(traj, grid.iloc[0:0], events) is True


def test_all_dependencies_skipped():
    from wt_task.skip import SKIP_SENTINEL

    from occupancy_model_tasks.tasks import all_dependencies_skipped

    assert all_dependencies_skipped(SKIP_SENTINEL, ("k", SKIP_SENTINEL))
    assert all_dependencies_skipped()
    assert not all_dependencies_skipped(("k", "layer"), SKIP_SENTINEL)
    assert all_dependencies_skipped([(None, SKIP_SENTINEL)], [(None, SKIP_SENTINEL)])
    assert not all_dependencies_skipped([("k", "layer")], [(None, SKIP_SENTINEL)])


def test_check_target_event_types_flags_wrong_slug(design, tmp_path):
    import pandas as pd

    from ecoscope.platform.tasks.filter import set_time_range
    from ecoscope.platform.tasks.io import set_patrols_and_patrol_events_params

    from occupancy_model_tasks.tasks import check_target_event_types

    patrols = pd.read_parquet("dev/fixtures/get-patrols-from-combined-params.example-return.parquet")
    tr = set_time_range(since="2015-01-01T00:00:00", until="2016-12-31T23:59:59",
                        timezone={"label": "UTC", "tzCode": "UTC", "name": "UTC", "utc": "+00:00"})

    def params(types):
        return set_patrols_and_patrol_events_params(client="x", time_range=tr, patrol_types=["ecoscope_patrol"], event_types=types)

    bad = check_target_event_types(combined_params=params(["bird_sighting"]), patrols_df=patrols)
    assert bad["matched"] == 0 and "bird_sighting_rep" in bad["available"]
    n_target = sum(e["event_type"] == "bird_sighting_rep" for segs in patrols.patrol_segments for s in segs for e in s["events"])
    good = check_target_event_types(combined_params=params(["bird_sighting_rep"]), patrols_df=patrols)
    assert good["matched"] == n_target > 0 and good["unknown"] == []
    assert check_target_event_types(combined_params=params(["x"]), patrols_df=None)["available"] == {}

    no_events = design.copy()
    no_events["y"] = 0
    fit = fit_occupancy_glm(design=no_events, gee_covariates=[_cov("x1")], distance_covariates=[], mcmc=FAST)
    txt = summarize_occupancy_run(fit=fit, event_type_check=bad)
    assert "None of the requested event types (bird_sighting)" in txt and f"bird_sighting_rep ({n_target})" in txt
    assert "None of the requested" not in summarize_occupancy_run(fit=fit, event_type_check=good)
