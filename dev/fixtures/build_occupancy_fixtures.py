"""Build the synthetic mock-io fixtures for the occupancy-model workflow.

Run from the repo root in the task-dev env:

    pixi run --manifest-path pixi.toml -e tasks python dev/fixtures/build_occupancy_fixtures.py

Everything is generated (seeded RNG, uuid5 ids, made-up names); only the *model* comes from
EarthRanger: the patrol / segment / event / observation schemas are copied from the packaged
ecoscope-platform fixtures, and the slugs (``ecoscope_patrol``, ``bird_sighting_rep``) are the
mep_dev test configuration. Never hand-edit the parquets: change this script and re-run it.

What is planted (and what each test case relies on):

- AOI: one ~20 x 20 km polygon near 38.6E 2.5S (display name "Synthetic Occupancy AOI").
- Grid: ``create_meshgrid(AOI, 1000 m, EPSG:3857, intersecting_only=True)`` -- the exact grid the
  base case builds, so the covariate fixtures join cell-for-cell.
- GEE covariates (the base case's three): elevation (m), slope (deg), ndvi
  (0-1), smooth random fields. A ``dist_station`` distance covariate (m) for the distance case.
- Truth: encounter intensity per km  log(lambda) = ALPHA + sum(BETA[k] * z(k)); detections are
  drawn per track segment with P = 1 - exp(-lambda * km), i.e. the effort-adjusted model. The
  base case must recover sign(BETA) for elevation (-) and ndvi (+).
- Patrols: ``N_PATROLS`` random-walk ``ecoscope_patrol`` patrols confined to the western ~65% of
  the AOI, leaving unsurveyed cells to predict into. Fixes every 10 min at ~2 km/h (inside the
  default trajectory segment filter).
- Events: detections embedded as ``bird_sighting_rep`` events on the patrol segments, plus
  ``fire_rep`` distractors that the event-type filter must drop. All inside 2015-2016.
- Empty variants (``*.empty.parquet``) keep the schemas with zero rows (the ``empty`` case).
- ``get-patrols-from-combined-params.no-target-events.parquet``: the same patrols with every
  ``bird_sighting_rep`` removed (the ``no_events`` case: tracks but no presences).
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

import ecoscope.platform.tasks.io as platform_io
from ecoscope.platform.tasks.analysis import create_meshgrid
from ecoscope.platform.tasks.analysis._time_density import CustomGridCellSize
from occupancy_model_tasks.tasks._grid import with_cell_keys

SEED = 20260924
NS = uuid.UUID("5b0b7c1e-3f3e-4c8e-9a51-0c0c0ffee000")
OUT = Path(__file__).parent
PKG = Path(__file__).resolve().parents[2] / "src" / "occupancy_model_tasks" / "tasks"
PLATFORM_FIXTURES = Path(platform_io.__file__).parent

AOI_BOUNDS = (38.50, -2.60, 38.68, -2.42)  # lon/lat, ~20 x 20 km
CELL_SIZE = 1000.0
N_PATROLS = 60
FIX_MINUTES = 10
FIXES_PER_PATROL = 36  # 6 h
TARGET_TYPE = "bird_sighting_rep"
DISTRACTOR_TYPE = "fire_rep"
PATROL_TYPE = "ecoscope_patrol"
START = datetime(2015, 1, 5, tzinfo=timezone.utc)
END = datetime(2016, 12, 20, tzinfo=timezone.utc)

ALPHA = -1.2
BETA = {"elevation": -0.6, "slope": 0.0, "ndvi": 0.8}


def uid(*parts) -> str:
    return str(uuid.uuid5(NS, "/".join(map(str, parts))))


def smooth_field(rng, shape, scale=6):
    ny, nx = shape
    noise = rng.normal(size=(ny + 2 * scale, nx + 2 * scale))
    k = np.ones(scale) / scale
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, noise)
    sm = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, sm)
    f = sm[scale:-scale, scale:-scale]
    return (f - f.mean()) / f.std()


def write(gdf: pd.DataFrame, name: str, *, also_package: bool = False, index: bool = False) -> None:
    for target in [OUT] + ([PKG] if also_package else []):
        gdf.to_parquet(target / f"{name}.example-return.parquet", index=index)
    gdf.iloc[0:0].to_parquet(OUT / f"{name}.empty.parquet", index=index)
    print(f"wrote {name}: {len(gdf)} rows")


def build_aoi() -> gpd.GeoDataFrame:
    template = gpd.read_parquet(PLATFORM_FIXTURES / "get-spatial-features-group.example-return.parquet")
    row = template.iloc[[0]].copy()
    row["geometry"] = [box(*AOI_BOUNDS)]
    for col, val in {"name": "Synthetic Occupancy AOI", "short_name": "Synthetic AOI", "pk": uid("aoi")}.items():
        if col in row.columns:
            row[col] = val
    if "metadata" in row.columns:
        row["metadata"] = [{"id": uid("aoi-group"), "display_name": "Synthetic Occupancy AOI"}]
    for col in ("description", "external_id", "external_source"):
        if col in row.columns:
            row[col] = ""
    return gpd.GeoDataFrame(row, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)


def build_covariates(rng, grid: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame):
    keyed = with_cell_keys(grid)
    xs = np.sort(keyed.cell_x.unique())
    ys = np.sort(keyed.cell_y.unique())[::-1]
    fields = {k: smooth_field(rng, (len(ys), len(xs))) for k in BETA}
    col = np.searchsorted(xs, keyed.cell_x)
    row = np.searchsorted(-ys, -keyed.cell_y)
    z = {k: f[row, col] for k, f in fields.items()}
    gee = keyed[["cell_x", "cell_y", "geometry"]].copy()
    gee["elevation"] = 900 + 120 * z["elevation"]
    gee["slope"] = np.clip(4 + 2.5 * z["slope"], 0, None)
    gee["ndvi"] = np.clip(0.45 + 0.12 * z["ndvi"], -0.1, 0.95)
    # standardised versions of what was written, so the truth is on the model's scale
    zs = {k: (gee[k] - gee[k].mean()) / gee[k].std() for k in BETA}
    log_lambda = ALPHA + sum(BETA[k] * zs[k] for k in BETA)
    truth = pd.DataFrame({"cell_x": gee.cell_x, "cell_y": gee.cell_y, "log_lambda": log_lambda})

    station = gpd.GeoSeries([Point(38.53, -2.45)], crs="EPSG:4326").to_crs(keyed.estimate_utm_crs())
    centres = keyed.geometry.centroid.to_crs(station.crs)
    dist = keyed[["cell_x", "cell_y", "geometry"]].copy()
    dist["dist_station"] = centres.distance(station.iloc[0]).to_numpy()
    return gee, dist, truth


def build_patrols(rng, grid: gpd.GeoDataFrame, truth: pd.DataFrame):
    template = pd.read_parquet(PLATFORM_FIXTURES / "get-patrols-from-combined-params.example-return.parquet")
    seg_t = template.patrol_segments.iloc[0][0]
    ev_t = seg_t["events"][0]
    obs_t = gpd.read_parquet(
        PLATFORM_FIXTURES / "get-patrol-observations-from-patrols-df-and-combined-params.example-return.parquet"
    )

    keyed = with_cell_keys(grid)
    cell_lookup = truth.set_index(["cell_x", "cell_y"])["log_lambda"]
    to_3857 = lambda lon, lat: gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(3857)  # noqa: E731
    xmin, ymin, xmax, ymax = AOI_BOUNDS
    west_limit = xmin + 0.65 * (xmax - xmin)

    patrol_rows, obs_rows = [], []
    span = (END - START).total_seconds()
    for p in range(N_PATROLS):
        pid, sid = uid("patrol", p), uid("segment", p)
        leader_name = f"Synthetic Ranger {p % 7 + 1}"
        t0 = START + timedelta(seconds=float(rng.uniform(0, span - 86400)))
        lon, lat = rng.uniform(xmin + 0.01, west_limit), rng.uniform(ymin + 0.01, ymax - 0.01)
        heading = rng.uniform(0, 2 * np.pi)
        lons, lats, times = [lon], [lat], [t0]
        for i in range(1, FIXES_PER_PATROL):
            heading += rng.normal(0, 0.5)
            step = max(rng.normal(330, 60), 50) / 111_000  # ~2 km/h at 10-min fixes
            lon2, lat2 = lon + step * np.cos(heading), lat + step * np.sin(heading)
            if not (xmin < lon2 < west_limit and ymin < lat2 < ymax):
                heading += np.pi
                lon2, lat2 = lon + step * np.cos(heading), lat + step * np.sin(heading)
            lon, lat = lon2, lat2
            lons.append(lon)
            lats.append(lat)
            times.append(t0 + timedelta(minutes=FIX_MINUTES * i))
        t1 = times[-1]

        # detections per track segment, effort-adjusted truth
        pts = to_3857(lons, lats)
        events = []
        for i in range(len(pts) - 1):
            a, b = pts.iloc[i], pts.iloc[i + 1]
            mid = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
            key = (int(np.floor(mid.x / CELL_SIZE) * CELL_SIZE + CELL_SIZE / 2),
                   int(np.floor(mid.y / CELL_SIZE) * CELL_SIZE + CELL_SIZE / 2))
            if key not in cell_lookup.index:
                continue
            km = a.distance(b) / 1000
            if rng.random() < 1 - np.exp(-np.exp(cell_lookup.loc[key]) * km):
                events.append((TARGET_TYPE, (lons[i] + lons[i + 1]) / 2, (lats[i] + lats[i + 1]) / 2, times[i]))
        if rng.random() < 0.3:
            j = int(rng.integers(0, len(lons)))
            events.append((DISTRACTOR_TYPE, lons[j], lats[j], times[j]))

        ev_dicts = []
        for k, (etype, elon, elat, et) in enumerate(events):
            ev = copy.deepcopy(ev_t)
            ev.update(
                id=uid("event", p, k),
                event_type=etype,
                title=etype,
                serial_number=100000 + p * 100 + k,
                patrol_type=PATROL_TYPE,
                state="resolved",
                created_at=et,
                updated_at=et,
            )
            ev["geojson"] = copy.deepcopy(ev_t["geojson"])
            ev["geojson"]["geometry"] = {"coordinates": np.array([elon, elat]), "type": "Point"}
            ev["geojson"]["properties"]["datetime"] = et.isoformat()
            ev_dicts.append(ev)

        seg = copy.deepcopy(seg_t)
        seg.update(
            id=sid,
            patrol_type=PATROL_TYPE,
            start_location={"latitude": lats[0], "longitude": lons[0]},
            end_location={"latitude": lats[-1], "longitude": lons[-1]},
            time_range={"start_time": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "end_time": t1.strftime("%Y-%m-%dT%H:%M:%SZ")},
            events=np.array(ev_dicts, dtype=object) if ev_dicts else np.array([ev_t], dtype=object)[:0],
        )
        leader = copy.deepcopy(seg_t["leader"]) or {}
        leader.update(id=uid("leader", p % 7), name=leader_name)
        for k in ("created_at", "updated_at"):
            if k in leader:
                leader[k] = START.isoformat()
        seg["leader"] = leader
        patrol_rows.append(
            {
                "id": pid,
                "priority": 0,
                "state": "done",
                "objective": "Synthetic patrol",
                "serial_number": 50000 + p,
                "title": f"Synthetic Patrol {p + 1}",
                "files": np.array([], dtype=object),
                "notes": np.array([], dtype=object),
                "patrol_segments": np.array([seg], dtype=object),
                "updates": template["updates"].iloc[0][:0],
            }
        )

        for i, (lo, la, t) in enumerate(zip(lons, lats, times)):
            oid = uid("obs", p, i)
            obs_rows.append(
                {
                    "id": oid,
                    "extra__id": oid,
                    "extra__location": {"latitude": la, "longitude": lo},
                    "extra__created_at": pd.Timestamp(t),
                    "extra__recorded_at": pd.Timestamp(t),
                    "extra__source": uid("source", p % 7),
                    "extra__exclusion_flags": 0,
                    "extra__subject_id": uid("leader", p % 7),
                    "geometry": Point(lo, la),
                    "groupby_col": pid,
                    "fixtime": pd.Timestamp(t),
                    "junk_status": False,
                    "patrol_id": pid,
                    "patrol_title": f"Synthetic Patrol {p + 1}",
                    "patrol_serial_number": 50000 + p,
                    "patrol_start_time": t0.isoformat(),
                    "patrol_end_time": t1.isoformat(),
                    "patrol_type": obs_t["patrol_type"].iloc[0],
                    "patrol_type__value": PATROL_TYPE,
                    "patrol_type__display": "Ecoscope Patrol",
                    "patrol_status": "done",
                    "patrol_subject": leader_name,
                }
            )

    patrols = pd.DataFrame(patrol_rows, columns=template.columns)
    obs = gpd.GeoDataFrame(obs_rows, geometry="geometry", crs="EPSG:4326").set_index("id")
    obs = obs[[c for c in obs_t.columns]]
    return patrols, obs


def main() -> None:
    rng = np.random.default_rng(SEED)
    aoi = build_aoi()
    grid = create_meshgrid(
        aoi=aoi,
        auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=CELL_SIZE),
        crs="EPSG:3857",
        intersecting_only=True,
    )
    gee, dist, truth = build_covariates(rng, grid, aoi)
    patrols, obs = build_patrols(rng, grid, truth)

    write(aoi, "get-spatial-features-group")
    write(gee, "label-grid-with-gee-covariates", also_package=True)
    write(dist, "label-grid-with-distance-covariates", also_package=True)
    write(patrols, "get-patrols-from-combined-params")
    write(obs, "get-patrol-observations-from-patrols-df-and-combined-params", index=True)
    truth.to_parquet(OUT / "truth.parquet", index=False)

    # no_events case: same patrols, target events removed (distractors kept, so the
    # event-type filter still has something to drop)
    no_target = patrols.copy()
    no_target["patrol_segments"] = [
        np.array([{**seg, "events": np.array([e for e in seg["events"] if e["event_type"] != TARGET_TYPE], dtype=object)}
                  for seg in segs], dtype=object)
        for segs in patrols.patrol_segments
    ]
    no_target.to_parquet(OUT / "get-patrols-from-combined-params.no-target-events.parquet", index=False)
    print(f"wrote get-patrols-from-combined-params.no-target-events: {len(no_target)} rows")

    n_target = sum(e["event_type"] == TARGET_TYPE for s in patrols.patrol_segments for e in s[0]["events"])
    n_other = sum(e["event_type"] != TARGET_TYPE for s in patrols.patrol_segments for e in s[0]["events"])
    print(f"grid cells {len(grid)}; patrols {len(patrols)}; fixes {len(obs)}; "
          f"{TARGET_TYPE} {n_target}; distractors {n_other}")


if __name__ == "__main__":
    main()
