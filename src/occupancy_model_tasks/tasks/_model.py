"""Bayesian occupancy GLM (PyMC): fit once, carry the posterior draws, predict every cell.

Units are grid cells. A surveyed cell (patrol effort >= threshold) is a presence when a
target event was recorded on a patrol in it, otherwise an absence. Two links:

- ``cloglog_effort`` (default): P(detect >= 1 event) = 1 - exp(-exp(a + Xb) * effort_km),
  i.e. a cloglog link with log(effort) offset. exp(a + Xb) is the encounter intensity per km
  patrolled, so predictions are "probability of >= 1 detection per N km of patrol".
- ``logit``: plain logistic presence/absence, as in the Feldmeier snare-risk GLM.

Covariates are standardised on the training cells; coefficients get Normal(0, prior_sd)
priors, which replaces stepwise AICc selection. ``fit_occupancy_glm`` returns a plain dict
holding the thinned posterior draws, so downstream tasks never refit (compare
``fit_trend_model``, which must refit because it cannot carry a fitted estimator).
``get_posterior_draws`` turns them into a table for the results download.
"""


import logging
from typing import Annotated, Any, Literal, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from ecoscope.platform.annotations import AdvancedField, AnyDataFrame, AnyGeoDataFrame
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from occupancy_model_tasks.tasks._covariates import (
    DistanceCovariate,
    GeeCovariate,
    covariate_terms,
)

logger = logging.getLogger(__name__)

MAX_STORED_DRAWS = 2000
MIN_CLASS_CELLS = 5
PREDICTION_COLUMNS = ["p_mean", "p_sd", "p_q05", "p_q95"]
SUMMARY_COLUMNS = ["Term", "Mean", "SD", "3%", "97%", "R-hat", "ESS"]

MODEL_TYPE_LABELS = {
    "cloglog_effort": "Effort-adjusted (cloglog + log effort)",
    "logit": "Logistic (presence/absence)",
}


def _labelled_enum(labels: dict[str, str]):
    def apply(schema: dict) -> None:
        schema.pop("enum", None)
        schema.pop("type", None)
        schema["oneOf"] = [{"const": k, "title": v} for k, v in labels.items()]

    return apply


class McmcSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draws: Annotated[int, Field(title="Posterior Draws", description="Draws per chain, after tuning.", ge=50)] = 1000
    tune: Annotated[int, Field(title="Tuning Steps", description="Warm-up steps per chain.", ge=50)] = 1000
    chains: Annotated[int, Field(title="Chains", description="Independent MCMC chains.", ge=1, le=8)] = 2
    random_seed: Annotated[
        int, Field(title="Random Seed", description="Seed for the holdout split and the sampler.")
    ] = 42


# ------------------------------------------------------------------ helpers


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from scipy.stats import rankdata  # type: ignore[import-untyped]

    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _inverse_link(eta: np.ndarray, model_type: str) -> np.ndarray:
    if model_type == "cloglog_effort":
        return -np.expm1(-np.exp(np.clip(eta, -30, 30)))
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def _design_matrix(df: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    cols = []
    for term in fit["terms"]:
        s = fit["scaling"][term["covariate"]]
        z = (df[term["covariate"]].to_numpy(dtype=float) - s["mean"]) / s["sd"]
        cols.append(z**2 if term["quadratic"] else z)
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def _stratified_holdout(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    test = np.zeros(len(y), dtype=bool)
    if fraction <= 0:
        return test
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        n_test = int(round(len(idx) * fraction))
        if n_test and len(idx) - n_test >= 1:
            test[rng.choice(idx, size=n_test, replace=False)] = True
    return test


def _thin(values: np.ndarray, max_draws: int) -> np.ndarray:
    if len(values) <= max_draws:
        return values
    return values[np.linspace(0, len(values) - 1, max_draws).round().astype(int)]


def _counts(df: pd.DataFrame) -> dict[str, int]:
    surveyed = df["surveyed"].astype(bool)
    return {
        "n_cells": int(len(df)),
        "n_surveyed": int(surveyed.sum()),
        "n_presence": int(df.loc[surveyed, "y"].sum()),
        "n_events": int(df["event_count"].sum()),
        "n_events_surveyed": int(df.loc[surveyed, "event_count"].sum()),
    }


def _not_fitted(base: dict[str, Any], message: str) -> dict[str, Any]:
    logger.warning(message)
    return {**base, "status": "not_fitted", "message": message}


# ------------------------------------------------------------------ tasks


@register()
def fit_occupancy_glm(
    design: Annotated[
        AnyGeoDataFrame,
        Field(description="Design frame (from build_occupancy_design).", exclude=True),
    ],
    gee_covariates: Annotated[
        list[GeeCovariate],
        Field(description="Earth Engine covariates (from set_gee_covariates).", exclude=True),
    ],
    distance_covariates: Annotated[
        list[DistanceCovariate],
        Field(description="Distance covariates (from set_distance_covariates).", exclude=True),
    ],
    model_type: Annotated[
        Literal["cloglog_effort", "logit"],
        Field(
            title="Model Type",
            description="Effort-adjusted accounts for how far patrols went in each cell; Logistic treats every surveyed cell alike.",
            json_schema_extra=_labelled_enum(MODEL_TYPE_LABELS),
        ),
    ] = "cloglog_effort",
    prior_sd: Annotated[
        float,
        AdvancedField(
            default=1.0,
            title="Coefficient Prior SD",
            description="Standard deviation of the Normal prior on each standardised coefficient; smaller shrinks harder.",
            gt=0,
        ),
    ] = 1.0,
    holdout_fraction: Annotated[
        float,
        AdvancedField(
            default=0.2,
            title="Holdout Fraction",
            description="Share of surveyed cells held out to measure predictive accuracy (AUC). 0 uses all cells.",
            ge=0,
            lt=0.9,
        ),
    ] = 0.2,
    mcmc: Annotated[
        McmcSettings,
        AdvancedField(default=McmcSettings(), title="MCMC Sampling"),
    ] = McmcSettings(),
) -> dict:
    """Fit the occupancy GLM on surveyed cells with complete covariates.

    Returns a JSON-safe dict: ``status`` ("ok" | "not_fitted"), ``message``, the terms and
    their standardisation, thinned posterior draws, a coefficient summary and metrics. When
    there are too few presences or absences it returns ``status="not_fitted"`` instead of
    raising, so the dashboard can explain why.
    """
    df = pd.DataFrame(cast(gpd.GeoDataFrame, design).drop(columns="geometry", errors="ignore"))
    terms_cfg = covariate_terms(gee_covariates, distance_covariates)
    names = [n for n, _ in terms_cfg]
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError(f"Covariate columns missing from the design frame: {missing}")

    base: dict[str, Any] = {
        "model_type": model_type,
        "model_label": MODEL_TYPE_LABELS[model_type],
        **_counts(df),
        "covariates": names,
        "warnings": [],
    }
    surveyed = df[df["surveyed"].astype(bool)]
    complete = surveyed.dropna(subset=names)
    base["n_dropped_missing_covariates"] = int(len(surveyed) - len(complete))
    if base["n_dropped_missing_covariates"]:
        base["warnings"].append(
            f"{base['n_dropped_missing_covariates']} surveyed cells had missing covariate values and were left out."
        )
    n_pres = int(complete["y"].sum())
    n_abs = int(len(complete) - n_pres)
    if n_pres < MIN_CLASS_CELLS or n_abs < MIN_CLASS_CELLS:
        return _not_fitted(
            base,
            f"Not enough data to fit the model: {n_pres} presence and {n_abs} absence cells "
            f"(at least {MIN_CLASS_CELLS} of each are needed). Try a longer time range, more "
            "patrol or event types, a larger grid cell size or a lower minimum patrol effort.",
        )

    scaling: dict[str, dict[str, float]] = {}
    terms: list[dict[str, Any]] = []
    for name, quad in terms_cfg:
        x = complete[name].to_numpy(dtype=float)
        sd = float(np.std(x))
        if not np.isfinite(sd) or sd == 0:
            base["warnings"].append(f"Covariate '{name}' is constant over the surveyed cells and was left out.")
            continue
        scaling[name] = {"mean": float(np.mean(x)), "sd": sd}
        terms.append({"label": name, "covariate": name, "quadratic": False})
        if quad:
            terms.append({"label": f"{name}^2", "covariate": name, "quadratic": True})
    fit: dict[str, Any] = {**base, "terms": terms, "scaling": scaling}

    X = _design_matrix(complete, fit)
    y = complete["y"].to_numpy(dtype=int)
    log_effort = np.log(np.maximum(complete["effort_km"].to_numpy(dtype=float), 1e-6))
    rng = np.random.default_rng(mcmc.random_seed)
    test = _stratified_holdout(y, holdout_fraction, rng)
    train = ~test

    import pymc as pm  # type: ignore[import-untyped]
    import pytensor.tensor as pt  # type: ignore[import-untyped]

    prevalence = float(np.clip(y[train].mean(), 1e-3, 1 - 1e-3))
    if model_type == "cloglog_effort":
        alpha_mu = float(np.log(-np.log1p(-prevalence)) - log_effort[train].mean())
    else:
        alpha_mu = float(np.log(prevalence / (1 - prevalence)))

    labels = [t["label"] for t in terms]
    with pm.Model(coords={"term": labels}):
        alpha = pm.Normal("alpha", mu=alpha_mu, sigma=2.5)
        eta = alpha
        if labels:
            beta = pm.Normal("beta", mu=0.0, sigma=prior_sd, dims="term")
            eta = eta + pt.dot(X[train], beta)
        if model_type == "cloglog_effort":
            p = 1.0 - pt.exp(-pt.exp(eta + log_effort[train]))
        else:
            p = pm.math.invlogit(eta)
        pm.Bernoulli("y", p=pt.clip(p, 1e-9, 1 - 1e-9), observed=y[train])
        idata = pm.sample(
            draws=mcmc.draws,
            tune=mcmc.tune,
            chains=mcmc.chains,
            cores=1,
            random_seed=mcmc.random_seed,
            target_accept=0.9,
            progressbar=False,
            compute_convergence_checks=False,
        )
        pm.compute_log_likelihood(idata, progressbar=False)

    post = idata["posterior"]
    alpha_draws = np.asarray(post["alpha"]).reshape(-1)
    beta_draws = np.asarray(post["beta"]).reshape(-1, len(labels)) if labels else np.zeros((len(alpha_draws), 0))
    chain_idx = np.repeat(np.arange(mcmc.chains), mcmc.draws)
    draw_idx = np.tile(np.arange(mcmc.draws), mcmc.chains)

    rhat, ess = _diagnostics(idata, labels)
    summary = []
    for j, lab in enumerate(["Intercept"] + labels):
        d = alpha_draws if j == 0 else beta_draws[:, j - 1]
        key = "alpha" if j == 0 else lab
        summary.append(
            {
                "Term": lab,
                "Mean": float(np.mean(d)),
                "SD": float(np.std(d)),
                "3%": float(np.quantile(d, 0.03)),
                "97%": float(np.quantile(d, 0.97)),
                "R-hat": rhat.get(key),
                "ESS": ess.get(key),
            }
        )

    keep = _thin(np.arange(len(alpha_draws)), MAX_STORED_DRAWS)
    fit.update(
        {
            "status": "ok",
            "message": "",
            "alpha_draws": alpha_draws[keep].tolist(),
            "beta_draws": beta_draws[keep].tolist(),
            "draw_chain": chain_idx[keep].tolist(),
            "draw_index": draw_idx[keep].tolist(),
            "summary": summary,
        }
    )

    def _p(rows: np.ndarray) -> np.ndarray:
        eta_d = fit_eta(X[rows], fit, keep=None)
        if model_type == "cloglog_effort":
            eta_d = eta_d + log_effort[rows][:, None]
        return _inverse_link(eta_d, model_type).mean(axis=1)

    auc_rows = test if test.any() else train
    fit["metrics"] = {
        "auc": _auc(y[auc_rows], _p(auc_rows)),
        "auc_basis": "holdout" if test.any() else "in-sample",
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "loo_elpd": _loo(idata),
        "max_r_hat": max((v for v in rhat.values() if v is not None), default=None),
        "min_ess": min((v for v in ess.values() if v is not None), default=None),
        "divergences": _divergences(idata),
    }
    if fit["metrics"]["max_r_hat"] is not None and fit["metrics"]["max_r_hat"] > 1.05:
        fit["warnings"].append("Some R-hat values exceed 1.05: chains have not converged; increase tuning steps or draws.")
    if fit["metrics"]["divergences"]:
        fit["warnings"].append(f"{fit['metrics']['divergences']} divergent transitions during sampling.")

    return fit


def fit_eta(X: np.ndarray, fit: dict[str, Any], keep: np.ndarray | None = None) -> np.ndarray:
    """Linear predictor (cells x draws) without any offset."""
    alpha = np.asarray(fit["alpha_draws"], dtype=float)
    beta = np.asarray(fit["beta_draws"], dtype=float).reshape(len(alpha), -1)
    if keep is not None:
        alpha, beta = alpha[keep], beta[keep]
    return alpha[None, :] + X @ beta.T


def _diagnostics(idata, labels: list[str]) -> tuple[dict[str, float | None], dict[str, float | None]]:
    import arviz as az  # type: ignore[import-untyped]

    out_r: dict[str, float | None] = {}
    out_e: dict[str, float | None] = {}
    for fn, out in ((az.rhat, out_r), (az.ess, out_e)):
        try:
            res = fn(idata, var_names=["alpha", "beta"] if labels else ["alpha"])
            res = res.to_dataset() if hasattr(res, "to_dataset") and not hasattr(res, "data_vars") else res
            out["alpha"] = float(np.asarray(res["alpha"]))
            if labels:
                vals = np.asarray(res["beta"]).reshape(-1)
                out.update({lab: float(v) for lab, v in zip(labels, vals)})
        except Exception as e:  # noqa: BLE001 - diagnostics are best-effort
            logger.warning("Could not compute %s: %s", getattr(fn, "__name__", fn), e)
    return out_r, out_e


def _loo(idata) -> float | None:
    import arviz as az  # type: ignore[import-untyped]

    try:
        res = az.loo(idata)
        for attr in ("elpd", "elpd_loo"):
            if hasattr(res, attr):
                return float(getattr(res, attr))
            if hasattr(res, "__getitem__"):
                try:
                    return float(res[attr])
                except Exception:  # noqa: BLE001
                    pass
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not compute LOO: %s", e)
    return None


def _divergences(idata) -> int | None:
    try:
        return int(np.asarray(idata["sample_stats"]["diverging"]).sum())
    except Exception:  # noqa: BLE001
        return None


@register()
def predict_occupancy_surface(
    fit: Annotated[dict, Field(description="Result of fit_occupancy_glm.", exclude=True)],
    design: Annotated[
        AnyGeoDataFrame,
        Field(description="Design frame (from build_occupancy_design).", exclude=True),
    ],
    reference_effort_km: Annotated[
        float,
        Field(
            title="Reference Patrol Effort (km)",
            description="Effort-adjusted model only: predictions are the probability of at least one detection per this distance of patrolling.",
            gt=0,
        ),
    ] = 1.0,
    n_draws: Annotated[
        int,
        Field(description="Posterior draws used for the prediction summaries.", gt=0),
    ] = 500,
) -> AnyGeoDataFrame:
    """Posterior mean, SD and 5%/95% quantiles of occurrence probability for every cell
    with complete covariates. Returns an empty frame when the model was not fitted, so the
    downstream map/COG/zone tasks skip.
    """
    gdf = cast(gpd.GeoDataFrame, design).copy()
    for c in PREDICTION_COLUMNS:
        gdf[c] = np.nan
    if fit.get("status") != "ok":
        return cast(AnyGeoDataFrame, gdf.iloc[0:0])

    names = sorted({t["covariate"] for t in fit["terms"]})
    ok = gdf[names].notna().all(axis=1).to_numpy() if names else np.ones(len(gdf), dtype=bool)
    n_total = len(fit["alpha_draws"])
    keep = _thin(np.arange(n_total), n_draws)
    X_all = _design_matrix(pd.DataFrame(gdf.loc[ok]), fit)
    offset = np.log(reference_effort_km) if fit["model_type"] == "cloglog_effort" else 0.0
    results = np.full((int(ok.sum()), 4), np.nan)
    for start in range(0, len(X_all), 5000):
        p = _inverse_link(fit_eta(X_all[start : start + 5000], fit, keep) + offset, fit["model_type"])
        results[start : start + 5000] = np.column_stack(
            [p.mean(axis=1), p.std(axis=1), np.quantile(p, 0.05, axis=1), np.quantile(p, 0.95, axis=1)]
        )
    gdf.loc[ok, PREDICTION_COLUMNS] = results
    return cast(AnyGeoDataFrame, gdf)


@register()
def get_occupancy_fit_summary(
    fit: Annotated[dict, Field(description="Result of fit_occupancy_glm.", exclude=True)],
) -> AnyDataFrame:
    """Posterior summary per term (standardised scale): mean, SD, 94% interval, R-hat, ESS."""
    if fit.get("status") != "ok":
        return cast(AnyDataFrame, pd.DataFrame(columns=SUMMARY_COLUMNS))
    df = pd.DataFrame(fit["summary"], columns=SUMMARY_COLUMNS)
    return cast(AnyDataFrame, df.round({"Mean": 3, "SD": 3, "3%": 3, "97%": 3, "R-hat": 3, "ESS": 0}))


@register()
def get_posterior_draws(
    fit: Annotated[dict, Field(description="Result of fit_occupancy_glm.", exclude=True)],
) -> AnyDataFrame:
    """Posterior draws (standardised scale) as a table: chain, draw, Intercept and one column
    per term -- every draw up to 2,000, evenly thinned beyond that. Empty when not fitted.
    """
    if fit.get("status") != "ok":
        return cast(AnyDataFrame, pd.DataFrame())
    labels = [t["label"] for t in fit["terms"]]
    alpha = np.asarray(fit["alpha_draws"], dtype=float)
    df = pd.DataFrame(np.asarray(fit["beta_draws"], dtype=float).reshape(len(alpha), -1), columns=labels)
    df.insert(0, "Intercept", alpha)
    df.insert(0, "draw", fit["draw_index"])
    df.insert(0, "chain", fit["draw_chain"])
    return cast(AnyDataFrame, df)


@register()
def extract_fit_metric(
    fit: Annotated[dict, Field(description="Result of fit_occupancy_glm.", exclude=True)],
    metric: Annotated[
        Literal["n_events", "n_events_surveyed", "n_cells", "n_surveyed", "n_presence", "auc", "max_r_hat"],
        Field(description="Which count or metric to return."),
    ],
) -> float | None:
    """A single count or metric from the fit, for a stat widget (None if unavailable)."""
    if metric in ("auc", "max_r_hat"):
        value = (fit.get("metrics") or {}).get(metric)
    else:
        value = fit.get(metric)
    return None if value is None else float(value)


@register()
def summarize_occupancy_run(
    fit: Annotated[dict, Field(description="Result of fit_occupancy_glm.", exclude=True)],
    event_type_check: Annotated[
        dict | SkipJsonSchema[None],
        Field(description="Result of check_target_event_types.", exclude=True),
    ] = None,
) -> str:
    """Markdown summary of the data, the model and any warnings, for a text widget."""
    from occupancy_model_tasks.tasks._checks import event_type_warnings

    lines = [
        f"**Model:** {fit.get('model_label', '')}",
        "",
        f"- Grid cells: {fit.get('n_cells', 0):,}; surveyed: {fit.get('n_surveyed', 0):,}; "
        f"with a target event: {fit.get('n_presence', 0):,}",
        f"- Target events in the study area: {fit.get('n_events', 0):,} "
        f"({fit.get('n_events_surveyed', 0):,} in surveyed cells)",
    ]
    if fit.get("status") != "ok":
        lines += ["", f"**No model was fitted.** {fit.get('message', '')}"]
    else:
        m = fit.get("metrics", {})
        auc = m.get("auc")
        auc_txt = f"{auc:.3f} ({m.get('auc_basis')}, {m.get('n_test') or m.get('n_train')} cells)" if auc is not None else "n/a"
        rhat = m.get("max_r_hat")
        lines += [
            f"- Covariates: {', '.join(t['label'] for t in fit.get('terms', [])) or 'none (intercept only)'}",
            f"- AUC: {auc_txt}",
            f"- Max R-hat: {rhat:.3f}" if rhat is not None else "- Max R-hat: n/a",
        ]
    for w in event_type_warnings(event_type_check) + list(fit.get("warnings", [])):
        lines.append(f"- ⚠ {w}")
    return "\n".join(lines)
