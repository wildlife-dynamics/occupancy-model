"""Coefficient forest plot for the occupancy GLM."""


from typing import Annotated, cast

import pandas as pd
from ecoscope.platform.annotations import AnyDataFrame
from pydantic import Field
from wt_registry import register

POINT_COLOR = "#3e35a3"


@register()
def draw_coefficient_plot(
    summary: Annotated[
        AnyDataFrame,
        Field(description="Coefficient summary (from get_occupancy_fit_summary).", exclude=True),
    ],
    title: Annotated[str, Field(description="Chart title.")] = "",
) -> Annotated[str, Field(description="The chart as HTML.")]:
    """Posterior mean and 94% interval per covariate term (standardised scale), with a zero
    line; the intercept is left out so the covariate effects share a readable axis.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    df = cast(pd.DataFrame, summary)
    df = df[df["Term"] != "Intercept"].iloc[::-1]
    fig = go.Figure(
        go.Scatter(
            x=df["Mean"],
            y=df["Term"],
            mode="markers",
            marker={"color": POINT_COLOR, "size": 9},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (df["97%"] - df["Mean"]).tolist(),
                "arrayminus": (df["Mean"] - df["3%"]).tolist(),
                "color": POINT_COLOR,
                "thickness": 2,
            },
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#888888")
    fig.update_layout(
        title={"text": title, "x": 0.5} if title else None,
        xaxis_title="Effect on occurrence (per SD of covariate, link scale)",
        yaxis_title=None,
        showlegend=False,
        hovermode="closest",
        margin={"l": 10, "r": 10, "t": 40 if title else 10, "b": 10},
        template="plotly_white",
    )
    return fig.to_html(full_html=True, include_plotlyjs="cdn")
