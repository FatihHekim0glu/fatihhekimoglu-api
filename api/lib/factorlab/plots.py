"""Plotly chart helpers for factor model attribution, exposures, and loadings."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def attribution_waterfall(
    contributions: dict[str, float],
    total: float,
    title: str = "Return Attribution",
) -> go.Figure:
    """Build a waterfall chart of return contributions with a final total bar."""
    labels = list(contributions.keys())
    values = list(contributions.values())
    measures = ["relative"] * len(labels) + ["total"]
    x = labels + ["Total"]
    y = values + [total]
    text = [f"{v:+.2%}" for v in values] + [f"{total:+.2%}"]
    fig = go.Figure(
        go.Waterfall(
            x=x,
            y=y,
            measure=measures,
            text=text,
            textposition="outside",
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={"marker": {"color": "#1f77b4"}},
            connector={"line": {"color": "rgba(120, 120, 120, 0.5)"}},
        )
    )
    fig.update_layout(title=title, showlegend=False)
    fig.update_yaxes(tickformat=".1%")
    return fig


def rolling_beta_chart(
    rolling_df: pd.DataFrame,
    factor: str,
    ci: float = 2.0,
) -> go.Figure:
    """Plot a rolling beta line with a shaded +/- ci*SE band for one factor."""
    sub = rolling_df[rolling_df["factor"] == factor].copy()
    sub = sub.sort_index()
    beta = sub["beta"]
    se = sub["se"]
    upper = beta + ci * se
    lower = beta - ci * se
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=upper.index,
            y=upper.values,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
            name="upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lower.index,
            y=lower.values,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            showlegend=False,
            hoverinfo="skip",
            name="lower",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=beta.index,
            y=beta.values,
            mode="lines",
            line={"color": "#1f77b4", "width": 2},
            name=f"{factor} beta",
        )
    )
    if factor == "Mkt-RF":
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Rolling Beta: {factor}",
        xaxis_title="Date",
        yaxis_title="Beta",
    )
    return fig


def factor_heatmap(
    loadings: pd.DataFrame,
    title: str = "Factor Loadings",
) -> go.Figure:
    """Render a diverging-palette heatmap of factor loadings centered at zero."""
    fig = px.imshow(
        loadings,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto",
        title=title,
    )
    fig.update_xaxes(side="top")
    return fig


def factor_cumulative_chart(
    factors: pd.DataFrame,
    title: str = "Factor Cumulative Returns",
) -> go.Figure:
    """Plot cumulative compounded returns for each factor column."""
    cum = (1 + factors).cumprod()
    fig = go.Figure()
    for col in cum.columns:
        fig.add_trace(
            go.Scatter(x=cum.index, y=cum[col].values, mode="lines", name=str(col))
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
    )
    return fig
