"""Figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from .scaling import master_curve

INK = "#1c252d"
MUTED = "#68737d"
BLUE = "#315f9e"
ORANGE = "#a3621b"
GREEN = "#3c7654"
PURPLE = "#7562a6"
LINE = "#d4d8dc"


def _prepare_path(path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _style() -> dict[str, str]:
    palette = {
        "background": "#ffffff",
        "ink": INK,
        "muted": MUTED,
        "line": LINE,
        "blue": BLUE,
        "orange": ORANGE,
        "green": GREEN,
        "purple": PURPLE,
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.2,
            "axes.edgecolor": palette["line"],
            "axes.labelcolor": palette["ink"],
            "axes.titlecolor": palette["ink"],
            "xtick.color": palette["muted"],
            "ytick.color": palette["muted"],
            "text.color": palette["ink"],
            "legend.frameon": False,
            "legend.fontsize": 8.2,
            "figure.facecolor": palette["background"],
            "axes.facecolor": palette["background"],
            "savefig.facecolor": palette["background"],
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return palette


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)


def plot_aggregate_impact(curve: pd.DataFrame, path: Path | str) -> Path:
    """Plot conditional mean impact at four event-time horizons."""

    palette = _style()
    horizons = [5, 10, 20, 50]
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), layout="constrained")
    for panel, (axis, horizon) in enumerate(zip(axes.ravel(), horizons, strict=True)):
        part = curve.loc[curve["horizon"].eq(horizon)]
        axis.plot(
            part["volume_imbalance"],
            part["impact"] * 100,
            marker="o",
            markersize=2.4,
            markerfacecolor=palette["background"],
            markeredgewidth=0.7,
            color=palette["blue"],
            linewidth=1.0,
        )
        axis.axhline(0, color=palette["line"], linewidth=0.7)
        axis.axvline(0, color=palette["line"], linewidth=0.7)
        axis.set_title(f"({chr(97 + panel)})  T = {horizon}", loc="left")
        axis.set_xlabel("Signed volume (shares)")
        axis.set_ylabel("Mean mid-price change (cents)")
        _clean_axis(axis)
    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_holdout_calibration(
    calibration: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path | str,
) -> Path:
    """Plot observed against fitted holdout impact by prediction quantile."""

    palette = _style()
    models = ["OLS volume transforms", "OLS volume and count"]
    colors = [palette["blue"], palette["orange"]]
    labels = ["Volume transforms", "Plus raw signed count"]
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), layout="constrained")
    combined = calibration.loc[calibration["model"].isin(models)]
    lower = float(
        min(combined["mean_fitted_cents"].min(), combined["mean_realised_cents"].min())
    )
    upper = float(
        max(combined["mean_fitted_cents"].max(), combined["mean_realised_cents"].max())
    )
    padding = 0.05 * (upper - lower)
    limits = (lower - padding, upper + padding)

    for panel, (axis, model, color, label) in enumerate(
        zip(axes, models, colors, labels, strict=True)
    ):
        part = calibration.loc[calibration["model"].eq(model)]
        score = float(metrics.loc[metrics["model"].eq(model), "test_r_squared"].iloc[0])
        axis.plot(limits, limits, color=palette["line"], linewidth=1.0, zorder=1)
        axis.errorbar(
            part["mean_fitted_cents"],
            part["mean_realised_cents"],
            yerr=1.96 * part["standard_error_cents"],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=3.2,
            elinewidth=0.7,
            capsize=1.5,
            alpha=0.95,
            zorder=2,
        )
        axis.set(xlim=limits, ylim=limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"({chr(97 + panel)})  {label}", loc="left")
        axis.text(
            0.05,
            0.92,
            rf"holdout $R^2$ = {score:.3f}",
            transform=axis.transAxes,
            color=palette["muted"],
        )
        axis.set_xlabel("Mean fitted impact (cents)")
        axis.set_ylabel("Mean realised impact (cents)")
        _clean_axis(axis)
    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_count_residuals(
    curve: pd.DataFrame,
    path: Path | str,
) -> Path:
    """Plot holdout volume-model residuals against signed order count."""

    palette = _style()
    figure, axis = plt.subplots(figsize=(6.6, 3.9), layout="constrained")
    x = curve["order_flow_imbalance"].to_numpy(dtype=float)
    y = curve["mean_residual_cents"].to_numpy(dtype=float)
    interval = 1.96 * curve["standard_error_cents"].to_numpy(dtype=float)
    axis.axhline(0, color=palette["line"], linewidth=1.0)
    axis.fill_between(
        x,
        y - interval,
        y + interval,
        color=palette["orange"],
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(
        x,
        y,
        color=palette["orange"],
        marker="o",
        markersize=4.0,
        linewidth=1.5,
    )
    axis.set_xlabel("Signed count imbalance in a 10-order window")
    axis.set_ylabel("Mean residual impact (cents)")
    axis.set_title("Holdout residual after fitting signed-volume transforms", loc="left")
    axis.text(
        0.01,
        0.04,
        "Points are date-level means; band is 1.96 standard errors across dates.",
        transform=axis.transAxes,
        color=palette["muted"],
        fontsize=7.8,
    )
    axis.set_xticks(x)
    _clean_axis(axis)
    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_horizon_robustness(metrics: pd.DataFrame, path: Path | str) -> Path:
    """Plot holdout model scores over five event-time horizons."""

    palette = _style()
    selected = [
        ("OLS signed volume", "Signed volume", palette["blue"]),
        ("OLS volume transforms", "Volume transforms", palette["green"]),
        ("OLS volume and count", "+ raw signed count", palette["orange"]),
        ("OLS count transforms", "+ count transforms", palette["purple"]),
    ]
    figure, axis = plt.subplots(figsize=(7.15, 3.45), layout="constrained")
    for model, label, color in selected:
        part = metrics.loc[metrics["model"].eq(model)].sort_values("horizon")
        axis.plot(
            part["horizon"],
            part["test_r_squared"],
            marker="o",
            markersize=3.8,
            linewidth=1.25,
            color=color,
            label=label,
        )
    axis.set_xscale("log")
    horizons = sorted(metrics["horizon"].unique())
    axis.set_xticks(horizons, [str(value) for value in horizons])
    axis.set_xlabel("Window length T (market orders)")
    axis.set_ylabel(r"Full-holdout $R^2$")
    axis.legend(ncol=2, loc="lower right")
    axis.axhline(0, color=palette["line"], linewidth=0.8)
    _clean_axis(axis)
    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_walk_forward(results: pd.DataFrame, path: Path | str) -> Path:
    """Plot volume-only and raw-count R-squared across expanding test folds."""

    palette = _style()
    figure, axis = plt.subplots(figsize=(7.15, 3.35), layout="constrained")
    for model, label, color in [
        ("OLS volume transforms", "Volume transforms", palette["green"]),
        ("OLS volume and count", "Plus raw signed count", palette["orange"]),
    ]:
        part = results.loc[results["model"].eq(model)].sort_values("fold")
        axis.plot(
            part["fold"],
            part["test_r_squared"],
            marker="o",
            markersize=4.2,
            linewidth=1.35,
            color=color,
            label=label,
        )
    labels = (
        results.loc[results["model"].eq("OLS volume transforms")]
        .sort_values("fold")["test_first_date"]
        .str.slice(5, 10)
        .tolist()
    )
    axis.set_xticks(range(1, len(labels) + 1), labels)
    axis.set_xlabel("First date of each test block in 2019")
    axis.set_ylabel(r"Test-block $R^2$")
    axis.legend(loc="lower right")
    axis.axhline(0, color=palette["line"], linewidth=0.8)
    _clean_axis(axis)
    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_scaling_fits(
    scales: pd.DataFrame,
    fits: dict[str, dict[str, float]],
    path: Path | str,
) -> Path:
    """Plot scale fits with their log-space residuals."""

    palette = _style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 4.55),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.05], "hspace": 0.08, "wspace": 0.30},
    )
    specifications = [
        ("Q_N", "width", palette["blue"], r"Width scale $Q_N$"),
        ("R_N", "height", palette["green"], r"Height scale $R_N$"),
    ]
    horizon_grid = np.geomspace(scales["horizon"].min(), scales["horizon"].max(), 300)

    for index, (column, fit_key, color, label) in enumerate(specifications):
        main = axes[0, index]
        residual = axes[1, index]
        fit = fits[fit_key]
        predicted = fit["prefactor"] * scales["horizon"] ** fit["exponent"]
        log_residual = np.log(scales[column]) - np.log(predicted)
        main.loglog(scales["horizon"], scales[column], "o", color=color, markersize=3.5)
        main.loglog(
            horizon_grid,
            fit["prefactor"] * horizon_grid ** fit["exponent"],
            color=color,
            linewidth=1.4,
        )
        main.text(
            0.04,
            0.95,
            (
                rf"$b$ = {fit['exponent']:.3f} $\pm$ {fit['standard_error']:.3f}" "\n"
                rf"log-fit $R^2$ = {fit['r_squared']:.4f}"
            ),
            transform=main.transAxes,
            va="top",
            fontsize=8.2,
        )
        main.set_ylabel(label)
        _clean_axis(main)
        residual.axhline(0, color=palette["muted"], linewidth=0.7)
        residual.semilogx(
            scales["horizon"],
            log_residual,
            marker="o",
            color=color,
            markersize=2.6,
            linewidth=0.8,
        )
        residual.set_xlabel("Horizon N (transactions)")
        residual.set_ylabel("Log residual")
        _clean_axis(residual)

    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_queue_imbalance_forecast(
    calibration: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path | str,
) -> Path:
    """Plot chronological next-move calibration by queue imbalance."""

    palette = _style()
    sample = "best_quote_updates"
    panels = [
        ("all_spreads", "All spreads"),
        ("one_tick", "One-tick spread"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), layout="constrained")
    for panel, (axis, (bucket, title)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        curve = calibration.loc[
            calibration["sample"].eq(sample)
            & calibration["spread_bucket"].eq(bucket)
        ].sort_values("bin_center")
        score = metrics.loc[
            metrics["sample"].eq(sample)
            & metrics["spread_bucket"].eq(bucket)
        ].iloc[0]
        visible = curve.loc[curve["observations"].ge(100)]
        axis.axhline(0.5, color=palette["line"], linewidth=0.9)
        axis.plot(
            curve["bin_center"],
            curve["model_probability"],
            color=palette["orange"],
            linewidth=1.5,
            label="Train-fitted logistic model",
        )
        axis.scatter(
            visible["bin_center"],
            visible["empirical_up_probability"],
            s=10,
            facecolor=palette["background"],
            edgecolor=palette["blue"],
            linewidth=0.8,
            label="Late-2019 holdout",
            zorder=3,
        )
        axis.set_xlim(-1, 1)
        axis.set_ylim(0.3, 0.7)
        axis.set_title(f"({chr(97 + panel)})  {title}", loc="left")
        axis.set_xlabel("Level-1 queue imbalance")
        axis.set_ylabel("Probability next mid-price move is up")
        axis.text(
            0.04,
            0.94,
            (
                f"AUC = {score['model_roc_auc']:.3f}\n"
                f"Brier reduction = {100 * score['relative_brier_reduction']:.1f}%"
            ),
            transform=axis.transAxes,
            va="top",
            color=palette["muted"],
            fontsize=8.2,
        )
        _clean_axis(axis)
    axes[1].legend(loc="lower right")
    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_order_flow_signal_ablation(
    comparisons: pd.DataFrame,
    path: Path | str,
) -> Path:
    """Plot late-date Brier reductions for linear order-flow signals."""

    palette = _style()
    panels = [
        ("all_spreads", "All spreads"),
        ("one_tick", "One-tick spread"),
    ]
    models = [
        ("queue", "Queue imbalance"),
        ("ofi", "Cumulative OFI"),
        ("queue_and_ofi", "Queue + OFI"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), layout="constrained")
    for panel, (axis, (bucket, title)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        part = comparisons.loc[
            comparisons["sample"].eq("best_quote_updates")
            & comparisons["spread_bucket"].eq(bucket)
            & comparisons["reference"].eq("intercept")
        ].set_index("challenger")
        y_positions = np.arange(len(models))[::-1]
        for y_position, (model, label) in zip(
            y_positions,
            models,
            strict=True,
        ):
            row = part.loc[model]
            point = 100 * row["relative_brier_reduction"]
            lower = 100 * row["relative_brier_reduction_lower_95"]
            upper = 100 * row["relative_brier_reduction_upper_95"]
            axis.errorbar(
                point,
                y_position,
                xerr=np.array([[point - lower], [upper - point]]),
                fmt="o",
                color=(
                    palette["blue"]
                    if model != "queue_and_ofi"
                    else palette["orange"]
                ),
                capsize=2.5,
                markersize=4.5,
                linewidth=1.0,
                label=label,
            )
        axis.axvline(0, color=palette["line"], linewidth=0.9)
        axis.set_yticks(y_positions, [label for _, label in models])
        axis.set_title(f"({chr(97 + panel)})  {title}", loc="left")
        axis.set_xlabel("Brier reduction vs train-frequency baseline (%)")
        _clean_axis(axis)
    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(figure)
    return output


def plot_scaling_collapse(
    collapsed: pd.DataFrame,
    shape: dict[str, float],
    path: Path | str,
) -> Path:
    """Plot rescaled conditional curves and the estimated shared shape."""

    palette = _style()
    figure, axis = plt.subplots(figsize=(7.15, 4.15), layout="constrained")
    horizons = np.sort(collapsed["horizon"].unique())
    normalizer = LogNorm(vmin=float(horizons.min()), vmax=float(horizons.max()))
    color_map = plt.get_cmap("viridis")
    for horizon, part in collapsed.groupby("horizon", sort=True, observed=True):
        axis.plot(
            part["x_scaled"],
            part["y_scaled"],
            color=color_map(normalizer(float(horizon))),
            linewidth=0.75,
            alpha=0.58,
        )
    x_limit = float(collapsed["x_scaled"].abs().quantile(0.99))
    x_grid = np.linspace(-x_limit, x_limit, 500)
    axis.plot(
        x_grid,
        master_curve(x_grid, float(shape["alpha"]), float(shape["beta"])),
        color=palette["orange"],
        linewidth=1.8,
        label="Estimated shared curve",
    )
    axis.axhline(0, color=palette["line"], linewidth=0.7)
    axis.axvline(0, color=palette["line"], linewidth=0.7)
    axis.set_xlim(-x_limit, x_limit)
    axis.set_xlabel(r"Rescaled imbalance $Q / Q_N$")
    axis.set_ylabel(r"Rescaled mean impact $R / R_N$")
    axis.legend(loc="lower right")
    _clean_axis(axis)
    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
    colorbar = figure.colorbar(scalar, ax=axis, pad=0.02)
    colorbar.set_label("Horizon N")
    output = _prepare_path(path)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output
