"""Analysis pipeline for the prepared TSLA transaction tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .impact import (
    aggregate_prediction_windows,
    estimate_kyle_lambdas,
    fit_kyle_decay,
    quantile_impact_curve,
)
from .models import (
    apply_training_sigma_filter,
    calibration_curve,
    day_cluster_bootstrap_improvement,
    evaluate_expanding_splits,
    evaluate_impact_models,
    model_specifications,
    prepare_impact_dataset,
    regression_scores,
    residual_curve_by_count,
    temporal_train_test_split,
)
from .plots import (
    plot_aggregate_impact,
    plot_count_residuals,
    plot_holdout_calibration,
    plot_horizon_robustness,
    plot_scaling_collapse,
    plot_scaling_fits,
    plot_walk_forward,
)
from .scaling import (
    DEFAULT_HORIZONS,
    collapse_curve,
    compute_scaling_windows,
    fit_scaling_laws,
    fit_variance_scaling_statistics,
    fit_volume_curve_scales,
    quantile_curve,
)

VISIBLE_COLUMNS = [
    "date",
    "seconds",
    "first_event_row",
    "size",
    "trade_sign",
    "mid_price_before",
    "spread_before",
]
SCALING_COLUMNS = [
    "date",
    "seconds",
    "first_event_row",
    "size",
    "trade_sign",
    "mid_price_before",
]
BASELINE_MODEL = "OLS signed volume"
VOLUME_MODEL = "OLS volume transforms"
COUNT_MODEL = "OLS volume and count"
AUGMENTED_MODEL = "OLS count transforms"


def _number(value: np.floating | np.integer | float | int) -> float | int:
    return value.item() if isinstance(value, np.generic) else value


def _score_record(row: pd.Series) -> dict[str, float | str]:
    return {
        "name": str(row["model"]),
        "test_mse_cents_squared": _number(row["test_mse_cents_squared"]),
        "test_rmse_cents": _number(row["test_rmse_cents"]),
        "test_mae_cents": _number(row["test_mae_cents"]),
        "test_r_squared": _number(row["test_r_squared"]),
        "test_sign_accuracy": _number(row["test_sign_accuracy"]),
    }


def _selected_specifications(names: set[str]) -> list[dict[str, object]]:
    return [
        specification
        for specification in model_specifications()
        if specification["name"] in names
    ]


def _monthly_scores(
    test: pd.DataFrame,
    baseline_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
) -> pd.DataFrame:
    frame = test[["date", "impact_cents"]].copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m")
    frame["baseline_prediction"] = baseline_prediction
    frame["augmented_prediction"] = augmented_prediction
    rows: list[dict[str, object]] = []
    for month, part in frame.groupby("month", sort=True, observed=True):
        y = part["impact_cents"].to_numpy(dtype=float)
        for model, column in [
            (BASELINE_MODEL, "baseline_prediction"),
            (AUGMENTED_MODEL, "augmented_prediction"),
        ]:
            rows.append(
                {
                    "month": month,
                    "model": model,
                    "observations": len(part),
                    **regression_scores(y, part[column].to_numpy(dtype=float)),
                }
            )
    return pd.DataFrame(rows)


def _horizon_scores(
    windows: pd.DataFrame,
    test_start_date: str | None = None,
) -> pd.DataFrame:
    specifications = _selected_specifications(
        {BASELINE_MODEL, VOLUME_MODEL, COUNT_MODEL, AUGMENTED_MODEL}
    )
    frames: list[pd.DataFrame] = []
    for horizon in sorted(windows["horizon"].unique()):
        data = prepare_impact_dataset(windows, horizon=int(horizon))
        train, test = temporal_train_test_split(
            data,
            test_date_fraction=0.20,
            test_start_date=test_start_date,
        )
        metrics, _ = evaluate_impact_models(train, test, specifications)
        metrics.insert(0, "horizon", int(horizon))
        metrics.insert(1, "train_observations", len(train))
        metrics.insert(2, "test_observations", len(test))
        metrics.insert(3, "test_first_date", str(test["date"].min()))
        metrics.insert(4, "test_last_date", str(test["date"].max()))
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def _sample_summary(visible: pd.DataFrame, windows: pd.DataFrame) -> dict[str, float | int]:
    daily_orders = visible.groupby("date", sort=True, observed=True).size()
    ten_order = windows.loc[windows["horizon"].eq(10)]
    return {
        "visible_market_orders": len(visible),
        "market_orders_per_session_median": float(daily_orders.median()),
        "market_orders_per_session_p10": float(daily_orders.quantile(0.10)),
        "market_orders_per_session_p90": float(daily_orders.quantile(0.90)),
        "order_size_shares_median": float(visible["size"].median()),
        "order_size_shares_p90": float(visible["size"].quantile(0.90)),
        "order_size_shares_p99": float(visible["size"].quantile(0.99)),
        "spread_cents_median": float(100 * visible["spread_before"].median()),
        "spread_cents_p90": float(100 * visible["spread_before"].quantile(0.90)),
        "buyer_initiated_fraction": float(visible["trade_sign"].eq(1).mean()),
        "ten_order_windows": len(ten_order),
        "ten_order_impact_cents_standard_deviation": float(100 * ten_order["impact"].std()),
        "ten_order_zero_impact_fraction": float(ten_order["impact"].eq(0).mean()),
    }


def run_analysis(
    visible_path: Path | str,
    scaling_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str,
    test_start_date: str | None = None,
) -> dict[str, object]:
    """Run the study and write aggregate tables and report figures."""

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    visible = pd.read_parquet(visible_path, columns=VISIBLE_COLUMNS)
    windows = aggregate_prediction_windows(
        visible,
        horizons=(5, 10, 20, 50, 100),
    )
    sample_summary = _sample_summary(visible, windows)
    pd.DataFrame(
        [
            {"quantity": key, "value": value}
            for key, value in sample_summary.items()
        ]
    ).to_csv(output / "sample_summary.csv", index=False)
    aggregate_curve = quantile_impact_curve(
        windows.loc[windows["horizon"].isin([5, 10, 20, 50])],
        quantiles=51,
    )
    aggregate_curve.to_csv(output / "aggregate_impact_curve.csv", index=False)

    impact_data = prepare_impact_dataset(windows, horizon=10)
    train, test = temporal_train_test_split(
        impact_data,
        test_date_fraction=0.20,
        test_start_date=test_start_date,
    )
    model_metrics, fitted = evaluate_impact_models(train, test)
    metric_columns = [
        "model",
        "features",
        "test_mse_cents_squared",
        "test_rmse_cents",
        "test_mae_cents",
        "test_r_squared",
        "test_sign_accuracy",
    ]
    model_metrics[metric_columns].to_csv(output / "model_metrics.csv", index=False)

    basis_point_data = impact_data.copy()
    basis_point_data["impact_cents"] = basis_point_data["impact_bps"]
    basis_point_train, basis_point_test = temporal_train_test_split(
        basis_point_data,
        test_date_fraction=0.20,
        test_start_date=test_start_date,
    )
    basis_point_metrics, _ = evaluate_impact_models(
        basis_point_train,
        basis_point_test,
        _selected_specifications({BASELINE_MODEL, AUGMENTED_MODEL}),
    )
    basis_point_metrics.rename(
        columns={
            "test_mse_cents_squared": "test_mse_bps_squared",
            "test_rmse_cents": "test_rmse_bps",
            "test_mae_cents": "test_mae_bps",
        }
    )[
        [
            "model",
            "features",
            "test_mse_bps_squared",
            "test_rmse_bps",
            "test_mae_bps",
            "test_r_squared",
            "test_sign_accuracy",
        ]
    ].to_csv(output / "basis_point_robustness.csv", index=False)

    baseline_features = next(
        specification["features"]
        for specification in model_specifications()
        if specification["name"] == BASELINE_MODEL
    )
    volume_features = next(
        specification["features"]
        for specification in model_specifications()
        if specification["name"] == VOLUME_MODEL
    )
    augmented_features = next(
        specification["features"]
        for specification in model_specifications()
        if specification["name"] == AUGMENTED_MODEL
    )
    baseline_prediction = fitted[BASELINE_MODEL].predict(
        test[baseline_features].to_numpy(dtype=float)
    )
    volume_prediction = fitted[VOLUME_MODEL].predict(
        test[volume_features].to_numpy(dtype=float)
    )
    augmented_prediction = fitted[AUGMENTED_MODEL].predict(
        test[augmented_features].to_numpy(dtype=float)
    )
    bootstrap = day_cluster_bootstrap_improvement(
        test,
        baseline_prediction,
        augmented_prediction,
        replicates=10_000,
        random_state=42,
    )
    calibration = calibration_curve(
        test["impact_cents"].to_numpy(dtype=float),
        {
            BASELINE_MODEL: baseline_prediction,
            AUGMENTED_MODEL: augmented_prediction,
        },
        quantiles=20,
    )
    count_residuals = residual_curve_by_count(test, volume_prediction)
    monthly = _monthly_scores(test, baseline_prediction, augmented_prediction)
    calibration.to_csv(output / "holdout_calibration.csv", index=False)
    count_residuals.to_csv(output / "holdout_count_residuals.csv", index=False)
    monthly.to_csv(output / "monthly_holdout.csv", index=False)

    trimmed_train, trimmed_test, trimmed_bounds = apply_training_sigma_filter(
        train,
        test,
        columns=["volume_imbalance", "impact_cents"],
        standard_deviations=3.0,
    )
    trimmed_metrics, _ = evaluate_impact_models(trimmed_train, trimmed_test)
    trimmed_metrics[metric_columns].to_csv(output / "model_metrics_trimmed.csv", index=False)

    horizon_metrics = _horizon_scores(windows, test_start_date=test_start_date)
    horizon_metrics.to_csv(output / "horizon_robustness.csv", index=False)
    walk_forward = evaluate_expanding_splits(
        impact_data,
        _selected_specifications({BASELINE_MODEL, AUGMENTED_MODEL}),
        n_splits=5,
        initial_train_fraction=0.40,
    )
    walk_forward.to_csv(output / "walk_forward.csv", index=False)

    scaling_transactions = pd.read_parquet(scaling_path, columns=SCALING_COLUMNS)
    scaling_windows = compute_scaling_windows(
        scaling_transactions,
        horizons=DEFAULT_HORIZONS,
    )
    scaling_curve = quantile_curve(scaling_windows, "Q", "impact_log", quantiles=31)
    scales, shape = fit_volume_curve_scales(scaling_curve)
    scale_fits = fit_scaling_laws(scales)
    variance_scaling_fits = fit_variance_scaling_statistics(scaling_windows)
    collapsed = collapse_curve(scaling_curve, scales)
    scales.to_csv(output / "scaling_scales.csv", index=False)
    scaling_curve.to_csv(output / "scaling_curve.csv", index=False)
    collapsed.to_csv(output / "scaling_collapse.csv", index=False)
    scale_fit_table = pd.DataFrame(
        [
            {"scale": "Q_N", **scale_fits["width"]},
            {"scale": "R_N", **scale_fits["height"]},
        ]
    )
    scale_fit_table.to_csv(output / "scaling_fits.csv", index=False)

    kyle_lambdas = estimate_kyle_lambdas(windows, central_quantile=0.35)
    kyle_decay = fit_kyle_decay(kyle_lambdas)
    kyle_lambdas.to_csv(output / "kyle_lambdas.csv", index=False)

    baseline = model_metrics.loc[model_metrics["model"].eq(BASELINE_MODEL)].iloc[0]
    augmented = model_metrics.loc[model_metrics["model"].eq(AUGMENTED_MODEL)].iloc[0]
    trimmed_baseline = trimmed_metrics.loc[trimmed_metrics["model"].eq(BASELINE_MODEL)].iloc[0]
    trimmed_augmented = trimmed_metrics.loc[
        trimmed_metrics["model"].eq(AUGMENTED_MODEL)
    ].iloc[0]
    basis_point_baseline = basis_point_metrics.loc[
        basis_point_metrics["model"].eq(BASELINE_MODEL)
    ].iloc[0]
    basis_point_augmented = basis_point_metrics.loc[
        basis_point_metrics["model"].eq(AUGMENTED_MODEL)
    ].iloc[0]
    relative_mse_reduction = 1 - (
        augmented["test_mse_cents_squared"] / baseline["test_mse_cents_squared"]
    )
    t10_lambda = kyle_lambdas.loc[kyle_lambdas["horizon"].eq(10)].iloc[0]

    horizon_summary = []
    for horizon in sorted(horizon_metrics["horizon"].unique()):
        part = horizon_metrics.loc[horizon_metrics["horizon"].eq(horizon)]
        base = part.loc[part["model"].eq(BASELINE_MODEL)].iloc[0]
        full = part.loc[part["model"].eq(AUGMENTED_MODEL)].iloc[0]
        horizon_summary.append(
            {
                "horizon": int(horizon),
                "baseline_r_squared": _number(base["test_r_squared"]),
                "augmented_r_squared": _number(full["test_r_squared"]),
                "relative_mse_reduction": _number(
                    1 - full["test_mse_cents_squared"] / base["test_mse_cents_squared"]
                ),
            }
        )

    result: dict[str, object] = {
        "scope": {
            "symbol": "TSLA",
            "year": 2019,
            "sessions": int(visible["date"].nunique()),
            "first_session": str(visible["date"].min()),
            "last_session": str(visible["date"].max()),
            "timestamp_aggregated_visible_market_orders": len(visible),
            "scaling_transactions_type_4_5": len(scaling_transactions),
            "sample_summary": sample_summary,
        },
        "prediction": {
            "horizon_market_orders": 10,
            "train_first_date": str(train["date"].min()),
            "train_last_date": str(train["date"].max()),
            "test_first_date": str(test["date"].min()),
            "test_last_date": str(test["date"].max()),
            "train_observations": len(train),
            "test_observations": len(test),
            "baseline": _score_record(baseline),
            "augmented": _score_record(augmented),
            "delta_r_squared": _number(
                augmented["test_r_squared"] - baseline["test_r_squared"]
            ),
            "relative_mse_reduction": _number(relative_mse_reduction),
            "day_cluster_bootstrap": bootstrap,
            "feature_ablation": [
                _score_record(row)
                for _, row in model_metrics.loc[
                    model_metrics["model"].isin(
                        [BASELINE_MODEL, VOLUME_MODEL, COUNT_MODEL, AUGMENTED_MODEL]
                    )
                ].iterrows()
            ],
            "basis_point_robustness": {
                "baseline_r_squared": _number(basis_point_baseline["test_r_squared"]),
                "augmented_r_squared": _number(basis_point_augmented["test_r_squared"]),
                "relative_mse_reduction": _number(
                    1
                    - basis_point_augmented["test_mse_cents_squared"]
                    / basis_point_baseline["test_mse_cents_squared"]
                ),
            },
            "trimmed_reference_protocol": {
                "bounds": trimmed_bounds,
                "train_observations": len(trimmed_train),
                "test_observations": len(trimmed_test),
                "baseline": _score_record(trimmed_baseline),
                "augmented": _score_record(trimmed_augmented),
                "relative_mse_reduction": _number(
                    1
                    - trimmed_augmented["test_mse_cents_squared"]
                    / trimmed_baseline["test_mse_cents_squared"]
                ),
            },
            "horizon_robustness": horizon_summary,
        },
        "scaling": {
            "horizons": len(DEFAULT_HORIZONS),
            "minimum_horizon": min(DEFAULT_HORIZONS),
            "maximum_horizon": max(DEFAULT_HORIZONS),
            "shape": shape,
            "width_scale": scale_fits["width"],
            "height_scale": scale_fits["height"],
            "sign_variance_scaling": variance_scaling_fits["sign"],
        },
        "liquidity": {
            "kyle_lambda_t10_cents_per_share": _number(
                t10_lambda["kyle_lambda_cents_per_share"]
            ),
            "kyle_decay_exponent": kyle_decay["exponent"],
        },
    }
    serialized = json.dumps(
        result,
        indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    )
    (output / "results.json").write_text(serialized + "\n")

    plot_aggregate_impact(aggregate_curve, figures / "aggregate-impact.pdf")
    plot_holdout_calibration(calibration, model_metrics, figures / "holdout-calibration.pdf")
    plot_count_residuals(count_residuals, figures / "count-residuals.pdf")
    plot_horizon_robustness(horizon_metrics, figures / "horizon-robustness.pdf")
    plot_walk_forward(walk_forward, figures / "walk-forward.pdf")
    plot_scaling_fits(scales, scale_fits, figures / "scaling-fits.pdf")
    plot_scaling_collapse(collapsed, shape, figures / "scaling-collapse.pdf")

    return json.loads(serialized)
