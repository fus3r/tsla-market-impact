"""Chronological evaluation of next-move queue-imbalance forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import required_columns
from .next_move import (
    compressed_binary_rows,
    day_cluster_brier_comparison,
    weighted_binary_scores,
)
from .plots import plot_queue_imbalance_forecast
from .policy import (
    AnalysisPolicy,
    split_index_for_test_start,
    validate_analysis_universe,
)

QUEUE_BIN_COLUMNS = [
    "date",
    "sample",
    "spread_bucket",
    "bin",
    "bin_left",
    "bin_right",
    "bin_center",
    "observations",
    "up_moves",
    "down_moves",
]


def evaluate_queue_imbalance(
    bins: pd.DataFrame,
    test_start_date: str,
    bootstrap_replicates: int = 10_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit one-feature logistic models on daily queue-imbalance aggregates."""

    required_columns(bins, QUEUE_BIN_COLUMNS)
    dates = np.array(sorted(bins["date"].astype(str).unique()))
    if len(dates) < 2:
        raise ValueError("At least two dates are required")
    split = split_index_for_test_start(dates.tolist(), test_start_date)
    train_dates = set(dates[:split])

    metric_rows: list[dict[str, object]] = []
    calibration_frames: list[pd.DataFrame] = []
    for (sample, spread_bucket), group in bins.groupby(
        ["sample", "spread_bucket"],
        sort=True,
        observed=True,
    ):
        train = group.loc[group["date"].astype(str).isin(train_dates)].copy()
        test = group.loc[~group["date"].astype(str).isin(train_dates)].copy()
        x_train, y_train, w_train = compressed_binary_rows(train, ["bin_center"])
        x_test, y_test, w_test = compressed_binary_rows(test, ["bin_center"])
        if w_train.sum() == 0 or w_test.sum() == 0:
            continue
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1_000,
        )
        model.fit(x_train, y_train, sample_weight=w_train)
        model_probability = model.predict_proba(x_test)[:, 1]
        baseline_probability = float(np.average(y_train, weights=w_train))
        baseline_predictions = np.full(len(y_test), baseline_probability)
        model_scores = weighted_binary_scores(y_test, model_probability, w_test)
        baseline_scores = weighted_binary_scores(
            y_test,
            baseline_predictions,
            w_test,
        )

        row_probability = model.predict_proba(
            test[["bin_center"]].to_numpy(dtype=float)
        )[:, 1]
        bootstrap = day_cluster_brier_comparison(
            test,
            row_probability,
            np.full(len(test), baseline_probability),
            bootstrap_replicates,
            random_state,
        )
        bootstrap.pop("relative_brier_reduction")
        brier_reduction = (
            1 - model_scores["brier_score"] / baseline_scores["brier_score"]
        )
        metric_rows.append(
            {
                "sample": sample,
                "spread_bucket": spread_bucket,
                "train_first_date": str(dates[0]),
                "train_last_date": str(dates[split - 1]),
                "test_first_date": str(dates[split]),
                "test_last_date": str(dates[-1]),
                "train_observations": int(w_train.sum()),
                "test_observations": int(w_test.sum()),
                "train_upward_fraction": baseline_probability,
                "test_upward_fraction": float(np.average(y_test, weights=w_test)),
                "intercept": float(model.intercept_[0]),
                "queue_imbalance_coefficient": float(model.coef_[0, 0]),
                **{f"baseline_{key}": value for key, value in baseline_scores.items()},
                **{f"model_{key}": value for key, value in model_scores.items()},
                "relative_brier_reduction": float(brier_reduction),
                **bootstrap,
            }
        )

        calibration = (
            test.assign(model_probability=row_probability)
            .groupby(
                ["sample", "spread_bucket", "bin", "bin_center"],
                sort=True,
                observed=True,
            )
            .agg(
                observations=("observations", "sum"),
                up_moves=("up_moves", "sum"),
                down_moves=("down_moves", "sum"),
                model_probability=("model_probability", "first"),
            )
            .reset_index()
        )
        calibration["empirical_up_probability"] = (
            calibration["up_moves"] / calibration["observations"]
        )
        calibration_frames.append(calibration)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["sample", "spread_bucket"]
    ).reset_index(drop=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    result = {
        "protocol": {
            "dates": int(len(dates)),
            "train_dates": int(split),
            "test_dates": int(len(dates) - split),
            "train_first_date": str(dates[0]),
            "train_last_date": str(dates[split - 1]),
            "test_first_date": str(dates[split]),
            "test_last_date": str(dates[-1]),
            "target": "direction of the next same-session mid-price change",
            "feature": "level-1 queue imbalance after the current event",
            "bootstrap_unit": "trading date",
        },
        "metric_rows": int(len(metrics)),
        "calibration_rows": int(len(calibration)),
    }
    return metrics, calibration, result


def run_queue_analysis(
    bins_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Evaluate queue imbalance and persist aggregate tables."""

    bins = pd.read_csv(bins_path)
    validate_analysis_universe(bins["date"], analysis_policy)
    metrics, calibration, result = evaluate_queue_imbalance(
        bins,
        test_start_date=analysis_policy.test_start,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "queue_model_metrics.csv", index=False)
    calibration.to_csv(output / "queue_calibration.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "queue_model.json").write_text(serialized + "\n")
    plot_queue_imbalance_forecast(
        calibration,
        metrics,
        Path(figures_dir) / "queue-imbalance.pdf",
    )
    return json.loads(serialized)
