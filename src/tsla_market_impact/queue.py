"""Chronological evaluation of next-move queue-imbalance forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .data import required_columns
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


def _compressed_binary_rows(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.repeat(frame[["bin_center"]].to_numpy(dtype=float), 2, axis=0)
    targets = np.tile([1, 0], len(frame))
    weights = frame[["up_moves", "down_moves"]].to_numpy(dtype=float).reshape(-1)
    keep = weights > 0
    return features[keep], targets[keep], weights[keep]


def _weighted_binary_scores(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    predicted = probabilities >= 0.5
    brier = np.average((targets - probabilities) ** 2, weights=weights)
    return {
        "roc_auc": float(
            roc_auc_score(targets, probabilities, sample_weight=weights)
        ),
        "log_loss": float(log_loss(targets, probabilities, sample_weight=weights)),
        "brier_score": float(brier),
        "accuracy": float(np.average(predicted == targets, weights=weights)),
    }


def _day_cluster_brier_improvement(
    test: pd.DataFrame,
    model_probability: np.ndarray,
    baseline_probability: float,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    if len(test) != len(model_probability):
        raise ValueError("model_probability must have one value per aggregate row")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    up = test["up_moves"].to_numpy(dtype=float)
    down = test["down_moves"].to_numpy(dtype=float)
    model_sse = up * (1 - model_probability) ** 2 + down * model_probability**2
    baseline_sse = (
        up * (1 - baseline_probability) ** 2 + down * baseline_probability**2
    )
    daily = (
        pd.DataFrame(
            {
                "date": test["date"].to_numpy(),
                "model_sse": model_sse,
                "baseline_sse": baseline_sse,
            }
        )
        .groupby("date", sort=True, observed=True)
        .sum()
    )
    values = daily[["model_sse", "baseline_sse"]].to_numpy(dtype=float)
    random = np.random.default_rng(random_state)
    sampled = random.integers(0, len(values), size=(replicates, len(values)))
    totals = values[sampled].sum(axis=1)
    improvements = 1 - totals[:, 0] / totals[:, 1]
    lower, median, upper = np.quantile(improvements, [0.025, 0.5, 0.975])
    return {
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "relative_brier_reduction_lower_95": float(lower),
        "relative_brier_reduction_median": float(median),
        "relative_brier_reduction_upper_95": float(upper),
        "probability_nonpositive": float(np.mean(improvements <= 0)),
    }


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
        x_train, y_train, w_train = _compressed_binary_rows(train)
        x_test, y_test, w_test = _compressed_binary_rows(test)
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
        model_scores = _weighted_binary_scores(y_test, model_probability, w_test)
        baseline_scores = _weighted_binary_scores(
            y_test,
            baseline_predictions,
            w_test,
        )

        row_probability = model.predict_proba(
            test[["bin_center"]].to_numpy(dtype=float)
        )[:, 1]
        bootstrap = _day_cluster_brier_improvement(
            test,
            row_probability,
            baseline_probability,
            bootstrap_replicates,
            random_state,
        )
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
        "models": metrics.to_dict(orient="records"),
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
