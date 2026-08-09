"""Chronological ablation of queue imbalance and top-of-book order flow."""

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
from .plots import plot_order_flow_signal_ablation
from .policy import (
    AnalysisPolicy,
    split_index_for_test_start,
    validate_analysis_universe,
)

ORDER_FLOW_BIN_COLUMNS = [
    "date",
    "sample",
    "spread_bucket",
    "queue_bin",
    "ofi_bin",
    "queue_center",
    "ofi_center",
    "observations",
    "up_moves",
    "down_moves",
]

MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "intercept": (),
    "queue": ("queue_center",),
    "ofi": ("ofi_center",),
    "queue_and_ofi": ("queue_center", "ofi_center"),
}

PAIRWISE_COMPARISONS = [
    ("queue", "intercept"),
    ("ofi", "intercept"),
    ("queue_and_ofi", "intercept"),
    ("queue_and_ofi", "queue"),
    ("queue_and_ofi", "ofi"),
]


def _weighted_quantile_edges(
    values: np.ndarray,
    weights: np.ndarray,
    bins: int,
) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    edges = np.interp(
        np.linspace(0.0, 1.0, bins + 1),
        cumulative,
        ordered_values,
    )
    return np.unique(edges)


def _calibration_frame(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_probability: np.ndarray,
    test_probability: np.ndarray,
    sample: str,
    spread_bucket: str,
    model: str,
    bins: int = 10,
) -> pd.DataFrame:
    edges = _weighted_quantile_edges(
        train_probability,
        train["observations"].to_numpy(dtype=float),
        bins,
    )
    if len(edges) == 1:
        calibration_bin = np.zeros(len(test), dtype=int)
    else:
        calibration_bin = np.searchsorted(
            edges[1:-1],
            test_probability,
            side="right",
        )
    frame = test[["observations", "up_moves", "down_moves"]].copy()
    frame["calibration_bin"] = calibration_bin
    frame["probability_sum"] = (
        test_probability * frame["observations"].to_numpy(dtype=float)
    )
    calibration = (
        frame.groupby("calibration_bin", sort=True, observed=True)
        .agg(
            observations=("observations", "sum"),
            up_moves=("up_moves", "sum"),
            down_moves=("down_moves", "sum"),
            probability_sum=("probability_sum", "sum"),
        )
        .reset_index()
    )
    calibration["predicted_up_probability"] = (
        calibration["probability_sum"] / calibration["observations"]
    )
    calibration["empirical_up_probability"] = (
        calibration["up_moves"] / calibration["observations"]
    )
    calibration.insert(0, "model", model)
    calibration.insert(0, "spread_bucket", spread_bucket)
    calibration.insert(0, "sample", sample)
    return calibration.drop(columns="probability_sum")


def evaluate_order_flow_signals(
    bins: pd.DataFrame,
    test_start_date: str,
    bootstrap_replicates: int = 10_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare linear queue, OFI, and combined next-move probabilities."""

    required_columns(bins, ORDER_FLOW_BIN_COLUMNS)
    dates = np.array(sorted(bins["date"].astype(str).unique()))
    if len(dates) < 2:
        raise ValueError("At least two dates are required")
    split = split_index_for_test_start(dates.tolist(), test_start_date)
    train_dates = set(dates[:split])

    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    calibration_frames: list[pd.DataFrame] = []
    for (sample, spread_bucket), group in bins.groupby(
        ["sample", "spread_bucket"],
        sort=True,
        observed=True,
    ):
        train = group.loc[group["date"].astype(str).isin(train_dates)].copy()
        test = group.loc[~group["date"].astype(str).isin(train_dates)].copy()
        if train["observations"].sum() == 0 or test["observations"].sum() == 0:
            continue
        if (
            train["up_moves"].sum() == 0
            or train["down_moves"].sum() == 0
            or test["up_moves"].sum() == 0
            or test["down_moves"].sum() == 0
        ):
            continue

        baseline_probability = float(
            train["up_moves"].sum() / train["observations"].sum()
        )
        test_upward_fraction = float(
            test["up_moves"].sum() / test["observations"].sum()
        )
        train_row_probability: dict[str, np.ndarray] = {}
        test_row_probability: dict[str, np.ndarray] = {}

        for model_name, feature_columns in MODEL_FEATURES.items():
            coefficients = {"queue_center": np.nan, "ofi_center": np.nan}
            if feature_columns:
                x_train, y_train, w_train = compressed_binary_rows(
                    train,
                    feature_columns,
                )
                x_test, y_test, w_test = compressed_binary_rows(
                    test,
                    feature_columns,
                )
                model = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1_000,
                )
                model.fit(x_train, y_train, sample_weight=w_train)
                train_probability = model.predict_proba(
                    train[list(feature_columns)].to_numpy(dtype=float)
                )[:, 1]
                test_probability = model.predict_proba(
                    test[list(feature_columns)].to_numpy(dtype=float)
                )[:, 1]
                expanded_probability = model.predict_proba(x_test)[:, 1]
                intercept = float(model.intercept_[0])
                coefficients.update(
                    {
                        feature: float(coefficient)
                        for feature, coefficient in zip(
                            feature_columns,
                            model.coef_[0],
                            strict=True,
                        )
                    }
                )
            else:
                _, y_test, w_test = compressed_binary_rows(
                    test,
                    ["queue_center"],
                )
                train_probability = np.full(len(train), baseline_probability)
                test_probability = np.full(len(test), baseline_probability)
                expanded_probability = np.full(len(y_test), baseline_probability)
                intercept = float(
                    np.log(baseline_probability / (1 - baseline_probability))
                )

            train_row_probability[model_name] = train_probability
            test_row_probability[model_name] = test_probability
            scores = weighted_binary_scores(y_test, expanded_probability, w_test)
            metric_rows.append(
                {
                    "sample": sample,
                    "spread_bucket": spread_bucket,
                    "model": model_name,
                    "train_first_date": str(dates[0]),
                    "train_last_date": str(dates[split - 1]),
                    "test_first_date": str(dates[split]),
                    "test_last_date": str(dates[-1]),
                    "train_observations": int(train["observations"].sum()),
                    "test_observations": int(test["observations"].sum()),
                    "train_upward_fraction": baseline_probability,
                    "test_upward_fraction": test_upward_fraction,
                    "intercept": intercept,
                    "queue_coefficient": coefficients["queue_center"],
                    "ofi_coefficient": coefficients["ofi_center"],
                    **scores,
                }
            )
            calibration_frames.append(
                _calibration_frame(
                    train,
                    test,
                    train_probability,
                    test_probability,
                    sample,
                    spread_bucket,
                    model_name,
                )
            )

        for challenger, reference in PAIRWISE_COMPARISONS:
            comparison = day_cluster_brier_comparison(
                test,
                test_row_probability[challenger],
                test_row_probability[reference],
                bootstrap_replicates,
                random_state,
            )
            comparison_rows.append(
                {
                    "sample": sample,
                    "spread_bucket": spread_bucket,
                    "challenger": challenger,
                    "reference": reference,
                    **comparison,
                }
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["sample", "spread_bucket", "model"]
    ).reset_index(drop=True)
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["sample", "spread_bucket", "reference", "challenger"]
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
            "sample": "post-event states after a best price or size update",
            "feature_grid_bins_per_axis": int(
                max(bins["queue_bin"].max(), bins["ofi_bin"].max()) + 1
            ),
            "queue_feature": "post-event level-1 queue imbalance",
            "ofi_feature": (
                "top-of-book OFI accumulated since the current mid-price was "
                "established, divided by current displayed depth and mapped by "
                "2*atan(x)/pi"
            ),
            "bootstrap_unit": "trading date",
        },
        "metric_rows": int(len(metrics)),
        "comparison_rows": int(len(comparisons)),
        "calibration_rows": int(len(calibration)),
    }
    return metrics, comparisons, calibration, result


def run_order_flow_analysis(
    bins_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Evaluate order-flow signals and persist aggregate evidence."""

    bins = pd.read_csv(bins_path)
    validate_analysis_universe(bins["date"], analysis_policy)
    metrics, comparisons, calibration, result = evaluate_order_flow_signals(
        bins,
        test_start_date=analysis_policy.test_start,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "order_flow_model_metrics.csv", index=False)
    comparisons.to_csv(output / "order_flow_comparisons.csv", index=False)
    calibration.to_csv(output / "order_flow_calibration.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "order_flow_model.json").write_text(serialized + "\n")
    plot_order_flow_signal_ablation(
        comparisons,
        Path(figures_dir) / "order-flow-ablation.pdf",
    )
    return json.loads(serialized)


def evaluate_order_flow_grid_robustness(
    grid_bins: dict[int, pd.DataFrame],
    test_start_date: str,
    bootstrap_replicates: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Summarize whether the ablation changes with fixed grid resolution."""

    rows: list[pd.DataFrame] = []
    for grid_size, bins in sorted(grid_bins.items()):
        metrics, comparisons, _, _ = evaluate_order_flow_signals(
            bins,
            test_start_date=test_start_date,
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state,
        )
        versus_intercept = comparisons.loc[
            comparisons["reference"].eq("intercept")
        ].rename(columns={"challenger": "model"})
        summary = metrics.merge(
            versus_intercept,
            on=["sample", "spread_bucket", "model"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_comparison"),
        )[
            [
                "sample",
                "spread_bucket",
                "model",
                "test_observations",
                "roc_auc",
                "brier_score",
                "relative_brier_reduction",
                "relative_brier_reduction_lower_95",
                "relative_brier_reduction_upper_95",
                "probability_nonpositive",
            ]
        ]
        summary.insert(0, "grid_bins_per_axis", grid_size)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["sample", "spread_bucket", "model", "grid_bins_per_axis"]
    ).reset_index(drop=True)


def run_order_flow_grid_robustness(
    bin_specs: list[tuple[int, Path]],
    results_dir: Path | str,
    *,
    analysis_policy: AnalysisPolicy,
) -> list[dict[str, object]]:
    """Evaluate precomputed grids and persist the compact comparison."""

    grid_bins: dict[int, pd.DataFrame] = {}
    for grid_size, path in bin_specs:
        if grid_size in grid_bins:
            raise ValueError(f"duplicate grid size: {grid_size}")
        frame = pd.read_csv(path)
        required_columns(frame, ORDER_FLOW_BIN_COLUMNS)
        validate_analysis_universe(frame["date"], analysis_policy)
        observed_grid = int(max(frame["queue_bin"].max(), frame["ofi_bin"].max()) + 1)
        if observed_grid != grid_size:
            raise ValueError(
                f"{path} declares grid {grid_size}, but its largest bin implies "
                f"{observed_grid}"
            )
        grid_bins[grid_size] = frame
    if len(grid_bins) < 2:
        raise ValueError("At least two order-flow grids are required")

    robustness = evaluate_order_flow_grid_robustness(
        grid_bins,
        test_start_date=analysis_policy.test_start,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    robustness.to_csv(output / "order_flow_grid_robustness.csv", index=False)
    return json.loads(robustness.to_json(orient="records"))
