"""Train-only selection of causal order-flow imbalance lookback horizons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import required_columns
from .next_move import compressed_binary_rows, weighted_binary_scores
from .plots import plot_ofi_horizon_selection
from .policy import (
    AnalysisPolicy,
    split_index_for_test_start,
    validate_analysis_universe,
)

OFI_HORIZON_BIN_COLUMNS = [
    "date",
    "sample",
    "spread_bucket",
    "horizon_kind",
    "horizon_value",
    "queue_bin",
    "ofi_bin",
    "queue_center",
    "ofi_center",
    "observations",
    "up_moves",
    "down_moves",
]

HORIZON_MODELS: dict[str, tuple[str, ...]] = {
    "ofi": ("ofi_center",),
    "queue_and_ofi": ("queue_center", "ofi_center"),
}

_HORIZON_KIND_ORDER = {
    "price_spell": 0,
    "quote_updates": 1,
    "clock_us": 2,
}


def _horizon_key(kind: str, value: int) -> tuple[int, int]:
    return _HORIZON_KIND_ORDER[kind], value


def _validate_horizon_bins(bins: pd.DataFrame) -> None:
    required_columns(bins, OFI_HORIZON_BIN_COLUMNS)
    counts = bins[["observations", "up_moves", "down_moves"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("OFI horizon counts must be finite and non-negative")
    if not np.array_equal(
        bins["observations"].to_numpy(),
        (bins["up_moves"] + bins["down_moves"]).to_numpy(),
    ):
        raise ValueError("observations must equal up_moves plus down_moves")

    kinds = set(bins["horizon_kind"].astype(str))
    if not kinds or not kinds.issubset(_HORIZON_KIND_ORDER):
        raise ValueError("unknown OFI horizon kind")
    values = bins["horizon_value"].to_numpy(dtype=np.int64)
    price_spell = bins["horizon_kind"].eq("price_spell").to_numpy()
    if not price_spell.any() or (values[price_spell] != 0).any():
        raise ValueError("price_spell must be present with horizon value zero")
    if (values[~price_spell] <= 0).any():
        raise ValueError("fixed OFI horizons must be positive")

    dimensions = [
        "date",
        "sample",
        "spread_bucket",
        "horizon_kind",
        "horizon_value",
        "queue_bin",
        "ofi_bin",
    ]
    if bins.duplicated(dimensions).any():
        raise ValueError("duplicate OFI horizon aggregate cell")

    totals = (
        bins.groupby(
            [
                "sample",
                "spread_bucket",
                "horizon_kind",
                "horizon_value",
                "date",
            ],
            sort=True,
            observed=True,
        )[["observations", "up_moves", "down_moves"]]
        .sum()
        .reset_index()
    )
    for (sample, spread_bucket), group in totals.groupby(
        ["sample", "spread_bucket"],
        sort=True,
        observed=True,
    ):
        reference = group.loc[
            group["horizon_kind"].eq("price_spell") & group["horizon_value"].eq(0),
            ["date", "observations", "up_moves", "down_moves"],
        ].sort_values("date")
        if reference.empty:
            raise ValueError(f"missing price_spell reference for {sample}/{spread_bucket}")
        reference_values = reference[["observations", "up_moves", "down_moves"]].to_numpy(
            dtype=np.int64
        )
        reference_dates = reference["date"].astype(str).tolist()
        for (kind, value), candidate in group.groupby(
            ["horizon_kind", "horizon_value"],
            sort=True,
            observed=True,
        ):
            candidate = candidate.sort_values("date")
            if candidate["date"].astype(str).tolist() != reference_dates or not np.array_equal(
                candidate[["observations", "up_moves", "down_moves"]].to_numpy(dtype=np.int64),
                reference_values,
            ):
                raise ValueError(
                    "every OFI horizon must preserve date-level labels and counts; "
                    f"mismatch for {sample}/{spread_bucket}/{kind}/{value}"
                )


def _fit_probabilities(
    train: pd.DataFrame,
    score: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[LogisticRegression, np.ndarray, dict[str, float]]:
    x_train, y_train, w_train = compressed_binary_rows(train, feature_columns)
    x_score, y_score, w_score = compressed_binary_rows(score, feature_columns)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_score)) != 2:
        raise ValueError("each chronological segment must contain both directions")
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1_000,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    row_probability = model.predict_proba(score[list(feature_columns)].to_numpy(dtype=float))[:, 1]
    expanded_probability = model.predict_proba(x_score)[:, 1]
    scores = weighted_binary_scores(
        y_score,
        expanded_probability,
        w_score,
    )
    return model, row_probability, scores


def _daily_sse(frame: pd.DataFrame, probability: np.ndarray) -> pd.Series:
    up = frame["up_moves"].to_numpy(dtype=float)
    down = frame["down_moves"].to_numpy(dtype=float)
    sse = up * (1 - probability) ** 2 + down * probability**2
    return (
        pd.DataFrame(
            {
                "date": frame["date"].astype(str).to_numpy(),
                "sse": sse,
            }
        )
        .groupby("date", sort=True, observed=True)["sse"]
        .sum()
    )


def _paired_relative_sse(
    challenger: pd.Series,
    reference: pd.Series,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not challenger.index.equals(reference.index):
        raise ValueError("paired daily losses must cover the same dates")
    values = np.column_stack(
        [
            challenger.to_numpy(dtype=float),
            reference.to_numpy(dtype=float),
        ]
    )
    point = 1 - values[:, 0].sum() / values[:, 1].sum()
    random = np.random.default_rng(random_state)
    sampled = random.integers(0, len(values), size=(replicates, len(values)))
    totals = values[sampled].sum(axis=1)
    improvements = 1 - totals[:, 0] / totals[:, 1]
    lower, median, upper = np.quantile(improvements, [0.025, 0.5, 0.975])
    return {
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "relative_brier_reduction": float(point),
        "relative_brier_reduction_lower_95": float(lower),
        "relative_brier_reduction_median": float(median),
        "relative_brier_reduction_upper_95": float(upper),
        "probability_nonpositive": float(np.mean(improvements <= 0)),
    }


def _baseline_daily_sse(frame: pd.DataFrame, probability: float) -> pd.Series:
    return _daily_sse(frame, np.full(len(frame), probability))


def evaluate_ofi_horizons(
    bins: pd.DataFrame,
    test_date_fraction: float = 0.20,
    selection_date_fraction: float = 0.25,
    bootstrap_replicates: int = 10_000,
    random_state: int = 42,
    development_end_date: str | None = None,
    selection_start_date: str | None = None,
    selection_end_date: str | None = None,
    test_start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select an OFI lookback on early dates and evaluate it once on late dates."""

    fixed_boundaries = {
        development_end_date,
        selection_start_date,
        selection_end_date,
        test_start_date,
    }
    if None in fixed_boundaries and len(fixed_boundaries) > 1:
        raise ValueError("All fixed OFI horizon boundaries must be provided together")
    if test_start_date is None and not 0 < test_date_fraction < 1:
        raise ValueError("test_date_fraction must lie between zero and one")
    if selection_start_date is None and not 0 < selection_date_fraction < 1:
        raise ValueError("selection_date_fraction must lie between zero and one")
    _validate_horizon_bins(bins)
    dates = np.array(sorted(bins["date"].astype(str).unique()))
    if len(dates) < 4:
        raise ValueError("At least four dates are required for nested selection")

    if test_start_date is None:
        outer_split = int(np.floor((1 - test_date_fraction) * len(dates)))
        outer_split = min(max(outer_split, 2), len(dates) - 1)
        selection_split = int(np.floor((1 - selection_date_fraction) * outer_split))
        selection_split = min(max(selection_split, 1), outer_split - 1)
    else:
        selection_split = split_index_for_test_start(
            dates.tolist(),
            str(selection_start_date),
        )
        outer_split = split_index_for_test_start(dates.tolist(), test_start_date)
        if (
            dates[selection_split - 1] != development_end_date
            or dates[outer_split - 1] != selection_end_date
        ):
            raise ValueError("OFI horizon boundaries do not match the included calendar")
    development_dates = set(dates[:selection_split])
    selection_dates = set(dates[selection_split:outer_split])
    outer_train_dates = set(dates[:outer_split])

    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    daily_test_losses: dict[
        tuple[str, str, str, str, int],
        pd.Series,
    ] = {}

    group_columns = ["sample", "spread_bucket"]
    for (sample, spread_bucket), signal_group in bins.groupby(
        group_columns,
        sort=True,
        observed=True,
    ):
        candidates = sorted(
            {
                (str(kind), int(value))
                for kind, value in signal_group[["horizon_kind", "horizon_value"]].itertuples(
                    index=False, name=None
                )
            },
            key=lambda item: _horizon_key(*item),
        )
        for model_name, feature_columns in HORIZON_MODELS.items():
            candidate_rows: list[dict[str, object]] = []
            for kind, value in candidates:
                candidate = signal_group.loc[
                    signal_group["horizon_kind"].eq(kind) & signal_group["horizon_value"].eq(value)
                ].copy()
                date_values = candidate["date"].astype(str)
                development = candidate.loc[date_values.isin(development_dates)]
                selection = candidate.loc[date_values.isin(selection_dates)]
                outer_train = candidate.loc[date_values.isin(outer_train_dates)]
                test = candidate.loc[~date_values.isin(outer_train_dates)]

                _, selection_probability, selection_scores = _fit_probabilities(
                    development,
                    selection,
                    feature_columns,
                )
                development_frequency = float(
                    development["up_moves"].sum() / development["observations"].sum()
                )
                selection_baseline_sse = _baseline_daily_sse(
                    selection,
                    development_frequency,
                )
                selection_sse = _daily_sse(selection, selection_probability)
                selection_improvement = 1 - selection_sse.sum() / selection_baseline_sse.sum()

                row = {
                    "sample": sample,
                    "spread_bucket": spread_bucket,
                    "model": model_name,
                    "horizon_kind": kind,
                    "horizon_value": value,
                    "development_first_date": str(dates[0]),
                    "development_last_date": str(dates[selection_split - 1]),
                    "selection_first_date": str(dates[selection_split]),
                    "selection_last_date": str(dates[outer_split - 1]),
                    "test_first_date": str(dates[outer_split]),
                    "test_last_date": str(dates[-1]),
                    "development_observations": int(development["observations"].sum()),
                    "selection_observations": int(selection["observations"].sum()),
                    "test_observations": int(test["observations"].sum()),
                    "selection_roc_auc": selection_scores["roc_auc"],
                    "selection_log_loss": selection_scores["log_loss"],
                    "selection_brier_score": selection_scores["brier_score"],
                    "selection_relative_brier_reduction": float(selection_improvement),
                    "test_roc_auc": np.nan,
                    "test_log_loss": np.nan,
                    "test_brier_score": np.nan,
                    "test_accuracy": np.nan,
                    "test_relative_brier_reduction": np.nan,
                    "intercept": np.nan,
                    "queue_coefficient": np.nan,
                    "ofi_coefficient": np.nan,
                    "selected_fixed_on_train_dates": False,
                }
                candidate_rows.append(row)
                metric_rows.append(row)

            fixed_rows = [row for row in candidate_rows if row["horizon_kind"] != "price_spell"]
            if not fixed_rows:
                raise ValueError("at least one fixed OFI horizon is required")
            selected = min(
                fixed_rows,
                key=lambda row: (
                    row["selection_brier_score"],
                    _horizon_key(
                        str(row["horizon_kind"]),
                        int(row["horizon_value"]),
                    ),
                ),
            )
            selected["selected_fixed_on_train_dates"] = True
            price_spell = next(
                row
                for row in candidate_rows
                if row["horizon_kind"] == "price_spell" and row["horizon_value"] == 0
            )

            for evaluated in (selected, price_spell):
                evaluated_kind = str(evaluated["horizon_kind"])
                evaluated_value = int(evaluated["horizon_value"])
                candidate = signal_group.loc[
                    signal_group["horizon_kind"].eq(evaluated_kind)
                    & signal_group["horizon_value"].eq(evaluated_value)
                ].copy()
                date_values = candidate["date"].astype(str)
                outer_train = candidate.loc[date_values.isin(outer_train_dates)]
                test = candidate.loc[~date_values.isin(outer_train_dates)]
                outer_model, test_probability, test_scores = _fit_probabilities(
                    outer_train,
                    test,
                    feature_columns,
                )
                outer_frequency = float(
                    outer_train["up_moves"].sum() / outer_train["observations"].sum()
                )
                test_sse = _daily_sse(test, test_probability)
                test_baseline_sse = _baseline_daily_sse(test, outer_frequency)
                test_improvement = 1 - test_sse.sum() / test_baseline_sse.sum()
                daily_test_losses[
                    (
                        sample,
                        spread_bucket,
                        model_name,
                        evaluated_kind,
                        evaluated_value,
                    )
                ] = test_sse
                coefficients = dict.fromkeys(
                    ["queue_center", "ofi_center"],
                    np.nan,
                )
                coefficients.update(
                    {
                        feature: float(coefficient)
                        for feature, coefficient in zip(
                            feature_columns,
                            outer_model.coef_[0],
                            strict=True,
                        )
                    }
                )
                evaluated.update(
                    {
                        "test_roc_auc": test_scores["roc_auc"],
                        "test_log_loss": test_scores["log_loss"],
                        "test_brier_score": test_scores["brier_score"],
                        "test_accuracy": test_scores["accuracy"],
                        "test_relative_brier_reduction": float(test_improvement),
                        "intercept": float(outer_model.intercept_[0]),
                        "queue_coefficient": coefficients["queue_center"],
                        "ofi_coefficient": coefficients["ofi_center"],
                    }
                )

            selected_key = (
                sample,
                spread_bucket,
                model_name,
                str(selected["horizon_kind"]),
                int(selected["horizon_value"]),
            )
            reference_key = (
                sample,
                spread_bucket,
                model_name,
                "price_spell",
                0,
            )
            test = signal_group.loc[
                signal_group["horizon_kind"].eq(selected["horizon_kind"])
                & signal_group["horizon_value"].eq(selected["horizon_value"])
                & ~signal_group["date"].astype(str).isin(outer_train_dates)
            ]
            baseline_probability = float(
                signal_group.loc[
                    signal_group["horizon_kind"].eq(selected["horizon_kind"])
                    & signal_group["horizon_value"].eq(selected["horizon_value"])
                    & signal_group["date"].astype(str).isin(outer_train_dates),
                    "up_moves",
                ].sum()
                / signal_group.loc[
                    signal_group["horizon_kind"].eq(selected["horizon_kind"])
                    & signal_group["horizon_value"].eq(selected["horizon_value"])
                    & signal_group["date"].astype(str).isin(outer_train_dates),
                    "observations",
                ].sum()
            )
            versus_baseline = _paired_relative_sse(
                daily_test_losses[selected_key],
                _baseline_daily_sse(test, baseline_probability),
                bootstrap_replicates,
                random_state,
            )
            versus_price_spell = _paired_relative_sse(
                daily_test_losses[selected_key],
                daily_test_losses[reference_key],
                bootstrap_replicates,
                random_state,
            )
            selection_rows.append(
                {
                    "sample": sample,
                    "spread_bucket": spread_bucket,
                    "model": model_name,
                    "selected_fixed_horizon_kind": selected["horizon_kind"],
                    "selected_fixed_horizon_value": selected["horizon_value"],
                    "selection_brier_score": selected["selection_brier_score"],
                    "selection_relative_brier_reduction": selected[
                        "selection_relative_brier_reduction"
                    ],
                    "test_observations": selected["test_observations"],
                    "test_roc_auc": selected["test_roc_auc"],
                    "test_brier_score": selected["test_brier_score"],
                    "test_relative_brier_reduction": selected["test_relative_brier_reduction"],
                    "price_spell_test_roc_auc": price_spell["test_roc_auc"],
                    "price_spell_test_brier_score": price_spell["test_brier_score"],
                    "price_spell_test_relative_brier_reduction": price_spell[
                        "test_relative_brier_reduction"
                    ],
                    **{f"versus_baseline_{key}": value for key, value in versus_baseline.items()},
                    **{
                        f"versus_price_spell_{key}": value
                        for key, value in versus_price_spell.items()
                    },
                }
            )

    metrics = (
        pd.DataFrame(metric_rows)
        .sort_values(
            [
                "sample",
                "spread_bucket",
                "model",
                "horizon_kind",
                "horizon_value",
            ]
        )
        .reset_index(drop=True)
    )
    selection = (
        pd.DataFrame(selection_rows)
        .sort_values(["sample", "spread_bucket", "model"])
        .reset_index(drop=True)
    )
    result = {
        "protocol": {
            "dates": int(len(dates)),
            "development_dates": int(selection_split),
            "selection_dates": int(outer_split - selection_split),
            "test_dates": int(len(dates) - outer_split),
            "development_first_date": str(dates[0]),
            "development_last_date": str(dates[selection_split - 1]),
            "selection_first_date": str(dates[selection_split]),
            "selection_last_date": str(dates[outer_split - 1]),
            "test_first_date": str(dates[outer_split]),
            "test_last_date": str(dates[-1]),
            "selection_rule": (
                "among fixed quote-update and clock-time windows, minimize "
                "validation Brier score after fitting on the earliest "
                "development dates; break exact ties toward quote-update "
                "windows, then shorter windows"
            ),
            "refit": (
                "refit the selected specification on all pre-test dates, then "
                "score the final dates once"
            ),
            "final_evaluation": (
                "compute final-period scores only for the selected fixed "
                "horizon and the pre-specified adaptive reference; rejected "
                "fixed candidates retain null test metrics"
            ),
            "target": "direction of the next same-session mid-price change",
            "sample": "post-event states after a best price or size update",
            "queue_feature": "post-event level-1 queue imbalance",
            "ofi_transform": (
                "top-of-book OFI divided by current displayed depth and mapped by 2*atan(x)/pi"
            ),
            "price_spell_window": (
                "increments strictly after the event that established the "
                "current mid-price; retained as the adaptive reference and "
                "excluded from fixed-window selection"
            ),
            "quote_update_window": ("current and preceding best-price-or-size OFI increments"),
            "clock_window": (
                "best-price-or-size OFI increments with timestamps in "
                "(signal_time - horizon, signal_time]"
            ),
            "feature_grid_bins_per_axis": int(
                max(bins["queue_bin"].max(), bins["ofi_bin"].max()) + 1
            ),
            "bootstrap_unit": "trading date",
            "bootstrap_replicates": int(bootstrap_replicates),
        },
        "metric_rows": int(len(metrics)),
        "selection_rows": int(len(selection)),
    }
    return metrics, selection, result


def run_ofi_horizon_analysis(
    bins_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Evaluate causal OFI lookbacks and persist compact aggregate evidence."""

    bins = pd.read_csv(
        bins_path,
        dtype={
            "date": "string",
            "sample": "category",
            "spread_bucket": "category",
            "horizon_kind": "category",
            "horizon_value": "uint64",
            "queue_bin": "uint16",
            "ofi_bin": "uint16",
            "queue_center": "float64",
            "ofi_center": "float64",
            "observations": "uint64",
            "up_moves": "uint64",
            "down_moves": "uint64",
        },
    )
    validate_analysis_universe(bins["date"], analysis_policy)
    metrics, selection, result = evaluate_ofi_horizons(
        bins,
        development_end_date=analysis_policy.development_end,
        selection_start_date=analysis_policy.selection_start,
        selection_end_date=analysis_policy.selection_end,
        test_start_date=analysis_policy.test_start,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "ofi_horizon_metrics.csv", index=False)
    selection.to_csv(output / "ofi_horizon_selection.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "ofi_horizon_model.json").write_text(serialized + "\n")
    if figures_dir is not None:
        plot_ofi_horizon_selection(
            metrics,
            Path(figures_dir) / "ofi-horizon-selection.pdf",
        )
    return json.loads(serialized)
