"""Rolling-origin stability audit for non-overlapping next-move signals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import required_columns
from .next_move import compressed_binary_rows, weighted_binary_scores
from .orderflow import MODEL_FEATURES
from .plots import plot_signal_stability
from .policy import AnalysisPolicy, validate_analysis_universe

STABILITY_BIN_COLUMNS = [
    "date",
    "sample",
    "spread_bucket",
    "landmark_age_us",
    "latency_us",
    "queue_bin",
    "ofi_bin",
    "queue_center",
    "ofi_center",
    "signals",
    "up_moves",
    "down_moves",
]

STABILITY_BUCKETS = ("all_spreads", "one_tick")
STABILITY_MODELS = {
    name: features
    for name, features in MODEL_FEATURES.items()
    if name in {"intercept", "queue", "ofi", "queue_and_ofi"}
}
STABILITY_COMPARISONS = (
    ("queue", "intercept"),
    ("ofi", "intercept"),
    ("queue_and_ofi", "intercept"),
    ("queue_and_ofi", "ofi"),
)
DEFAULT_BLOCK_LENGTHS = (1, 5, 10)


def _validate_stability_bins(bins: pd.DataFrame) -> None:
    required_columns(bins, STABILITY_BIN_COLUMNS)
    if bins.empty:
        raise ValueError("landmark bins must not be empty")
    if set(bins["sample"].astype(str)) != {"price_spell_landmarks"}:
        raise ValueError("stability audit requires price_spell_landmarks")
    ages = bins["landmark_age_us"].drop_duplicates().to_numpy(dtype=np.int64)
    if len(ages) != 1 or ages[0] <= 0:
        raise ValueError("landmark bins must contain one positive landmark age")

    counts = bins[["signals", "up_moves", "down_moves"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("landmark counts must be finite and non-negative")
    if not np.array_equal(
        bins["signals"].to_numpy(),
        (bins["up_moves"] + bins["down_moves"]).to_numpy(),
    ):
        raise ValueError("signals must equal up_moves plus down_moves")

    zero_latency = bins.loc[bins["latency_us"].eq(0)]
    if zero_latency.empty:
        raise ValueError("zero-latency landmark rows are required")
    missing_buckets = set(STABILITY_BUCKETS) - set(zero_latency["spread_bucket"].astype(str))
    if missing_buckets:
        raise ValueError(
            "stability audit is missing spread buckets: " + ", ".join(sorted(missing_buckets))
        )
    duplicate_keys = [
        "date",
        "spread_bucket",
        "queue_bin",
        "ofi_bin",
    ]
    selected = zero_latency.loc[zero_latency["spread_bucket"].isin(STABILITY_BUCKETS)]
    if selected.duplicated(duplicate_keys).any():
        raise ValueError("zero-latency landmark grid contains duplicate cells")


def _expanded_binary_scores(
    frame: pd.DataFrame,
    row_probability: np.ndarray,
) -> dict[str, float]:
    weights = frame[["up_moves", "down_moves"]].to_numpy(dtype=float).reshape(-1)
    keep = weights > 0
    targets = np.tile([1, 0], len(frame))[keep]
    probabilities = np.repeat(np.asarray(row_probability, dtype=float), 2)[keep]
    return weighted_binary_scores(targets, probabilities, weights[keep])


def _row_brier_loss(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> np.ndarray:
    fitted = np.asarray(probability, dtype=float)
    if len(frame) != len(fitted):
        raise ValueError("probability must contain one value per aggregate row")
    up = frame["up_moves"].to_numpy(dtype=float)
    down = frame["down_moves"].to_numpy(dtype=float)
    return up * (1 - fitted) ** 2 + down * fitted**2


def circular_block_brier_comparison(
    challenger_loss: np.ndarray,
    reference_loss: np.ndarray,
    block_length: int,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    """Compare paired daily Brier losses with circular blocks of trading dates."""

    challenger = np.asarray(challenger_loss, dtype=float)
    reference = np.asarray(reference_loss, dtype=float)
    if challenger.ndim != 1 or reference.ndim != 1 or len(challenger) != len(reference):
        raise ValueError("daily loss arrays must be one-dimensional and aligned")
    if len(challenger) < 2:
        raise ValueError("at least two evaluation dates are required")
    if not np.isfinite(challenger).all() or not np.isfinite(reference).all():
        raise ValueError("daily losses must be finite")
    if (challenger < 0).any() or (reference <= 0).any():
        raise ValueError("daily challenger losses must be non-negative and references positive")
    if not 1 <= block_length <= len(challenger):
        raise ValueError("block length must lie between one and the number of dates")
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")

    point = 1 - challenger.sum() / reference.sum()
    blocks_per_draw = int(np.ceil(len(challenger) / block_length))
    random = np.random.default_rng(random_state)
    starts = random.integers(
        0,
        len(challenger),
        size=(replicates, blocks_per_draw),
    )
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(challenger)
    indices = indices.reshape(replicates, -1)[:, : len(challenger)]
    improvements = 1 - challenger[indices].sum(axis=1) / reference[indices].sum(axis=1)
    lower, median, upper = np.quantile(improvements, [0.025, 0.5, 0.975])
    return {
        "dates": int(len(challenger)),
        "block_length_dates": int(block_length),
        "replicates": int(replicates),
        "relative_brier_reduction": float(point),
        "relative_brier_reduction_lower_95": float(lower),
        "relative_brier_reduction_median": float(median),
        "relative_brier_reduction_upper_95": float(upper),
        "probability_nonpositive": float(np.mean(improvements <= 0)),
    }


def evaluate_signal_stability(
    bins: pd.DataFrame,
    initial_train_months: int = 6,
    block_lengths: tuple[int, ...] = DEFAULT_BLOCK_LENGTHS,
    bootstrap_replicates: int = 10_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Refit at month boundaries and score every later month once."""

    if initial_train_months < 1:
        raise ValueError("initial training months must be positive")
    if not block_lengths or len(set(block_lengths)) != len(block_lengths):
        raise ValueError("block lengths must be non-empty and unique")
    _validate_stability_bins(bins)

    selected = bins.loc[
        bins["latency_us"].eq(0) & bins["spread_bucket"].isin(STABILITY_BUCKETS)
    ].copy()
    selected["date"] = selected["date"].astype(str)
    dates = np.array(sorted(selected["date"].unique()))
    date_periods = pd.PeriodIndex(pd.to_datetime(dates), freq="M")
    months = date_periods.unique().sort_values()
    if len(months) <= initial_train_months:
        raise ValueError("not enough calendar months for rolling-origin evaluation")
    evaluation_months = months[initial_train_months:]
    evaluation_dates = dates[date_periods.isin(evaluation_months)]
    if max(block_lengths) > len(evaluation_dates):
        raise ValueError("block lengths cannot exceed the number of evaluation dates")

    fold_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    for spread_bucket in STABILITY_BUCKETS:
        group = selected.loc[selected["spread_bucket"].eq(spread_bucket)].copy()
        daily_losses: dict[str, list[pd.Series]] = {model: [] for model in STABILITY_MODELS}
        test_frames: list[pd.DataFrame] = []
        row_probabilities: dict[str, list[np.ndarray]] = {model: [] for model in STABILITY_MODELS}

        for fold, month in enumerate(evaluation_months, start=1):
            test_dates = dates[date_periods == month]
            train = group.loc[group["date"] < test_dates[0]].copy()
            test = group.loc[group["date"].isin(test_dates)].copy()
            if train.empty or test.empty:
                raise ValueError(
                    f"{spread_bucket} has no observations for rolling-origin month {month}"
                )
            if train["up_moves"].sum() == 0 or train["down_moves"].sum() == 0:
                raise ValueError("each training origin must contain both directions")
            if test["up_moves"].sum() == 0 or test["down_moves"].sum() == 0:
                raise ValueError("each evaluation month must contain both directions")

            baseline_probability = float(train["up_moves"].sum() / train["signals"].sum())
            test_frames.append(test)
            fold_losses: dict[str, float] = {}
            fitted_rows: list[dict[str, object]] = []

            for model_name, feature_columns in STABILITY_MODELS.items():
                coefficients = {"queue_center": np.nan, "ofi_center": np.nan}
                if feature_columns:
                    x_train, y_train, w_train = compressed_binary_rows(
                        train,
                        feature_columns,
                    )
                    model = LogisticRegression(
                        penalty=None,
                        solver="lbfgs",
                        max_iter=1_000,
                    )
                    model.fit(x_train, y_train, sample_weight=w_train)
                    probability = model.predict_proba(
                        test[list(feature_columns)].to_numpy(dtype=float)
                    )[:, 1]
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
                    probability = np.full(len(test), baseline_probability)
                    intercept = float(np.log(baseline_probability / (1 - baseline_probability)))

                scores = _expanded_binary_scores(test, probability)
                row_loss = _row_brier_loss(test, probability)
                brier_sse = float(row_loss.sum())
                fold_losses[model_name] = brier_sse
                daily = (
                    pd.DataFrame({"date": test["date"], "brier_sse": row_loss})
                    .groupby("date", sort=True, observed=True)["brier_sse"]
                    .sum()
                    .reindex(test_dates, fill_value=0)
                )
                daily_losses[model_name].append(daily)
                row_probabilities[model_name].append(probability)
                fitted_rows.append(
                    {
                        "sample": "price_spell_landmarks",
                        "spread_bucket": spread_bucket,
                        "model": model_name,
                        "fold": fold,
                        "evaluation_month": str(month),
                        "train_first_date": str(train["date"].min()),
                        "train_last_date": str(train["date"].max()),
                        "test_first_date": str(test_dates[0]),
                        "test_last_date": str(test_dates[-1]),
                        "train_dates": int(train["date"].nunique()),
                        "test_dates": int(len(test_dates)),
                        "train_observations": int(train["signals"].sum()),
                        "test_observations": int(test["signals"].sum()),
                        "train_upward_fraction": baseline_probability,
                        "test_upward_fraction": float(
                            test["up_moves"].sum() / test["signals"].sum()
                        ),
                        "intercept": intercept,
                        "queue_coefficient": coefficients["queue_center"],
                        "ofi_coefficient": coefficients["ofi_center"],
                        "brier_sse": brier_sse,
                        **scores,
                    }
                )

            baseline_loss = fold_losses["intercept"]
            for row in fitted_rows:
                row["relative_brier_reduction_vs_intercept"] = float(
                    1 - float(row["brier_sse"]) / baseline_loss
                )
                fold_rows.append(row)

        evaluated = pd.concat(test_frames, ignore_index=True)
        aligned_daily_losses = {
            model: pd.concat(parts).reindex(evaluation_dates).to_numpy(dtype=float)
            for model, parts in daily_losses.items()
        }
        baseline_total_loss = aligned_daily_losses["intercept"].sum()
        for model_name in STABILITY_MODELS:
            probability = np.concatenate(row_probabilities[model_name])
            scores = _expanded_binary_scores(evaluated, probability)
            overall_rows.append(
                {
                    "sample": "price_spell_landmarks",
                    "spread_bucket": spread_bucket,
                    "model": model_name,
                    "train_months_before_first_origin": int(initial_train_months),
                    "evaluation_months": int(len(evaluation_months)),
                    "evaluation_dates": int(len(evaluation_dates)),
                    "evaluation_first_date": str(evaluation_dates[0]),
                    "evaluation_last_date": str(evaluation_dates[-1]),
                    "test_observations": int(evaluated["signals"].sum()),
                    "relative_brier_reduction_vs_intercept": float(
                        1 - aligned_daily_losses[model_name].sum() / baseline_total_loss
                    ),
                    **scores,
                }
            )

        bucket_folds = pd.DataFrame(
            row for row in fold_rows if row["spread_bucket"] == spread_bucket
        )
        for challenger, reference in STABILITY_COMPARISONS:
            challenger_months = bucket_folds.loc[
                bucket_folds["model"].eq(challenger),
                ["evaluation_month", "brier_sse"],
            ].rename(columns={"brier_sse": "challenger_sse"})
            reference_months = bucket_folds.loc[
                bucket_folds["model"].eq(reference),
                ["evaluation_month", "brier_sse"],
            ].rename(columns={"brier_sse": "reference_sse"})
            monthly = challenger_months.merge(
                reference_months,
                on="evaluation_month",
                validate="one_to_one",
            )
            monthly_improvement = 1 - monthly["challenger_sse"] / monthly["reference_sse"]
            for block_length in block_lengths:
                comparison = circular_block_brier_comparison(
                    aligned_daily_losses[challenger],
                    aligned_daily_losses[reference],
                    block_length,
                    bootstrap_replicates,
                    random_state,
                )
                comparison_rows.append(
                    {
                        "sample": "price_spell_landmarks",
                        "spread_bucket": spread_bucket,
                        "challenger": challenger,
                        "reference": reference,
                        "evaluation_first_date": str(evaluation_dates[0]),
                        "evaluation_last_date": str(evaluation_dates[-1]),
                        "positive_months": int((monthly_improvement > 0).sum()),
                        "months": int(len(monthly)),
                        **comparison,
                    }
                )

    folds = (
        pd.DataFrame(fold_rows)
        .sort_values(["spread_bucket", "fold", "model"])
        .reset_index(drop=True)
    )
    overall = (
        pd.DataFrame(overall_rows).sort_values(["spread_bucket", "model"]).reset_index(drop=True)
    )
    comparisons = (
        pd.DataFrame(comparison_rows)
        .sort_values(["spread_bucket", "reference", "challenger", "block_length_dates"])
        .reset_index(drop=True)
    )
    result = {
        "protocol": {
            "sample": "one 100-microsecond landmark per eligible price spell",
            "models": list(STABILITY_MODELS),
            "spread_buckets": list(STABILITY_BUCKETS),
            "initial_train_months": int(initial_train_months),
            "initial_train_first_date": str(dates[0]),
            "initial_train_last_date": str(
                dates[date_periods == months[initial_train_months - 1]][-1]
            ),
            "evaluation_months": [str(month) for month in evaluation_months],
            "evaluation_dates": int(len(evaluation_dates)),
            "evaluation_first_date": str(evaluation_dates[0]),
            "evaluation_last_date": str(evaluation_dates[-1]),
            "refit_rule": (
                "expanding window refit at each calendar-month boundary using only preceding dates"
            ),
            "baseline": "up-move frequency on dates preceding each origin",
            "primary_comparison": "queue_and_ofi versus intercept across all spreads",
            "block_bootstrap": {
                "method": "paired circular blocks of consecutive trading dates",
                "block_lengths_dates": list(block_lengths),
                "replicates": int(bootstrap_replicates),
                "random_state": int(random_state),
            },
            "interpretation": (
                "post-hoc split-sensitivity audit on the same stock-year; it is "
                "not an untouched confirmation or an execution study"
            ),
        },
        "fold_rows": int(len(folds)),
        "metric_rows": int(len(overall)),
        "comparison_rows": int(len(comparisons)),
    }
    return folds, overall, comparisons, result


def run_signal_stability_analysis(
    bins_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Run the rolling-origin audit and persist compact aggregate evidence."""

    bins = pd.read_csv(
        bins_path,
        usecols=lambda column: column in STABILITY_BIN_COLUMNS,
    )
    validate_analysis_universe(bins["date"], analysis_policy)
    folds, overall, comparisons, result = evaluate_signal_stability(bins)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "price_spell_signal_stability_folds.csv", index=False)
    overall.to_csv(output / "price_spell_signal_stability_metrics.csv", index=False)
    comparisons.to_csv(
        output / "price_spell_signal_stability_comparisons.csv",
        index=False,
    )
    serialized = json.dumps(result, indent=2)
    (output / "price_spell_signal_stability_model.json").write_text(serialized + "\n")
    if figures_dir is not None:
        plot_signal_stability(
            folds,
            Path(figures_dir) / "price-spell-signal-stability.pdf",
        )
    return json.loads(serialized)
