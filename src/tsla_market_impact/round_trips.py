"""Depth-constrained round trips from non-overlapping price-spell landmarks."""

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
    weighted_confidence_cutoff,
)
from .orderflow import MODEL_FEATURES
from .plots import plot_price_spell_round_trips
from .policy import (
    AnalysisPolicy,
    split_index_for_test_start,
    validate_analysis_universe,
)

ROUND_TRIP_BIN_COLUMNS = [
    "date",
    "sample",
    "spread_bucket",
    "landmark_age_us",
    "entry_latency_us",
    "shares",
    "queue_bin",
    "ofi_bin",
    "queue_center",
    "ofi_center",
    "signals",
    "arrived",
    "stale",
    "up_moves",
    "down_moves",
    "long_fills",
    "long_capacity_censored",
    "long_midpoint_move_sum_bps",
    "long_entry_cost_sum_bps",
    "long_exit_cost_sum_bps",
    "long_quoted_pnl_sum_bps",
    "short_fills",
    "short_capacity_censored",
    "short_midpoint_move_sum_bps",
    "short_entry_cost_sum_bps",
    "short_exit_cost_sum_bps",
    "short_quoted_pnl_sum_bps",
]

ROUND_TRIP_MODELS = {
    name: features
    for name, features in MODEL_FEATURES.items()
    if name in {"queue", "ofi", "queue_and_ofi"}
}

DEFAULT_TRAIN_SIGNAL_FRACTIONS = (1.0, 0.20, 0.10, 0.05)

COUNT_COLUMNS = [
    "signals",
    "arrived",
    "stale",
    "up_moves",
    "down_moves",
    "long_fills",
    "long_capacity_censored",
    "short_fills",
    "short_capacity_censored",
]

SUM_COLUMNS = [
    f"{side}_{measure}_sum_bps"
    for side in ("long", "short")
    for measure in ("midpoint_move", "entry_cost", "exit_cost", "quoted_pnl")
]


def _validate_round_trip_bins(bins: pd.DataFrame) -> None:
    required_columns(bins, ROUND_TRIP_BIN_COLUMNS)
    if set(bins["sample"]) != {"price_spell_round_trips"}:
        raise ValueError("round-trip bins must use the price_spell_round_trips sample")
    if set(bins["spread_bucket"]) - {"all_spreads", "one_tick"}:
        raise ValueError("round-trip bins must contain only pre-specified spread samples")

    ages = bins["landmark_age_us"].drop_duplicates().to_numpy(dtype=np.int64)
    if len(ages) != 1 or ages[0] <= 0:
        raise ValueError("round-trip bins must contain one positive landmark age")
    if (bins["entry_latency_us"] < 0).any():
        raise ValueError("entry latencies must be non-negative")
    if not bins["entry_latency_us"].eq(0).any():
        raise ValueError("zero entry latency is required to fit the direction model")
    if (bins["shares"] <= 0).any() or not bins["shares"].eq(1).any():
        raise ValueError("positive order sizes including one share are required")

    counts = bins[COUNT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("round-trip counts must be finite and non-negative")
    if not np.array_equal(
        bins["signals"].to_numpy(),
        (bins["arrived"] + bins["stale"]).to_numpy(),
    ):
        raise ValueError("signals must equal arrived plus stale")
    if not np.array_equal(
        bins["signals"].to_numpy(),
        (bins["up_moves"] + bins["down_moves"]).to_numpy(),
    ):
        raise ValueError("signals must equal up_moves plus down_moves")
    for side in ("long", "short"):
        if not np.array_equal(
            bins["arrived"].to_numpy(),
            (bins[f"{side}_fills"] + bins[f"{side}_capacity_censored"]).to_numpy(),
        ):
            raise ValueError(f"arrived must equal {side} fills plus capacity-censored signals")

    one_share = bins["shares"].eq(1)
    if (
        bins.loc[
            one_share,
            ["long_capacity_censored", "short_capacity_censored"],
        ].to_numpy()
        != 0
    ).any():
        raise ValueError("one-share round trips cannot be capacity-censored")

    sums = bins[SUM_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(sums).all():
        raise ValueError("round-trip sums must be finite")
    if (
        bins[
            [
                "long_entry_cost_sum_bps",
                "long_exit_cost_sum_bps",
                "short_entry_cost_sum_bps",
                "short_exit_cost_sum_bps",
            ]
        ].to_numpy(dtype=float)
        < -1e-9
    ).any():
        raise ValueError("entry and exit costs must be non-negative")
    for side in ("long", "short"):
        expected = (
            bins[f"{side}_midpoint_move_sum_bps"]
            - bins[f"{side}_entry_cost_sum_bps"]
            - bins[f"{side}_exit_cost_sum_bps"]
        )
        if not np.allclose(
            bins[f"{side}_quoted_pnl_sum_bps"],
            expected,
            rtol=1e-9,
            atol=1e-7,
        ):
            raise ValueError(
                f"{side} quoted PnL must equal midpoint move minus both execution costs"
            )


def _cluster_round_trip(
    frame: pd.DataFrame,
    action: np.ndarray,
    selected: np.ndarray,
    dates: np.ndarray,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    selected_frame = frame.loc[selected].copy()
    selected_action = action[selected]
    long_action = selected_action > 0
    selected_frame["fills"] = np.where(
        long_action,
        selected_frame["long_fills"],
        selected_frame["short_fills"],
    )
    for measure in (
        "midpoint_move",
        "entry_cost",
        "exit_cost",
        "quoted_pnl",
    ):
        selected_frame[f"{measure}_sum_bps"] = np.where(
            long_action,
            selected_frame[f"long_{measure}_sum_bps"],
            selected_frame[f"short_{measure}_sum_bps"],
        )

    daily = (
        selected_frame.groupby("date", sort=True, observed=True)
        .agg(
            fills=("fills", "sum"),
            midpoint_move_sum_bps=("midpoint_move_sum_bps", "sum"),
            entry_cost_sum_bps=("entry_cost_sum_bps", "sum"),
            exit_cost_sum_bps=("exit_cost_sum_bps", "sum"),
            quoted_pnl_sum_bps=("quoted_pnl_sum_bps", "sum"),
        )
        .reindex(dates, fill_value=0)
    )
    values = daily[
        [
            "fills",
            "midpoint_move_sum_bps",
            "entry_cost_sum_bps",
            "exit_cost_sum_bps",
            "quoted_pnl_sum_bps",
        ]
    ].to_numpy(dtype=float)
    fills = values[:, 0].sum()
    if fills <= 0:
        raise ValueError("confidence rule produces no filled round trips")
    means = values[:, 1:].sum(axis=0) / fills

    random = np.random.default_rng(random_state)
    sampled = random.integers(0, len(values), size=(replicates, len(values)))
    totals = values[sampled].sum(axis=1)
    valid = totals[:, 0] > 0
    if not valid.any():
        raise ValueError("bootstrap samples contain no filled round trips")
    pnl_means = totals[valid, 4] / totals[valid, 0]
    lower, median, upper = np.quantile(pnl_means, [0.025, 0.5, 0.975])
    return {
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "midpoint_move_mean_bps": float(means[0]),
        "entry_cost_mean_bps": float(means[1]),
        "exit_cost_mean_bps": float(means[2]),
        "quoted_round_trip_mean_bps": float(means[3]),
        "quoted_round_trip_lower_95_bps": float(lower),
        "quoted_round_trip_median_bps": float(median),
        "quoted_round_trip_upper_95_bps": float(upper),
        "probability_quoted_pnl_nonpositive": float(np.mean(pnl_means <= 0)),
    }


def evaluate_price_spell_round_trips(
    bins: pd.DataFrame,
    test_date_fraction: float = 0.20,
    train_signal_fractions: tuple[float, ...] = DEFAULT_TRAIN_SIGNAL_FRACTIONS,
    bootstrap_replicates: int = 10_000,
    random_state: int = 42,
    test_start_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate quoted entry and exit VWAPs at fixed price-spell landmarks."""

    if test_start_date is None and not 0 < test_date_fraction < 1:
        raise ValueError("test_date_fraction must lie between zero and one")
    _validate_round_trip_bins(bins)
    dates = np.array(sorted(bins["date"].astype(str).unique()))
    if len(dates) < 2:
        raise ValueError("At least two dates are required")
    if test_start_date is None:
        split = int(np.floor((1 - test_date_fraction) * len(dates)))
        split = min(max(split, 1), len(dates) - 1)
    else:
        split = split_index_for_test_start(dates.tolist(), test_start_date)
    train_dates = set(dates[:split])
    test_dates = dates[split:]
    latencies = sorted(int(value) for value in bins["entry_latency_us"].unique())
    order_sizes = sorted(int(value) for value in bins["shares"].unique())
    reference = bins.loc[bins["entry_latency_us"].eq(0) & bins["shares"].eq(1)]

    rows: list[dict[str, object]] = []
    for (sample, spread_bucket), group_reference in reference.groupby(
        ["sample", "spread_bucket"],
        sort=True,
        observed=True,
    ):
        train_reference = group_reference.loc[
            group_reference["date"].astype(str).isin(train_dates)
        ].copy()
        test_reference = group_reference.loc[
            ~group_reference["date"].astype(str).isin(train_dates)
        ].copy()
        if (
            train_reference["up_moves"].sum() == 0
            or train_reference["down_moves"].sum() == 0
            or test_reference["up_moves"].sum() == 0
            or test_reference["down_moves"].sum() == 0
        ):
            continue

        group = bins.loc[bins["sample"].eq(sample) & bins["spread_bucket"].eq(spread_bucket)]
        for model_name, feature_columns in ROUND_TRIP_MODELS.items():
            x_train, y_train, w_train = compressed_binary_rows(
                train_reference.rename(columns={"signals": "observations"}),
                feature_columns,
            )
            x_test, y_test, w_test = compressed_binary_rows(
                test_reference.rename(columns={"signals": "observations"}),
                feature_columns,
            )
            model = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=1_000,
            )
            model.fit(x_train, y_train, sample_weight=w_train)
            train_probability = model.predict_proba(
                train_reference[list(feature_columns)].to_numpy(dtype=float)
            )[:, 1]
            test_probability = model.predict_proba(
                test_reference[list(feature_columns)].to_numpy(dtype=float)
            )[:, 1]
            direction_scores = weighted_binary_scores(
                y_test,
                model.predict_proba(x_test)[:, 1],
                w_test,
            )
            baseline_probability = float(np.average(y_train, weights=w_train))
            baseline_scores = weighted_binary_scores(
                y_test,
                np.full(len(y_test), baseline_probability),
                w_test,
            )
            direction_comparison = day_cluster_brier_comparison(
                test_reference,
                test_probability,
                np.full(len(test_reference), baseline_probability),
                bootstrap_replicates,
                random_state,
            )
            train_confidence = np.abs(train_probability - 0.5)
            train_weights = train_reference["signals"].to_numpy(dtype=float)

            for target_fraction in train_signal_fractions:
                cutoff = weighted_confidence_cutoff(
                    train_confidence,
                    train_weights,
                    target_fraction,
                )
                train_selected = train_confidence >= cutoff
                achieved_train_fraction = float(
                    train_weights[train_selected].sum() / train_weights.sum()
                )

                for latency in latencies:
                    for shares in order_sizes:
                        scenario = group.loc[
                            group["entry_latency_us"].eq(latency) & group["shares"].eq(shares)
                        ]
                        test = scenario.loc[~scenario["date"].astype(str).isin(train_dates)].copy()
                        probability = model.predict_proba(
                            test[list(feature_columns)].to_numpy(dtype=float)
                        )[:, 1]
                        confidence = np.abs(probability - 0.5)
                        selected = confidence >= cutoff
                        action = np.where(probability >= 0.5, 1.0, -1.0)
                        selected_signals = int(test.loc[selected, "signals"].sum())
                        arrived = int(test.loc[selected, "arrived"].sum())
                        chosen_fills = np.where(
                            action[selected] > 0,
                            test.loc[selected, "long_fills"],
                            test.loc[selected, "short_fills"],
                        ).astype(float)
                        fills = int(chosen_fills.sum())
                        if selected_signals == 0:
                            continue
                        if fills > 0:
                            economics = _cluster_round_trip(
                                test,
                                action,
                                selected,
                                test_dates,
                                bootstrap_replicates,
                                random_state,
                            )
                            buy_fills = np.where(
                                action[selected] > 0,
                                test.loc[selected, "long_fills"],
                                0,
                            ).astype(float)
                            buy_fraction = float(buy_fills.sum() / fills)
                        else:
                            economics = {
                                "clusters": int(len(test_dates)),
                                "replicates": 0,
                                "midpoint_move_mean_bps": np.nan,
                                "entry_cost_mean_bps": np.nan,
                                "exit_cost_mean_bps": np.nan,
                                "quoted_round_trip_mean_bps": np.nan,
                                "quoted_round_trip_lower_95_bps": np.nan,
                                "quoted_round_trip_median_bps": np.nan,
                                "quoted_round_trip_upper_95_bps": np.nan,
                                "probability_quoted_pnl_nonpositive": np.nan,
                            }
                            buy_fraction = np.nan
                        rows.append(
                            {
                                "sample": sample,
                                "spread_bucket": spread_bucket,
                                "model": model_name,
                                "target_train_signal_fraction": target_fraction,
                                "achieved_train_signal_fraction": (achieved_train_fraction),
                                "test_signal_fraction": (
                                    selected_signals / int(test["signals"].sum())
                                ),
                                "confidence_cutoff": cutoff,
                                "landmark_age_us": int(bins["landmark_age_us"].iloc[0]),
                                "entry_latency_us": latency,
                                "shares": shares,
                                "train_first_date": str(dates[0]),
                                "train_last_date": str(dates[split - 1]),
                                "test_first_date": str(dates[split]),
                                "test_last_date": str(dates[-1]),
                                "train_signals": int(w_train.sum()),
                                "test_signals": int(w_test.sum()),
                                "selected_test_signals": selected_signals,
                                "arrived_test_signals": arrived,
                                "filled_round_trips": fills,
                                "stale_fraction": 1 - arrived / selected_signals,
                                "capacity_censored_fraction": (
                                    1 - fills / arrived if arrived else np.nan
                                ),
                                "buy_fraction": buy_fraction,
                                "direction_roc_auc": direction_scores["roc_auc"],
                                "direction_accuracy": direction_scores["accuracy"],
                                "direction_log_loss": direction_scores["log_loss"],
                                "direction_brier_score": direction_scores["brier_score"],
                                "direction_baseline_brier_score": baseline_scores["brier_score"],
                                **{
                                    f"direction_{key}": value
                                    for key, value in direction_comparison.items()
                                },
                                "intercept": float(model.intercept_[0]),
                                "queue_coefficient": (
                                    float(
                                        model.coef_[
                                            0,
                                            feature_columns.index("queue_center"),
                                        ]
                                    )
                                    if "queue_center" in feature_columns
                                    else np.nan
                                ),
                                "ofi_coefficient": (
                                    float(
                                        model.coef_[
                                            0,
                                            feature_columns.index("ofi_center"),
                                        ]
                                    )
                                    if "ofi_center" in feature_columns
                                    else np.nan
                                ),
                                **economics,
                            }
                        )

    metrics = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "sample",
                "spread_bucket",
                "model",
                "target_train_signal_fraction",
                "shares",
                "entry_latency_us",
            ]
        )
        .reset_index(drop=True)
    )
    headline = metrics.loc[
        metrics["model"].eq("queue_and_ofi")
        & metrics["target_train_signal_fraction"].eq(0.10)
        & metrics["shares"].eq(1)
        & metrics["entry_latency_us"].eq(0)
    ]
    result = {
        "protocol": {
            "dates": int(len(dates)),
            "train_dates": int(split),
            "test_dates": int(len(dates) - split),
            "train_first_date": str(dates[0]),
            "train_last_date": str(dates[split - 1]),
            "test_first_date": str(dates[split]),
            "test_last_date": str(dates[-1]),
            "sample": ("one fixed clock-time landmark per eligible constant-mid-price spell"),
            "spread_samples": ["all_spreads", "one_tick"],
            "landmark_age_us": int(bins["landmark_age_us"].iloc[0]),
            "entry_latencies_us": latencies,
            "order_sizes_shares": order_sizes,
            "entry": (
                "marketable VWAP across the two displayed levels immediately before "
                "the entry deadline"
            ),
            "exit": (
                "opposite marketable VWAP across the two displayed levels in the "
                "snapshot immediately after the next mid-price change"
            ),
            "capacity": (
                "a round trip is retained only when the requested shares are "
                "displayed across both entry and exit level-2 books"
            ),
            "fees": (
                "zero; quoted PnL is an optimistic benchmark and any non-negative fee lowers it"
            ),
            "bootstrap_unit": "trading date",
            "selection_warning": (
                "this stress test was specified after inspecting the preceding "
                "midpoint-markout result; it is not a new untouched validation"
            ),
            "interpretation_warning": (
                "the exit has zero reaction latency and displayed liquidity is "
                "assumed simultaneously fillable; impact, queue competition, "
                "inventory, capital, and risk limits remain outside the model"
            ),
        },
        "metric_rows": int(len(metrics)),
        "headline": json.loads(headline.to_json(orient="records")),
    }
    return metrics, result


def run_price_spell_round_trip_analysis(
    bins_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Evaluate depth-constrained landmark round trips and persist evidence."""

    bins = pd.read_csv(
        bins_path,
        dtype={
            "date": "string",
            "sample": "category",
            "spread_bucket": "category",
        },
    )
    validate_analysis_universe(bins["date"], analysis_policy)
    metrics, result = evaluate_price_spell_round_trips(
        bins,
        test_start_date=analysis_policy.test_start,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "price_spell_round_trip_metrics.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "price_spell_round_trip_model.json").write_text(serialized + "\n")
    if figures_dir is not None:
        plot_price_spell_round_trips(
            metrics,
            Path(figures_dir) / "price-spell-round-trips.pdf",
        )
    return json.loads(serialized)
