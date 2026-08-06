"""Chronological marketable-markout diagnostic for next-move signals."""

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
from .plots import plot_marketable_markouts
from .policy import (
    AnalysisPolicy,
    split_index_for_test_start,
    validate_analysis_universe,
)

MARKOUT_BIN_COLUMNS = [
    "date",
    "sample",
    "spread_bucket",
    "latency_us",
    "queue_bin",
    "ofi_bin",
    "queue_center",
    "ofi_center",
    "signals",
    "executable",
    "stale",
    "up_moves",
    "down_moves",
    "midpoint_move_sum_bps",
    "half_spread_sum_bps",
]

LANDMARK_BIN_COLUMNS = [*MARKOUT_BIN_COLUMNS, "landmark_age_us"]

MARKOUT_MODELS = {
    name: features
    for name, features in MODEL_FEATURES.items()
    if name in {"queue", "ofi", "queue_and_ofi"}
}

DEFAULT_TRAIN_SIGNAL_FRACTIONS = (1.0, 0.20, 0.10, 0.05)


def _validate_markout_bins(bins: pd.DataFrame) -> None:
    required_columns(bins, MARKOUT_BIN_COLUMNS)
    counts = bins[["signals", "executable", "stale", "up_moves", "down_moves"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("markout counts must be finite and non-negative")
    if not np.array_equal(
        bins["signals"].to_numpy(),
        (bins["up_moves"] + bins["down_moves"]).to_numpy(),
    ):
        raise ValueError("signals must equal up_moves plus down_moves")
    if not np.array_equal(
        bins["signals"].to_numpy(),
        (bins["executable"] + bins["stale"]).to_numpy(),
    ):
        raise ValueError("signals must equal executable plus stale")
    if (bins["latency_us"] < 0).any():
        raise ValueError("latencies must be non-negative")
    zero_latency = bins["latency_us"].eq(0)
    if not zero_latency.any():
        raise ValueError("zero-latency rows are required to fit the direction model")
    if bins.loc[zero_latency, "stale"].sum() != 0:
        raise ValueError("zero-latency signals cannot be stale")
    markouts = bins[["midpoint_move_sum_bps", "half_spread_sum_bps"]].to_numpy(dtype=float)
    if not np.isfinite(markouts).all():
        raise ValueError("markout sums must be finite")
    if (bins["half_spread_sum_bps"] < 0).any():
        raise ValueError("half-spread costs must be non-negative")


def _cluster_markout(
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
    selected_frame["gross_sum_bps"] = selected_action * selected_frame[
        "midpoint_move_sum_bps"
    ].to_numpy(dtype=float)
    selected_frame["net_sum_bps"] = selected_frame["gross_sum_bps"] - selected_frame[
        "half_spread_sum_bps"
    ].to_numpy(dtype=float)
    daily = (
        selected_frame.groupby("date", sort=True, observed=True)
        .agg(
            executions=("executable", "sum"),
            gross_sum_bps=("gross_sum_bps", "sum"),
            half_spread_sum_bps=("half_spread_sum_bps", "sum"),
            net_sum_bps=("net_sum_bps", "sum"),
        )
        .reindex(dates, fill_value=0)
    )
    values = daily[["executions", "gross_sum_bps", "half_spread_sum_bps", "net_sum_bps"]].to_numpy(
        dtype=float
    )
    executions = values[:, 0].sum()
    if executions <= 0:
        raise ValueError("confidence rule produces no executable test signals")
    means = values[:, 1:].sum(axis=0) / executions

    random = np.random.default_rng(random_state)
    sampled = random.integers(0, len(values), size=(replicates, len(values)))
    totals = values[sampled].sum(axis=1)
    valid = totals[:, 0] > 0
    if not valid.any():
        raise ValueError("bootstrap samples contain no executable test signals")
    net_means = totals[valid, 3] / totals[valid, 0]
    lower, median, upper = np.quantile(net_means, [0.025, 0.5, 0.975])
    return {
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "gross_midpoint_markout_mean_bps": float(means[0]),
        "half_spread_mean_bps": float(means[1]),
        "net_markout_mean_bps": float(means[2]),
        "net_markout_lower_95_bps": float(lower),
        "net_markout_median_bps": float(median),
        "net_markout_upper_95_bps": float(upper),
        "probability_net_nonpositive": float(np.mean(net_means <= 0)),
    }


def evaluate_marketable_markouts(
    bins: pd.DataFrame,
    test_start_date: str,
    train_signal_fractions: tuple[float, ...] = DEFAULT_TRAIN_SIGNAL_FRACTIONS,
    bootstrap_replicates: int = 10_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate next-move signals after crossing the displayed spread."""

    _validate_markout_bins(bins)
    dates = np.array(sorted(bins["date"].astype(str).unique()))
    if len(dates) < 2:
        raise ValueError("At least two dates are required")
    split = split_index_for_test_start(dates.tolist(), test_start_date)
    train_dates = set(dates[:split])
    test_dates = dates[split:]
    latencies = sorted(int(value) for value in bins["latency_us"].unique())
    zero_latency = bins.loc[bins["latency_us"].eq(0)]

    rows: list[dict[str, object]] = []
    for (sample, spread_bucket), reference in zero_latency.groupby(
        ["sample", "spread_bucket"],
        sort=True,
        observed=True,
    ):
        train_reference = reference.loc[reference["date"].astype(str).isin(train_dates)].copy()
        test_reference = reference.loc[~reference["date"].astype(str).isin(train_dates)].copy()
        if (
            train_reference["up_moves"].sum() == 0
            or train_reference["down_moves"].sum() == 0
            or test_reference["up_moves"].sum() == 0
            or test_reference["down_moves"].sum() == 0
        ):
            continue

        group = bins.loc[bins["sample"].eq(sample) & bins["spread_bucket"].eq(spread_bucket)]
        for model_name, feature_columns in MARKOUT_MODELS.items():
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
            expanded_test_probability = model.predict_proba(x_test)[:, 1]
            direction_scores = weighted_binary_scores(
                y_test,
                expanded_test_probability,
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
                    latency_frame = group.loc[group["latency_us"].eq(latency)]
                    test = latency_frame.loc[
                        ~latency_frame["date"].astype(str).isin(train_dates)
                    ].copy()
                    probability = model.predict_proba(
                        test[list(feature_columns)].to_numpy(dtype=float)
                    )[:, 1]
                    confidence = np.abs(probability - 0.5)
                    selected = confidence >= cutoff
                    action = np.where(probability >= 0.5, 1.0, -1.0)
                    selected_signals = int(test.loc[selected, "signals"].sum())
                    executions = int(test.loc[selected, "executable"].sum())
                    if selected_signals == 0 or executions == 0:
                        continue
                    economics = _cluster_markout(
                        test,
                        action,
                        selected,
                        test_dates,
                        bootstrap_replicates,
                        random_state,
                    )
                    selected_executable = test.loc[selected, "executable"].to_numpy(dtype=float)
                    buy_fraction = float(
                        np.sum(selected_executable[action[selected] > 0]) / executions
                    )
                    rows.append(
                        {
                            "sample": sample,
                            "spread_bucket": spread_bucket,
                            "model": model_name,
                            "target_train_signal_fraction": target_fraction,
                            "achieved_train_signal_fraction": (achieved_train_fraction),
                            "test_signal_fraction": (selected_signals / int(test["signals"].sum())),
                            "confidence_cutoff": cutoff,
                            "latency_us": latency,
                            "train_first_date": str(dates[0]),
                            "train_last_date": str(dates[split - 1]),
                            "test_first_date": str(dates[split]),
                            "test_last_date": str(dates[-1]),
                            "train_signals": int(w_train.sum()),
                            "test_signals": int(w_test.sum()),
                            "selected_test_signals": selected_signals,
                            "executable_test_signals": executions,
                            "stale_fraction": 1 - executions / selected_signals,
                            "buy_fraction": buy_fraction,
                            "direction_roc_auc": direction_scores["roc_auc"],
                            "direction_accuracy": direction_scores["accuracy"],
                            "direction_log_loss": direction_scores["log_loss"],
                            "direction_brier_score": direction_scores["brier_score"],
                            "direction_baseline_brier_score": baseline_scores[
                                "brier_score"
                            ],
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
                "latency_us",
            ]
        )
        .reset_index(drop=True)
    )
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
            "decision_rule": (
                "buy at or above 0.5 and sell below 0.5; confidence cutoffs "
                "are weighted quantiles of absolute train-date probability "
                "distance from 0.5"
            ),
            "mark": "midpoint immediately after the next mid-price change",
            "execution": (
                "zero latency uses the signal's post-event displayed best; "
                "positive-latency entry uses the last displayed best strictly "
                "before the deadline, provided that deadline strictly precedes "
                "the next mid-price change"
            ),
            "cost": (
                "displayed half-spread; fees and impact excluded; the stated "
                "aggregate delay is applied to LOBSTER event time"
            ),
            "bootstrap_unit": "trading date",
            "overlap_warning": (
                "states within one constant-mid-price spell share the "
                "next-move outcome; results are diagnostics, not additive "
                "strategy returns"
            ),
        },
        "models": json.loads(metrics.to_json(orient="records")),
    }
    return metrics, result


def evaluate_price_spell_landmarks(
    bins: pd.DataFrame,
    test_start_date: str,
    train_signal_fractions: tuple[float, ...] = DEFAULT_TRAIN_SIGNAL_FRACTIONS,
    bootstrap_replicates: int = 10_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate one pre-specified clock-time signal per eligible price spell."""

    required_columns(bins, LANDMARK_BIN_COLUMNS)
    ages = bins["landmark_age_us"].drop_duplicates().to_numpy(dtype=np.int64)
    if len(ages) != 1 or ages[0] <= 0:
        raise ValueError("landmark bins must contain one positive landmark age")
    if set(bins["sample"]) != {"price_spell_landmarks"}:
        raise ValueError("landmark bins must use the price_spell_landmarks sample")

    metrics, result = evaluate_marketable_markouts(
        bins,
        test_start_date=test_start_date,
        train_signal_fractions=train_signal_fractions,
        bootstrap_replicates=bootstrap_replicates,
        random_state=random_state,
    )
    protocol = result["protocol"]
    protocol["landmark_age_us"] = int(ages[0])
    protocol["observation"] = (
        "one prevailing book state at the fixed clock-time landmark after each "
        "constant-mid-price spell begins; spells ending at or before the landmark "
        "do not produce a signal"
    )
    protocol["execution"] = (
        "one-unit marketable order at the displayed best quote prevailing after "
        "the stated post-landmark latency, provided the next mid-price change has "
        "not already occurred"
    )
    protocol["nonoverlap"] = (
        "each eligible constant-mid-price spell contributes at most one signal, "
        "and price spells are sequential within a session"
    )
    protocol["interpretation_warning"] = (
        "the terminal midpoint is a mark rather than an executable exit; fees, "
        "impact, fill uncertainty, inventory, capital, and risk limits are excluded"
    )
    protocol.pop("overlap_warning")
    return metrics, result


def run_marketable_markout_analysis(
    bins_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Evaluate marketable markouts and persist aggregate evidence."""

    bins = pd.read_csv(bins_path)
    validate_analysis_universe(bins["date"], analysis_policy)
    metrics, result = evaluate_marketable_markouts(
        bins,
        test_start_date=analysis_policy.test_start,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "marketable_markout_metrics.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "marketable_markout_model.json").write_text(serialized + "\n")
    if figures_dir is not None:
        plot_marketable_markouts(
            metrics,
            Path(figures_dir) / "marketable-markouts.pdf",
        )
    return json.loads(serialized)


def run_price_spell_landmark_analysis(
    bins_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Evaluate non-overlapping price-spell landmarks and persist aggregates."""

    bins = pd.read_csv(bins_path)
    validate_analysis_universe(bins["date"], analysis_policy)
    metrics, result = evaluate_price_spell_landmarks(
        bins,
        test_start_date=analysis_policy.test_start,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "price_spell_landmark_metrics.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "price_spell_landmark_model.json").write_text(serialized + "\n")
    if figures_dir is not None:
        plot_marketable_markouts(
            metrics,
            Path(figures_dir) / "price-spell-landmarks.pdf",
            sample="price_spell_landmarks",
        )
    return json.loads(serialized)
