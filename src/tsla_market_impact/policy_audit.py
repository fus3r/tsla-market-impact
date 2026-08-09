"""Selection-aware inference for the depth-constrained round-trip policy grid."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .data import required_columns
from .plots import plot_round_trip_policy_audit
from .policy import AnalysisPolicy, validate_analysis_universe
from .round_trips import _validate_round_trip_bins

POLICY_COLUMNS = [
    "sample",
    "spread_bucket",
    "model",
    "target_train_signal_fraction",
    "entry_latency_us",
    "shares",
]

POLICY_METRIC_COLUMNS = [
    *POLICY_COLUMNS,
    "confidence_cutoff",
    "test_first_date",
    "test_last_date",
    "selected_test_signals",
    "arrived_test_signals",
    "filled_round_trips",
    "intercept",
    "queue_coefficient",
    "ofi_coefficient",
    "clusters",
    "quoted_round_trip_mean_bps",
    "quoted_round_trip_lower_95_bps",
    "quoted_round_trip_upper_95_bps",
]


def _validate_policy_metrics(metrics: pd.DataFrame) -> None:
    required_columns(metrics, POLICY_METRIC_COLUMNS)
    if metrics.empty:
        raise ValueError("round-trip policy metrics cannot be empty")
    if metrics.duplicated(POLICY_COLUMNS).any():
        raise ValueError("round-trip policy metrics contain duplicate policies")

    dimensions = [sorted(metrics[column].drop_duplicates()) for column in POLICY_COLUMNS]
    expected = set(product(*dimensions))
    observed = set(metrics[POLICY_COLUMNS].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("round-trip policy metrics must contain the complete policy grid")

    for column in ("test_first_date", "test_last_date", "clusters"):
        if metrics[column].nunique(dropna=False) != 1:
            raise ValueError(f"round-trip policies must share one {column}")
    if int(metrics["clusters"].iloc[0]) < 2:
        raise ValueError("at least two test-date clusters are required")

    counts = metrics[
        ["selected_test_signals", "arrived_test_signals", "filled_round_trips"]
    ].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("round-trip policy counts must be finite and non-negative")
    if (metrics["arrived_test_signals"] > metrics["selected_test_signals"]).any():
        raise ValueError("arrived signals cannot exceed selected signals")
    if (metrics["filled_round_trips"] > metrics["arrived_test_signals"]).any():
        raise ValueError("filled round trips cannot exceed arrived signals")

    has_fills = metrics["filled_round_trips"] > 0
    if metrics.loc[has_fills, "quoted_round_trip_mean_bps"].isna().any():
        raise ValueError("filled policies require a quoted PnL estimate")
    if metrics.loc[~has_fills, "quoted_round_trip_mean_bps"].notna().any():
        raise ValueError("zero-fill policies cannot have a quoted PnL estimate")


def _policy_probability(frame: pd.DataFrame, metric: pd.Series) -> np.ndarray:
    log_odds = np.full(len(frame), float(metric["intercept"]), dtype=float)
    for feature, coefficient in (
        ("queue_center", metric["queue_coefficient"]),
        ("ofi_center", metric["ofi_coefficient"]),
    ):
        if pd.notna(coefficient):
            log_odds += float(coefficient) * frame[feature].to_numpy(dtype=float)
    return expit(log_odds)


def _daily_policy_outcome(
    frame: pd.DataFrame,
    metric: pd.Series,
    test_dates: np.ndarray,
) -> pd.DataFrame:
    probability = _policy_probability(frame, metric)
    confidence = np.abs(probability - 0.5)
    cutoff = float(metric["confidence_cutoff"])
    # Reconstructed logits can straddle their serialized fitted cutoff by a few ulps.
    selected = (confidence >= cutoff) | np.isclose(
        confidence,
        cutoff,
        rtol=0,
        atol=np.finfo(float).eps,
    )
    long_action = probability >= 0.5
    chosen_fills = np.where(
        long_action,
        frame["long_fills"],
        frame["short_fills"],
    ).astype(float)
    chosen_pnl = np.where(
        long_action,
        frame["long_quoted_pnl_sum_bps"],
        frame["short_quoted_pnl_sum_bps"],
    ).astype(float)

    selected_signals = int(frame.loc[selected, "signals"].sum())
    arrived_signals = int(frame.loc[selected, "arrived"].sum())
    filled_round_trips = int(chosen_fills[selected].sum())
    expected_counts = (
        int(metric["selected_test_signals"]),
        int(metric["arrived_test_signals"]),
        int(metric["filled_round_trips"]),
    )
    if (selected_signals, arrived_signals, filled_round_trips) != expected_counts:
        raise ValueError("policy reconstruction does not reproduce the stored counts")

    daily = (
        pd.DataFrame(
            {
                "date": frame["date"].astype(str).to_numpy(),
                "fills": np.where(selected, chosen_fills, 0),
                "quoted_pnl_sum_bps": np.where(selected, chosen_pnl, 0),
            }
        )
        .groupby("date", sort=True, observed=True)
        .sum()
        .reindex(test_dates, fill_value=0)
    )
    if filled_round_trips:
        reconstructed = daily["quoted_pnl_sum_bps"].sum() / daily["fills"].sum()
        if not np.isclose(
            reconstructed,
            float(metric["quoted_round_trip_mean_bps"]),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError("policy reconstruction does not reproduce quoted PnL")
    return daily


def _status(total_fills: float, filled_dates: int, cluster_scale: float) -> str:
    if total_fills == 0:
        return "zero_fills"
    if filled_dates < 2:
        return "one_filled_date"
    if cluster_scale == 0:
        return "zero_cluster_variance"
    return "estimable"


def _policy_record(row: pd.Series) -> dict[str, object]:
    return {
        "spread_bucket": str(row["spread_bucket"]),
        "model": str(row["model"]),
        "target_train_signal_fraction": float(row["target_train_signal_fraction"]),
        "entry_latency_us": int(row["entry_latency_us"]),
        "shares": int(row["shares"]),
        "filled_round_trips": int(row["filled_round_trips"]),
        "filled_test_dates": int(row["filled_test_dates"]),
        "quoted_round_trip_mean_bps": float(row["quoted_round_trip_mean_bps"]),
        "pointwise_lower_95_bps": float(row["pointwise_lower_95_bps"]),
        "pointwise_upper_95_bps": float(row["pointwise_upper_95_bps"]),
        "simultaneous_lower_95_bps": float(row["simultaneous_lower_95_bps"]),
        "simultaneous_upper_95_bps": float(row["simultaneous_upper_95_bps"]),
        "familywise_adjusted_p_value": float(row["familywise_adjusted_p_value"]),
        "inference_status": str(row["inference_status"]),
    }


def evaluate_round_trip_policy_family(
    bins: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap_replicates: int = 10_000,
    simultaneous_coverage: float = 0.95,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit the complete policy grid with date-clustered simultaneous intervals."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0 < simultaneous_coverage < 1:
        raise ValueError("simultaneous coverage must lie between zero and one")
    _validate_round_trip_bins(bins)
    _validate_policy_metrics(metrics)

    first_date = str(metrics["test_first_date"].iloc[0])
    last_date = str(metrics["test_last_date"].iloc[0])
    test_dates = np.array(
        sorted(
            date for date in bins["date"].astype(str).unique() if first_date <= date <= last_date
        )
    )
    if len(test_dates) != int(metrics["clusters"].iloc[0]):
        raise ValueError("policy metrics and bins disagree on test-date clusters")

    ordered = metrics.sort_values(POLICY_COLUMNS).reset_index(drop=True)
    daily_fills = np.zeros((len(test_dates), len(ordered)), dtype=float)
    daily_pnl = np.zeros_like(daily_fills)
    test = bins.loc[bins["date"].astype(str).between(first_date, last_date)].copy()
    scenarios = {
        key: group
        for key, group in test.groupby(
            ["sample", "spread_bucket", "entry_latency_us", "shares"],
            sort=False,
            observed=True,
        )
    }
    for position, metric in ordered.iterrows():
        key = (
            metric["sample"],
            metric["spread_bucket"],
            metric["entry_latency_us"],
            metric["shares"],
        )
        if key not in scenarios:
            raise ValueError(f"round-trip bins are missing policy scenario {key!r}")
        daily = _daily_policy_outcome(scenarios[key], metric, test_dates)
        daily_fills[:, position] = daily["fills"].to_numpy(dtype=float)
        daily_pnl[:, position] = daily["quoted_pnl_sum_bps"].to_numpy(dtype=float)

    total_fills = daily_fills.sum(axis=0)
    filled = total_fills > 0
    point = np.full(len(ordered), np.nan)
    point[filled] = daily_pnl[:, filled].sum(axis=0) / total_fills[filled]
    influence_numerator = np.zeros_like(daily_pnl)
    influence_numerator[:, filled] = daily_pnl[:, filled] - daily_fills[:, filled] * point[filled]
    cluster_scale = influence_numerator.std(axis=0, ddof=1)
    filled_dates = (daily_fills > 0).sum(axis=0)
    estimable = filled & (filled_dates >= 2) & (cluster_scale > 0)

    standard_error = np.full(len(ordered), np.nan)
    standard_error[estimable] = cluster_scale[estimable] / (
        np.sqrt(len(test_dates)) * daily_fills[:, estimable].mean(axis=0)
    )
    random = np.random.default_rng(random_state)
    multipliers = random.choice(
        (-1.0, 1.0),
        size=(bootstrap_replicates, len(test_dates)),
    )
    bootstrap_t = (multipliers @ influence_numerator[:, estimable]) / (
        np.sqrt(len(test_dates)) * cluster_scale[estimable]
    )
    max_absolute_t = np.max(np.abs(bootstrap_t), axis=1)
    critical_value = float(np.quantile(max_absolute_t, simultaneous_coverage))

    simultaneous_lower = np.full(len(ordered), np.nan)
    simultaneous_upper = np.full(len(ordered), np.nan)
    simultaneous_lower[estimable] = point[estimable] - critical_value * standard_error[estimable]
    simultaneous_upper[estimable] = point[estimable] + critical_value * standard_error[estimable]
    absolute_t = np.abs(point[estimable] / standard_error[estimable])
    sorted_maximum = np.sort(max_absolute_t)
    exceedances = bootstrap_replicates - np.searchsorted(
        sorted_maximum,
        absolute_t,
        side="left",
    )
    adjusted_p_value = np.full(len(ordered), np.nan)
    adjusted_p_value[estimable] = (exceedances + 1) / (bootstrap_replicates + 1)

    audit = ordered[POLICY_COLUMNS].copy()
    audit["filled_round_trips"] = total_fills.astype(np.int64)
    audit["filled_test_dates"] = filled_dates.astype(np.int64)
    audit["quoted_round_trip_mean_bps"] = point
    audit["cluster_standard_error_bps"] = standard_error
    audit["pointwise_lower_95_bps"] = ordered["quoted_round_trip_lower_95_bps"].to_numpy(
        dtype=float
    )
    audit["pointwise_upper_95_bps"] = ordered["quoted_round_trip_upper_95_bps"].to_numpy(
        dtype=float
    )
    audit["simultaneous_lower_95_bps"] = simultaneous_lower
    audit["simultaneous_upper_95_bps"] = simultaneous_upper
    audit["familywise_adjusted_p_value"] = adjusted_p_value
    audit["inference_status"] = [
        _status(fills, dates, scale)
        for fills, dates, scale in zip(
            total_fills,
            filled_dates,
            cluster_scale,
            strict=True,
        )
    ]

    active = audit.loc[audit["filled_round_trips"] > 0]
    best_index = active["quoted_round_trip_mean_bps"].idxmax()
    best = audit.loc[best_index]
    result = {
        "protocol": {
            "family_policies": int(len(audit)),
            "filled_policies": int(filled.sum()),
            "estimable_policies": int(estimable.sum()),
            "zero_fill_policies": int((~filled).sum()),
            "unstudentized_filled_policies": int((filled & ~estimable).sum()),
            "test_dates": int(len(test_dates)),
            "test_first_date": first_date,
            "test_last_date": last_date,
            "bootstrap_replicates": int(bootstrap_replicates),
            "random_state": int(random_state),
            "simultaneous_coverage": float(simultaneous_coverage),
            "critical_value": critical_value,
            "method": (
                "two-sided studentized Rademacher multiplier intervals for "
                "ratio-of-sums estimates clustered by trading date"
            ),
            "selection_rule": (
                "highest observed zero-fee quoted PnL across the complete "
                "spread, model, confidence, latency, and size grid"
            ),
        },
        "best_observed_policy": _policy_record(best),
        "best_observed_upper_below_zero": bool(best["simultaneous_upper_95_bps"] < 0),
        "estimable_policies_with_upper_below_zero": int(
            (audit["simultaneous_upper_95_bps"] < 0).sum()
        ),
        "all_estimable_upper_below_zero": bool(
            (audit.loc[estimable, "simultaneous_upper_95_bps"] < 0).all()
        ),
    }
    return audit, result


def run_round_trip_policy_audit(
    bins_path: Path | str,
    metrics_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Run the selection-aware audit and persist aggregate evidence."""

    bins = pd.read_csv(
        bins_path,
        dtype={
            "date": "string",
            "sample": "category",
            "spread_bucket": "category",
        },
    )
    validate_analysis_universe(bins["date"], analysis_policy)
    metrics = pd.read_csv(
        metrics_path,
        dtype={
            "sample": "category",
            "spread_bucket": "category",
            "model": "category",
        },
    )
    audit, result = evaluate_round_trip_policy_family(bins, metrics)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "price_spell_round_trip_policy_audit.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "price_spell_round_trip_policy_audit.json").write_text(serialized + "\n")
    if figures_dir is not None:
        plot_round_trip_policy_audit(
            audit,
            Path(figures_dir) / "price-spell-round-trip-policy-audit.pdf",
        )
    return json.loads(serialized)
