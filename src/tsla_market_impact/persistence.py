"""Daily DFA1 audit for transaction-sign persistence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .data import required_columns
from .plots import plot_order_sign_persistence
from .policy import AnalysisPolicy, validate_analysis_universe

DFA_SCALES = (16, 32, 64, 128, 256, 512, 1024)
MINIMUM_SIGNED_TRANSACTIONS = 8_192
PERMUTATION_REPLICATES = 199
DATE_BLOCK_LENGTHS = (1, 5, 10)
DATE_BOOTSTRAP_REPLICATES = 10_000
RANDOM_STATE = 42
PERMUTATION_BATCH_SIZE = 16
TRANSACTION_COLUMNS = ["date", "first_event_row", "trade_sign"]


def _scale_array(scales: Sequence[int]) -> np.ndarray:
    values = np.asarray(scales, dtype=np.int64)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("DFA requires at least three transaction scales")
    if (values < 2).any() or not np.all(values[1:] > values[:-1]):
        raise ValueError("DFA scales must be strictly increasing integers of at least two")
    return values


def _dfa1_batch(
    sign_rows: np.ndarray,
    scales: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(sign_rows, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("sign rows must be a non-empty one- or two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("sign rows must be finite")

    scale_values = _scale_array(scales)
    row_count, transaction_count = values.shape
    if transaction_count < int(scale_values[-1]):
        raise ValueError("sign rows are shorter than the largest DFA scale")

    profile = np.cumsum(values - values.mean(axis=1, keepdims=True), axis=1)
    prefix_sum = np.empty((row_count, transaction_count + 1), dtype=float)
    prefix_squared_sum = np.empty_like(prefix_sum)
    prefix_indexed_sum = np.empty_like(prefix_sum)
    prefix_sum[:, 0] = 0.0
    prefix_squared_sum[:, 0] = 0.0
    prefix_indexed_sum[:, 0] = 0.0
    np.cumsum(profile, axis=1, out=prefix_sum[:, 1:])
    np.cumsum(np.square(profile), axis=1, out=prefix_squared_sum[:, 1:])
    np.cumsum(
        profile * np.arange(transaction_count, dtype=float),
        axis=1,
        out=prefix_indexed_sum[:, 1:],
    )

    fluctuations = np.empty((row_count, len(scale_values)), dtype=float)
    for column, scale_value in enumerate(scale_values):
        scale = int(scale_value)
        segments_per_direction = transaction_count // scale
        offsets = np.arange(segments_per_direction, dtype=np.int64) * scale
        starts = np.concatenate(
            [
                offsets,
                transaction_count - segments_per_direction * scale + offsets,
            ]
        )
        stops = starts + scale
        segment_sum = prefix_sum[:, stops] - prefix_sum[:, starts]
        segment_squared_sum = (
            prefix_squared_sum[:, stops] - prefix_squared_sum[:, starts]
        )
        segment_indexed_sum = (
            prefix_indexed_sum[:, stops] - prefix_indexed_sum[:, starts]
        )
        centered_index_dot = (
            segment_indexed_sum
            - starts * segment_sum
            - ((scale - 1) / 2) * segment_sum
        )
        centered_index_sum_squares = scale * (scale**2 - 1) / 12
        residual_sum_squares = (
            segment_squared_sum
            - np.square(segment_sum) / scale
            - np.square(centered_index_dot) / centered_index_sum_squares
        )
        total_residual = np.maximum(residual_sum_squares, 0.0).sum(axis=1)
        fluctuations[:, column] = np.sqrt(
            total_residual / (len(starts) * scale)
        )

    if not np.isfinite(fluctuations).all() or (fluctuations <= 0).any():
        raise ValueError("DFA fluctuations must be finite and positive")

    log_scales = np.log(scale_values.astype(float))
    centered_log_scales = log_scales - log_scales.mean()
    log_fluctuations = np.log(fluctuations)
    exponents = (
        log_fluctuations @ centered_log_scales
        / float(centered_log_scales @ centered_log_scales)
    )
    intercepts = log_fluctuations.mean(axis=1) - exponents * log_scales.mean()
    fitted = intercepts[:, None] + exponents[:, None] * log_scales
    residual_sum_squares = np.square(log_fluctuations - fitted).sum(axis=1)
    total_sum_squares = np.square(
        log_fluctuations - log_fluctuations.mean(axis=1, keepdims=True)
    ).sum(axis=1)
    r_squared = np.where(
        total_sum_squares > 0,
        1 - residual_sum_squares / total_sum_squares,
        np.nan,
    )
    return fluctuations, exponents, r_squared


def dfa1_statistics(
    signs: np.ndarray | Sequence[int],
    scales: Sequence[int] = DFA_SCALES,
) -> dict[str, object]:
    """Estimate one DFA1 exponent using equal-weight log fluctuations."""

    scale_values = _scale_array(scales)
    fluctuations, exponents, r_squared = _dfa1_batch(
        np.asarray(signs),
        scale_values,
    )
    return {
        "exponent": float(exponents[0]),
        "r_squared": float(r_squared[0]),
        "fluctuations": {
            int(scale): float(value)
            for scale, value in zip(
                scale_values,
                fluctuations[0],
                strict=True,
            )
        },
    }


def circular_block_median_interval(
    values: np.ndarray | Sequence[float],
    block_length: int,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    """Bootstrap a median with circular blocks of consecutive trading dates."""

    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or len(observations) < 2:
        raise ValueError("at least two daily exponents are required")
    if not np.isfinite(observations).all():
        raise ValueError("daily exponents must be finite")
    if not 1 <= block_length <= len(observations):
        raise ValueError("block length must lie between one and the number of dates")
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")

    blocks_per_draw = int(np.ceil(len(observations) / block_length))
    random = np.random.default_rng(random_state)
    starts = random.integers(
        0,
        len(observations),
        size=(replicates, blocks_per_draw),
    )
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(observations)
    indices = indices.reshape(replicates, -1)[:, : len(observations)]
    medians = np.median(observations[indices], axis=1)
    lower, bootstrap_median, upper = np.quantile(medians, [0.025, 0.5, 0.975])
    return {
        "dates": int(len(observations)),
        "block_length_dates": int(block_length),
        "replicates": int(replicates),
        "observed_median_exponent": float(np.median(observations)),
        "bootstrap_median_exponent": float(bootstrap_median),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _permuted_exponents(
    signs: np.ndarray,
    scales: Sequence[int],
    replicates: int,
    random: np.random.Generator,
) -> np.ndarray:
    exponents = np.empty(replicates, dtype=float)
    for start in range(0, replicates, PERMUTATION_BATCH_SIZE):
        stop = min(start + PERMUTATION_BATCH_SIZE, replicates)
        batch_size = stop - start
        shuffled = random.permuted(
            np.broadcast_to(signs, (batch_size, len(signs))),
            axis=1,
        )
        _, batch_exponents, _ = _dfa1_batch(shuffled, scales)
        exponents[start:stop] = batch_exponents
    return exponents


def evaluate_order_sign_persistence(
    transactions: pd.DataFrame,
    scales: Sequence[int] = DFA_SCALES,
    minimum_transactions: int = MINIMUM_SIGNED_TRANSACTIONS,
    permutation_replicates: int = PERMUTATION_REPLICATES,
    block_lengths: Sequence[int] = DATE_BLOCK_LENGTHS,
    bootstrap_replicates: int = DATE_BOOTSTRAP_REPLICATES,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the fixed daily DFA1, permutation-null, and date-block protocol."""

    required_columns(transactions, TRANSACTION_COLUMNS)
    if transactions.empty:
        raise ValueError("transaction table must not be empty")
    if minimum_transactions < int(max(scales)):
        raise ValueError("minimum transactions cannot be below the largest DFA scale")
    if permutation_replicates < 1:
        raise ValueError("permutation replicates must be positive")
    block_values = tuple(int(value) for value in block_lengths)
    if not block_values or len(set(block_values)) != len(block_values):
        raise ValueError("date block lengths must be non-empty and unique")

    ordered = transactions.sort_values(["date", "first_event_row"]).copy()
    signs = ordered["trade_sign"].to_numpy()
    if not np.isin(signs, [-1, 1]).all():
        raise ValueError("trade_sign must contain only -1 and 1")
    if ordered.duplicated(["date", "first_event_row"]).any():
        raise ValueError("transaction event order must be unique within each date")

    scale_values = _scale_array(scales)
    daily_rows: list[dict[str, object]] = []
    included_signs: list[np.ndarray] = []
    for date, day in ordered.groupby("date", sort=True, observed=True):
        day_signs = day["trade_sign"].to_numpy(dtype=np.int8)
        included = len(day_signs) >= minimum_transactions
        row: dict[str, object] = {
            "date": pd.Timestamp(date).date().isoformat(),
            "signed_transactions": int(len(day_signs)),
            "buy_fraction": float(np.mean(day_signs == 1)),
            "included": bool(included),
            "exclusion_reason": (
                "" if included else f"fewer_than_{minimum_transactions}_signed_transactions"
            ),
            "dfa_exponent": np.nan,
            "dfa_r_squared": np.nan,
        }
        row.update(
            {
                f"fluctuation_{int(scale)}": np.nan
                for scale in scale_values
            }
        )
        if included:
            statistics = dfa1_statistics(day_signs, scale_values)
            row["dfa_exponent"] = statistics["exponent"]
            row["dfa_r_squared"] = statistics["r_squared"]
            row.update(
                {
                    f"fluctuation_{scale}": value
                    for scale, value in statistics["fluctuations"].items()
                }
            )
            included_signs.append(day_signs)
        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    included = daily.loc[daily["included"]].copy()
    if len(included) < 2:
        raise ValueError("at least two dates must meet the DFA inclusion rule")
    if max(block_values) > len(included):
        raise ValueError("date block lengths cannot exceed the included date count")

    random = np.random.default_rng(random_state)
    null_by_date = np.empty(
        (permutation_replicates, len(included_signs)),
        dtype=float,
    )
    for column, day_signs in enumerate(included_signs):
        null_by_date[:, column] = _permuted_exponents(
            day_signs,
            scale_values,
            permutation_replicates,
            random,
        )
    null_medians = np.median(null_by_date, axis=1)
    observed_exponents = included["dfa_exponent"].to_numpy(dtype=float)
    observed_median = float(np.median(observed_exponents))
    observed_q25, observed_q75 = np.quantile(observed_exponents, [0.25, 0.75])
    null_lower, null_median, null_upper = np.quantile(
        null_medians,
        [0.025, 0.5, 0.975],
    )
    upper_tail_exceedances = int(np.sum(null_medians >= observed_median))
    monte_carlo_p_value = (upper_tail_exceedances + 1) / (
        permutation_replicates + 1
    )
    date_intervals = [
        circular_block_median_interval(
            observed_exponents,
            block_length,
            bootstrap_replicates,
            random_state,
        )
        for block_length in block_values
    ]

    total_transactions = int(daily["signed_transactions"].sum())
    included_transactions = int(included["signed_transactions"].sum())
    result: dict[str, object] = {
        "protocol": {
            "sample": (
                "timestamp-aggregated type-4/type-5 transactions signed against "
                "the pre-event midpoint"
            ),
            "date_boundary": "signs are centered, integrated, and analyzed within each date",
            "dfa_order": 1,
            "scales_transactions": [int(value) for value in scale_values],
            "minimum_signed_transactions": int(minimum_transactions),
            "segment_rule": (
                "non-overlapping segments from the beginning and end; "
                "linear detrending within every segment"
            ),
            "scale_fit": "equal-weight OLS of log fluctuation on log scale",
            "permutation_null": {
                "method": "independent within-date sign permutations",
                "replicates": int(permutation_replicates),
                "random_state": int(random_state),
                "preserves": ["date length", "date buy fraction"],
                "test": "upper-tail annual median exponent",
            },
            "date_bootstrap": {
                "method": "circular blocks of consecutive trading dates",
                "block_lengths_dates": list(block_values),
                "replicates": int(bootstrap_replicates),
                "random_state": int(random_state),
            },
            "interpretation": (
                "rejection of the permutation null demonstrates serial dependence, "
                "not specifically long memory; DFA1 does not remove every intraday "
                "seasonal pattern or structural break"
            ),
        },
        "sample": {
            "dates": int(len(daily)),
            "included_dates": int(len(included)),
            "excluded_dates": int(len(daily) - len(included)),
            "first_included_date": str(included["date"].min()),
            "last_included_date": str(included["date"].max()),
            "signed_transactions": total_transactions,
            "included_signed_transactions": included_transactions,
            "covered_transaction_share": (
                float(included_transactions / total_transactions)
                if total_transactions
                else float("nan")
            ),
            "excluded_date_values": daily.loc[
                ~daily["included"],
                "date",
            ].tolist(),
        },
        "observed": {
            "median_exponent": observed_median,
            "q25_exponent": float(observed_q25),
            "q75_exponent": float(observed_q75),
            "median_fit_r_squared": float(included["dfa_r_squared"].median()),
        },
        "permutation_null": {
            "null_median_exponent": float(null_median),
            "null_median_lower_95": float(null_lower),
            "null_median_upper_95": float(null_upper),
            "upper_tail_exceedances": upper_tail_exceedances,
            "monte_carlo_p_value": float(monte_carlo_p_value),
        },
        "date_block_bootstrap": date_intervals,
    }
    return daily, result


def run_order_sign_persistence_analysis(
    transactions_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Run the daily persistence audit and persist compact aggregate evidence."""

    transactions = pd.read_parquet(
        transactions_path,
        columns=TRANSACTION_COLUMNS,
    )
    validate_analysis_universe(transactions["date"], analysis_policy)
    daily, result = evaluate_order_sign_persistence(transactions)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "order_sign_persistence_daily.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "order_sign_persistence.json").write_text(serialized + "\n")
    if figures_dir is not None:
        plot_order_sign_persistence(
            daily,
            result,
            Path(figures_dir) / "order-sign-persistence.pdf",
        )
    return json.loads(serialized)
