"""Order-sign predictability and the liquidity offered against visible orders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import required_columns
from .plots import plot_asymmetric_liquidity
from .policy import AnalysisPolicy, validate_analysis_universe

ORDER_COLUMNS = [
    "date",
    "seconds",
    "first_event_row",
    "last_event_row",
    "execution_count",
    "trade_sign",
    "size",
    "spread_before",
    "opposite_best_size_before",
    "same_side_best_size_before",
    "midpoint_twice_before_raw",
    "midpoint_twice_after_raw",
]
DEFAULT_LAG_ORDER = 50
DEFAULT_QUANTILES = 10
DEFAULT_BLOCK_LENGTHS = (1, 5, 10)
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
RANDOM_STATE = 42
SIDES = (-1, 1)
SESSION_OPEN_SECONDS = 34_200
INTRADAY_BUCKET_SECONDS = 1_800
TAIL_METRICS = (
    "penetrated_best",
    "midpoint_changed",
    "signed_midpoint_response_bp",
    "log_order_size",
    "log_opposite_depth",
    "log_size_to_depth",
    "log_same_side_depth",
    "log_opposite_to_same_depth",
    "log_spread",
)
TAIL_METRIC_INDEX = {
    metric: index for index, metric in enumerate(TAIL_METRICS, start=1)
}


def _lagged_signs(
    signs: np.ndarray,
    lag_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(signs) <= lag_order:
        raise ValueError("Each session must contain more signs than the AR order")
    windows = np.lib.stride_tricks.sliding_window_view(signs, lag_order + 1)
    return windows[:, :-1][:, ::-1], windows[:, -1]


def _validate_orders(orders: pd.DataFrame, lag_order: int) -> pd.DataFrame:
    required_columns(orders, ORDER_COLUMNS)
    if lag_order < 1:
        raise ValueError("AR order must be positive")
    frame = orders[ORDER_COLUMNS].copy()
    frame["date"] = frame["date"].astype(str)
    frame = frame.sort_values(["date", "first_event_row"]).reset_index(drop=True)
    if not frame["trade_sign"].isin(SIDES).all():
        raise ValueError("Visible-order signs must be either -1 or 1")
    numeric_columns = [
        "seconds",
        "size",
        "spread_before",
        "opposite_best_size_before",
        "same_side_best_size_before",
        "midpoint_twice_before_raw",
        "midpoint_twice_after_raw",
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Order time, size, book state, and spread must be finite")
    positive_columns = [
        "size",
        "spread_before",
        "opposite_best_size_before",
        "same_side_best_size_before",
        "midpoint_twice_before_raw",
        "midpoint_twice_after_raw",
    ]
    if frame[positive_columns].le(0).any(axis=None):
        raise ValueError("Order size, book state, and spread must be positive")
    spread_ticks = 100 * frame["spread_before"].to_numpy(dtype=float)
    if not np.allclose(spread_ticks, np.rint(spread_ticks), atol=1e-8, rtol=0):
        raise ValueError("Displayed spreads must be integer tick multiples")
    midpoint_raw = frame[
        ["midpoint_twice_before_raw", "midpoint_twice_after_raw"]
    ].to_numpy(dtype=float)
    if not np.array_equal(midpoint_raw, np.rint(midpoint_raw)):
        raise ValueError("Raw twice-midpoints must be fixed-point integers")
    event_span = frame["last_event_row"] - frame["first_event_row"] + 1
    if frame["execution_count"].le(0).any() or event_span.lt(
        frame["execution_count"]
    ).any():
        raise ValueError("Visible-fill row spans must cover every grouped execution")
    if frame["seconds"].lt(SESSION_OPEN_SECONDS).any():
        raise ValueError("Visible orders cannot precede the regular session open")
    daily_counts = frame.groupby("date", sort=True, observed=True).size()
    if daily_counts.le(lag_order).any():
        raise ValueError("Every session must contain more signs than the AR order")
    return frame


def _fit_sign_ar(
    train: pd.DataFrame,
    lag_order: int,
) -> tuple[np.ndarray, int, float]:
    normal = np.zeros((lag_order + 1, lag_order + 1), dtype=float)
    right_hand_side = np.zeros(lag_order + 1, dtype=float)
    observations = 0
    for _, day in train.groupby("date", sort=True, observed=True):
        signs = day["trade_sign"].to_numpy(dtype=float)
        lags, target = _lagged_signs(signs, lag_order)
        count = len(target)
        normal[0, 0] += count
        lag_sums = lags.sum(axis=0)
        normal[0, 1:] += lag_sums
        normal[1:, 0] += lag_sums
        normal[1:, 1:] += lags.T @ lags
        right_hand_side[0] += target.sum()
        right_hand_side[1:] += lags.T @ target
        observations += count
    condition_number = float(np.linalg.cond(normal))
    if not np.isfinite(condition_number) or condition_number > 1e10:
        raise ValueError("Sign autoregression normal equations are ill-conditioned")
    return np.linalg.solve(normal, right_hand_side), observations, condition_number


def _score_orders(
    orders: pd.DataFrame,
    coefficients: np.ndarray,
) -> pd.DataFrame:
    lag_order = len(coefficients) - 1
    frames: list[pd.DataFrame] = []
    for date_value, day in orders.groupby("date", sort=True, observed=True):
        signs = day["trade_sign"].to_numpy(dtype=float)
        lags, target = _lagged_signs(signs, lag_order)
        selected = day.iloc[lag_order:].copy()
        selected["predicted_sign"] = coefficients[0] + lags @ coefficients[1:]
        selected["expectedness"] = target * selected["predicted_sign"]
        selected["penetrated_best"] = selected["size"].ge(
            selected["opposite_best_size_before"]
        )
        selected["log_order_size"] = np.log(selected["size"])
        selected["log_opposite_depth"] = np.log(
            selected["opposite_best_size_before"]
        )
        selected["log_size_to_depth"] = (
            selected["log_order_size"] - selected["log_opposite_depth"]
        )
        selected["log_same_side_depth"] = np.log(
            selected["same_side_best_size_before"]
        )
        selected["log_opposite_to_same_depth"] = (
            selected["log_opposite_depth"] - selected["log_same_side_depth"]
        )
        selected["log_spread"] = np.log(selected["spread_before"])
        midpoint_delta_raw = (
            selected["midpoint_twice_after_raw"]
            - selected["midpoint_twice_before_raw"]
        )
        selected["midpoint_changed"] = midpoint_delta_raw.ne(0)
        selected["signed_midpoint_response_bp"] = (
            10_000
            * selected["trade_sign"]
            * midpoint_delta_raw
            / selected["midpoint_twice_before_raw"]
        )
        selected["intervening_events"] = (
            selected["last_event_row"]
            - selected["first_event_row"]
            + 1
            - selected["execution_count"]
        )
        selected["intraday_bucket"] = np.floor_divide(
            selected["seconds"] - SESSION_OPEN_SECONDS,
            INTRADAY_BUCKET_SECONDS,
        )
        selected["date"] = date_value
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def _circular_indices(
    dates: int,
    block_length: int,
    replicates: int,
    random_state: int,
) -> np.ndarray:
    if not 1 <= block_length <= dates:
        raise ValueError("Block length must lie between one and the number of dates")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    blocks_per_draw = int(np.ceil(dates / block_length))
    random = np.random.default_rng(random_state)
    starts = random.integers(0, dates, size=(replicates, blocks_per_draw))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % dates
    return indices.reshape(replicates, -1)[:, :dates]


def _prediction_interval(
    daily_losses: np.ndarray,
    block_length: int,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    indices = _circular_indices(
        len(daily_losses),
        block_length,
        replicates,
        random_state,
    )
    sampled = daily_losses[indices].sum(axis=1)
    reductions = 1 - sampled[:, 0] / sampled[:, 1]
    point = 1 - daily_losses[:, 0].sum() / daily_losses[:, 1].sum()
    lower, median, upper = np.quantile(reductions, [0.025, 0.5, 0.975])
    return {
        "block_length_dates": block_length,
        "relative_mse_reduction": float(point),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "probability_nonpositive": float(np.mean(reductions <= 0)),
    }


def _tail_statistics(
    scored: pd.DataFrame,
    quantiles: int,
) -> np.ndarray:
    dates = sorted(scored["date"].unique())
    tails = (0, quantiles - 1)
    shape = (len(dates), len(SIDES), len(tails), len(TAIL_METRICS) + 1)
    statistics = np.zeros(shape, dtype=float)
    for date_index, date_value in enumerate(dates):
        day = scored.loc[scored["date"].eq(date_value)]
        for side_index, side in enumerate(SIDES):
            side_rows = day.loc[day["trade_sign"].eq(side)]
            for tail_index, tail in enumerate(tails):
                rows = side_rows.loc[side_rows["expectedness_bin"].eq(tail)]
                cell = statistics[date_index, side_index, tail_index]
                cell[0] = len(rows)
                for metric, metric_index in TAIL_METRIC_INDEX.items():
                    cell[metric_index] = rows[metric].sum()
    if (statistics[..., 0].sum(axis=0) == 0).any():
        raise ValueError("A side-tail cell contains no evaluation order")
    return statistics


def _stratified_tail_statistics(
    scored: pd.DataFrame,
    quantiles: int,
    strata: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    dates = sorted(scored["date"].unique())
    date_positions = {date: index for index, date in enumerate(dates)}
    side_positions = {side: index for index, side in enumerate(SIDES)}
    numerators = np.zeros((len(dates), len(SIDES), len(TAIL_METRICS)))
    denominators = np.zeros((len(dates), len(SIDES)))
    tails = scored.loc[scored["expectedness_bin"].isin((0, quantiles - 1))]
    overlap_orders = 0
    overlap_strata = 0
    total_strata = 0

    group_columns = ["date", "trade_sign", *strata]
    for key, rows in tails.groupby(
        group_columns,
        sort=True,
        observed=True,
    ):
        date_value, side = key[:2]
        total_strata += 1
        surprising = rows.loc[rows["expectedness_bin"].eq(0)]
        expected = rows.loc[rows["expectedness_bin"].eq(quantiles - 1)]
        if surprising.empty or expected.empty:
            continue
        overlap_strata += 1
        overlap_orders += len(surprising) + len(expected)
        weight = len(surprising) * len(expected) / len(rows)
        date_index = date_positions[date_value]
        side_index = side_positions[side]
        denominators[date_index, side_index] += weight
        for metric_index, metric in enumerate(TAIL_METRICS):
            difference = expected[metric].mean() - surprising[metric].mean()
            numerators[date_index, side_index, metric_index] += weight * difference

    if (denominators.sum(axis=0) == 0).any():
        raise ValueError("An order side has no stratified tail overlap")
    total_orders = len(tails)
    coverage: dict[str, int | float] = {
        "tail_orders": total_orders,
        "overlap_tail_orders": overlap_orders,
        "overlap_order_fraction": overlap_orders / total_orders,
        "strata": total_strata,
        "overlap_strata": overlap_strata,
    }
    return numerators, denominators, coverage


def _conditional_response_statistics(
    scored: pd.DataFrame,
    quantiles: int,
    strata: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    dates = sorted(scored["date"].unique())
    date_positions = {date: index for index, date in enumerate(dates)}
    side_positions = {side: index for index, side in enumerate(SIDES)}
    numerators = np.zeros((len(dates), len(SIDES), 1))
    denominators = np.zeros((len(dates), len(SIDES)))
    movers = scored.loc[
        scored["midpoint_changed"]
        & scored["expectedness_bin"].isin((0, quantiles - 1))
    ]
    overlap_movers = 0
    overlap_strata = 0
    total_strata = 0

    group_columns = ["date", "trade_sign", *strata]
    for key, rows in movers.groupby(
        group_columns,
        sort=True,
        observed=True,
    ):
        total_strata += 1
        date_value, side = key[:2]
        surprising = rows.loc[rows["expectedness_bin"].eq(0)]
        expected = rows.loc[rows["expectedness_bin"].eq(quantiles - 1)]
        if surprising.empty or expected.empty:
            continue
        overlap_strata += 1
        overlap_movers += len(surprising) + len(expected)
        weight = len(surprising) * len(expected) / len(rows)
        difference = (
            expected["signed_midpoint_response_bp"].mean()
            - surprising["signed_midpoint_response_bp"].mean()
        )
        date_index = date_positions[date_value]
        side_index = side_positions[side]
        denominators[date_index, side_index] += weight
        numerators[date_index, side_index, 0] += weight * difference

    if (denominators.sum(axis=0) == 0).any():
        raise ValueError("An order side has no conditional-response tail overlap")
    coverage: dict[str, int | float] = {
        "tail_movers": len(movers),
        "overlap_tail_movers": overlap_movers,
        "overlap_mover_fraction": overlap_movers / len(movers),
        "strata": total_strata,
        "overlap_strata": overlap_strata,
    }
    return numerators, denominators, coverage


def _contrast_interval(
    statistics: np.ndarray,
    metric_index: int,
    block_length: int,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    def contrast(sums: np.ndarray) -> np.ndarray:
        means = sums[..., metric_index] / sums[..., 0]
        return np.mean(means[..., 1] - means[..., 0], axis=-1)

    point = float(contrast(statistics.sum(axis=0)))
    indices = _circular_indices(
        len(statistics),
        block_length,
        replicates,
        random_state,
    )
    sampled = statistics[indices].sum(axis=1)
    contrasts = contrast(sampled)
    lower, median, upper = np.quantile(contrasts, [0.025, 0.5, 0.975])
    return {
        "block_length_dates": block_length,
        "expected_minus_surprising": point,
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "probability_nonnegative": float(np.mean(contrasts >= 0)),
    }


def _ratio_contrast_interval(
    statistics: np.ndarray,
    numerator_index: int,
    denominator_index: int,
    block_length: int,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    def contrast(sums: np.ndarray) -> np.ndarray:
        means = sums[..., numerator_index] / sums[..., denominator_index]
        return np.mean(means[..., 1] - means[..., 0], axis=-1)

    totals = statistics.sum(axis=0)
    if (totals[..., denominator_index] == 0).any():
        raise ValueError("A side-tail cell contains no midpoint move")
    point = float(contrast(totals))
    indices = _circular_indices(
        len(statistics),
        block_length,
        replicates,
        random_state,
    )
    sampled = statistics[indices].sum(axis=1)
    if (sampled[..., denominator_index] == 0).any():
        raise ValueError("A bootstrap side-tail cell contains no midpoint move")
    contrasts = contrast(sampled)
    lower, median, upper = np.quantile(contrasts, [0.025, 0.5, 0.975])
    return {
        "block_length_dates": block_length,
        "expected_minus_surprising": point,
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "probability_nonnegative": float(np.mean(contrasts >= 0)),
    }


def _stratified_contrast_interval(
    numerators: np.ndarray,
    denominators: np.ndarray,
    metric_index: int,
    block_length: int,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    def contrast(
        numerator_sums: np.ndarray,
        denominator_sums: np.ndarray,
    ) -> np.ndarray:
        side_contrasts = numerator_sums[..., metric_index] / denominator_sums
        return side_contrasts.mean(axis=-1)

    point = float(contrast(numerators.sum(axis=0), denominators.sum(axis=0)))
    indices = _circular_indices(
        len(numerators),
        block_length,
        replicates,
        random_state,
    )
    sampled_numerators = numerators[indices].sum(axis=1)
    sampled_denominators = denominators[indices].sum(axis=1)
    contrasts = contrast(sampled_numerators, sampled_denominators)
    lower, median, upper = np.quantile(contrasts, [0.025, 0.5, 0.975])
    return {
        "block_length_dates": block_length,
        "expected_minus_surprising": point,
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "probability_nonnegative": float(np.mean(contrasts >= 0)),
    }


def evaluate_asymmetric_liquidity(
    orders: pd.DataFrame,
    test_start_date: str,
    lag_order: int = DEFAULT_LAG_ORDER,
    quantiles: int = DEFAULT_QUANTILES,
    block_lengths: tuple[int, ...] = DEFAULT_BLOCK_LENGTHS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit a fixed sign AR and test whether expected orders face more depth."""

    if quantiles < 2:
        raise ValueError("At least two expectedness quantiles are required")
    if not block_lengths or len(set(block_lengths)) != len(block_lengths):
        raise ValueError("Block lengths must be non-empty and unique")
    frame = _validate_orders(orders, lag_order)
    train = frame.loc[frame["date"].lt(test_start_date)].copy()
    test = frame.loc[frame["date"].ge(test_start_date)].copy()
    if train.empty or test.empty:
        raise ValueError("Fixed test boundary must leave training and test orders")
    test_dates = sorted(test["date"].unique())
    if max(block_lengths) > len(test_dates):
        raise ValueError("Block lengths cannot exceed the number of test dates")

    coefficients, train_observations, condition_number = _fit_sign_ar(
        train,
        lag_order,
    )
    scored_train = _score_orders(train, coefficients)
    scored_test = _score_orders(test, coefficients)
    quantile_edges: dict[str, list[float]] = {}
    scored_test["expectedness_bin"] = -1
    for side in SIDES:
        train_values = scored_train.loc[
            scored_train["trade_sign"].eq(side),
            "expectedness",
        ].to_numpy(dtype=float)
        edges = np.quantile(train_values, np.linspace(0, 1, quantiles + 1))
        scored_test.loc[
            scored_test["trade_sign"].eq(side),
            "expectedness_bin",
        ] = np.searchsorted(
            edges[1:-1],
            scored_test.loc[
                scored_test["trade_sign"].eq(side),
                "expectedness",
            ],
            side="right",
        )
        quantile_edges[str(side)] = [float(value) for value in edges]

    bins = (
        scored_test.groupby(
            ["trade_sign", "expectedness_bin"],
            sort=True,
            observed=True,
        )
        .agg(
            observations=("expectedness", "size"),
            mean_expectedness=("expectedness", "mean"),
            penetration_probability=("penetrated_best", "mean"),
            midpoint_moves=("midpoint_changed", "sum"),
            midpoint_move_probability=("midpoint_changed", "mean"),
            signed_midpoint_response_sum_bp=(
                "signed_midpoint_response_bp",
                "sum",
            ),
            mean_signed_midpoint_response_bp=(
                "signed_midpoint_response_bp",
                "mean",
            ),
            mean_log_order_size=("log_order_size", "mean"),
            mean_log_opposite_depth=("log_opposite_depth", "mean"),
            mean_log_size_to_depth=("log_size_to_depth", "mean"),
            mean_log_same_side_depth=("log_same_side_depth", "mean"),
            mean_log_opposite_to_same_depth=(
                "log_opposite_to_same_depth",
                "mean",
            ),
            mean_spread=("spread_before", "mean"),
            mean_log_spread=("log_spread", "mean"),
        )
        .reset_index()
    )
    if bins["midpoint_moves"].eq(0).any():
        raise ValueError("Every expectedness bin must contain a midpoint move")
    bins["conditional_signed_midpoint_response_bp"] = (
        bins["signed_midpoint_response_sum_bp"] / bins["midpoint_moves"]
    )
    bins["side"] = bins["trade_sign"].map({-1: "sell", 1: "buy"})
    bins = bins[
        [
            "side",
            "trade_sign",
            "expectedness_bin",
            "observations",
            "mean_expectedness",
            "penetration_probability",
            "midpoint_moves",
            "midpoint_move_probability",
            "mean_signed_midpoint_response_bp",
            "conditional_signed_midpoint_response_bp",
            "mean_log_order_size",
            "mean_log_opposite_depth",
            "mean_log_size_to_depth",
            "mean_log_same_side_depth",
            "mean_log_opposite_to_same_depth",
            "mean_spread",
            "mean_log_spread",
        ]
    ]

    train_mean = float(scored_train["trade_sign"].mean())
    scored_test["ar_squared_error"] = (
        scored_test["trade_sign"] - scored_test["predicted_sign"]
    ) ** 2
    scored_test["reference_squared_error"] = (
        scored_test["trade_sign"] - train_mean
    ) ** 2
    daily_losses = (
        scored_test.groupby("date", sort=True, observed=True)[
            ["ar_squared_error", "reference_squared_error"]
        ]
        .sum()
        .to_numpy(dtype=float)
    )
    prediction_intervals = [
        _prediction_interval(
            daily_losses,
            block_length,
            bootstrap_replicates,
            random_state + block_length,
        )
        for block_length in block_lengths
    ]
    tail_statistics = _tail_statistics(scored_test, quantiles)

    def metric_intervals(
        statistics: np.ndarray,
        metric: str,
        seed_offset: int,
    ) -> list[dict[str, float | int]]:
        return [
            _contrast_interval(
                statistics,
                TAIL_METRIC_INDEX[metric],
                block_length,
                bootstrap_replicates,
                random_state + seed_offset + block_length,
            )
            for block_length in block_lengths
        ]

    def conditional_response_intervals(
        statistics: np.ndarray,
        seed_offset: int,
    ) -> list[dict[str, float | int]]:
        return [
            _ratio_contrast_interval(
                statistics,
                TAIL_METRIC_INDEX["signed_midpoint_response_bp"],
                TAIL_METRIC_INDEX["midpoint_changed"],
                block_length,
                bootstrap_replicates,
                random_state + seed_offset + block_length,
            )
            for block_length in block_lengths
        ]

    raw_intervals = {
        "penetrated_best": metric_intervals(
            tail_statistics,
            "penetrated_best",
            100,
        ),
        **{
            metric: metric_intervals(tail_statistics, metric, 200)
            for metric in (
                "log_order_size",
                "log_opposite_depth",
                "log_size_to_depth",
                "log_same_side_depth",
                "log_opposite_to_same_depth",
                "log_spread",
            )
        },
        **{
            metric: metric_intervals(tail_statistics, metric, 400)
            for metric in (
                "midpoint_changed",
                "signed_midpoint_response_bp",
            )
        },
        "conditional_signed_midpoint_response_bp": (
            conditional_response_intervals(tail_statistics, 400)
        ),
    }

    intraday_numerators, intraday_denominators, intraday_coverage = (
        _stratified_tail_statistics(
            scored_test,
            quantiles,
            ("intraday_bucket",),
        )
    )
    intraday_conditional = _conditional_response_statistics(
        scored_test,
        quantiles,
        ("intraday_bucket",),
    )
    intraday_intervals = {
        metric: [
            _stratified_contrast_interval(
                intraday_numerators,
                intraday_denominators,
                metric_index,
                block_length,
                bootstrap_replicates,
                random_state + 300 + block_length,
            )
            for block_length in block_lengths
        ]
        for metric_index, metric in enumerate(TAIL_METRICS)
    }
    intraday_conditional_intervals = [
        _stratified_contrast_interval(
            intraday_conditional[0],
            intraday_conditional[1],
            0,
            block_length,
            bootstrap_replicates,
            random_state + 300 + block_length,
        )
        for block_length in block_lengths
    ]

    primary_block = 5 if 5 in block_lengths else block_lengths[0]
    primary_prediction = next(
        row
        for row in prediction_intervals
        if row["block_length_dates"] == primary_block
    )
    if primary_prediction["relative_mse_reduction"] <= 0:
        raise ValueError("Sign-prediction gate failed on the fixed test dates")

    proxy = scored_test["penetrated_best"].to_numpy(dtype=bool)
    observed_move = scored_test["midpoint_changed"].to_numpy(dtype=bool)
    true_positive = int(np.sum(proxy & observed_move))
    false_positive = int(np.sum(proxy & ~observed_move))
    false_negative = int(np.sum(~proxy & observed_move))
    true_negative = int(np.sum(~proxy & ~observed_move))
    intervening_groups = int(scored_test["intervening_events"].gt(0).sum())
    adverse_responses = int(scored_test["signed_midpoint_response_bp"].lt(0).sum())

    result: dict[str, object] = {
        "protocol": {
            "model": f"OLS AR({lag_order}) of visible-order signs",
            "lags_reset_at_session_boundary": True,
            "fit": "once on dates before the fixed test boundary",
            "expectedness": "realised sign times predicted sign",
            "expectedness_bins": (
                f"{quantiles} train-defined quantiles fitted separately by side"
            ),
            "penetration": (
                "proxy: order size at least opposite best depth before first fill"
            ),
            "liquidity_decomposition": (
                "log order size minus log initial opposite best depth"
            ),
            "best_quote_asymmetry": (
                "log initial opposite best depth minus log initial same-side "
                "best depth"
            ),
            "immediate_response": (
                "signed midpoint change from the snapshot before first_event_row "
                "to the snapshot after last_event_row; the move indicator uses "
                "the exact fixed-point twice-midpoint difference"
            ),
            "primary_post_hoc_endpoint": (
                "most-expected minus most-surprising mean signed immediate "
                "midpoint response in basis points"
            ),
            "intraday_control": (
                "post-hoc date, realised side, and fixed 30-minute session "
                "bucket effects; no bucket-width search"
            ),
            "primary_bootstrap_block_dates": primary_block,
            "bootstrap_replicates": bootstrap_replicates,
            "test_status": (
                "chronological for this model; dates were inspected by other "
                "project analyses, and the response and depth falsifications "
                "were selected after the original liquidity result"
            ),
        },
        "scope": {
            "train_first_date": str(train["date"].min()),
            "train_last_date": str(train["date"].max()),
            "test_first_date": str(test["date"].min()),
            "test_last_date": str(test["date"].max()),
            "train_dates": int(train["date"].nunique()),
            "test_dates": int(test["date"].nunique()),
            "train_scored_orders": train_observations,
            "test_scored_orders": len(scored_test),
            "test_groups_with_intervening_events": intervening_groups,
            "test_groups_with_intervening_events_fraction": (
                intervening_groups / len(scored_test)
            ),
            "test_adverse_signed_responses": adverse_responses,
        },
        "sign_prediction": {
            "training_mean_sign": train_mean,
            "normal_matrix_condition_number": condition_number,
            "coefficients": [float(value) for value in coefficients],
            "test_mse": float(scored_test["ar_squared_error"].mean()),
            "reference_test_mse": float(
                scored_test["reference_squared_error"].mean()
            ),
            "test_sign_accuracy": float(
                np.sign(scored_test["predicted_sign"])
                .eq(scored_test["trade_sign"])
                .mean()
            ),
            "date_block_bootstrap": prediction_intervals,
        },
        "liquidity_response": {
            "comparison": "most expected minus most surprising train-defined bin",
            "side_standardisation": "equal weight to buys and sells",
            "penetration_probability_contrast": raw_intervals["penetrated_best"],
            "midpoint_move_audit": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "proxy_precision": true_positive / (true_positive + false_positive),
                "proxy_recall": true_positive / (true_positive + false_negative),
                "interpretation": (
                    "observed midpoint moves validate the penetration proxy; "
                    "they are not an independent outcome family"
                ),
            },
            "midpoint_move_probability_contrast": raw_intervals[
                "midpoint_changed"
            ],
            "mean_signed_midpoint_response_bp_contrast": raw_intervals[
                "signed_midpoint_response_bp"
            ],
            "conditional_signed_midpoint_response_bp_contrast": raw_intervals[
                "conditional_signed_midpoint_response_bp"
            ],
            "mean_log_order_size_contrast": raw_intervals["log_order_size"],
            "mean_log_opposite_depth_contrast": raw_intervals[
                "log_opposite_depth"
            ],
            "mean_log_size_to_depth_contrast": raw_intervals[
                "log_size_to_depth"
            ],
            "mean_log_same_side_depth_contrast": raw_intervals[
                "log_same_side_depth"
            ],
            "mean_log_opposite_to_same_depth_contrast": raw_intervals[
                "log_opposite_to_same_depth"
            ],
            "mean_log_spread_contrast": raw_intervals["log_spread"],
            "intraday_adjusted": {
                "method": (
                    "within-stratum tail differences weighted by "
                    "n_surprising*n_expected/(n_surprising+n_expected), then "
                    "equal weight to sides"
                ),
                "session_open_seconds": SESSION_OPEN_SECONDS,
                "bucket_seconds": INTRADAY_BUCKET_SECONDS,
                "coverage": intraday_coverage,
                "penetration_probability_contrast": intraday_intervals[
                    "penetrated_best"
                ],
                "mean_log_order_size_contrast": intraday_intervals[
                    "log_order_size"
                ],
                "mean_log_opposite_depth_contrast": intraday_intervals[
                    "log_opposite_depth"
                ],
                "mean_log_size_to_depth_contrast": intraday_intervals[
                    "log_size_to_depth"
                ],
                "mean_log_same_side_depth_contrast": intraday_intervals[
                    "log_same_side_depth"
                ],
                "mean_log_opposite_to_same_depth_contrast": intraday_intervals[
                    "log_opposite_to_same_depth"
                ],
                "mean_log_spread_contrast": intraday_intervals["log_spread"],
                "midpoint_move_probability_contrast": intraday_intervals[
                    "midpoint_changed"
                ],
                "mean_signed_midpoint_response_bp_contrast": intraday_intervals[
                    "signed_midpoint_response_bp"
                ],
                "conditional_signed_midpoint_response_bp_contrast": (
                    intraday_conditional_intervals
                ),
                "conditional_response_coverage": intraday_conditional[2],
            },
            "train_quantile_edges_by_sign": quantile_edges,
        },
    }
    return bins, result


def run_asymmetric_liquidity_analysis(
    orders_path: Path | str,
    results_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    analysis_policy: AnalysisPolicy,
) -> dict[str, object]:
    """Run the predictability-conditioned liquidity audit and persist aggregates."""

    orders = pd.read_parquet(orders_path, columns=ORDER_COLUMNS)
    validate_analysis_universe(orders["date"], analysis_policy)
    bins, result = evaluate_asymmetric_liquidity(
        orders,
        test_start_date=analysis_policy.test_start,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    bins.to_csv(output / "asymmetric_liquidity_bins.csv", index=False)
    serialized = json.dumps(result, indent=2)
    (output / "asymmetric_liquidity.json").write_text(serialized + "\n")
    if figures_dir is not None:
        plot_asymmetric_liquidity(
            bins,
            Path(figures_dir) / "asymmetric-liquidity.pdf",
        )
    return json.loads(serialized)
