from pathlib import Path

import numpy as np
import pandas as pd

from tsla_market_impact.data import (
    DailyFiles,
    aggregate_visible_market_orders,
    read_visible_executions,
)
from tsla_market_impact.liquidity import evaluate_asymmetric_liquidity


def test_market_order_keeps_the_initial_opposite_depth(tmp_path: Path) -> None:
    message = tmp_path / "message.csv"
    orderbook = tmp_path / "orderbook.csv"
    message.write_text(
        "0,1,1,100,1000000,1\n"
        "1,4,2,60,1010000,-1\n"
        "1,4,3,50,1010000,-1\n"
        "1,4,4,30,990000,1\n"
    )
    orderbook.write_text(
        "1010000,100,990000,700\n"
        "1010000,40,990000,700\n"
        "1020000,350,990000,700\n"
        "1020000,350,990000,670\n"
    )
    day = DailyFiles(
        symbol="TSLA",
        date="2019-01-02",
        start="34200000",
        end="57600000",
        levels=1,
        message_path=message,
        orderbook_path=orderbook,
    )

    orders = aggregate_visible_market_orders(read_visible_executions(day))
    buy = orders.loc[orders["trade_sign"].eq(1)].iloc[0]
    sell = orders.loc[orders["trade_sign"].eq(-1)].iloc[0]

    assert buy["size"] == 110
    assert buy["opposite_best_size_before"] == 100
    assert buy["same_side_best_size_before"] == 700
    assert buy["midpoint_twice_before_raw"] == 2_000_000
    assert buy["midpoint_twice_after_raw"] == 2_010_000
    assert sell["opposite_best_size_before"] == 700
    assert sell["same_side_best_size_before"] == 350
    assert sell["midpoint_twice_before_raw"] == 2_010_000
    assert sell["midpoint_twice_after_raw"] == 2_010_000


def test_expected_orders_face_more_depth_in_a_fixed_later_sample() -> None:
    random = np.random.default_rng(7)
    frames = []
    dates = pd.bdate_range("2019-01-02", periods=8).strftime("%Y-%m-%d")
    for date_value in dates:
        signs = np.empty(500, dtype=int)
        signs[0] = random.choice([-1, 1])
        switches = random.random(len(signs) - 1) < 0.20
        for index, switch in enumerate(switches, start=1):
            signs[index] = -signs[index - 1] if switch else signs[index - 1]
        expected = np.r_[False, signs[1:] == signs[:-1]]
        seconds = np.linspace(34_200, 57_599, len(signs))
        depth_regime = np.where(seconds < 45_000, 10, 1)
        row = np.arange(len(signs))
        spread_ticks = np.where(
            expected,
            np.where(row % 5, 6, 2),
            np.where(row % 5, 2, 6),
        )
        midpoint_move = ~expected | (row % 3 != 0)
        response_delta = np.where(expected, 100, 300) + 100 * (
            spread_ticks > 2
        )
        signed_delta = signs * np.where(midpoint_move, response_delta, 0)
        frames.append(
            pd.DataFrame(
                {
                    "date": date_value,
                    "seconds": seconds,
                    "first_event_row": row,
                    "last_event_row": row,
                    "execution_count": 1,
                    "trade_sign": signs,
                    "size": np.where(expected, 12, 10),
                    "spread_before": spread_ticks / 100,
                    "opposite_best_size_before": (
                        np.where(expected, 24, 5) * depth_regime
                    ),
                    "same_side_best_size_before": (
                        np.where(expected, 4, 8) * depth_regime
                    ),
                    "midpoint_twice_before_raw": 2_000_000,
                    "midpoint_twice_after_raw": 2_000_000 + signed_delta,
                }
            )
        )
    orders = pd.concat(frames, ignore_index=True)

    bins, result = evaluate_asymmetric_liquidity(
        orders,
        test_start_date=dates[5],
        lag_order=1,
        quantiles=2,
        block_lengths=(1,),
        bootstrap_replicates=200,
    )

    prediction = result["sign_prediction"]["date_block_bootstrap"][0]
    penetration = result["liquidity_response"][
        "penetration_probability_contrast"
    ][0]
    order_size = result["liquidity_response"]["mean_log_order_size_contrast"][0]
    opposite_depth = result["liquidity_response"][
        "mean_log_opposite_depth_contrast"
    ][0]
    relative_liquidity = result["liquidity_response"][
        "mean_log_size_to_depth_contrast"
    ][0]
    adjusted = result["liquidity_response"]["intraday_adjusted"]
    assert result["scope"]["train_scored_orders"] == 5 * 499
    assert len(bins) == 4
    assert prediction["relative_mse_reduction"] > 0
    assert penetration["expected_minus_surprising"] < 0
    assert order_size["expected_minus_surprising"] > 0
    assert (
        opposite_depth["expected_minus_surprising"]
        > order_size["expected_minus_surprising"]
    )
    assert np.isclose(
        relative_liquidity["expected_minus_surprising"],
        order_size["expected_minus_surprising"]
        - opposite_depth["expected_minus_surprising"],
    )
    adjusted_order_size = adjusted["mean_log_order_size_contrast"][0][
        "expected_minus_surprising"
    ]
    adjusted_depth = adjusted["mean_log_opposite_depth_contrast"][0][
        "expected_minus_surprising"
    ]
    adjusted_relative_liquidity = adjusted[
        "mean_log_size_to_depth_contrast"
    ][0]["expected_minus_surprising"]
    midpoint_move = result["liquidity_response"][
        "midpoint_move_probability_contrast"
    ][0]["expected_minus_surprising"]
    midpoint_response = result["liquidity_response"][
        "mean_signed_midpoint_response_bp_contrast"
    ][0]["expected_minus_surprising"]
    conditional_response = result["liquidity_response"][
        "conditional_signed_midpoint_response_bp_contrast"
    ][0]["expected_minus_surprising"]
    depth_asymmetry = result["liquidity_response"][
        "mean_log_opposite_to_same_depth_contrast"
    ][0]["expected_minus_surprising"]
    same_side_depth = result["liquidity_response"][
        "mean_log_same_side_depth_contrast"
    ][0]["expected_minus_surprising"]
    adjusted_same_side_depth = adjusted[
        "mean_log_same_side_depth_contrast"
    ][0]["expected_minus_surprising"]
    adjusted_depth_asymmetry = adjusted[
        "mean_log_opposite_to_same_depth_contrast"
    ][0]["expected_minus_surprising"]
    spread = result["liquidity_response"]["mean_log_spread_contrast"][0][
        "expected_minus_surprising"
    ]
    assert adjusted_depth > adjusted_order_size > 0
    assert np.isclose(adjusted_order_size, np.log(12 / 10))
    assert np.isclose(adjusted_depth, np.log(24 / 5))
    assert np.isclose(
        adjusted_relative_liquidity,
        adjusted_order_size - adjusted_depth,
    )
    assert midpoint_move < 0
    assert midpoint_response < 0
    assert depth_asymmetry > 0
    assert np.isclose(
        depth_asymmetry,
        opposite_depth["expected_minus_surprising"] - same_side_depth,
    )
    assert np.isclose(
        adjusted_depth_asymmetry,
        adjusted_depth - adjusted_same_side_depth,
    )
    tail_bins = bins.loc[bins["expectedness_bin"].isin((0, 1))]
    manual_response = tail_bins.pivot(
        index="side",
        columns="expectedness_bin",
        values="mean_signed_midpoint_response_bp",
    )
    manual_conditional = tail_bins.pivot(
        index="side",
        columns="expectedness_bin",
        values="conditional_signed_midpoint_response_bp",
    )
    manual_spread = tail_bins.pivot(
        index="side",
        columns="expectedness_bin",
        values="mean_log_spread",
    )
    assert np.isclose(
        midpoint_response,
        (manual_response[1] - manual_response[0]).mean(),
    )
    assert np.isclose(
        conditional_response,
        (manual_conditional[1] - manual_conditional[0]).mean(),
    )
    assert np.isclose(spread, (manual_spread[1] - manual_spread[0]).mean())
    assert np.allclose(
        bins["mean_signed_midpoint_response_bp"],
        bins["midpoint_move_probability"]
        * bins["conditional_signed_midpoint_response_bp"],
    )
    assert not np.isclose(
        adjusted_depth,
        opposite_depth["expected_minus_surprising"],
    )
