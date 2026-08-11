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
    assert sell["opposite_best_size_before"] == 700


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
        frames.append(
            pd.DataFrame(
                {
                    "date": date_value,
                    "seconds": seconds,
                    "first_event_row": np.arange(len(signs)),
                    "trade_sign": signs,
                    "size": np.where(expected, 12, 10),
                    "opposite_best_size_before": (
                        np.where(expected, 24, 5) * depth_regime
                    ),
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
    assert adjusted_depth > adjusted_order_size > 0
    assert np.isclose(adjusted_order_size, np.log(12 / 10))
    assert np.isclose(adjusted_depth, np.log(24 / 5))
    assert np.isclose(
        adjusted_relative_liquidity,
        adjusted_order_size - adjusted_depth,
    )
    assert not np.isclose(
        adjusted_depth,
        opposite_depth["expected_minus_surprising"],
    )
