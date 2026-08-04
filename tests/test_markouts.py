import numpy as np
import pandas as pd
import pytest

from tsla_market_impact.markouts import (
    evaluate_marketable_markouts,
    evaluate_price_spell_landmarks,
)


def _markout_bins() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2019-01-02", periods=5, freq="B"):
        for queue_bin, center, up_moves in [(0, -0.8, 20), (2, 0.8, 80)]:
            direction = -1 if center < 0 else 1
            for latency_us, executable, move_sum, spread_sum in [
                (0, 100, 10 * direction, 5),
                (1_000, 50, 4 * direction, 2),
            ]:
                rows.append(
                    {
                        "date": str(date.date()),
                        "sample": "best_quote_updates",
                        "spread_bucket": "all_spreads",
                        "latency_us": latency_us,
                        "queue_bin": queue_bin,
                        "ofi_bin": queue_bin,
                        "queue_center": center,
                        "ofi_center": center,
                        "signals": 100,
                        "executable": executable,
                        "stale": 100 - executable,
                        "up_moves": up_moves,
                        "down_moves": 100 - up_moves,
                        "midpoint_move_sum_bps": move_sum,
                        "half_spread_sum_bps": spread_sum,
                    }
                )
    return pd.DataFrame(rows)


def test_markouts_use_fixed_later_dates_and_date_cluster_accounting() -> None:
    metrics, result = evaluate_marketable_markouts(
        _markout_bins(),
        test_start_date="2019-01-07",
        train_signal_fractions=(1.0,),
        bootstrap_replicates=200,
        random_state=7,
    )

    combined = metrics.loc[metrics["model"].eq("queue_and_ofi")].set_index("latency_us")
    immediate = combined.loc[0]
    delayed = combined.loc[1_000]

    assert result["protocol"]["train_dates"] == 3
    assert result["protocol"]["test_dates"] == 2
    assert immediate["train_last_date"] == "2019-01-04"
    assert immediate["test_first_date"] == "2019-01-07"
    assert immediate["selected_test_signals"] == 400
    assert immediate["executable_test_signals"] == 400
    assert immediate["stale_fraction"] == 0
    assert np.isclose(immediate["gross_midpoint_markout_mean_bps"], 0.10)
    assert np.isclose(immediate["half_spread_mean_bps"], 0.05)
    assert np.isclose(immediate["net_markout_mean_bps"], 0.05)
    assert immediate["probability_net_nonpositive"] == 0

    assert delayed["selected_test_signals"] == 400
    assert delayed["executable_test_signals"] == 200
    assert np.isclose(delayed["stale_fraction"], 0.5)
    assert np.isclose(delayed["net_markout_mean_bps"], 0.04)


def test_markouts_reject_inconsistent_counts() -> None:
    bins = _markout_bins()
    bins.loc[0, "stale"] = 1

    with pytest.raises(
        ValueError,
        match="signals must equal executable plus stale",
    ):
        evaluate_marketable_markouts(
            bins,
            test_start_date="2019-01-07",
        )


def test_price_spell_landmarks_use_one_pre_specified_clock_time() -> None:
    bins = _markout_bins()
    bins["sample"] = "price_spell_landmarks"
    bins["landmark_age_us"] = 100

    metrics, result = evaluate_price_spell_landmarks(
        bins,
        test_start_date="2019-01-07",
        train_signal_fractions=(1.0,),
        bootstrap_replicates=200,
        random_state=7,
    )

    assert set(metrics["sample"]) == {"price_spell_landmarks"}
    assert metrics["direction_relative_brier_reduction"].min() > 0
    assert result["protocol"]["landmark_age_us"] == 100
    assert "at most one signal" in result["protocol"]["nonoverlap"]
    assert "executable exit" in result["protocol"]["interpretation_warning"]
    assert "overlap_warning" not in result["protocol"]

    bins.loc[0, "landmark_age_us"] = 200
    with pytest.raises(ValueError, match="one positive landmark age"):
        evaluate_price_spell_landmarks(
            bins,
            test_start_date="2019-01-07",
        )
