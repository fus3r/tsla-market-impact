import numpy as np
import pandas as pd
import pytest

from tsla_market_impact.policy_audit import evaluate_round_trip_policy_family
from tsla_market_impact.round_trips import evaluate_price_spell_round_trips


def _round_trip_bins() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2019-01-02", periods=5, freq="B"):
        for queue_bin, center, up_moves in [(0, -0.8, 20), (2, 0.8, 80)]:
            for latency_us, arrived in [(0, 100), (1_000, 50)]:
                for shares, fill_fraction, entry_cost, exit_cost in [
                    (1, 1.0, 0.03, 0.02),
                    (100, 0.5, 0.04, 0.03),
                ]:
                    fills = int(arrived * fill_fraction)
                    long_midpoint = 0.10 if center > 0 else -0.10
                    short_midpoint = -long_midpoint
                    rows.append(
                        {
                            "date": str(date.date()),
                            "sample": "price_spell_round_trips",
                            "spread_bucket": "all_spreads",
                            "landmark_age_us": 100,
                            "entry_latency_us": latency_us,
                            "shares": shares,
                            "queue_bin": queue_bin,
                            "ofi_bin": queue_bin,
                            "queue_center": center,
                            "ofi_center": center,
                            "signals": 100,
                            "arrived": arrived,
                            "stale": 100 - arrived,
                            "up_moves": up_moves,
                            "down_moves": 100 - up_moves,
                            "long_fills": fills,
                            "long_capacity_censored": arrived - fills,
                            "long_midpoint_move_sum_bps": fills * long_midpoint,
                            "long_entry_cost_sum_bps": fills * entry_cost,
                            "long_exit_cost_sum_bps": fills * exit_cost,
                            "long_quoted_pnl_sum_bps": fills
                            * (long_midpoint - entry_cost - exit_cost),
                            "short_fills": fills,
                            "short_capacity_censored": arrived - fills,
                            "short_midpoint_move_sum_bps": fills * short_midpoint,
                            "short_entry_cost_sum_bps": fills * entry_cost,
                            "short_exit_cost_sum_bps": fills * exit_cost,
                            "short_quoted_pnl_sum_bps": fills
                            * (short_midpoint - entry_cost - exit_cost),
                        }
                    )
    return pd.DataFrame(rows)


def test_round_trip_analysis_applies_depth_to_both_legs():
    metrics, result = evaluate_price_spell_round_trips(
        _round_trip_bins(),
        test_date_fraction=0.40,
        train_signal_fractions=(1.0,),
        bootstrap_replicates=200,
        random_state=7,
    )

    combined = metrics.loc[metrics["model"].eq("queue_and_ofi")].set_index(
        ["entry_latency_us", "shares"]
    )
    immediate = combined.loc[(0, 1)]
    sized = combined.loc[(0, 100)]
    delayed = combined.loc[(1_000, 1)]

    assert result["protocol"]["train_dates"] == 3
    assert result["protocol"]["test_dates"] == 2
    assert immediate["selected_test_signals"] == 400
    assert immediate["arrived_test_signals"] == 400
    assert immediate["filled_round_trips"] == 400
    assert immediate["capacity_censored_fraction"] == 0
    assert np.isclose(immediate["quoted_round_trip_mean_bps"], 0.05)
    assert immediate["probability_quoted_pnl_nonpositive"] == 0

    assert sized["filled_round_trips"] == 200
    assert np.isclose(sized["capacity_censored_fraction"], 0.5)
    assert np.isclose(sized["quoted_round_trip_mean_bps"], 0.03)

    assert delayed["arrived_test_signals"] == 200
    assert delayed["filled_round_trips"] == 200
    assert np.isclose(delayed["stale_fraction"], 0.5)


def test_round_trip_analysis_rejects_one_share_capacity_censoring():
    bins = _round_trip_bins()
    row = bins["shares"].eq(1).idxmax()
    bins.loc[row, "long_fills"] -= 1
    bins.loc[row, "long_capacity_censored"] += 1

    with pytest.raises(ValueError, match="one-share round trips"):
        evaluate_price_spell_round_trips(bins)


def test_policy_audit_reconstructs_the_complete_grid_and_adjusts_selection():
    bins = _round_trip_bins()
    date_index = bins["date"].map(
        {date: index for index, date in enumerate(sorted(bins["date"].unique()))}
    )
    extra_exit_cost = 0.002 * date_index
    for side in ("long", "short"):
        fills = bins[f"{side}_fills"]
        bins[f"{side}_exit_cost_sum_bps"] += fills * extra_exit_cost
        bins[f"{side}_quoted_pnl_sum_bps"] -= fills * extra_exit_cost

    metrics, _ = evaluate_price_spell_round_trips(
        bins,
        test_date_fraction=0.40,
        train_signal_fractions=(1.0,),
        bootstrap_replicates=200,
        random_state=7,
    )
    metrics["confidence_cutoff"] = metrics["confidence_cutoff"].map(
        lambda cutoff: np.nextafter(cutoff, np.inf)
    )
    audit, result = evaluate_round_trip_policy_family(
        bins,
        metrics,
        bootstrap_replicates=500,
        random_state=7,
    )

    assert len(audit) == 12
    assert result["protocol"]["family_policies"] == 12
    assert result["protocol"]["estimable_policies"] == 12
    assert np.allclose(
        audit["quoted_round_trip_mean_bps"],
        metrics.sort_values(
            [
                "sample",
                "spread_bucket",
                "model",
                "target_train_signal_fraction",
                "entry_latency_us",
                "shares",
            ]
        )["quoted_round_trip_mean_bps"],
    )
    assert (audit["simultaneous_upper_95_bps"] > audit["quoted_round_trip_mean_bps"]).all()
    best = audit.loc[audit["quoted_round_trip_mean_bps"].idxmax()]
    assert np.isclose(
        result["best_observed_policy"]["quoted_round_trip_mean_bps"],
        best["quoted_round_trip_mean_bps"],
    )
