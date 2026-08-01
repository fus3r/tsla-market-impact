import numpy as np
import pandas as pd

from tsla_market_impact.orderflow import (
    evaluate_order_flow_grid_robustness,
    evaluate_order_flow_signals,
)


def _joint_signal_bins() -> pd.DataFrame:
    rows = []
    centers = [-0.75, -0.25, 0.25, 0.75]
    for day in range(1, 31):
        for queue_index, queue in enumerate(centers):
            for ofi_index, ofi in enumerate(centers):
                probability = 1 / (1 + np.exp(-(1.3 * queue + 1.1 * ofi)))
                observations = 240
                up_moves = int(round(observations * probability))
                rows.append(
                    {
                        "date": f"2019-01-{day:02d}",
                        "sample": "best_quote_updates",
                        "spread_bucket": "all_spreads",
                        "queue_bin": queue_index,
                        "ofi_bin": ofi_index,
                        "queue_center": queue,
                        "ofi_center": ofi,
                        "observations": observations,
                        "up_moves": up_moves,
                        "down_moves": observations - up_moves,
                    }
                )
    return pd.DataFrame(rows)


def test_combined_order_flow_model_uses_later_dates_and_both_signals() -> None:
    bins = _joint_signal_bins()
    bins.loc[0, ["up_moves", "down_moves"]] = [0, bins.loc[0, "observations"]]
    metrics, comparisons, calibration, result = evaluate_order_flow_signals(
        bins,
        test_start_date="2019-01-25",
        bootstrap_replicates=200,
        random_state=7,
    )

    scores = metrics.set_index("model")
    assert scores.loc["queue_and_ofi", "test_first_date"] == "2019-01-25"
    assert scores.loc["queue_and_ofi", "roc_auc"] > scores.loc["queue", "roc_auc"]
    assert scores.loc["queue_and_ofi", "roc_auc"] > scores.loc["ofi", "roc_auc"]
    comparison = comparisons.loc[
        comparisons["challenger"].eq("queue_and_ofi")
        & comparisons["reference"].eq("queue")
    ].iloc[0]
    assert comparison["relative_brier_reduction_lower_95"] > 0
    assert set(calibration["model"]) == {
        "intercept",
        "queue",
        "ofi",
        "queue_and_ofi",
    }
    assert result["protocol"]["test_dates"] == 6


def test_grid_robustness_keeps_each_resolution_explicit() -> None:
    bins = _joint_signal_bins()
    coarse = bins.copy()
    fine = bins.copy()
    fine["queue_bin"] *= 2
    fine["ofi_bin"] *= 2
    robustness = evaluate_order_flow_grid_robustness(
        {4: coarse, 7: fine},
        test_start_date="2019-01-25",
        bootstrap_replicates=100,
        random_state=3,
    )

    combined = robustness.loc[robustness["model"].eq("queue_and_ofi")]
    assert set(combined["grid_bins_per_axis"]) == {4, 7}
    assert combined["roc_auc"].min() > 0.7
