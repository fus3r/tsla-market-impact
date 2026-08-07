import numpy as np
import pandas as pd

from tsla_market_impact.stability import (
    circular_block_brier_comparison,
    evaluate_signal_stability,
)


def _landmark_bins() -> pd.DataFrame:
    rows = []
    centers = [-0.75, -0.25, 0.25, 0.75]
    for month in range(1, 9):
        for day in [5, 20]:
            date = f"2019-{month:02d}-{day:02d}"
            for spread_bucket in ["all_spreads", "one_tick"]:
                for queue_bin, queue in enumerate(centers):
                    for ofi_bin, ofi in enumerate(centers):
                        probability = 1 / (1 + np.exp(-(0.8 * queue + 1.5 * ofi)))
                        signals = 200
                        up_moves = int(round(signals * probability))
                        rows.append(
                            {
                                "date": date,
                                "sample": "price_spell_landmarks",
                                "spread_bucket": spread_bucket,
                                "landmark_age_us": 100,
                                "latency_us": 0,
                                "queue_bin": queue_bin,
                                "ofi_bin": ofi_bin,
                                "queue_center": queue,
                                "ofi_center": ofi,
                                "signals": signals,
                                "up_moves": up_moves,
                                "down_moves": signals - up_moves,
                            }
                        )
    return pd.DataFrame(rows)


def test_monthly_origins_use_only_preceding_landmark_dates() -> None:
    folds, overall, comparisons, result = evaluate_signal_stability(
        _landmark_bins(),
        initial_train_months=3,
        block_lengths=(1, 2),
        bootstrap_replicates=200,
        random_state=7,
    )

    assert result["protocol"]["evaluation_months"] == [
        "2019-04",
        "2019-05",
        "2019-06",
        "2019-07",
        "2019-08",
    ]
    origins = folds[["fold", "train_last_date", "test_first_date"]].drop_duplicates()
    assert (origins["train_last_date"] < origins["test_first_date"]).all()
    assert set(folds["model"]) == {"intercept", "queue", "ofi", "queue_and_ofi"}

    combined_folds = folds.loc[folds["model"].eq("queue_and_ofi")]
    assert (combined_folds["relative_brier_reduction_vs_intercept"] > 0).all()
    combined_overall = overall.loc[overall["model"].eq("queue_and_ofi")]
    assert (combined_overall["roc_auc"] > 0.70).all()

    primary = comparisons.loc[
        comparisons["challenger"].eq("queue_and_ofi")
        & comparisons["reference"].eq("intercept")
        & comparisons["block_length_dates"].eq(2)
    ]
    assert set(primary["positive_months"]) == {5}
    assert (primary["relative_brier_reduction_lower_95"] > 0).all()


def test_circular_blocks_keep_paired_loss_ratios() -> None:
    reference = np.array([4.0, 2.0, 5.0, 3.0, 6.0])
    comparison = circular_block_brier_comparison(
        0.8 * reference,
        reference,
        block_length=3,
        replicates=200,
        random_state=11,
    )

    assert np.isclose(comparison["relative_brier_reduction"], 0.20)
    assert np.isclose(comparison["relative_brier_reduction_lower_95"], 0.20)
    assert np.isclose(comparison["relative_brier_reduction_upper_95"], 0.20)
    assert comparison["probability_nonpositive"] == 0
