import numpy as np
import pandas as pd

from tsla_market_impact.queue import evaluate_queue_imbalance


def _queue_bins() -> pd.DataFrame:
    rows = []
    centers = np.linspace(-0.8, 0.8, 9)
    for day in range(1, 21):
        for index, center in enumerate(centers):
            probability = 1 / (1 + np.exp(-2.5 * center))
            observations = 200
            up_moves = int(round(observations * probability))
            rows.append(
                {
                    "date": f"2019-01-{day:02d}",
                    "sample": "best_quote_updates",
                    "spread_bucket": "all_spreads",
                    "bin": index,
                    "bin_left": center - 0.1,
                    "bin_right": center + 0.1,
                    "bin_center": center,
                    "observations": observations,
                    "up_moves": up_moves,
                    "down_moves": observations - up_moves,
                }
            )
    return pd.DataFrame(rows)


def test_queue_model_uses_fixed_later_date_holdout() -> None:
    metrics, _, result = evaluate_queue_imbalance(
        _queue_bins(),
        test_start_date="2019-01-17",
        bootstrap_replicates=200,
        random_state=7,
    )

    row = metrics.iloc[0]
    assert row["train_last_date"] == "2019-01-16"
    assert row["test_first_date"] == "2019-01-17"
    assert row["baseline_roc_auc"] == 0.5
    assert row["model_roc_auc"] > 0.75
    assert row["relative_brier_reduction"] > 0.1
    assert row["clusters"] == 4
    assert result["protocol"]["train_dates"] == 16
    assert result["protocol"]["test_dates"] == 4
