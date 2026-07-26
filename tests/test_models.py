import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tsla_market_impact.models import (
    AUGMENTED_FEATURES,
    BASE_FEATURES,
    add_impact_features,
    apply_training_sigma_filter,
    day_cluster_bootstrap_improvement,
    evaluate_impact_models,
    expanding_date_splits,
    temporal_train_test_split,
)


def test_temporal_split_holds_out_latest_dates() -> None:
    data = pd.DataFrame(
        {
            "date": np.repeat([f"2019-01-{day:02d}" for day in range(1, 11)], 2),
            "value": np.arange(20),
        }
    )

    train, test = temporal_train_test_split(data, test_date_fraction=0.2)

    assert train["date"].max() == "2019-01-08"
    assert test["date"].min() == "2019-01-09"
    assert set(train["date"]).isdisjoint(test["date"])


def test_signed_count_adds_holdout_signal() -> None:
    random = np.random.default_rng(12)
    rows = []
    for day in range(1, 21):
        for _ in range(80):
            volume = random.normal(0, 500)
            count = random.integers(-10, 11)
            target = 0.002 * volume + 1.8 * count + random.normal(0, 1.2)
            rows.append(
                {
                    "date": f"2019-01-{day:02d}",
                    "volume_imbalance": volume,
                    "order_flow_imbalance": count,
                    "impact_cents": target,
                }
            )
    data = add_impact_features(pd.DataFrame(rows))
    train, test = temporal_train_test_split(data)
    specifications = [
        {"name": "baseline", "features": BASE_FEATURES, "estimator": LinearRegression()},
        {
            "name": "augmented",
            "features": AUGMENTED_FEATURES,
            "estimator": LinearRegression(),
        },
    ]

    metrics, _ = evaluate_impact_models(train, test, specifications)
    scores = metrics.set_index("model")["test_r_squared"]

    assert scores["augmented"] > 0.95
    assert scores["augmented"] > scores["baseline"] + 0.5


def test_sigma_filter_uses_training_bounds_only() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 100.0]})
    test = pd.DataFrame({"x": [1.0, 2.0, 500.0]})

    filtered_train, filtered_test, bounds = apply_training_sigma_filter(
        train, test, ["x"], standard_deviations=1.5
    )

    assert bounds["x"][1] < 500
    assert 500.0 not in filtered_test["x"].tolist()
    assert len(filtered_train) == 3


def test_expanding_splits_keep_test_dates_in_order() -> None:
    data = pd.DataFrame(
        {
            "date": np.repeat([f"2019-01-{day:02d}" for day in range(1, 11)], 2),
            "impact_cents": np.arange(20),
        }
    )

    splits = expanding_date_splits(data, n_splits=3, initial_train_fraction=0.4)

    assert len(splits) == 3
    assert splits[0][0]["date"].max() < splits[0][1]["date"].min()
    assert len(splits[-1][0]["date"].unique()) > len(splits[0][0]["date"].unique())
    assert splits[-1][1]["date"].max() == "2019-01-10"


def test_day_cluster_bootstrap_detects_better_predictions() -> None:
    test = pd.DataFrame(
        {
            "date": np.repeat(["2019-01-01", "2019-01-02", "2019-01-03"], 4),
            "impact_cents": np.tile([-2.0, -1.0, 1.0, 2.0], 3),
        }
    )
    baseline = np.zeros(len(test))
    augmented = test["impact_cents"].to_numpy() * 0.9

    result = day_cluster_bootstrap_improvement(
        test,
        baseline,
        augmented,
        replicates=200,
        random_state=7,
    )

    assert result["delta_r_squared_lower_95"] > 0
    assert result["relative_mse_reduction_lower_95"] > 0
