"""Shared utilities for aggregate next-mid-price-move evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def compressed_binary_rows(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent binomial aggregate rows as two weighted class rows."""

    features = np.repeat(frame[list(feature_columns)].to_numpy(dtype=float), 2, axis=0)
    targets = np.tile([1, 0], len(frame))
    weights = frame[["up_moves", "down_moves"]].to_numpy(dtype=float).reshape(-1)
    keep = weights > 0
    return features[keep], targets[keep], weights[keep]


def weighted_binary_scores(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Score binary probabilities without expanding aggregate counts."""

    predicted = probabilities >= 0.5
    brier = np.average((targets - probabilities) ** 2, weights=weights)
    return {
        "roc_auc": float(
            roc_auc_score(targets, probabilities, sample_weight=weights)
        ),
        "log_loss": float(log_loss(targets, probabilities, sample_weight=weights)),
        "brier_score": float(brier),
        "accuracy": float(np.average(predicted == targets, weights=weights)),
    }


def day_cluster_brier_comparison(
    test: pd.DataFrame,
    challenger_probability: np.ndarray,
    reference_probability: np.ndarray,
    replicates: int,
    random_state: int,
) -> dict[str, float | int]:
    """Bootstrap a relative Brier reduction over complete trading dates."""

    if len(test) != len(challenger_probability) or len(test) != len(
        reference_probability
    ):
        raise ValueError("each probability array must have one value per aggregate row")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    up = test["up_moves"].to_numpy(dtype=float)
    down = test["down_moves"].to_numpy(dtype=float)
    challenger_sse = (
        up * (1 - challenger_probability) ** 2
        + down * challenger_probability**2
    )
    reference_sse = (
        up * (1 - reference_probability) ** 2 + down * reference_probability**2
    )
    daily = (
        pd.DataFrame(
            {
                "date": test["date"].to_numpy(),
                "challenger_sse": challenger_sse,
                "reference_sse": reference_sse,
            }
        )
        .groupby("date", sort=True, observed=True)
        .sum()
    )
    values = daily[["challenger_sse", "reference_sse"]].to_numpy(dtype=float)
    point = 1 - values[:, 0].sum() / values[:, 1].sum()
    random = np.random.default_rng(random_state)
    sampled = random.integers(0, len(values), size=(replicates, len(values)))
    totals = values[sampled].sum(axis=1)
    improvements = 1 - totals[:, 0] / totals[:, 1]
    lower, median, upper = np.quantile(improvements, [0.025, 0.5, 0.975])
    return {
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "relative_brier_reduction": float(point),
        "relative_brier_reduction_lower_95": float(lower),
        "relative_brier_reduction_median": float(median),
        "relative_brier_reduction_upper_95": float(upper),
        "probability_nonpositive": float(np.mean(improvements <= 0)),
    }
