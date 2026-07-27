"""Chronological evaluation for aggregate-impact regression."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import required_columns
from .policy import split_index_for_test_start

BASE_FEATURES = ["volume_imbalance"]
VOLUME_FEATURES = [
    "volume_imbalance",
    "abs_volume_imbalance",
    "signed_sqrt_volume_imbalance",
    "signed_log_volume_imbalance",
]
VOLUME_COUNT_FEATURES = [*VOLUME_FEATURES, "order_flow_imbalance"]
AUGMENTED_FEATURES = [
    "volume_imbalance",
    "order_flow_imbalance",
    "abs_volume_imbalance",
    "signed_sqrt_volume_imbalance",
    "signed_log_volume_imbalance",
    "abs_order_flow_imbalance",
    "signed_sqrt_order_flow_imbalance",
]


def add_impact_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add fixed transforms of signed volume and signed order count."""

    required_columns(data, ["volume_imbalance", "order_flow_imbalance"])
    result = data.copy()
    volume = result["volume_imbalance"].to_numpy(dtype=float)
    count = result["order_flow_imbalance"].to_numpy(dtype=float)
    result["abs_volume_imbalance"] = np.abs(volume)
    result["signed_sqrt_volume_imbalance"] = np.sign(volume) * np.sqrt(np.abs(volume))
    result["signed_log_volume_imbalance"] = np.sign(volume) * np.log1p(np.abs(volume))
    result["abs_order_flow_imbalance"] = np.abs(count)
    result["signed_sqrt_order_flow_imbalance"] = np.sign(count) * np.sqrt(np.abs(count))
    return result


def prepare_impact_dataset(
    windows: pd.DataFrame,
    horizon: int = 10,
) -> pd.DataFrame:
    """Select one horizon, convert impact to cents, and add fixed features."""

    required_columns(
        windows,
        ["date", "horizon", "volume_imbalance", "order_flow_imbalance", "impact"],
    )
    selected = [
        "date",
        "horizon",
        "volume_imbalance",
        "order_flow_imbalance",
        "impact",
    ]
    if "impact_bps" in windows.columns:
        selected.append("impact_bps")
    data = windows.loc[windows["horizon"].eq(horizon), selected].copy()
    data["impact_cents"] = data["impact"] * 100
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "volume_imbalance", "order_flow_imbalance", "impact_cents"]
    )
    return (
        add_impact_features(data).sort_values(["date", "volume_imbalance"]).reset_index(drop=True)
    )


def temporal_train_test_split(
    data: pd.DataFrame,
    test_date_fraction: float = 0.20,
    test_start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out dates from a fixed boundary, or a final fraction when omitted."""

    if test_start_date is None and not 0 < test_date_fraction < 1:
        raise ValueError("test_date_fraction must lie between zero and one")
    required_columns(data, ["date"])
    dates = np.array(sorted(data["date"].unique()))
    if len(dates) < 2:
        raise ValueError("At least two distinct dates are required")
    if test_start_date is None:
        split = int(np.floor((1 - test_date_fraction) * len(dates)))
        split = min(max(split, 1), len(dates) - 1)
    else:
        split = split_index_for_test_start(dates.astype(str).tolist(), test_start_date)
    train_dates = set(dates[:split])
    train = data.loc[data["date"].isin(train_dates)].copy()
    test = data.loc[~data["date"].isin(train_dates)].copy()
    return train, test


def apply_training_sigma_filter(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    standard_deviations: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    """Estimate filter bounds on training dates and apply them to both splits."""

    if standard_deviations <= 0:
        raise ValueError("standard_deviations must be positive")
    bounds: dict[str, tuple[float, float]] = {}
    train_mask = np.ones(len(train), dtype=bool)
    test_mask = np.ones(len(test), dtype=bool)
    for column in columns:
        required_columns(train, [column])
        required_columns(test, [column])
        mean = float(train[column].mean())
        standard_deviation = float(train[column].std(ddof=0))
        lower = mean - standard_deviations * standard_deviation
        upper = mean + standard_deviations * standard_deviation
        bounds[column] = (lower, upper)
        train_mask &= train[column].between(lower, upper, inclusive="both").to_numpy()
        test_mask &= test[column].between(lower, upper, inclusive="both").to_numpy()
    return train.loc[train_mask].copy(), test.loc[test_mask].copy(), bounds


def regression_scores(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
    """Return the predictive metrics used in the report."""

    y_true = np.asarray(y_true, dtype=float)
    y_predicted = np.asarray(y_predicted, dtype=float)
    mse = float(mean_squared_error(y_true, y_predicted))
    nonzero = (y_true != 0) & (y_predicted != 0)
    sign_accuracy = (
        float(np.mean(np.sign(y_true[nonzero]) == np.sign(y_predicted[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    return {
        "mse_cents_squared": mse,
        "rmse_cents": float(np.sqrt(mse)),
        "mae_cents": float(mean_absolute_error(y_true, y_predicted)),
        "r_squared": float(r2_score(y_true, y_predicted)),
        "sign_accuracy": sign_accuracy,
    }


def model_specifications() -> list[dict[str, object]]:
    """Return the nested linear specifications used in the report."""

    return [
        {
            "name": "OLS signed volume",
            "features": BASE_FEATURES,
            "estimator": LinearRegression(),
        },
        {
            "name": "OLS volume transforms",
            "features": VOLUME_FEATURES,
            "estimator": LinearRegression(),
        },
        {
            "name": "OLS volume and count",
            "features": VOLUME_COUNT_FEATURES,
            "estimator": LinearRegression(),
        },
        {
            "name": "OLS count transforms",
            "features": AUGMENTED_FEATURES,
            "estimator": LinearRegression(),
        },
        {
            "name": "Ridge count transforms",
            "features": AUGMENTED_FEATURES,
            "estimator": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        },
    ]


def evaluate_impact_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specifications: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit each specification and score it on one chronological holdout."""

    required_columns(train, ["impact_cents"])
    required_columns(test, ["impact_cents"])
    specifications = specifications or model_specifications()
    rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    y_train = train["impact_cents"].to_numpy(dtype=float)
    y_test = test["impact_cents"].to_numpy(dtype=float)

    for specification in specifications:
        name = str(specification["name"])
        features = list(specification["features"])
        required_columns(train, features)
        required_columns(test, features)
        estimator = clone(specification["estimator"])
        estimator.fit(train[features].to_numpy(dtype=float), y_train)
        fitted[name] = estimator
        train_scores = regression_scores(
            y_train,
            estimator.predict(train[features].to_numpy(dtype=float)),
        )
        test_scores = regression_scores(
            y_test,
            estimator.predict(test[features].to_numpy(dtype=float)),
        )
        row: dict[str, object] = {"model": name, "features": ", ".join(features)}
        row.update({f"train_{key}": value for key, value in train_scores.items()})
        row.update({f"test_{key}": value for key, value in test_scores.items()})
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("test_mse_cents_squared").reset_index(drop=True)
    return results, fitted


def expanding_date_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    initial_train_fraction: float = 0.40,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return expanding training sets followed by contiguous date blocks."""

    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must lie between zero and one")
    required_columns(data, ["date"])
    dates = np.array(sorted(data["date"].unique()))
    if len(dates) <= n_splits:
        raise ValueError("Not enough dates for the requested number of splits")
    initial = max(1, int(np.floor(initial_train_fraction * len(dates))))
    if len(dates) - initial < n_splits:
        initial = len(dates) - n_splits

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for test_dates in np.array_split(dates[initial:], n_splits):
        first_test = np.flatnonzero(dates == test_dates[0])[0]
        train_dates = set(dates[:first_test])
        test_date_set = set(test_dates)
        splits.append(
            (
                data.loc[data["date"].isin(train_dates)].copy(),
                data.loc[data["date"].isin(test_date_set)].copy(),
            )
        )
    return splits


def evaluate_expanding_splits(
    data: pd.DataFrame,
    specifications: list[dict[str, object]],
    n_splits: int = 5,
    initial_train_fraction: float = 0.40,
) -> pd.DataFrame:
    """Score model specifications over expanding chronological folds."""

    rows: list[dict[str, object]] = []
    splits = expanding_date_splits(data, n_splits, initial_train_fraction)
    for fold, (train, test) in enumerate(splits, start=1):
        metrics, _ = evaluate_impact_models(train, test, specifications)
        for result in metrics.to_dict(orient="records"):
            rows.append(
                {
                    "fold": fold,
                    "train_first_date": str(train["date"].min()),
                    "train_last_date": str(train["date"].max()),
                    "test_first_date": str(test["date"].min()),
                    "test_last_date": str(test["date"].max()),
                    "train_observations": len(train),
                    "test_observations": len(test),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def day_cluster_bootstrap_improvement(
    test: pd.DataFrame,
    baseline_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    replicates: int = 10_000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Bootstrap score improvement by resampling complete test dates."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    required_columns(test, ["date", "impact_cents"])
    y = test["impact_cents"].to_numpy(dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    augmented = np.asarray(augmented_prediction, dtype=float)
    if len(y) != len(baseline) or len(y) != len(augmented):
        raise ValueError("Predictions must have the same length as test")

    contributions = (
        pd.DataFrame(
            {
                "date": test["date"].to_numpy(),
                "y": y,
                "y_squared": y**2,
                "baseline_sse": (y - baseline) ** 2,
                "augmented_sse": (y - augmented) ** 2,
            }
        )
        .groupby("date", sort=True, observed=True)
        .agg(
            observations=("y", "size"),
            y_sum=("y", "sum"),
            y_squared_sum=("y_squared", "sum"),
            baseline_sse=("baseline_sse", "sum"),
            augmented_sse=("augmented_sse", "sum"),
        )
    )
    values = contributions.to_numpy(dtype=float)
    random = np.random.default_rng(random_state)
    sampled = random.integers(0, len(values), size=(replicates, len(values)))
    totals = values[sampled].sum(axis=1)
    observations, y_sum, y_squared_sum, baseline_sse, augmented_sse = totals.T
    total_sum_squares = y_squared_sum - y_sum**2 / observations
    delta_r_squared = (baseline_sse - augmented_sse) / total_sum_squares
    relative_mse_reduction = 1 - augmented_sse / baseline_sse

    def interval(values_: np.ndarray) -> tuple[float, float, float]:
        lower, median, upper = np.quantile(values_, [0.025, 0.5, 0.975])
        return float(lower), float(median), float(upper)

    delta_lower, delta_median, delta_upper = interval(delta_r_squared)
    mse_lower, mse_median, mse_upper = interval(relative_mse_reduction)
    return {
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "delta_r_squared_lower_95": delta_lower,
        "delta_r_squared_median": delta_median,
        "delta_r_squared_upper_95": delta_upper,
        "relative_mse_reduction_lower_95": mse_lower,
        "relative_mse_reduction_median": mse_median,
        "relative_mse_reduction_upper_95": mse_upper,
        "probability_delta_r_squared_nonpositive": float(np.mean(delta_r_squared <= 0)),
    }


def calibration_curve(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    quantiles: int = 20,
) -> pd.DataFrame:
    """Aggregate realised and fitted impact by prediction quantile."""

    y = np.asarray(y_true, dtype=float)
    frames: list[pd.DataFrame] = []
    for model, prediction in predictions.items():
        fitted = np.asarray(prediction, dtype=float)
        if len(y) != len(fitted):
            raise ValueError("Predictions must have the same length as y_true")
        bins = pd.qcut(fitted, q=quantiles, duplicates="drop")
        curve = (
            pd.DataFrame({"realised": y, "fitted": fitted, "bin": bins})
            .groupby("bin", observed=True)
            .agg(
                mean_fitted_cents=("fitted", "mean"),
                mean_realised_cents=("realised", "mean"),
                standard_error_cents=("realised", "sem"),
                observations=("realised", "size"),
            )
            .reset_index(drop=True)
        )
        curve.insert(0, "model", model)
        curve.insert(1, "bin", np.arange(1, len(curve) + 1))
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def residual_curve_by_count(
    test: pd.DataFrame,
    volume_prediction: np.ndarray,
) -> pd.DataFrame:
    """Summarise holdout residuals from a volume-only model by signed count."""

    required_columns(test, ["date", "order_flow_imbalance", "impact_cents"])
    prediction = np.asarray(volume_prediction, dtype=float)
    if len(test) != len(prediction):
        raise ValueError("volume_prediction must have the same length as test")
    residuals = test[["date", "order_flow_imbalance"]].copy()
    residuals["residual_cents"] = test["impact_cents"].to_numpy(dtype=float) - prediction
    daily = (
        residuals.groupby(["order_flow_imbalance", "date"], observed=True)["residual_cents"]
        .mean()
        .reset_index()
    )
    curve = (
        daily.groupby("order_flow_imbalance", sort=True, observed=True)
        .agg(
            mean_residual_cents=("residual_cents", "mean"),
            standard_error_cents=("residual_cents", "sem"),
            dates=("date", "nunique"),
        )
        .reset_index()
    )
    observations = (
        residuals.groupby("order_flow_imbalance", observed=True)
        .size()
        .rename("observations")
        .reset_index()
    )
    return curve.merge(observations, on="order_flow_imbalance", validate="one_to_one")
