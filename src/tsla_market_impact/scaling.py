"""Finite-size scaling analysis for aggregate market impact."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .data import required_columns

REGULAR_SESSION_OPEN = 9.5 * 60 * 60
REGULAR_SESSION_CLOSE = 16 * 60 * 60
DEFAULT_HORIZONS = tuple(np.unique(np.rint(np.geomspace(10, 5_000, 28)).astype(int)))


def master_curve(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Sigmoidal curve ``x / (1 + |x|**alpha)**(beta/alpha)``."""

    values = np.asarray(x, dtype=float)
    return values / np.power(1 + np.power(np.abs(values), alpha), beta / alpha)


def compute_scaling_windows(
    transactions: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    open_seconds: float = REGULAR_SESSION_OPEN,
    close_seconds: float = REGULAR_SESSION_CLOSE,
) -> pd.DataFrame:
    """Build non-overlapping same-session windows using the paper definitions."""

    required_columns(
        transactions,
        ["date", "seconds", "size", "trade_sign", "mid_price_before"],
    )
    trimmed = transactions.loc[
        transactions["seconds"].between(open_seconds, close_seconds, inclusive="both")
    ].copy()
    daily_volume = trimmed.groupby("date", observed=True)["size"].sum()
    mean_daily_volume = float(daily_volume.mean())
    frames: list[pd.DataFrame] = []

    for date, day in trimmed.groupby("date", sort=True, observed=True):
        sort_column = "first_event_row" if "first_event_row" in day else "seconds"
        ordered = day.sort_values(sort_column)
        mid = ordered["mid_price_before"].to_numpy(dtype=float)
        log_mid = np.log(mid)
        signs = ordered["trade_sign"].to_numpy(dtype=float)
        signed_volume = signs * ordered["size"].to_numpy(dtype=float)
        one_step_change = np.r_[np.diff(mid) != 0, False].astype(float)
        return_sign = np.r_[np.sign(np.diff(log_mid)), 0.0]

        volume_cumsum = np.r_[0.0, np.cumsum(signed_volume)]
        sign_cumsum = np.r_[0.0, np.cumsum(signs)]
        change_cumsum = np.r_[0.0, np.cumsum(one_step_change)]
        return_sign_cumsum = np.r_[0.0, np.cumsum(return_sign)]
        volume_factor = mean_daily_volume / float(daily_volume.loc[date])

        for horizon in horizons:
            starts = np.arange(0, len(ordered) - horizon, horizon)
            if starts.size == 0:
                continue
            stops = starts + horizon
            sign_sum = sign_cumsum[stops] - sign_cumsum[starts]
            frames.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "horizon": horizon,
                        "Q": volume_factor * (volume_cumsum[stops] - volume_cumsum[starts]),
                        "E": sign_sum,
                        "mean_sign": sign_sum / horizon,
                        "impact_log": log_mid[stops] - log_mid[starts],
                        "price_change_probability": (change_cumsum[stops] - change_cumsum[starts])
                        / horizon,
                        "return_sign_imbalance": (
                            return_sign_cumsum[stops] - return_sign_cumsum[starts]
                        ),
                    }
                )
            )
    if not frames:
        raise ValueError("No scaling windows could be constructed")
    return pd.concat(frames, ignore_index=True)


def quantile_curve(
    windows: pd.DataFrame,
    x_column: str = "Q",
    y_column: str = "impact_log",
    quantiles: int = 31,
) -> pd.DataFrame:
    """Average a response in separate x-quantiles for every horizon."""

    required_columns(windows, ["horizon", x_column, y_column])
    frames: list[pd.DataFrame] = []
    for horizon, part in windows.groupby("horizon", sort=True, observed=True):
        labels = pd.qcut(part[x_column], q=quantiles, duplicates="drop")
        curve = (
            part.assign(_bin=labels)
            .groupby("_bin", observed=True)
            .agg(
                x=(x_column, "mean"),
                y=(y_column, "mean"),
                observations=(y_column, "size"),
            )
            .reset_index(drop=True)
        )
        curve.insert(0, "horizon", int(horizon))
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def fit_volume_curve_scales(curve: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Estimate one width and height per horizon plus a shared curve shape."""

    required_columns(curve, ["horizon", "x", "y", "observations"])
    horizons = np.sort(curve["horizon"].unique())
    lookup = {horizon: index for index, horizon in enumerate(horizons)}
    horizon_index = curve["horizon"].map(lookup).to_numpy(dtype=int)
    x = curve["x"].to_numpy(dtype=float)
    y = curve["y"].to_numpy(dtype=float)
    weights = np.sqrt(curve["observations"].to_numpy(dtype=float))
    weights /= weights.max()
    grouped = curve.groupby("horizon", sort=True, observed=True)
    initial_q = grouped["x"].std().clip(lower=1e-9).to_numpy()
    initial_r = grouped["y"].std().clip(lower=1e-12).to_numpy()
    initial = np.r_[np.log(initial_q), np.log(initial_r), np.log(1.2), np.log(1.3)]

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        count = len(horizons)
        q_scale = np.exp(parameters[:count])
        r_scale = np.exp(parameters[count : 2 * count])
        alpha, beta = np.exp(parameters[-2:])
        return q_scale, r_scale, float(alpha), float(beta)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        q_scale, r_scale, alpha, beta = unpack(parameters)
        predicted = r_scale[horizon_index] * master_curve(x / q_scale[horizon_index], alpha, beta)
        return weights * (predicted - y)

    count = len(horizons)
    lower = np.r_[
        np.full(count, -30.0),
        np.full(count, -30.0),
        np.log(0.1),
        np.log(0.1),
    ]
    upper = np.r_[
        np.full(count, 30.0),
        np.full(count, 5.0),
        np.log(8.0),
        np.log(5.0),
    ]
    result = least_squares(
        residuals,
        x0=initial,
        bounds=(lower, upper),
        loss="soft_l1",
        max_nfev=10_000,
    )
    q_scale, r_scale, alpha, beta = unpack(result.x)
    scales = pd.DataFrame({"horizon": horizons, "Q_N": q_scale, "R_N": r_scale})
    shape = {
        "alpha": alpha,
        "beta": beta,
        "success": bool(result.success),
        "cost": float(result.cost),
    }
    return scales, shape


def fit_loglog_scale(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, float]:
    """Fit ``y = prefactor * x**exponent`` and return log-space diagnostics."""

    x_log = np.log(np.asarray(x, dtype=float))
    y_log = np.log(np.asarray(y, dtype=float))
    if len(x_log) < 3:
        raise ValueError("At least three observations are required for a scaling fit")
    design = np.column_stack([np.ones(len(x_log)), x_log])
    intercept, exponent = np.linalg.lstsq(design, y_log, rcond=None)[0]
    fitted = design @ np.array([intercept, exponent])
    residuals = y_log - fitted
    degrees_of_freedom = len(x_log) - 2
    sum_squared_error = float(residuals @ residuals)
    centered = y_log - y_log.mean()
    total_sum_squares = float(centered @ centered)
    covariance = np.linalg.pinv(design.T @ design) * (sum_squared_error / degrees_of_freedom)
    return {
        "prefactor": float(np.exp(intercept)),
        "exponent": float(exponent),
        "standard_error": float(np.sqrt(covariance[1, 1])),
        "r_squared": (
            float(1 - sum_squared_error / total_sum_squares)
            if total_sum_squares > 0
            else float("nan")
        ),
        "log_residual_standard_deviation": float(np.sqrt(sum_squared_error / degrees_of_freedom)),
        "observations": int(len(x_log)),
    }


def fit_scaling_laws(scales: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit the horizon dependence of the width and height scales."""

    required_columns(scales, ["horizon", "Q_N", "R_N"])
    return {
        "width": fit_loglog_scale(scales["horizon"], scales["Q_N"]),
        "height": fit_loglog_scale(scales["horizon"], scales["R_N"]),
    }


def fit_variance_scaling_statistics(
    windows: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Fit standard-deviation scaling for volume, returns, and order signs."""

    required_columns(windows, ["horizon", "Q", "E", "impact_log"])
    statistics = (
        windows.groupby("horizon", sort=True, observed=True)
        .agg(q_std=("Q", "std"), return_std=("impact_log", "std"), sign_std=("E", "std"))
        .reset_index()
    )
    return {
        "volume": fit_loglog_scale(statistics["horizon"], statistics["q_std"]),
        "return": fit_loglog_scale(statistics["horizon"], statistics["return_std"]),
        "sign": fit_loglog_scale(statistics["horizon"], statistics["sign_std"]),
    }


def collapse_curve(curve: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    """Attach rescaled coordinates for each horizon."""

    lookup = scales.set_index("horizon")
    collapsed = curve.copy()
    collapsed["x_scaled"] = [row.x / lookup.loc[row.horizon, "Q_N"] for row in curve.itertuples()]
    collapsed["y_scaled"] = [row.y / lookup.loc[row.horizon, "R_N"] for row in curve.itertuples()]
    return collapsed
