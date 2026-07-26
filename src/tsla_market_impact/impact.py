"""Event-time returns, aggregate impact, and local Kyle-lambda estimates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data import required_columns


def _sort_columns(frame: pd.DataFrame, include_date: bool = True) -> list[str]:
    prefix = ["date"] if include_date else []
    if "first_event_row" in frame.columns:
        return prefix + ["first_event_row"]
    return prefix + ["seconds", "trade_sign"]


def aggregate_prediction_windows(
    trades: pd.DataFrame,
    horizons: Sequence[int] = (5, 10, 20, 50, 100),
    stride: str | int = "horizon",
) -> pd.DataFrame:
    """Build same-session windows of order flow and mid-price impact."""

    required_columns(
        trades,
        ["date", "size", "trade_sign", "mid_price_before"],
    )
    if isinstance(stride, int) and stride < 1:
        raise ValueError("stride must be a positive integer or 'horizon'")
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")

    frames: list[pd.DataFrame] = []
    for date, day in trades.groupby("date", sort=True, observed=True):
        ordered = day.sort_values(_sort_columns(day, include_date=False))
        mid = ordered["mid_price_before"].to_numpy(dtype=float)
        size = ordered["size"].to_numpy(dtype=float)
        sign = ordered["trade_sign"].to_numpy(dtype=float)
        signed_volume = size * sign
        volume_cumsum = np.r_[0.0, np.cumsum(signed_volume)]
        sign_cumsum = np.r_[0.0, np.cumsum(sign)]

        for horizon in horizons:
            step = horizon if stride == "horizon" else int(stride)
            starts = np.arange(0, len(ordered) - horizon, step)
            if starts.size == 0:
                continue
            stops = starts + horizon
            start_mid = mid[starts]
            stop_mid = mid[stops]
            frames.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "horizon": horizon,
                        "volume_imbalance": (
                            volume_cumsum[stops] - volume_cumsum[starts]
                        ),
                        "order_flow_imbalance": (
                            sign_cumsum[stops] - sign_cumsum[starts]
                        ),
                        "impact": stop_mid - start_mid,
                        "impact_bps": 10_000 * np.log(stop_mid / start_mid),
                    }
                )
            )

    columns = [
        "date",
        "horizon",
        "volume_imbalance",
        "order_flow_imbalance",
        "impact",
        "impact_bps",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def quantile_impact_curve(
    windows: pd.DataFrame,
    quantiles: int = 31,
    bin_column: str = "volume_imbalance",
) -> pd.DataFrame:
    """Average impact within separate imbalance quantiles for each horizon."""

    required_columns(windows, ["horizon", bin_column, "impact"])
    frames: list[pd.DataFrame] = []
    for horizon, part in windows.groupby("horizon", sort=True, observed=True):
        labels = pd.qcut(part[bin_column], q=quantiles, duplicates="drop")
        curve = (
            part.assign(_bin=labels)
            .groupby("_bin", observed=True)
            .agg(
                volume_imbalance=("volume_imbalance", "mean"),
                order_flow_imbalance=("order_flow_imbalance", "mean"),
                impact=("impact", "mean"),
                observations=("impact", "size"),
            )
            .reset_index(drop=True)
        )
        curve.insert(0, "horizon", int(horizon))
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def estimate_kyle_lambdas(
    windows: pd.DataFrame,
    central_quantile: float = 0.35,
) -> pd.DataFrame:
    """Fit local impact slopes in the central imbalance region."""

    if not 0 < central_quantile < 1:
        raise ValueError("central_quantile must lie between zero and one")
    required_columns(windows, ["horizon", "volume_imbalance", "impact"])
    rows: list[dict[str, float | int]] = []
    for horizon, part in windows.groupby("horizon", sort=True, observed=True):
        threshold = float(part["volume_imbalance"].abs().quantile(central_quantile))
        central = part.loc[part["volume_imbalance"].abs() <= threshold]
        x = central["volume_imbalance"].to_numpy(dtype=float)
        y = central["impact"].to_numpy(dtype=float) * 100
        design = np.column_stack([np.ones(len(x)), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        rows.append(
            {
                "horizon": int(horizon),
                "threshold_shares": threshold,
                "intercept_cents": float(intercept),
                "kyle_lambda_cents_per_share": float(slope),
                "observations": len(central),
            }
        )
    return pd.DataFrame(rows)


def fit_kyle_decay(lambdas: pd.DataFrame) -> dict[str, float]:
    """Fit ``lambda(T) = prefactor * T ** (-exponent)`` in log space."""

    required_columns(lambdas, ["horizon", "kyle_lambda_cents_per_share"])
    clean = lambdas.loc[lambdas["kyle_lambda_cents_per_share"] > 0]
    x = np.log(clean["horizon"].to_numpy(dtype=float))
    y = np.log(clean["kyle_lambda_cents_per_share"].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"prefactor": float(np.exp(intercept)), "exponent": float(-slope)}
