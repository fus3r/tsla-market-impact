import numpy as np
import pandas as pd

from tsla_market_impact.impact import aggregate_prediction_windows


def _market_orders() -> pd.DataFrame:
    frames = []
    for day_index, date in enumerate(["2019-01-02", "2019-01-03"]):
        count = 6
        frames.append(
            pd.DataFrame(
                {
                    "date": date,
                    "seconds": np.arange(count, dtype=float) + 34_200,
                    "first_event_row": np.arange(count),
                    "size": [10, 20, 30, 40, 50, 60],
                    "trade_sign": np.ones(count),
                    "mid_price_before": 100 + day_index * 100 + np.arange(count) * 0.1,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_prediction_windows_never_cross_a_session() -> None:
    windows = aggregate_prediction_windows(_market_orders(), horizons=[2])

    assert windows.groupby("date").size().to_dict() == {
        "2019-01-02": 2,
        "2019-01-03": 2,
    }
    assert np.allclose(windows["impact"], 0.2)
    assert windows["volume_imbalance"].tolist() == [30.0, 70.0, 30.0, 70.0]

def test_prediction_windows_do_not_require_daily_impact_normalisation() -> None:
    trades = _market_orders()
    second_day = trades["date"].eq("2019-01-03")
    trades.loc[second_day, "mid_price_before"] = 200 - np.arange(6) * 0.1

    windows = aggregate_prediction_windows(trades, horizons=[2])

    assert windows.groupby("date").size().to_dict() == {
        "2019-01-02": 2,
        "2019-01-03": 2,
    }
    assert np.allclose(windows.loc[windows["date"].eq("2019-01-03"), "impact"], -0.2)
