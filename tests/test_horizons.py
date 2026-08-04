import pandas as pd
import pytest

from tsla_market_impact.horizons import evaluate_ofi_horizons


def _horizon_bins() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2019-01-02", periods=40, freq="B")
    for date_index, date in enumerate(dates):
        test_date = date_index >= 32
        for kind, value in [
            ("price_spell", 0),
            ("quote_updates", 5),
            ("clock_us", 10),
        ]:
            for ofi_bin, center in [(0, -0.8), (1, 0.8)]:
                if kind == "price_spell":
                    up_moves = 50
                elif kind == "clock_us":
                    up_moves = 40 if center < 0 else 60
                elif test_date:
                    up_moves = 80 if center < 0 else 20
                else:
                    up_moves = 20 if center < 0 else 80
                rows.append(
                    {
                        "date": str(date.date()),
                        "sample": "best_quote_updates",
                        "spread_bucket": "all_spreads",
                        "horizon_kind": kind,
                        "horizon_value": value,
                        "queue_bin": 1,
                        "ofi_bin": ofi_bin,
                        "queue_center": 0.0,
                        "ofi_center": center,
                        "observations": 100,
                        "up_moves": up_moves,
                        "down_moves": 100 - up_moves,
                    }
                )
    return pd.DataFrame(rows)


def test_horizon_selection_never_uses_final_dates() -> None:
    bins = _horizon_bins()
    dates = sorted(bins["date"].unique())
    metrics, selection, result = evaluate_ofi_horizons(
        bins,
        bootstrap_replicates=200,
        random_state=7,
        development_end_date=dates[23],
        selection_start_date=dates[24],
        selection_end_date=dates[31],
        test_start_date=dates[32],
    )

    combined = selection.loc[selection["model"].eq("queue_and_ofi")].iloc[0]
    assert result["protocol"]["development_dates"] == 24
    assert result["protocol"]["selection_dates"] == 8
    assert result["protocol"]["test_dates"] == 8
    assert combined["selected_fixed_horizon_kind"] == "quote_updates"
    assert combined["selected_fixed_horizon_value"] == 5
    assert combined["versus_price_spell_relative_brier_reduction"] < 0

    chosen = metrics.loc[
        metrics["model"].eq("queue_and_ofi") & metrics["selected_fixed_on_train_dates"]
    ].iloc[0]
    assert chosen["selection_relative_brier_reduction"] > 0
    assert chosen["test_relative_brier_reduction"] < 0
    rejected = metrics.loc[
        metrics["model"].eq("queue_and_ofi") & metrics["horizon_kind"].eq("clock_us")
    ].iloc[0]
    assert pd.isna(rejected["test_brier_score"])


def test_horizon_selection_rejects_changed_label_totals() -> None:
    bins = _horizon_bins()
    candidate = bins["horizon_kind"].eq("quote_updates")
    row = bins.index[candidate][0]
    bins.loc[row, "up_moves"] += 1
    bins.loc[row, "observations"] += 1

    with pytest.raises(ValueError, match="preserve date-level labels and counts"):
        evaluate_ofi_horizons(bins, bootstrap_replicates=20)
