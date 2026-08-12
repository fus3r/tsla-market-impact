import numpy as np
import pandas as pd

from tsla_market_impact.persistence import (
    dfa1_statistics,
    evaluate_order_sign_persistence,
)
from tsla_market_impact.scaling import fit_loglog_scale, master_curve


def test_loglog_fit_recovers_known_exponent() -> None:
    x = pd.Series(np.geomspace(10, 5_000, 28))
    y = 2.7 * x**0.64

    fit = fit_loglog_scale(x, y)

    assert np.isclose(fit["exponent"], 0.64, atol=1e-10)
    assert np.isclose(fit["prefactor"], 2.7, atol=1e-10)
    assert np.isclose(fit["r_squared"], 1.0)


def test_master_curve_is_odd() -> None:
    x = np.array([0.1, 0.5, 2.0])

    assert np.allclose(master_curve(-x, 0.8, 1.5), -master_curve(x, 0.8, 1.5))


def _reference_dfa1(signs: np.ndarray, scales: tuple[int, ...]) -> tuple[np.ndarray, float]:
    profile = np.cumsum(signs - signs.mean())
    fluctuations = []
    for scale in scales:
        segment_count = len(profile) // scale
        segments = [
            profile[start : start + scale]
            for start in range(0, segment_count * scale, scale)
        ]
        segments.extend(
            profile[start : start + scale]
            for start in range(
                len(profile) - segment_count * scale,
                len(profile),
                scale,
            )
        )
        residual_squares = []
        positions = np.arange(scale)
        for segment in segments:
            slope, intercept = np.polyfit(positions, segment, 1)
            residual_squares.extend(
                np.square(segment - (intercept + slope * positions))
            )
        fluctuations.append(np.sqrt(np.mean(residual_squares)))
    exponent = np.polyfit(np.log(scales), np.log(fluctuations), 1)[0]
    return np.asarray(fluctuations), float(exponent)


def test_dfa1_matches_direct_two_direction_detrending() -> None:
    random = np.random.default_rng(17)
    signs = random.choice([-1, 1], size=73, p=[0.43, 0.57])
    scales = (4, 8, 16, 32)

    expected_fluctuations, expected_exponent = _reference_dfa1(signs, scales)
    statistics = dfa1_statistics(signs, scales)

    assert np.allclose(
        list(statistics["fluctuations"].values()),
        expected_fluctuations,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(statistics["exponent"], expected_exponent, atol=1e-12)


def test_daily_persistence_protocol_keeps_fixed_date_rules() -> None:
    random = np.random.default_rng(23)
    rows = []
    lengths = [96, 96, 96, 96, 96, 63]
    for day, length in enumerate(lengths, start=1):
        innovations = random.choice([-1, 1], size=(length + 1) // 2)
        signs = np.repeat(innovations, 2)[:length]
        rows.extend(
            {
                "date": f"2019-01-{day:02d}",
                "first_event_row": event_row,
                "trade_sign": int(sign),
            }
            for event_row, sign in enumerate(signs)
        )
    transactions = pd.DataFrame(rows)

    daily, result = evaluate_order_sign_persistence(
        transactions,
        scales=(4, 8, 16),
        minimum_transactions=64,
        permutation_replicates=19,
        block_lengths=(1, 2),
        bootstrap_replicates=100,
        random_state=7,
    )

    assert daily["included"].tolist() == [True, True, True, True, True, False]
    assert daily.loc[~daily["included"], "dfa_exponent"].isna().all()
    assert result["sample"]["included_dates"] == 5
    assert result["sample"]["excluded_date_values"] == ["2019-01-06"]
    assert np.isclose(
        result["sample"]["covered_transaction_share"],
        5 * 96 / sum(lengths),
    )
    assert result["protocol"]["scales_transactions"] == [4, 8, 16]
    assert result["protocol"]["permutation_null"]["replicates"] == 19
    assert {
        interval["block_length_dates"]
        for interval in result["date_block_bootstrap"]
    } == {1, 2}
    assert result["permutation_null"]["monte_carlo_p_value"] in {
        value / 20 for value in range(1, 21)
    }
