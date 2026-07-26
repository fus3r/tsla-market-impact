import numpy as np
import pandas as pd

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
