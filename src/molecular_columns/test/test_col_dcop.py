import numpy as np
import pytest
import astropy.units as u

try:
    from astropy.units.errors import UnitsError
except ImportError:  # pragma: no cover
    from astropy.units.core import UnitsError

import molecular_columns.col_dcop as col_dcop


def test_col_dcop_Q_DCOp():
    value = col_dcop.Q_DCOp(Tex=5 * u.K)  # type: ignore
    np.testing.assert_almost_equal(value, 3.2504, decimal=4)

    value = col_dcop.Q_DCOp(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    np.testing.assert_almost_equal(value, [3.2504, 3.2504, 3.2504], decimal=4)


def test_col_dcop_invalid_units():
    with pytest.raises(UnitsError):
        col_dcop.Q_DCOp(Tex=5 * u.m)  # type: ignore


def test_col_dcop_thin():
    result = col_dcop.DCOp_thin(
        J_up=30, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s  # type: ignore
    )
    assert np.isnan(result.value)
    result = col_dcop.DCOp_thin(
        J_up=1, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s  # type: ignore
    )
    assert pytest.approx(result.to(u.cm**-2).value, rel=0.001) == 1.6109e12  # type: ignore
