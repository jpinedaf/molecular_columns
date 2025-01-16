import astropy.units as u
import molecular_columns.col_dcop as col_dcop
import pytest
from astropy.units.errors import UnitsError
import numpy as np


def test_col_dcop_Q_DCOp():
    value = col_dcop.Q_DCOp(Tex=5 * u.K)
    assert pytest.approx(value, abs=0.001) == 3.2504

    value = col_dcop.Q_DCOp(Tex=[5.0, 5.0, 5.0] * u.K)
    np.testing.assert_almost_equal(value, [3.2504, 3.2504, 3.2504])


def test_col_dcop_invalid_units():
    with pytest.raises(UnitsError):
        col_dcop.Q_DCOp(Tex=5 * u.m)


def test_col_dcop_thin():
    result = col_dcop.DCOp_thin(
        J_up=1, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s)
    assert pytest.approx(result.to(u.cm**-2).value) == 1.61091087e21
