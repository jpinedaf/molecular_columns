import numpy as np
import pytest
import astropy.units as u
import molecular_columns.col_dcn as col_dcn
try:
    from astropy.units.errors import UnitsError
except ImportError:
    from astropy.units.core import UnitsError


def test_col_dcn_Q_DCN():
    value = col_dcn.Q_DCN(Tex=5 * u.K)
    # Replace with the expected value
    assert pytest.approx(value.value, abs=0.001) == 6.70685842

    value = col_dcn.Q_DCN(Tex=[5., 5., 5.] * u.K)
    np.testing.assert_almost_equal(value, [6.70685842, 6.70685842, 6.70685842])


def test_col_dcn_invalid_units():
    with pytest.raises(UnitsError):
        col_dcn.Q_DCN(Tex=5 * u.m)


def test_col_dcn_thin():
    result = col_dcn.DCN_thin(J_up=63, Tex=5 * u.K,
                              TdV=1000.0 * u.K * u.km / u.s)
    assert np.isnan(result.value)
    result = col_dcn.DCN_thin(J_up=1, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s)
    assert pytest.approx(result.to(u.cm**-2).value,
                         rel=0.001) == 4.56431689e+12
