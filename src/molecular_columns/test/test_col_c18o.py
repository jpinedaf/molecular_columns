import numpy as np
import pytest
import astropy.units as u
try:
    from astropy.units.errors import UnitsError
except ImportError:
    from astropy.units.core import UnitsError
import molecular_columns.col_c18o as col_c18o


def test_col_c18o_Q_C18O():
    value = col_c18o.Q_C18O(Tex=5 * u.K)
    assert pytest.approx(value.value, abs=0.001) == 1.27059314

    value = col_c18o.Q_C18O(Tex=[5., 5., 5.] * u.K)
    np.testing.assert_almost_equal(
        value, [1.27059, 1.27059, 1.27059], decimal=4)


def test_col_c18o_invalid_units():
    with pytest.raises(UnitsError):
        col_c18o.Q_C18O(Tex=5 * u.m)


def test_col_c18o_thin():
    result = col_c18o.C18O_thin(
        J_up=60, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s)
    assert np.isnan(result.value)
    result = col_c18o.C18O_thin(
        J_up=1, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s)
    assert pytest.approx(result.to(u.cm**-2).value, rel=0.0001) == 3.2835e+15


def test_col_c18o_Ncol_C18O_3_2_Curtis2010():
    result = col_c18o.Ncol_C18O_3_2_Curtis2010(
        TdV=1.0 * u.K * u.km / u.s, Tex=10 * u.K)
    assert pytest.approx(result.to(u.cm**-2).value) == 1.1785298e+15

    result = col_c18o.Ncol_C18O_3_2_Curtis2010(
        TdV=1.0 * u.K * u.km / u.s, Tex=31.6 * u.K)
    assert pytest.approx(result.to(u.cm**-2).value) == 4.29488529e+14
