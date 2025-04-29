import numpy as np
import pytest
import astropy.units as u
try:
    from astropy.units.errors import UnitsError
except ImportError:
    from astropy.units.core import UnitsError
import molecular_columns.col_h13cop as col_h13cop


def test_col_h13cop_Q_H13COp():
    value = col_h13cop.Q_H13COp(Tex=5 * u.K)
    np.testing.assert_almost_equal(value, 2.7653, decimal=4)

    value = col_h13cop.Q_H13COp(Tex=[5., 5., 5.] * u.K)
    np.testing.assert_almost_equal(value, [2.7653, 2.7653, 2.7653], decimal=4)


def test_col_h13cop_invalid_units():
    with pytest.raises(UnitsError):
        col_h13cop.Q_H13COp(Tex=5 * u.m)


def test_col_h13cop_thin():
    result = col_h13cop.H13COp_thin(
        J_up=30, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s)
    assert np.isnan(result.value)
    result = col_h13cop.H13COp_thin(
        J_up=1, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s)
    assert pytest.approx(result.to(u.cm**-2).value, rel=0.001) == 1.26008e+12


def test_col_h13cop_thick():
    result = col_h13cop.H13COp_thick(
        J_up=1, Tex=5 * u.K, sigma_v=0.2 * u.km / u.s, tau=2.0)
    assert pytest.approx(result.to(u.cm**-2).value, rel=0.001) == 1.2634e+12
