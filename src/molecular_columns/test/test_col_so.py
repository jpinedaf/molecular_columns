import numpy as np
import pytest
import astropy.units as u
try:
    from astropy.units.errors import UnitsError
except ImportError:
    from astropy.units.core import UnitsError
import molecular_columns.col_so as col_so


def test_col_so_Q_SO():
    value = col_so.Q_SO(Tex=5 * u.K)
    assert pytest.approx(value, abs=0.001) == 14.97061

    value = col_so.Q_SO(Tex=[5., 5., np.nan] * u.K)
    np.testing.assert_almost_equal(
        value, [14.97061, 14.97061, np.nan], decimal=4)


def test_col_so_invalid_units_transition():
    with pytest.raises(UnitsError):
        col_so.Q_SO(Tex=5 * u.m)

    assert np.isnan(col_so.Q_SO(Tex=np.nan * u.K))
    assert np.isnan(col_so.Q_SO(Tex=-1 * u.K))

    result = col_so.SO_thin(N_J_up='0_1', N_J_low='0_0',
                            Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s)
    assert np.isnan(result.value)

    result = col_so.SO_thin_Nu_Rot(
        N_J_up='0_1', N_J_low='0_0', TdV=1.0*u.K*u.km/u.s, give_Eup=False)
    assert np.isnan(result.value)


def test_col_so_thin():
    result = col_so.SO_thin(N_J_up='2_1', N_J_low='1_1',
                            Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s)
    assert pytest.approx(result.to(u.cm**-2).value,
                         rel=0.0001) == 2.77374e+15


def test_SO_thin_Nu_Rot():
    result = col_so.SO_thin_Nu_Rot(
        N_J_up='2_1', N_J_low='1_1', TdV=1.0*u.K*u.km/u.s, give_Eup=False)
    assert pytest.approx(result.to(u.cm**-2).value,
                         rel=0.0001) == 3.78660734e+12

    result, result2 = col_so.SO_thin_Nu_Rot(
        N_J_up='2_1', N_J_low='1_1', TdV=1.0*u.K*u.km/u.s, give_Eup=True)
    np.testing.assert_almost_equal(
        [result.value*1e-12, result2.value], [3.78660734, 15.80798543], decimal=4)
