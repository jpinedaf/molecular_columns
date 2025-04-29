import numpy as np
import pytest
import astropy.units as u
try:
    from astropy.units.errors import UnitsError
except ImportError:
    from astropy.units.core import UnitsError

import molecular_columns.col_nh2d as col_nh2d


def test_col_nh2d_Q_op_NH2D() -> None:
    value = col_nh2d.Q_o_NH2D(Tex=5 * u.K)
    np.testing.assert_almost_equal(value, 9.69344271)
    #
    value = col_nh2d.Q_p_NH2D(Tex=5 * u.K).value
    assert pytest.approx(value, abs=0.001) == 3.5770964
    #
    # test with temperature arrays
    value = col_nh2d.Q_o_NH2D(Tex=[5.0, 5.0, 5.0] * u.K)
    np.testing.assert_almost_equal(value, 9.69344271)
    #
    value = col_nh2d.Q_p_NH2D(Tex=[5.0, 5.0, 5.0] * u.K)
    np.testing.assert_almost_equal(value, 3.5770964)


def test_col_nh2d_invalid_units() -> None:
    with pytest.raises(UnitsError):
        col_nh2d.Q_p_NH2D(Tex=5 * u.m)
    with pytest.raises(UnitsError):
        col_nh2d.Q_o_NH2D(Tex=5 * u.m)


def test_col_nh2d_values() -> None:
    result = col_nh2d.o_NH2D_thick(
        Tex=5 * u.K, sigma_v=0.2 * u.km / u.s, tau=2.0)
    assert pytest.approx(result.to(u.cm**-2).value) == 1.32950178e14

    result = col_nh2d.p_NH2D_thick(
        Tex=5 * u.K, sigma_v=0.2 * u.km / u.s, tau=2.0)
    assert pytest.approx(result.to(u.cm**-2).value) == 1.12581955e14
