import astropy.units as u

import molecular_columns.common_functions as common_functions
import pytest
try:
    from astropy.units.errors import UnitsError
except ImportError:
    from astropy.units.core import UnitsError
import numpy as np


def test_J_nu() -> None:
    # handles the cases where the units are not correct
    with pytest.raises(TypeError):
        common_functions.J_nu(Tex=5, freq=100)
    with pytest.raises(UnitsError):
        common_functions.J_nu(Tex=5*u.K, freq=100*u.K)
    assert pytest.approx(common_functions.J_nu(
        5*u.K, 100*u.GHz).to(u.K).value) == 2.97848924


def test_c_tau() -> None:
    assert np.isnan(common_functions.c_tau(-1))
    assert np.isnan(common_functions.c_tau(np.nan))
    assert (np.isnan(common_functions.c_tau(
        [np.nan, 1.0])) == [True, False]).all()
    assert pytest.approx(common_functions.c_tau(1.0)) == 1.5819767068693265
    assert pytest.approx(common_functions.c_tau(0.5)) == 1.2707470412683992


def test_tau_nu() -> None:
    with pytest.raises(ValueError):
        common_functions.tau_nu(Tex=2*u.K, Tbg=2.73*u.K,
                                freq=100*u.GHz, Tp=1*u.K)
    assert np.isnan(common_functions.tau_nu(
        Tex=0*u.K, Tbg=2.73*u.K, freq=100*u.GHz, Tp=1*u.K))
    assert pytest.approx(common_functions.tau_nu(
        Tex=10*u.K, Tbg=2.73*u.K, freq=100*u.GHz, Tp=1*u.K)) == 0.15927105888724097
