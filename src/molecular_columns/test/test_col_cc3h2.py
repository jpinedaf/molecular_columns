import astropy.units as u
import numpy as np
import pytest
import molecular_columns.col_cc3h2 as col_cc3h2

try:
    from astropy.units.errors import UnitsError
except ImportError:  # pragma: no cover
    from astropy.units.core import UnitsError


def test_col_cc3h2_Q_o_C3H2():
    value = col_cc3h2.Q_o_C3H2(Tex=5 * u.K)  # type: ignore
    assert value > 0
    value = col_cc3h2.Q_o_C3H2(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    assert np.all(value > 0)


def test_col_cc3h2_Q_p_C3H2():
    value = col_cc3h2.Q_p_C3H2(Tex=5 * u.K)  # type: ignore
    assert value > 0
    value = col_cc3h2.Q_p_C3H2(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    assert np.all(value > 0)


def test_col_cc3h2_invalid_units():
    with pytest.raises(UnitsError):
        col_cc3h2.Q_o_C3H2(Tex=5 * u.m)  # type: ignore
    with pytest.raises(UnitsError):
        col_cc3h2.Q_p_C3H2(Tex=5 * u.m)  # type: ignore
