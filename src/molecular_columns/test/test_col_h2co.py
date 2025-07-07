import astropy.units as u
import numpy as np
import pytest
import molecular_columns.col_h2co as col_h2co

try:
    from astropy.units.errors import UnitsError
except ImportError:  # pragma: no cover
    from astropy.units.core import UnitsError


def test_col_h2co_Q_p_H2CO():
    value = col_h2co.Q_p_H2CO(Tex=5 * u.K)  # type: ignore
    assert value > 0
    value = col_h2co.Q_p_H2CO(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    assert np.all(value > 0)


def test_col_h2co_Q_o_H2CO():
    value = col_h2co.Q_o_H2CO(Tex=5 * u.K)  # type: ignore
    assert value > 0
    value = col_h2co.Q_o_H2CO(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    assert np.all(value > 0)


def test_col_h2co_invalid_units():
    with pytest.raises(UnitsError):
        col_h2co.Q_p_H2CO(Tex=5 * u.m)  # type: ignore
    with pytest.raises(UnitsError):
        col_h2co.Q_o_H2CO(Tex=5 * u.m)  # type: ignore
