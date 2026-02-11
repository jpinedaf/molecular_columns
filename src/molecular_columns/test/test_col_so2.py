import astropy.units as u
import numpy as np
import pytest
import molecular_columns.col_so2 as col_so2

try:
    from astropy.units.errors import UnitsError
except ImportError:  # pragma: no cover
    from astropy.units.core import UnitsError


def test_col_so2_Q_SO2():
    value = col_so2.Q_SO2(Tex=5 * u.K)  # type: ignore
    assert value > 0
    value = col_so2.Q_SO2(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    assert np.all(value > 0)


def test_col_so2_invalid_units():
    with pytest.raises(UnitsError):
        col_so2.Q_SO2(Tex=5 * u.m)  # type: ignore
