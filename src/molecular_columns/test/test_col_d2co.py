import astropy.units as u
from astropy.constants import h, k_B  # type: ignore
import numpy as np
import pytest
import molecular_columns.col_d2co as col_d2co

try:
    from astropy.units.errors import UnitsError
except ImportError:  # pragma: no cover
    from astropy.units.core import UnitsError


def test_col_d2co_parse_qns():
    upper, lower = col_d2co.parse_qns("10(6,4) - 10(6,5)")
    assert upper == (10, 6, 4)
    assert lower == (10, 6, 5)
    upper, lower = col_d2co.parse_qns("4(4,0) - 4(4,1)")
    assert upper == (4, 4, 0)
    assert lower == (4, 4, 1)


def test_col_d2co_levels():
    # 0(0,0) only appears as a lower state; it is the ortho ground state
    i = col_d2co.o_level_dict["J_Ka_Kc"].index((0, 0, 0))
    assert col_d2co.E_u_o_list[i] == 0 * u.K
    assert col_d2co.gu_o_list[i] == 2  # g_I = 2 for ortho, (2J+1) = 1
    # 3(0,3): g = 2 * (2*3 + 1)
    i = col_d2co.o_level_dict["J_Ka_Kc"].index((3, 0, 3))
    assert col_d2co.gu_o_list[i] == 14
    assert np.isclose(col_d2co.E_u_o_list[i].value, 16.77682)
    # para levels have g_I = 1
    i = col_d2co.p_level_dict["J_Ka_Kc"].index((1, 1, 1))
    assert col_d2co.gu_p_list[i] == 3


def test_col_d2co_Q_p_D2CO():
    value = col_d2co.Q_p_D2CO(Tex=5 * u.K)  # type: ignore
    assert value > 0
    value = col_d2co.Q_p_D2CO(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    assert np.all(value > 0)


def test_col_d2co_Q_o_D2CO():
    value = col_d2co.Q_o_D2CO(Tex=5 * u.K)  # type: ignore
    assert value > 0
    value = col_d2co.Q_o_D2CO(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    assert np.all(value > 0)


def test_col_d2co_Q_rigid_rotor():
    # Q_ortho + Q_para against the rigid-rotor approximation, with
    # spin weights ortho:para = 2:1 (CDMS convention)
    A, B, C = 141.653, 32.283, 26.215  # GHz
    Tex = 50 * u.K
    kT_h = (k_B * Tex / h).to(u.GHz).value
    Q_rr = 1.5 * np.sqrt(np.pi / (A * B * C)) * kT_h**1.5
    Q_tot = col_d2co.Q_o_D2CO(Tex=Tex) + col_d2co.Q_p_D2CO(Tex=Tex)  # type: ignore
    assert float(Q_tot) == pytest.approx(Q_rr, rel=0.03)


def test_col_d2co_invalid_units():
    with pytest.raises(UnitsError):
        col_d2co.Q_p_D2CO(Tex=5 * u.m)  # type: ignore
    with pytest.raises(UnitsError):
        col_d2co.Q_o_D2CO(Tex=5 * u.m)  # type: ignore
    with pytest.raises(UnitsError):
        col_d2co.o_D2CO_thin(freq=231.410234 * u.K)  # type: ignore
    with pytest.raises(UnitsError):
        col_d2co.o_D2CO_thin(Tex=5 * u.m)  # type: ignore
    with pytest.raises(UnitsError):
        col_d2co.o_D2CO_thin(TdV=1.0 * u.K)  # type: ignore
    with pytest.raises(UnitsError):
        col_d2co.p_D2CO_thin(freq=221.1916613 * u.K)  # type: ignore


def test_col_d2co_unknown_frequency():
    with pytest.raises(ValueError):
        col_d2co.o_D2CO_thin(freq=100.0 * u.GHz)  # type: ignore
    # ortho line is not in the para list
    with pytest.raises(ValueError):
        col_d2co.p_D2CO_thin(freq=231.410234 * u.GHz)  # type: ignore


def test_col_d2co_defaults():
    assert np.isfinite(col_d2co.p_D2CO_thin().value)  # type: ignore
    assert np.isfinite(col_d2co.o_D2CO_thin().value)  # type: ignore


def test_col_d2co_freq_units():
    # same line given in MHz
    N_GHz = col_d2co.o_D2CO_thin(freq=231.410234 * u.GHz)  # type: ignore
    N_MHz = col_d2co.o_D2CO_thin(freq=231410.234 * u.MHz)  # type: ignore
    assert N_MHz == N_GHz


def test_col_d2co_o_D2CO_thin():
    # ortho-D2CO 4(0,4)-3(0,3)
    result = col_d2co.o_D2CO_thin(
        freq=231.410234 * u.GHz, Tex=20 * u.K, TdV=1.0 * u.K * u.km / u.s, T_bg=2.73 * u.K  # type: ignore
    )
    assert result.unit == u.cm**-2
    assert result.value == pytest.approx(3.02996e12, rel=1e-4)


def test_col_d2co_p_D2CO_thin():
    # para-D2CO 4(1,4)-3(1,3)
    result = col_d2co.p_D2CO_thin(
        freq=221.1916613 * u.GHz, Tex=20 * u.K, TdV=1.0 * u.K * u.km / u.s, T_bg=2.73 * u.K  # type: ignore
    )
    assert result.unit == u.cm**-2
    assert result.value == pytest.approx(4.13247e12, rel=1e-4)


def test_col_d2co_Tex_array():
    Tex = [5.0, 20.0] * u.K
    result = col_d2co.o_D2CO_thin(Tex=Tex)  # type: ignore
    assert result.shape == (2,)
    assert result[1] == col_d2co.o_D2CO_thin(Tex=20 * u.K)  # type: ignore
