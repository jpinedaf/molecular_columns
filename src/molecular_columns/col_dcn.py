import numpy as np
import astropy.units as u
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files
from astropy.constants import c, k_B, h

from .common_functions import J_nu

# transition properties obtained from splatalogue:
# dcn.dat (downloaded 2023 nov 7)
file_mol = files("molecular_columns").joinpath("dcn.dat")
gu_list = np.loadtxt(file_mol, usecols=4)
E_u_list = np.loadtxt(file_mol, usecols=3) * u.K
full_index = np.arange(np.size(E_u_list))
freq_list = np.loadtxt(file_mol, usecols=0) * u.GHz
Aij_list = 10.0 ** (np.loadtxt(file_mol, usecols=2)) / u.s


@u.quantity_input
def Q_DCN_i(index: int, Tex: u.K = 5 * u.K) -> float:
    """
    The function returns the individual elements of the partition function:
    the occupancy of each level dependent on degeneracy and energy level
    for a given excitation temperature.

    Parameters
    ----------
    index : int
        The index of the energy level.
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    float
        The occupancy of the level.
    """
    return gu_list[index] * np.exp(-E_u_list[index] / Tex)


@u.quantity_input
def Q_DCN(Tex: u.K = 5 * u.K) -> float:
    """
    It returns the partition function for DCN with an excitation
    temperature.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_DCN_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_DCN_i(full_index, Tex=Tex))
    else:
        Q_DCN_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_DCN_all[i] = np.sum(Q_DCN_i(full_index, Tex=Tex[i]))
        return Q_DCN_all


@u.quantity_input
def DCN_thin(
    J_up: int = 1,
    Tex: u.K = 5 * u.K,
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,
    T_bg: u.K = 2.73 * u.K,
) -> u.cm**-2:
    """
    Total column density determination from the DCN J_up -> J_up-1 transition.
    The A_ul, frequency and Einstein coefficient are obtained from CDMS.

    Parameters
    ----------
    J_up : int
        The upper level of the transition.
    Tex : u.K
        The excitation temperature.
    TdV : u.K * u.km / u.s
        The integrated intensity of the transition.
    T_bg : u.K
        The background temperature.

    Returns
    -------
    Ncol : u.cm**-2
        The column density.
    """
    if J_up < np.size(Aij_list):
        freq = freq_list[J_up - 1]
        A_ul = Aij_list[J_up - 1]
    else:
        print("J_up is not available")
        return np.nan * u.cm**-2
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)

    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_DCN(Tex=Tex)
        / A_ul
        / Q_DCN_i(J_up, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
        / (Jex - Jbg)
    )
    return Ncol
