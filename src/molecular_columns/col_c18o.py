import numpy as np
from numpy.typing import NDArray
import astropy.units as u
from astropy.constants import c, k_B, h  # type: ignore

try:
    from importlib.resources import files
except ImportError:  # pragma: no cover
    from importlib_resources import files

from .common_functions import J_nu

# transition properties obtained from splatalogue:
# c18o.dat (downloaded 2023 dec 19)
file_mol = str(files("molecular_columns").joinpath("c18o.dat"))

gu_list = np.loadtxt(file_mol, usecols=4)
E_u_list = np.loadtxt(file_mol, usecols=3) * u.K  # type: ignore
full_index = np.arange(np.size(E_u_list))
freq_list = np.loadtxt(file_mol, usecols=0) * 1e-3 * u.GHz  # type: ignore
Aij_list = 10.0 ** (np.loadtxt(file_mol, usecols=2)) / u.s  # type: ignore


@u.quantity_input
def Q_C18O_i(
    index: int | NDArray[np.int_],
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
    """
    Returns the individual elements of the partition function for C18O:
    the occupancy of each level dependent on degeneracy and energy level
    for a given excitation temperature.

    Parameters
    ----------
    index : int or NDArray[int]
        The index or indices of the energy level(s).
    Tex : astropy.units.Quantity
        The excitation temperature (must have temperature units).

    Returns
    -------
    float or ndarray
        The occupancy of the level(s).
    """
    return gu_list[index] * np.exp(-E_u_list[index] / Tex)


@u.quantity_input
def Q_C18O(
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
    """
    Returns the partition function for C18O with an excitation temperature.
    Uses all available energy levels.

    Parameters
    ----------
    Tex : astropy.units.Quantity
        The excitation temperature(s) (must have temperature units).

    Returns
    -------
    float or ndarray
        The partition function value(s).
    """
    if Tex.size == 1:
        return np.sum(Q_C18O_i(full_index, Tex=Tex))
    else:
        Q_C18O_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_C18O_all[i] = np.sum(Q_C18O_i(full_index, Tex=Tex[i]))
        return Q_C18O_all


@u.quantity_input
def C18O_thin(
    J_up: int = 1,
    Tex: u.K = 5 * u.K,  # type: ignore
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,  # type: ignore
    T_bg: u.K = 2.73 * u.K,  # type: ignore
) -> u.cm**-2:  # type: ignore
    """
    Calculates the total column density from the C18O J_up -> J_up-1 transition.
    The A_ul, frequency and Einstein coefficient are obtained from CDMS.

    Parameters
    ----------
    J_up : int
        The upper level of the transition (1-based index).
    Tex : astropy.units.Quantity
        The excitation temperature (must have temperature units).
    TdV : astropy.units.Quantity
        The integrated intensity of the transition (K km/s).
    T_bg : astropy.units.Quantity
        The background temperature (must have temperature units).

    Returns
    -------
    astropy.units.Quantity
        The column density (cm^-2).
    """
    if J_up < np.size(Aij_list):
        freq = freq_list[J_up - 1]
        A_ul = Aij_list[J_up - 1]
    else:
        print("J_up is not available")
        return np.nan * u.cm**-2  # type: ignore
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)

    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_C18O(Tex=Tex)
        / A_ul
        / Q_C18O_i(J_up, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
        / (Jex - Jbg)
    )
    return Ncol


@u.quantity_input
def Ncol_C18O_3_2_Curtis2010(TdV: u.K * u.km / u.s, Tex: u.K = 10 * u.K) -> u.cm**-2:  # type: ignore
    """
    Column density calculation using the expression from
    E. Curtis et al. (2010) doi:10.1111/j.1365-2966.2009.15658.x.
    Takes the integrated intensity and returns the C18O column density
    in units of cm^-2.

    Parameters
    ----------
    TdV : astropy.units.Quantity
        The integrated intensity of the transition (K km/s).
    Tex : astropy.units.Quantity, optional
        The excitation temperature (default is 10 K).

    Returns
    -------
    astropy.units.Quantity
        The column density (cm^-2).
    """
    return (
        5e12
        * (Tex / u.K)  # type: ignore
        * np.exp(31.6 * u.K / Tex)  # type: ignore
        * (TdV / (u.K * u.km / u.s))  # type: ignore
        * u.cm**-2  # type: ignore
    )
