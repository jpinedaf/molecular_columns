import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, h

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from .common_functions import J_nu

# transition properties obtained from splatalogue:
# c18o.dat (downloaded 2023 dec 19)
file_mol = files('molecular_columns').joinpath('c18o.dat')
# eml = files('email.tests.data').joinpath('message.eml').read_text()

gu_list = np.loadtxt(file_mol, usecols=4)
E_u_list = np.loadtxt(file_mol, usecols=3) * u.K
full_index = np.arange(np.size(E_u_list))
freq_list = np.loadtxt(file_mol, usecols=0) * 1e-3 * u.GHz
Aij_list = 10.0**(np.loadtxt(file_mol, usecols=2)) / u.s


@u.quantity_input
def Q_C18O_i(index: int, Tex: u.K = 5*u.K) -> float:
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
def Q_C18O(Tex: u.K = 5*u.K) -> float:
    """
    It returns the partition function for C18O with an excitation
    temperature.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_C18O_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_C18O_i(full_index, Tex=Tex))
    else:
        Q_C18O_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_C18O_all[i] = np.sum(Q_C18O_i(full_index, Tex=Tex[i]))
        return Q_C18O_all


@u.quantity_input
def C18O_thin(J_up: int = 1, Tex: u.K = 5*u.K, TdV: u.K*u.km/u.s = 1.0*u.K*u.km/u.s, T_bg: u.K = 2.73*u.K) -> u.cm**-2:
    """
    Total column density determination from the C18O J_up -> J_up-1 transition.
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
        print('J_up is not available')
        return np.nan * u.cm**-2
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)

    Ncol = (8*np.pi*freq**3/c**3) * Q_C18O(Tex=Tex) \
        / A_ul / Q_C18O_i(J_up, Tex=Tex) \
        / (np.exp(h*freq/k_B/Tex) - 1) * TdV / (Jex - Jbg)
    return Ncol


@u.quantity_input
def Ncol_C18O_3_2_Curtis2010(TdV: u.K*u.km/u.s, Tex: u.K = 10*u.K) -> u.cm**-2:
    """
    Column density calculation using expression from 
    E. Curtis et al. (2010) doi:10.1111/j.1365-2966.2009.15658.x
    It takes the integrated intensity and returns the C18O column density
    in units of cm^-2

    Parameters
    ----------
    TdV : u.K * u.km / u.s
        The integrated intensity of the transition.
    Tex : u.K
        The excitation temperature.

    Returns
    -------
    Ncol : u.cm**-2
        The column density.
    """
    return 5e12 * (Tex/u.K) * np.exp(31.6*u.K/Tex) * (TdV/(u.K*u.km/u.s)) * u.cm**-2
