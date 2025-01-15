import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, h

from importlib.resources import files

from .common_functions import J_nu

# transition properties obtained from splatalogue:
# dcn.dat (downloaded 2023 nov 7)
file_mol = files('molecular_columns').joinpath('dcn.dat')
gu_list = np.loadtxt(file_mol, usecols=4)

E_u_list = np.loadtxt(file_mol, usecols=3) * u.K

full_index = np.arange(np.size(E_u_list))

freq_list = np.loadtxt(file_mol, usecols=0) * u.GHz

Aij_list = 10.0**(np.loadtxt(file_mol, usecols=2)) / u.s

T_bg = 2.73 * u.K


def Q_DCN_i(index, Tex=5*u.K):
    """
    The function returns the individual elements of the partition function:
    the occupancy of each level dependent on degeneracy and energy level
    for a given excitation temperature.
    """
    return gu_list[index] * np.exp(-E_u_list[index] / Tex)


def Q_DCN(Tex=5*u.K):
    """
    It returns the particion function for DCN with an excitation
    temperature.
    """
    if Tex.size == 1:
        return np.sum(Q_DCN_i(full_index, Tex=Tex))
    else:
        Q_DCN_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_DCN_all[i] = np.sum(Q_DCN_i(full_index, Tex=Tex[i]))
        return Q_DCN_all


def DCN_thin(J_up=1, Tex=5*u.K, TdV=1.0*u.K*u.km/u.s):
    """
    Total column density determination from the DCN J_up -> J_up-1 transition.
    The A_ul, frequency and Einstein coefficient are obtained from CDMS.
    """
    if J_up < np.size(Aij_list):
        freq = freq_list[J_up - 1]
        A_ul = Aij_list[J_up - 1]
    else:
        print('J_up is not available')
        return np.nan
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)

    Ncol = (8*np.pi*freq**3/c**3) * Q_DCN(Tex=Tex) \
         / A_ul / Q_DCN_i(J_up, Tex=Tex) \
         / (np.exp(h*freq/k_B/Tex) - 1) * TdV / (Jex - Jbg)
    return Ncol.to(u.cm**-2)
