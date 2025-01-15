# from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, h
# from astropy.units.core import UnitConversionError


@u.quantity_input
def J_nu(Tex: u.Quantity[u.K] = 5 * u.K, freq: u.Quantity[u.GHz] = 100 * u.GHz) -> u.K:
    # def J_nu(Tex: u.K, freq: u.GHz):  # -> u.K:
    """
    Calculate the Planck function at a given frequency and excitation temperature.
    Parameters
    ----------
    Tex : Quantity
        The excitation temperature of the line.
    freq : Quantity
        The frequency of the line.
    Returns
    -------
    J : Quantity
        The Planck function at the given frequency and excitation temperature.
    """
    return (h*freq/k_B/(np.exp(h*freq/k_B/Tex) - 1.0))


def c_tau(tau: float) -> float:
    """
    Calculate the correction factor for the optical depth.
    Parameters
    ----------
    tau : float
        The optical depth of the line.
    Returns
    -------
    c : float
        The correction factor for the optical depth.    
    """
    if tau <= 0 or np.isnan(tau):
        return np.nan
    return tau / (1 - np.exp(-tau))


@u.quantity_input
def tau_nu(Tex=5*u.K, Tbg=2.73*u.K, freq=100 * u.GHz, Tp=1 * u.K) -> float:
    """
    Calculate the optical depth of a line given the peak temperature of the line.
    Parameters
    ----------
    Tex : u.K
        The excitation temperature of the line.
    Tbg : u.K
        The background temperature.
    freq : u.GHz
        The frequency of the line.
    Tp : u.K
        The peak temperature of the line.
    Returns
    -------
    tau : float
        The optical depth of the line.
    """
    if (Tex.value == 0) | np.isnan(Tex):
        return np.nan
    if Tex < Tbg:
        raise ValueError("Tex must be larger than Tbg")
    return -np.log(1 - Tp / (J_nu(Tex=Tex, freq=freq) - J_nu(Tex=Tbg, freq=freq)))
