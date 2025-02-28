import numpy as np
import astropy.units as u
from astropy.constants import k_B, h


@u.quantity_input
def J_nu(Tex: u.K = 5 * u.K, freq: u.GHz = 100 * u.GHz) -> u.K:
    """
    Calculate the Planck function at a given frequency and excitation temperature.
    Parameters
    ----------
    Tex : u.K
        The excitation temperature of the line.
    freq : u.GHz
        The frequency of the line.
    Returns
    -------
    J : u.K
        The Planck function at the given frequency and excitation temperature.
    """
    return h * freq / k_B / (np.exp(h * freq / k_B / Tex) - 1.0)


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
    tau = np.asarray(tau)
    scalar_input = False
    if tau.ndim == 0:
        tau = tau[np.newaxis]  # Makes x 1D
        scalar_input = True
    c = tau / (1 - np.exp(-tau))
    bad = (tau <= 0) | (np.isnan(tau))
    c[bad] = np.nan
    return np.squeeze(c) if scalar_input else c


@u.quantity_input
def tau_nu(Tex=5 * u.K, Tbg=2.73 * u.K, freq=100 * u.GHz, Tp=1 * u.K) -> float:
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
