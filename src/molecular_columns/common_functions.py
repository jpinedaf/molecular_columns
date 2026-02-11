import numpy as np
import astropy.units as u
from astropy.constants import k_B, h  # type: ignore
from numpy.typing import NDArray


@u.quantity_input
def J_nu(Tex: u.K = 5 * u.K, freq: u.GHz = 100 * u.GHz) -> u.K:  # type: ignore
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


from typing import Sequence


def c_tau(tau: float | Sequence[float]) -> float | NDArray[np.float64]:
    """
    Calculate the correction factor for the optical depth.
    Parameters
    ----------
    tau : float | Sequence[float]
        The optical depth of the line.
    Returns
    -------
    c : float | list[float]
        The correction factor for the optical depth.
    """
    # tau = np.asarray(tau)
    tau_array = np.asarray(tau, dtype=np.float64)  # Ensure tau is a float array

    if tau_array.ndim == 0:
        tau_array = tau_array[np.newaxis]  # Makes x 1D
        scalar_input = True
    else:
        scalar_input = False
    c = tau_array / (1 - np.exp(-tau_array))
    bad = (tau_array <= 0) | (np.isnan(tau_array))
    c[bad] = np.nan
    return float(np.squeeze(c)) if scalar_input else c


@u.quantity_input
def tau_nu(
    Tex=5 * u.K,  # type: ignore
    Tbg=2.73 * u.K,  # type: ignore
    freq=100 * u.GHz,  # type: ignore
    Tp=1 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
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
