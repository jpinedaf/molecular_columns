import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, h

def J_nu(Tex=5*u.K, freq=100 * u.GHz):
    return (h*freq/k_B/(np.exp(h*freq/k_B/Tex) - 1.0)).to(u.K)

def c_tau(tau):
    return tau / (1 - np.exp(-tau))
