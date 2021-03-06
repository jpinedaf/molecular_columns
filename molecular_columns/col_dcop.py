import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, h

from .common_functions import J_nu, c_tau

# g_u and E_u values obtained from LAMDA database
# https://home.strw.leidenuniv.nl/~moldata/datafiles/dco+@xpol.dat
gu_list = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 
                    17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 
                    31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0,
                    45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 
                    59.0, 61.0])
E_u_list = (np.array([0.0000, 2.4030, 7.2089, 14.4176, 24.0291, 
                      36.0430, 50.4593, 67.2776, 86.4976, 
                      108.1190, 132.1414, 158.5642, 187.3871, 
                      218.6094, 252.2307, 288.2502, 326.6673, 
                      367.4813, 410.6915, 456.2969, 504.2968,
                      554.6903, 607.4764, 662.6541, 720.2224, 
                      780.1803, 842.5266, 907.2602, 974.3798, 
                      1043.8842, 1115.7720]) 
            * (h*c/k_B) / u.cm).to(u.K)
full_index = np.arange(np.size(E_u_list))

freq_list = np.array([72.0393540, 144.0773190, 216.1126045, 
                      288.1439110, 360.1698810, 432.1890330, 
                      504.2002000, 576.2019724, 648.1930137, 
                      720.1719825, 792.1375375, 864.0883373, 
                      936.0230404, 1007.9403056, 1079.8387915, 
                      1151.7171566, 1223.5740595, 1295.4081590,
                      1367.2181135, 1439.0025818, 1510.7602224,
                      1582.4896939, 1654.1896551, 1725.8587644,
                      1797.4956805, 1869.0990620, 1940.6675676,
                      2012.1998558, 2083.6945853, 2155.1504147]) * u.GHz

Aij_list = np.array([2.2247e-05, 2.1358e-04, 7.7217e-04, 
                     1.8976e-03, 3.7892e-03, 6.6460e-03, 
                     1.0667e-02, 1.6045e-02, 2.2986e-02, 
                     3.1675e-02, 4.2311e-02, 5.5091e-02, 
                     7.0203e-02, 8.7829e-02, 1.0816e-01, 
                     1.3141e-01, 1.5772e-01, 1.8731e-01, 
                     2.2036e-01, 2.5705e-01, 2.9754e-01, 
                     3.4206e-01, 3.9074e-01, 4.4372e-01,
                     5.0127e-01, 5.6348e-01, 6.3071e-01,
                     7.0278e-01, 7.8028e-01, 8.6303e-01]) / u.s

T_bg = 2.73 * u.K

# def J_nu(Tex=5*u.K, freq=100 * u.GHz):
#     return (h*freq/k_B/(np.exp(h*freq/k_B/Tex) - 1.0)).to(u.K)

def Q_DCOp_i(index, Tex=5*u.K):
    """
    The function returns the individual elements of the partition function:
    the occupancy of each level dependent on degeneracy and energy level 
    for a given excitation temperature. 
    """
    return gu_list[index] * np.exp(-E_u_list[index] / Tex)


def Q_DCOp(Tex=5*u.K):
    """
    It returns the particion function for DCO^+ with an excitation 
    temperature.
    It uses the first 30-energy levels.
    """
    if Tex.size == 1:
        return np.sum(Q_DCOp_i(full_index, Tex=Tex))
    else:
        Q_DCOp_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_DCOp_all[i] = np.sum(Q_DCOp_i(full_index, Tex=Tex[i]))
        return Q_DCOp_all


def DCOp_thin(J_up=1, Tex=5*u.K, TdV=1.0*u.K*u.km/u.s):
    """
    Total column density determination from the DCO+ J_up -> J_up-1 transition.
    The A_ul, frequency and Einstein coefficient are obtained from LAMBDA database.
    """
    if J_up < np.size(Aij_list):
        freq = freq_list[J_up - 1] # 110.153594*u.GHz
        A_ul = Aij_list[J_up - 1] # 0.165e-4/u.s
    else:
        print('J_up is not available')
        return np.nan
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)
    # J_up = 2
    Ncol = (8*np.pi*freq**3/c**3) * Q_DCOp(Tex=Tex) \
         / A_ul / Q_DCOp_i(J_up, Tex=Tex) \
         / (np.exp(h*freq/k_B/Tex) - 1) * TdV / (Jex - Jbg)
    return Ncol.to(u.cm**-2)


# def DCOp_thick(J_up=1, Tex=5*u.K, sigma_v=0.2*u.km/u.s, tau=2.0):
#     """
#     Total column density determination from the HCO+ J_up -> J_up-1 transition.
#     The A_ul, frequency and Einstein coefficient are obtained from LAMBDA database.
#     """
#     TdV = np.sqrt(2*np.pi) * tau * sigma_v# * (Jex - Jbg)
#     return DCOp_thin(J_up=J_up, Tex=Tex, TdV=TdV)

# def c_tau(tau):
#     return tau / (1 - np.exp(-tau))
