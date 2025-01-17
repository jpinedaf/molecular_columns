import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, h

from .common_functions import J_nu, c_tau

# g_u and E_u values obtained from LAMDA database
# https://home.strw.leidenuniv.nl/~moldata/datafiles/h13co+@xpol.dat
gu_list = np.array(
    [
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
        17.0,
        19.0,
        21.0,
        23.0,
        25.0,
        27.0,
        29.0,
        31.0,
        33.0,
        35.0,
        37.0,
        39.0,
        41.0,
        43.0,
        45.0,
        47.0,
        49.0,
        51.0,
        53.0,
        55.0,
        57.0,
        59.0,
        61.0,
    ]
)
E_u_list = (
    np.array(
        [
            0.0000,
            2.8938,
            8.6814,
            17.3626,
            28.9372,
            43.4050,
            60.7656,
            81.0188,
            104.1640,
            130.2008,
            159.1286,
            190.9467,
            225.6545,
            263.2512,
            303.7360,
            347.1080,
            393.3663,
            442.5099,
            494.5377,
            549.4486,
            607.2413,
            667.9147,
            731.4674,
            797.8981,
            867.2052,
            939.3873,
            1014.4428,
            1092.3701,
            1173.1675,
            1256.8332,
            1343.3655,
        ]
    )
    * (h * c / k_B)
    / u.cm
).to(u.K)
full_index = np.arange(np.size(E_u_list))

freq_list = (
    np.array(
        [
            86.7542884,
            173.5066953,
            260.2553390,
            346.9983381,
            433.7338110,
            520.4598762,
            607.1746520,
            693.8762570,
            780.5628096,
            867.2324283,
            953.8832314,
            1040.5133375,
            1127.1208650,
            1213.7039324,
            1300.2606580,
            1386.7891604,
            1473.2875580,
            1559.7539693,
            1646.1865126,
            1732.5833065,
            1818.9424694,
            1905.2621198,
            1991.5403760,
            2077.7753566,
            2163.9651800,
            2250.1079647,
            2336.2018290,
            2422.2448915,
            2508.2352706,
            2594.1710848,
        ]
    )
    * u.GHz
)

Aij_list = (
    np.array(
        [
            3.8534e-05,
            3.6987e-04,
            1.3374e-03,
            3.2879e-03,
            6.5667e-03,
            1.1520e-02,
            1.8492e-02,
            2.7831e-02,
            3.9885e-02,
            5.4985e-02,
            7.3483e-02,
            9.5725e-02,
            1.2205e-01,
            1.5282e-01,
            1.8830e-01,
            2.2894e-01,
            2.7496e-01,
            3.2678e-01,
            3.8472e-01,
            4.4914e-01,
            5.2030e-01,
            5.9864e-01,
            6.8427e-01,
            7.7774e-01,
            8.7941e-01,
            9.8946e-01,
            1.1080e00,
            1.2358e00,
            1.3731e00,
            1.5199e00,
        ]
    )
    / u.s
)


@u.quantity_input
def Q_H13COp_i(index: int, Tex: u.K = 5 * u.K) -> float:
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
def Q_H13COp(Tex: u.K = 5 * u.K) -> float:
    """
    It returns the partition function for H^{13}CO^+ with an excitation
    temperature.
    It uses the first 30-energy levels.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_H13COp_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_H13COp_i(full_index, Tex=Tex))
    else:
        Q_H13COp_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_H13COp_all[i] = np.sum(Q_H13COp_i(full_index, Tex=Tex[i]))
        return Q_H13COp_all


@u.quantity_input
def H13COp_thin(
    J_up: int = 1,
    Tex: u.K = 5 * u.K,
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,
    T_bg: u.K = 2.73 * u.K,
) -> u.cm**-2:
    """
    Total column density determination from the HCO+ J_up -> J_up-1 transition.
    The A_ul, frequency and Einstein coefficient are obtained from LAMBDA database.

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
        return np.nan*u.cm**-2
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)
    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_H13COp(Tex=Tex)
        / A_ul
        / Q_H13COp_i(J_up, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
        / (Jex - Jbg)
    )
    return Ncol


@u.quantity_input
def H13COp_thick(
    J_up: int = 1,
    Tex: u.K = 5 * u.K,
    sigma_v: u.km / u.s = 0.2 * u.km / u.s,
    tau: float = 2.0,
    T_bg: u.K = 2.73 * u.K,
) -> u.cm**-2:
    """
    Total column density determination from the HCO+ J_up -> J_up-1 transition.
    The A_ul, frequency and Einstein coefficient are obtained from LAMBDA database.

    Parameters
    ----------
    J_up : int
        The upper level of the transition.
    Tex : u.K
        The excitation temperature.
    sigma_v : u.km / u.s
        The velocity dispersion.
    tau : float
        The optical depth.
    T_bg : u.K
        The background temperature.

    Returns
    -------
    Ncol : u.cm**-2
        The column density.
    """
    TdV = np.sqrt(2 * np.pi) * tau * sigma_v * u.K
    return H13COp_thin(J_up=J_up, Tex=Tex, TdV=TdV, T_bg=T_bg)
