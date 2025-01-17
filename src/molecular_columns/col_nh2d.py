import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, h

# from .common_functions import J_nu
# g_u and E_u values obtained from LAMBDA database
# https://home.strw.leidenuniv.nl/~moldata/datafiles/p-nh2d.dat
gu_p_list = np.array(
    [
        3.0,
        9.0,
        9.0,
        9.0,
        15.0,
        15.0,
        15.0,
        15.0,
        15.0,
        21.0,
        21.0,
        21.0,
        21.0,
        21.0,
        21.0,
        21.0,
        27.0,
        27.0,
        27.0,
        27.0,
        27.0,
        27.0,
        27.0,
        33.0,
        33.0,
        27.0,
        27.0,
        33.0,
        33.0,
        33.0,
    ]
)
E_u_p_list = (
    np.array(
        [
            0.0,
            11.1018,
            14.7761,
            16.4932,
            32.7820,
            35.2555,
            40.4052,
            49.8154,
            50.3348,
            64.2466,
            65.6823,
            75.8736,
            83.0973,
            85.4414,
            104.3779,
            104.4807,
            104.9901,
            105.8248,
            122.2922,
            127.0495,
            133.0003,
            149.5094,
            150.1873,
            154.9357,
            155.5045,
            177.2895,
            177.3063,
            178.7962,
            181.3445,
            192.6501,
        ]
    )
    * (h * c / k_B)
    / u.cm
).to(u.K)
p_full_index = np.arange(np.size(E_u_p_list))

# g_u and E_u values obtained from LAMBDA database
# https://home.strw.leidenuniv.nl/~moldata/datafiles/o-nh2d.dat
gu_o_list = np.array(
    [
        9.0,
        27.0,
        27.0,
        27.0,
        45.0,
        45.0,
        45.0,
        45.0,
        45.0,
        63.0,
        63.0,
        63.0,
        63.0,
        63.0,
        63.0,
        63.0,
        81.0,
        81.0,
        81.0,
        81.0,
        81.0,
        81.0,
        81.0,
        99.0,
        99.0,
        81.0,
        81.0,
        99.0,
        99.0,
        99.0,
    ]
)
E_u_o_list = (
    np.array(
        [
            0.4059,
            11.5063,
            14.3725,
            16.0925,
            33.1852,
            34.8518,
            40.0099,
            50.2077,
            50.7258,
            64.6502,
            65.2776,
            75.4853,
            83.4858,
            85.8243,
            104.0016,
            104.1048,
            105.3963,
            105.4179,
            121.9107,
            127.4339,
            133.3719,
            149.1397,
            149.8199,
            155.0938,
            155.3460,
            177.6453,
            177.6620,
            178.4192,
            181.7254,
            193.0089,
        ]
    )
    * (h * c / k_B)
    / u.cm
).to(u.K)
o_full_index = np.arange(np.size(E_u_o_list))


@u.quantity_input
def Q_p_NH2D_i(index: int, Tex: u.K = 5 * u.K) -> float:
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
    return gu_p_list[index] * np.exp(-E_u_p_list[index] / Tex)


@u.quantity_input
def Q_p_NH2D(Tex: u.K = 5 * u.K) -> float:
    """
    It returns the partition function for para-NH2D with an excitation
    temperature.
    It uses the first 30-energy levels.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_p_NH2D_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_p_NH2D_i(p_full_index, Tex=Tex))
    else:
        Q_p_NH2D_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_p_NH2D_all[i] = np.sum(Q_p_NH2D_i(p_full_index, Tex=Tex[i]))
        return Q_p_NH2D_all


@u.quantity_input
def Q_o_NH2D_i(index: int, Tex: u.K = 5 * u.K) -> float:
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
    return gu_o_list[index] * np.exp(-E_u_o_list[index] / Tex)


@u.quantity_input
def Q_o_NH2D(Tex: u.K = 5 * u.K) -> float:
    """
    It returns the partition function for ortho-NH2D with an excitation
    temperature.
    It uses the first 30-energy levels.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_o_NH2D_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_o_NH2D_i(o_full_index, Tex=Tex))
    else:
        Q_o_NH2D_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_o_NH2D_all[i] = np.sum(Q_o_NH2D_i(o_full_index, Tex=Tex[i]))
        return Q_o_NH2D_all


@u.quantity_input
def p_NH2D_thick(
    Tex: u.K = 5 * u.K, sigma_v: u.km / u.s = 0.2 * u.km / u.s, tau: float = 2.0
) -> u.cm**-2:
    """
    Column density determination for the para-NH2D (1_{11}-1{01}) transition.
    The frequency and Einstein coefficient are obtained from LAMBDA database.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    sigma_v : u.km / u.s
        The velocity dispersion.
    tau : float
        The optical depth.

    Returns
    -------
    Ncol : u.cm**-2
        The column density.
    """
    freq = 110.153594 * u.GHz
    A_ul = 0.165e-4 / u.s
    J_up = 2
    TdV = np.sqrt(2 * np.pi) * tau * sigma_v
    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_p_NH2D(Tex=Tex)
        / A_ul
        / Q_p_NH2D_i(J_up, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
    )
    return Ncol


@u.quantity_input
def o_NH2D_thick(
    Tex: u.K = 5 * u.K, sigma_v: u.km / u.s = 0.2 * u.km / u.s, tau: float = 2.0
) -> u.cm**-2:
    """
    Column density determination for the ortho-NH2D (1_{11}-1{01}) transition.
    The frequency and Einstein coefficient are obtained from LAMBDA database.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    sigma_v : u.km / u.s
        The velocity dispersion.
    tau : float
        The optical depth.

    Returns
    -------
    Ncol : u.cm**-2
        The column density.
    """
    freq = 85.92627 * u.GHz
    A_ul = 0.782e-5 / u.s
    J_up = 2
    TdV = np.sqrt(2 * np.pi) * tau * sigma_v
    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_o_NH2D(Tex=Tex)
        / A_ul
        / Q_o_NH2D_i(J_up, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
    )
    return Ncol  # .to(u.cm**-2)
