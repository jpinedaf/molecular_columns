import csv
import numpy as np
from numpy.typing import NDArray
import astropy.units as u
from astropy.constants import c, k_B, h  # type: ignore
from .common_functions import J_nu
from importlib.resources import files


def parse_qns(qn: str):
    """
    Parse a CDMS asymmetric-top transition label into (J, Ka, Kc) tuples.
    e.g. '10(6,4) - 10(6,5)' -> ((10,6,4),(10,6,5))
    """
    upper, lower = qn.split("-")

    def parse(state):
        state = state.strip()
        J = int(state.split("(")[0])
        Ka, Kc = state.split("(")[1].replace(")", "").split(",")
        return (J, int(Ka), int(Kc))

    return parse(upper), parse(lower)


# g_u, E_u, and A_ul values obtained from the CDMS line list
def extract_from_csv(file_name):
    # Read the content of the file
    file_mol = files("molecular_columns").joinpath(file_name)
    with open(str(file_mol), "r", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]  # type: ignore
        rows = list(reader)

    # Make an empty dictionary for J_Ka_Kc, E_u (K), and g_u values
    level_dict = {"J_Ka_Kc": [], "E_u": [], "g_u": []}
    # Make an empty dictionary for frequency (GHz), A_ul, and E_u (K) values
    trans_dict = {"freq": [], "A_ul": [], "E_u": [], "upper_level_no": []}

    # The line list only has transitions, so the levels are collected from
    # both ends of each one. The upper state takes E_U and the upper state
    # degeneracy; the lower state takes E_L and g_l = g_I (2 J_l + 1), where
    # the nuclear spin weight g_I = g_u / (2 J_u + 1) is the same for every
    # level of one spin species (ortho or para).
    level_no = {}

    def add_level(state, energy, weight):
        if state not in level_no:
            level_dict["J_Ka_Kc"].append(state)
            level_dict["E_u"].append(energy)
            level_dict["g_u"].append(weight)
            level_no[state] = len(level_dict["J_Ka_Kc"])  # 1-based, as in LAMBDA
        return level_no[state]

    for row in rows:
        upper, lower = parse_qns(row["Resolved QNs"])
        g_u = float(row["Upper State Degeneracy"])
        g_I = g_u / (2 * upper[0] + 1)
        upper_level_no = add_level(upper, float(row["E_U (K)"]), g_u)
        add_level(lower, float(row["E_L (K)"]), g_I * (2 * lower[0] + 1))

        trans_dict["upper_level_no"].append(upper_level_no)
        trans_dict["A_ul"].append(10 ** float(row["log10_aij"]))
        trans_dict["freq"].append(float(row["Frequency (MHz)"]) / 1e3)
        trans_dict["E_u"].append(float(row["E_U (K)"]))

    return level_dict, trans_dict


p_level_dict, p_trans_dict = extract_from_csv("p-d2co.csv")
o_level_dict, o_trans_dict = extract_from_csv("o-d2co.csv")
gu_p_list = np.array(p_level_dict["g_u"])
E_u_p_list = np.array(p_level_dict["E_u"]) * u.K
gu_o_list = np.array(o_level_dict["g_u"])
E_u_o_list = np.array(o_level_dict["E_u"]) * u.K

p_full_index = np.arange(np.size(E_u_p_list))
o_full_index = np.arange(np.size(E_u_o_list))


def _find_transition(trans_dict, freq):
    """Index of the line-list transition closest to freq, or raise if none matches."""
    freq_GHz = freq.to(u.GHz).value
    freq_list = np.array(trans_dict["freq"])
    trans_index = np.argmin(np.abs(freq_list - freq_GHz))
    if not np.isclose(freq_list[trans_index], freq_GHz):
        raise ValueError(f"No D2CO transition found at {freq}")
    return trans_index


@u.quantity_input
def Q_p_D2CO_i(
    index: int | NDArray[np.int_],
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
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
def Q_p_D2CO(
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
    """
    It returns the partition function for para-D2CO with an excitation
    temperature.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_p_D2CO_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_p_D2CO_i(p_full_index, Tex=Tex))
    else:
        Q_p_D2CO_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_p_D2CO_all[i] = np.sum(Q_p_D2CO_i(p_full_index, Tex=Tex[i]))
        return Q_p_D2CO_all


@u.quantity_input
def Q_o_D2CO_i(
    index: int | NDArray[np.int_],
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
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
def Q_o_D2CO(
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
    """
    It returns the partition function for ortho-D2CO with an excitation
    temperature.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_o_D2CO_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_o_D2CO_i(o_full_index, Tex=Tex))
    else:
        Q_o_D2CO_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_o_D2CO_all[i] = np.sum(Q_o_D2CO_i(o_full_index, Tex=Tex[i]))
        return Q_o_D2CO_all


@u.quantity_input
def p_D2CO_thin(
    freq: u.GHz = 221.1916613 * u.GHz,  # type: ignore
    Tex: u.K = 5 * u.K,  # type: ignore
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,  # type: ignore
    T_bg: u.K = 2.73 * u.K,  # type: ignore
) -> u.cm**-2:  # type: ignore
    """
    Column density determination for a para-D2CO transition, by default
    the (4_{14}-3_{13}) line. The frequency and Einstein coefficient are
    obtained from the CDMS line list.

    Parameters
    ----------
    freq : u.GHz
        The frequency of the transition.
    Tex : u.K
        The excitation temperature.
    TdV : u.K*u.km/u.s
        The integrated intensity.
    T_bg : u.K
        The background temperature.

    Returns
    -------
    Ncol : u.cm**-2
        The column density.
    """
    trans_index = _find_transition(p_trans_dict, freq)
    freq = p_trans_dict["freq"][trans_index] * u.GHz  # catalogue frequency
    A_ul = p_trans_dict["A_ul"][trans_index] / u.s  # type: ignore
    upper_level_no = p_trans_dict["upper_level_no"][trans_index]
    upper_level_index = upper_level_no - 1
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)
    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_p_D2CO(Tex=Tex)
        / A_ul
        / Q_p_D2CO_i(upper_level_index, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
        / (Jex - Jbg)
    )
    return Ncol


@u.quantity_input
def o_D2CO_thin(
    freq: u.GHz = 231.410234 * u.GHz,  # type: ignore
    Tex: u.K = 5 * u.K,  # type: ignore
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,  # type: ignore
    T_bg: u.K = 2.73 * u.K,  # type: ignore
) -> u.cm**-2:  # type: ignore
    """
    Column density determination for an ortho-D2CO transition, by default
    the (4_{04}-3_{03}) line. The frequency and Einstein coefficient are
    obtained from the CDMS line list.

    Parameters
    ----------
    freq : u.GHz
        The frequency of the transition.
    Tex : u.K
        The excitation temperature.
    TdV : u.K*u.km/u.s
        The integrated intensity.
    T_bg : u.K
        The background temperature.

    Returns
    -------
    Ncol : u.cm**-2
        The column density.
    """
    trans_index = _find_transition(o_trans_dict, freq)
    freq = o_trans_dict["freq"][trans_index] * u.GHz  # catalogue frequency
    A_ul = o_trans_dict["A_ul"][trans_index] / u.s  # type: ignore
    upper_level_no = o_trans_dict["upper_level_no"][trans_index]
    upper_level_index = upper_level_no - 1
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)
    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_o_D2CO(Tex=Tex)
        / A_ul
        / Q_o_D2CO_i(upper_level_index, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
        / (Jex - Jbg)
    )
    return Ncol
