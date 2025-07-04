import numpy as np
import requests
import astropy.units as u
from astropy.constants import c, k_B, h
from .common_functions import J_nu
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


# g_u, E_u, and A_ul values obtained from LAMBDA database


def extract_from_lambda(file_name):
    # Read the content of the file
    file_mol = files("molecular_columns").joinpath(file_name)
    f = open(file_mol, 'r')
    # Read the content line by line
    lines = f.readlines()
    f.close()
    
    # Make an empty dictionary for J_Kp_Ko, E_u (cm-1), and g_u values
    level_dict={'E_u':[], 'g_u':[]}
    # Make an empty dictionary for frequency (GHz), A_ul, and E_u (K) values
    trans_dict={'freq':[], 'A_ul':[], 'E_u':[], 'upper_level_no':[]}

    # Extract WEIGHT values
    i = 0
    for line in lines:
        if i == 5:
            n_levels = int(line.split()[0]) # number of energy levels
        if line=="!NUMBER OF COLL PARTNERS":
            break
        parts = line.split()
        if line[0]=="!":
            pass
        else:
            if len(parts) ==6: # LEVEL + ENERGIES(cm^-1) + WEIGHT + J + Kp + Ko
                try:
                    energy = float(parts[1]) 
                    level_dict['E_u'].append(energy) 
                    weight = float(parts[2])  
                    level_dict['g_u'].append(weight)
                except ValueError:
                    pass  # Skip lines that don't contain valid numbers
            if i > 5:
                if i > n_levels+5: #
                    if len(parts) ==6: #TRANS + UP + LOW + EINSTEINA(s^-1) + FREQ(GHz) + E_u(K)
                        try:
                            upper_level_no = int(parts[1])
                            trans_dict['upper_level_no'].append(upper_level_no)
                            einsteinA = float(parts[3])
                            trans_dict['A_ul'].append(einsteinA)
                            freq = float(parts[4])
                            trans_dict['freq'].append(freq)
                            upper_level_energy = float(parts[5])
                            trans_dict['E_u'].append(upper_level_energy)
                            
                        except ValueError:
                            pass
        i+=1
   
    return level_dict, trans_dict

p_level_dict, p_trans_dict = extract_from_lambda("p-c3h2.dat")
o_level_dict, o_trans_dict = extract_from_lambda("o-c3h2.dat")
gu_p_list = np.array(p_level_dict['g_u'])
E_u_p_list = (np.array(p_level_dict['E_u'])* (h * c / k_B) / u.cm).to(u.K)
gu_o_list = np.array(o_level_dict['g_u'])
E_u_o_list = (np.array(o_level_dict['E_u'])* (h * c / k_B) / u.cm).to(u.K)

p_full_index = np.arange(np.size(E_u_p_list))
o_full_index = np.arange(np.size(E_u_o_list))


@u.quantity_input
def Q_p_C3H2_i(index: int, Tex: u.K = 5 * u.K) -> float:
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
def Q_p_C3H2(Tex: u.K = 5 * u.K) -> float:
    """
    It returns the partition function for para-C3H2 with an excitation
    temperature.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_p_C3H2_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_p_C3H2_i(p_full_index, Tex=Tex))
    else:
        Q_p_C3H2_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_p_C3H2_all[i] = np.sum(Q_p_C3H2_i(p_full_index, Tex=Tex[i]))
        return Q_p_C3H2_all


@u.quantity_input
def Q_o_C3H2_i(index: int, Tex: u.K = 5 * u.K) -> float:
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
def Q_o_C3H2(Tex: u.K = 5 * u.K) -> float:
    """
    It returns the partition function for ortho-C3H2 with an excitation
    temperature.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    Q_o_C3H2_all : float
        The partition function.
    """
    if Tex.size == 1:
        return np.sum(Q_o_C3H2_i(o_full_index, Tex=Tex))
    else:
        Q_o_C3H2_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_o_C3H2_all[i] = np.sum(Q_o_C3H2_i(o_full_index, Tex=Tex[i]))
        return Q_o_C3H2_all


@u.quantity_input
def p_C3H2_thin(
    freq: float = 218.222192* u.GHz, Tex: u.K = 5*u.K, TdV: u.K*u.km/u.s = 1.0*u.K*u.km/u.s, T_bg: u.K= 2.73*u.K
) -> u.cm**-2:
    """
    Column density determination for the para-C3H2 (1_{11}-1{01}) transition.
    The frequency and Einstein coefficient are obtained from LAMBDA database.

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
    trans_index = np.where(p_trans_dict['freq'] == freq.value)[0][0]
    A_ul = p_trans_dict['A_ul'][trans_index] / u.s
    upper_level_no = p_trans_dict['upper_level_no'][trans_index] #new
    upper_level_index = upper_level_no-1 #new
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)
    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_p_C3H2(Tex=Tex)
        / A_ul
        / Q_p_C3H2_i(upper_level_index, Tex=Tex) #modified
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV/ (Jex - Jbg)
    )
    return Ncol


@u.quantity_input
def o_C3H2_thin(
    freq: float = 218.222192* u.GHz, Tex: u.K = 5*u.K, TdV: u.K*u.m/u.s = 1.0*u.K*u.m/u.s, T_bg: u.K= 2.73*u.K
) -> u.cm**-2:
    """
    Column density determination for the para-C3H2 (1_{11}-1{01}) transition.
    The frequency and Einstein coefficient are obtained from LAMBDA database.

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
    trans_index = np.where(o_trans_dict['freq'] == freq.value)[0][0]
    A_ul = o_trans_dict['A_ul'][trans_index] / u.s
    upper_level_no = o_trans_dict['upper_level_no'][trans_index] #new
    upper_level_index = upper_level_no-1 #new
    E_up = E_u_o_list[upper_level_index]
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)
    print(A_ul, upper_level_no, upper_level_index)
    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_o_C3H2(Tex=Tex)
        / A_ul
        / Q_o_C3H2_i(upper_level_index, Tex=Tex) #modified
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV/ (Jex - Jbg)
    )
    return Ncol
