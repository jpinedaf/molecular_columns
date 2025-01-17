import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, h

from .common_functions import J_nu

cm2K = ((h*c/k_B) / u.cm).to(u.K)
# g_u and E_u values obtained from LAMDA database
# https://home.strw.leidenuniv.nl/~moldata/datafiles/so@lique.dat
#!LEVEL + ENERGIES(cm^-1) + WEIGHT  + N_J  -->  'N_J'  Eup  gu
SO_levels = {
    '1_0': {'Eup': 0.0000 * cm2K, 'g_u': 1},
    '0_1': {'Eup': 1.0007 * cm2K, 'g_u': 3},
    '1_2': {'Eup': 3.0999 * cm2K, 'g_u': 5},
    '2_3': {'Eup': 6.4122 * cm2K, 'g_u': 7},
    '1_1': {'Eup': 10.5520 * cm2K, 'g_u': 3},
    '2_1': {'Eup': 10.9871 * cm2K, 'g_u': 3},
    '3_4': {'Eup': 11.0213 * cm2K, 'g_u': 9},
    '2_2': {'Eup': 13.4238 * cm2K, 'g_u': 5},
    '3_2': {'Eup': 14.6314 * cm2K, 'g_u': 5},
    '4_5': {'Eup': 16.9790 * cm2K, 'g_u': 11},
    '3_3': {'Eup': 17.7314 * cm2K, 'g_u': 7},
    '4_3': {'Eup': 19.9341 * cm2K, 'g_u': 7},
    '4_4': {'Eup': 23.4748 * cm2K, 'g_u': 9},
    '5_6': {'Eup': 24.3157 * cm2K, 'g_u': 13},
    '5_4': {'Eup': 26.8114 * cm2K, 'g_u': 9},
    '5_5': {'Eup': 30.6538 * cm2K, 'g_u': 11},
    '6_7': {'Eup': 33.0499 * cm2K, 'g_u': 15},
    '6_5': {'Eup': 35.2114 * cm2K, 'g_u': 11},
    '6_6': {'Eup': 39.2682 * cm2K, 'g_u': 13},
    '7_8': {'Eup': 43.1928 * cm2K, 'g_u': 17},
    '7_6': {'Eup': 45.1032 * cm2K, 'g_u': 13},
    '7_7': {'Eup': 49.3181 * cm2K, 'g_u': 15},
    '8_9': {'Eup': 54.7518 * cm2K, 'g_u': 19},
    '8_7': {'Eup': 56.4683 * cm2K, 'g_u': 15},
    '8_8': {'Eup': 60.8030 * cm2K, 'g_u': 17},
    '9_10': {'Eup': 67.7314 * cm2K, 'g_u': 21},
    '9_8': {'Eup': 69.2947 * cm2K, 'g_u': 17},
    '9_9': {'Eup': 73.7229 * cm2K, 'g_u': 19},
    '10_11': {'Eup': 82.1350 * cm2K, 'g_u': 23},
    '10_9': {'Eup': 83.5749 * cm2K, 'g_u': 19},
    '10_10': {'Eup': 88.0775 * cm2K, 'g_u': 21},
    '11_12': {'Eup': 97.9646 * cm2K, 'g_u': 25},
    '11_10': {'Eup': 99.3037 * cm2K, 'g_u': 21},
    '11_11': {'Eup': 03.8665 * cm2K, 'g_u': 23},
    '12_13': {'Eup': 115.2217 * cm2K, 'g_u': 27},
    '12_11': {'Eup': 116.4774 * cm2K, 'g_u': 23},
    '12_12': {'Eup': 121.0896 * cm2K, 'g_u': 25},
    '13_14': {'Eup': 133.9073 * cm2K, 'g_u': 29},
    '13_12': {'Eup': 135.0932 * cm2K, 'g_u': 25},
    '13_13': {'Eup': 139.7465 * cm2K, 'g_u': 27},
    '14_15': {'Eup': 154.0219 * cm2K, 'g_u': 31},
    '14_13': {'Eup': 155.1490 * cm2K, 'g_u': 27},
    '14_14': {'Eup': 159.8369 * cm2K, 'g_u': 29},
    '15_16': {'Eup': 175.5660 * cm2K, 'g_u': 33},
    '15_14': {'Eup': 176.6432 * cm2K, 'g_u': 29},
    '15_15': {'Eup': 181.3602 * cm2K, 'g_u': 31},
    '16_17': {'Eup': 198.5397 * cm2K, 'g_u': 35},
    '16_15': {'Eup': 199.5743 * cm2K, 'g_u': 31},
    '16_16': {'Eup': 204.3163 * cm2K, 'g_u': 33},
    '17_18': {'Eup': 222.9432 * cm2K, 'g_u': 37},
    '17_16': {'Eup': 223.9412 * cm2K, 'g_u': 33},
    '17_17': {'Eup': 228.7045 * cm2K, 'g_u': 35},
    '18_19': {'Eup': 248.7762 * cm2K, 'g_u': 39},
    '18_17': {'Eup': 249.7428 * cm2K, 'g_u': 35},
    '18_18': {'Eup': 254.5245 * cm2K, 'g_u': 37},
    '19_20': {'Eup': 276.0386 * cm2K, 'g_u': 41},
    '19_18': {'Eup': 276.9782 * cm2K, 'g_u': 37},
    '19_19': {'Eup': 281.7758 * cm2K, 'g_u': 39},
    '20_21': {'Eup': 304.7302 * cm2K, 'g_u': 43},
    '20_19': {'Eup': 305.6464 * cm2K, 'g_u': 39},
    '20_20': {'Eup': 310.4578 * cm2K, 'g_u': 41},
    '21_22': {'Eup': 334.8506 * cm2K, 'g_u': 45},
    '21_20': {'Eup': 335.7466 * cm2K, 'g_u': 41},
    '21_21': {'Eup': 340.5700 * cm2K, 'g_u': 43},
    '22_23': {'Eup': 366.3994 * cm2K, 'g_u': 47},
    '22_21': {'Eup': 367.2780 * cm2K, 'g_u': 43},
    '22_22': {'Eup': 372.1119 * cm2K, 'g_u': 45},
    '23_24': {'Eup': 399.3762 * cm2K, 'g_u': 49},
    '23_22': {'Eup': 400.2398 * cm2K, 'g_u': 45},
    '23_23': {'Eup': 405.0827 * cm2K, 'g_u': 47},
    '24_25': {'Eup': 433.7804 * cm2K, 'g_u': 51},
    '24_23': {'Eup': 434.6312 * cm2K, 'g_u': 47},
    '24_24': {'Eup': 439.4820 * cm2K, 'g_u': 49},
    '25_26': {'Eup': 469.6115 * cm2K, 'g_u': 53},
    '25_24': {'Eup': 470.4514 * cm2K, 'g_u': 49},
    '25_25': {'Eup': 475.3091 * cm2K, 'g_u': 51},
    '26_27': {'Eup': 506.8689 * cm2K, 'g_u': 55},
    '26_25': {'Eup': 507.6996 * cm2K, 'g_u': 51},
    '26_26': {'Eup': 512.5632 * cm2K, 'g_u': 53},
    '27_28': {'Eup': 545.5520 * cm2K, 'g_u': 57},
    '27_26': {'Eup': 546.3750 * cm2K, 'g_u': 53},
    '27_27': {'Eup': 551.2437 * cm2K, 'g_u': 55},
    '28_29': {'Eup': 585.6602 * cm2K, 'g_u': 59},
    '28_27': {'Eup': 586.4767 * cm2K, 'g_u': 55},
    '28_28': {'Eup': 591.3499 * cm2K, 'g_u': 57},
    '29_30': {'Eup': 627.1926 * cm2K, 'g_u': 61},
    '29_28': {'Eup': 628.0040 * cm2K, 'g_u': 57},
    '29_29': {'Eup': 632.8809 * cm2K, 'g_u': 59},
    '30_31': {'Eup': 670.1486 * cm2K, 'g_u': 63},
    '30_29': {'Eup': 670.9559 * cm2K, 'g_u': 59},
    '30_30': {'Eup': 675.8360 * cm2K, 'g_u': 61},
}
# full_index = np.arange(np.size(E_u_list))
full_index = [key for key in SO_levels]

line_list = {
    full_index[2-1]+'-'+full_index[1-1]: {'A_ij': 2.361e-07, 'freq': 30.0015800},
    full_index[3-1]+'-'+full_index[2-1]: {'A_ij': 2.646e-06, 'freq': 62.9318000},
    full_index[4-1]+'-'+full_index[3-1]: {'A_ij': 1.125e-05, 'freq': 99.2998700},
    full_index[5-1]+'-'+full_index[1-1]: {'A_ij': 8.468e-08, 'freq': 316.3416930},
    full_index[5-1]+'-'+full_index[2-1]: {'A_ij': 1.403e-05, 'freq': 286.3401520},
    full_index[6-1]+'-'+full_index[1-1]: {'A_ij': 1.423e-05, 'freq': 329.3854770},
    full_index[6-1]+'-'+full_index[2-1]: {'A_ij': 1.203e-07, 'freq': 299.3839573},
    full_index[6-1]+'-'+full_index[3-1]: {'A_ij': 1.417e-06, 'freq': 236.4522934},
    full_index[6-1]+'-'+full_index[5-1]: {'A_ij': 2.911e-08, 'freq': 13.0438070},
    full_index[7-1]+'-'+full_index[4-1]: {'A_ij': 3.165e-05, 'freq': 138.1786000},
    full_index[8-1]+'-'+full_index[2-1]: {'A_ij': 8.575e-08, 'freq': 372.4341073},
    full_index[8-1]+'-'+full_index[3-1]: {'A_ij': 1.419e-05, 'freq': 309.5024440},
    full_index[8-1]+'-'+full_index[4-1]: {'A_ij': 2.843e-08, 'freq': 210.2025571},
    full_index[8-1]+'-'+full_index[5-1]: {'A_ij': 5.250e-06, 'freq': 86.0939500},
    full_index[9-1]+'-'+full_index[2-1]: {'A_ij': 1.583e-05, 'freq': 408.6361383},
    full_index[9-1]+'-'+full_index[3-1]: {'A_ij': 1.390e-07, 'freq': 345.7044744},
    full_index[9-1]+'-'+full_index[4-1]: {'A_ij': 1.007e-06, 'freq': 246.4045881},
    full_index[9-1]+'-'+full_index[6-1]: {'A_ij': 1.080e-05, 'freq': 109.2522200},
    full_index[9-1]+'-'+full_index[8-1]: {'A_ij': 1.942e-07, 'freq': 36.2020220},
    full_index[10-1]+'-'+full_index[7-1]: {'A_ij': 7.021e-05, 'freq': 178.6054030},
    full_index[11-1]+'-'+full_index[3-1]: {'A_ij': 8.999e-08, 'freq': 438.6413448},
    full_index[11-1]+'-'+full_index[4-1]: {'A_ij': 1.455e-05, 'freq': 339.3414590},
    full_index[11-1]+'-'+full_index[7-1]: {'A_ij': 2.751e-08, 'freq': 201.1628047},
    full_index[11-1]+'-'+full_index[8-1]: {'A_ij': 2.250e-05, 'freq': 129.1389230},
    full_index[12-1]+'-'+full_index[3-1]: {'A_ij': 1.595e-05, 'freq': 504.6762856},
    full_index[12-1]+'-'+full_index[4-1]: {'A_ij': 1.629e-07, 'freq': 405.3763993},
    full_index[12-1]+'-'+full_index[7-1]: {'A_ij': 7.117e-07, 'freq': 267.1977455},
    full_index[12-1]+'-'+full_index[9-1]: {'A_ij': 4.233e-05, 'freq': 158.9718112},
    full_index[12-1]+'-'+full_index[11-1]: {'A_ij': 5.509e-07, 'freq': 66.0349400},
    full_index[13-1]+'-'+full_index[4-1]: {'A_ij': 9.537e-08, 'freq': 511.5228619},
    full_index[13-1]+'-'+full_index[7-1]: {'A_ij': 1.508e-05, 'freq': 373.3442081},
    full_index[13-1]+'-'+full_index[10-1]: {'A_ij': 2.632e-08, 'freq': 194.7388438},
    full_index[13-1]+'-'+full_index[11-1]: {'A_ij': 5.833e-05, 'freq': 172.1814034},
    full_index[14-1]+'-'+full_index[10-1]: {'A_ij': 1.335e-04, 'freq': 219.9494420},
    full_index[15-1]+'-'+full_index[4-1]: {'A_ij': 1.507e-05, 'freq': 611.5524120},
    full_index[15-1]+'-'+full_index[7-1]: {'A_ij': 1.903e-07, 'freq': 473.3737582},
    full_index[15-1]+'-'+full_index[10-1]: {'A_ij': 5.326e-07, 'freq': 294.7683939},
    full_index[15-1]+'-'+full_index[12-1]: {'A_ij': 1.010e-04, 'freq': 206.1760050},
    full_index[15-1]+'-'+full_index[13-1]: {'A_ij': 1.083e-06, 'freq': 100.0296400},
    full_index[16-1]+'-'+full_index[7-1]: {'A_ij': 1.015e-07, 'freq': 588.5648577},
    full_index[16-1]+'-'+full_index[10-1]: {'A_ij': 1.575e-05, 'freq': 409.9594934},
    full_index[16-1]+'-'+full_index[13-1]: {'A_ij': 1.193e-04, 'freq': 215.2206530},
    full_index[16-1]+'-'+full_index[14-1]: {'A_ij': 2.524e-08, 'freq': 190.0101048},
    full_index[17-1]+'-'+full_index[14-1]: {'A_ij': 2.282e-04, 'freq': 261.8437210},
    full_index[18-1]+'-'+full_index[7-1]: {'A_ij': 1.384e-05, 'freq': 725.1995176},
    full_index[18-1]+'-'+full_index[10-1]: {'A_ij': 2.197e-07, 'freq': 546.5941534},
    full_index[18-1]+'-'+full_index[14-1]: {'A_ij': 4.200e-07, 'freq': 326.6447648},
    full_index[18-1]+'-'+full_index[15-1]: {'A_ij': 1.925e-04, 'freq': 251.8257700},
    full_index[18-1]+'-'+full_index[16-1]: {'A_ij': 1.749e-06, 'freq': 136.6347990},
    full_index[19-1]+'-'+full_index[10-1]: {'A_ij': 1.081e-07, 'freq': 668.2153193},
    full_index[19-1]+'-'+full_index[14-1]: {'A_ij': 1.651e-05, 'freq': 448.2659307},
    full_index[19-1]+'-'+full_index[16-1]: {'A_ij': 2.120e-04, 'freq': 258.2558259},
    full_index[19-1]+'-'+full_index[17-1]: {'A_ij': 2.433e-0, 'freq': 186.4222261},
    full_index[19-1]+'-'+full_index[18-1]: {'A_ij': 6.473e-09, 'freq': 121.6211660},
    full_index[20-1]+'-'+full_index[17-1]: {'A_ij': 3.609e-04, 'freq': 304.0778440},
    full_index[21-1]+'-'+full_index[10-1]: {'A_ij': 1.259e-05, 'freq': 843.1442060},
    full_index[21-1]+'-'+full_index[14-1]: {'A_ij': 2.505e-07, 'freq': 623.1948174},
    full_index[21-1]+'-'+full_index[17-1]: {'A_ij': 3.450e-07, 'freq': 361.3511128},
    full_index[21-1]+'-'+full_index[18-1]: {'A_ij': 3.229e-04, 'freq': 296.5500640},
    full_index[21-1]+'-'+full_index[19-1]: {'A_ij': 2.514e-06, 'freq': 174.9288600},
    full_index[22-1]+'-'+full_index[14-1]: {'A_ij': 1.152e-07, 'freq': 749.5520491},
    full_index[22-1]+'-'+full_index[17-1]: {'A_ij': 1.735e-05, 'freq': 487.7083445},
    full_index[22-1]+'-'+full_index[19-1]: {'A_ij': 3.429e-04, 'freq': 301.2861240},
    full_index[22-1]+'-'+full_index[20-1]: {'A_ij': 2.359e-08, 'freq': 183.6304862},
    full_index[22-1]+'-'+full_index[21-1]: {'A_ij': 7.465e-09, 'freq': 126.3572317},
    full_index[23-1]+'-'+full_index[20-1]: {'A_ij': 5.382e-04, 'freq': 346.5284810},
    full_index[24-1]+'-'+full_index[14-1]: {'A_ij': 1.143e-05, 'freq': 963.9091039},
    full_index[24-1]+'-'+full_index[17-1]: {'A_ij': 2.823e-07, 'freq': 702.0653993},
    full_index[24-1]+'-'+full_index[19-1]: {'A_ij': 2.521e-08, 'freq': 515.6431731},
    full_index[24-1]+'-'+full_index[20-1]: {'A_ij': 2.921e-07, 'freq': 397.9875410},
    full_index[24-1]+'-'+full_index[21-1]: {'A_ij': 4.985e-04, 'freq': 340.7141550},
    full_index[24-1]+'-'+full_index[22-1]: {'A_ij': 3.351e-06, 'freq': 214.3570390},
    full_index[25-1]+'-'+full_index[17-1]: {'A_ij': 1.226e-07, 'freq': 832.0190575},
    full_index[25-1]+'-'+full_index[20-1]: {'A_ij': 1.823e-05, 'freq': 527.9411992},
    full_index[25-1]+'-'+full_index[22-1]: {'A_ij': 5.186e-04, 'freq': 344.3106120},
    full_index[25-1]+'-'+full_index[23-1]: {'A_ij': 2.297e-08, 'freq': 181.4126712},
    full_index[25-1]+'-'+full_index[24-1]: {'A_ij': 8.274e-09, 'freq': 129.9536582},
    full_index[26-1]+'-'+full_index[23-1]: {'A_ij': 7.665e-04, 'freq': 389.1209320},
    full_index[27-1]+'-'+full_index[17-1]: {'A_ij': 1.042e-05, 'freq': 1086.5926858},
    full_index[27-1]+'-'+full_index[20-1]: {'A_ij': 3.146e-07, 'freq': 782.5148275},
    full_index[27-1]+'-'+full_index[22-1]: {'A_ij': 3.211e-08, 'freq': 598.8843413},
    full_index[27-1]+'-'+full_index[23-1]: {'A_ij': 2.532e-07, 'freq': 435.9862996},
    full_index[27-1]+'-'+full_index[24-1]: {'A_ij': 7.255e-04, 'freq': 384.5272866},
    full_index[27-1]+'-'+full_index[25-1]: {'A_ij': 4.239e-06, 'freq': 254.5736284},
    full_index[28-1]+'-'+full_index[20-1]: {'A_ij': 1.302e-07, 'freq': 915.2699946},
    full_index[28-1]+'-'+full_index[23-1]: {'A_ij': 1.916e-05, 'freq': 568.7414667},
    full_index[28-1]+'-'+full_index[25-1]: {'A_ij': 7.456e-04, 'freq': 387.3287950},
    full_index[28-1]+'-'+full_index[26-1]: {'A_ij': 2.246e-08, 'freq': 179.6204980},
    full_index[28-1]+'-'+full_index[27-1]: {'A_ij': 8.935e-09, 'freq': 132.7551671},
    full_index[29-1]+'-'+full_index[26-1]: {'A_ij': 1.053e-03, 'freq': 431.8081960},
    full_index[30-1]+'-'+full_index[20-1]: {'A_ij': 9.541e-06, 'freq': 1210.6255742},
    full_index[30-1]+'-'+full_index[23-1]: {'A_ij': 3.475e-07, 'freq': 864.0970463},
    full_index[30-1]+'-'+full_index[25-1]: {'A_ij': 3.934e-08, 'freq': 682.6843751},
    full_index[30-1]+'-'+full_index[26-1]: {'A_ij': 2.234e-07, 'freq': 474.9760777},
    full_index[30-1]+'-'+full_index[27-1]: {'A_ij': 1.011e-03, 'freq': 428.1107467},
    full_index[30-1]+'-'+full_index[28-1]: {'A_ij': 5.167e-06, 'freq': 295.3556960},
    full_index[31-1]+'-'+full_index[23-1]: {'A_ij': 1.380e-07, 'freq': 999.0810184},
    full_index[31-1]+'-'+full_index[26-1]: {'A_ij': 2.012e-05, 'freq': 609.9600498},
    full_index[31-1]+'-'+full_index[28-1]: {'A_ij': 1.030e-03, 'freq': 430.3395440},
    full_index[31-1]+'-'+full_index[29-1]: {'A_ij': 2.204e-08, 'freq': 178.1518554},
    full_index[31-1]+'-'+full_index[30-1]: {'A_ij': 9.482e-09, 'freq': 134.9839721},
    full_index[32-1]+'-'+full_index[29-1]: {'A_ij': 1.403e-03, 'freq': 474.5596050},
    full_index[33-1]+'-'+full_index[23-1]: {'A_ij': 8.777e-06, 'freq': 1335.6348296},
    full_index[33-1]+'-'+full_index[26-1]: {'A_ij': 3.806e-07, 'freq': 946.5138609},
    full_index[33-1]+'-'+full_index[28-1]: {'A_ij': 4.679e-08, 'freq': 766.8933629},
    full_index[33-1]+'-'+full_index[29-1]: {'A_ij': 1.999e-07, 'freq': 514.7056665},
    full_index[33-1]+'-'+full_index[30-1]: {'A_ij': 1.360e-03, 'freq': 471.5378180},
    full_index[33-1]+'-'+full_index[31-1]: {'A_ij': 6.125e-06, 'freq': 336.5538112},
    full_index[34-1]+'-'+full_index[26-1]: {'A_ij': 1.460e-07, 'freq': 1083.3022174},
    full_index[34-1]+'-'+full_index[29-1]: {'A_ij': 2.110e-05, 'freq': 651.4940230},
    full_index[34-1]+'-'+full_index[31-1]: {'A_ij': 1.379e-03, 'freq': 473.3421677},
    full_index[34-1]+'-'+full_index[32-1]: {'A_ij': 2.169e-08, 'freq': 176.9344298},
    full_index[34-1]+'-'+full_index[33-1]: {'A_ij': 9.935e-09, 'freq': 136.7883565},
    full_index[35-1]+'-'+full_index[32-1]: {'A_ij': 1.824e-03, 'freq': 517.3545316},
    full_index[36-1]+'-'+full_index[26-1]: {'A_ij': 8.114e-06, 'freq': 1461.3676149},
    full_index[36-1]+'-'+full_index[29-1]: {'A_ij': 4.140e-07, 'freq': 1029.5594205},
    full_index[36-1]+'-'+full_index[31-1]: {'A_ij': 5.444e-08, 'freq': 851.4075651},
    full_index[36-1]+'-'+full_index[32-1]: {'A_ij': 1.809e-07, 'freq': 554.9998273},
    full_index[36-1]+'-'+full_index[33-1]: {'A_ij': 1.780e-03, 'freq': 514.8537539},
    full_index[36-1]+'-'+full_index[34-1]: {'A_ij': 7.104e-06, 'freq': 378.0653974},
    full_index[37-1]+'-'+full_index[29-1]: {'A_ij': 1.540e-07, 'freq': 1167.8298519},
    full_index[37-1]+'-'+full_index[32-1]: {'A_ij': 2.209e-05, 'freq': 693.2702588},
    full_index[37-1]+'-'+full_index[34-1]: {'A_ij': 1.799e-03, 'freq': 516.3358289},
    full_index[37-1]+'-'+full_index[35-1]: {'A_ij': 2.139e-08, 'freq': 175.9157271},
    full_index[37-1]+'-'+full_index[36-1]: {'A_ij': 1.031e-08, 'freq': 138.2704315},
    full_index[38-1]+'-'+full_index[35-1]: {'A_ij': 2.323e-03, 'freq': 560.1786500},
    full_index[39-1]+'-'+full_index[29-1]: {'A_ij': 7.536e-06, 'freq': 1587.6470627},
    full_index[39-1]+'-'+full_index[32-1]: {'A_ij': 4.477e-07, 'freq': 1113.0874695},
    full_index[39-1]+'-'+full_index[34-1]: {'A_ij': 6.222e-08, 'freq': 936.1530397},
    full_index[39-1]+'-'+full_index[35-1]: {'A_ij': 1.651e-07, 'freq': 595.7329379},
    full_index[39-1]+'-'+full_index[36-1]: {'A_ij': 2.278e-03, 'freq': 558.0876422},
    full_index[39-1]+'-'+full_index[37-1]: {'A_ij': 8.102e-06, 'freq': 419.8172108},
    full_index[40-1]+'-'+full_index[32-1]: {'A_ij': 1.621e-07, 'freq': 1252.5899799},
    full_index[40-1]+'-'+full_index[35-1]: {'A_ij': 2.311e-05, 'freq': 735.2354483},
    full_index[40-1]+'-'+full_index[37-1]: {'A_ij': 2.297e-03, 'freq': 559.3197212},
    full_index[40-1]+'-'+full_index[38-1]: {'A_ij': 2.113e-08, 'freq': 175.0567983},
    full_index[40-1]+'-'+full_index[39-1]: {'A_ij': 1.064e-08, 'freq': 139.5025104},
    full_index[41-1]+'-'+full_index[38-1]: {'A_ij': 2.905e-03, 'freq': 603.0216500},
    full_index[42-1]+'-'+full_index[32-1]: {'A_ij': 7.030e-06, 'freq': 1714.3459223},
    full_index[42-1]+'-'+full_index[35-1]: {'A_ij': 4.815e-07, 'freq': 1196.9913907},
    full_index[42-1]+'-'+full_index[37-1]: {'A_ij': 7.012e-08, 'freq': 1021.0756636},
    full_index[42-1]+'-'+full_index[38-1]: {'A_ij': 1.520e-07, 'freq': 636.8127407},
    full_index[42-1]+'-'+full_index[39-1]: {'A_ij': 2.859e-03, 'freq': 601.2584520},
    full_index[42-1]+'-'+full_index[40-1]: {'A_ij': 9.113e-06, 'freq': 461.7559424},
    full_index[43-1]+'-'+full_index[35-1]: {'A_ij': 1.703e-07, 'freq': 1337.5284784},
    full_index[43-1]+'-'+full_index[38-1]: {'A_ij': 2.413e-05, 'freq': 777.3498284},
    full_index[43-1]+'-'+full_index[40-1]: {'A_ij': 2.878e-03, 'freq': 602.2930210},
    full_index[43-1]+'-'+full_index[41-1]: {'A_ij': 2.092e-08, 'freq': 174.3281807},
    full_index[43-1]+'-'+full_index[42-1]: {'A_ij': 1.091e-08, 'freq': 140.5370877},
    full_index[44-1]+'-'+full_index[41-1]: {'A_ij': 3.577e-03, 'freq': 645.8759240},
    full_index[45-1]+'-'+full_index[35-1]: {'A_ij': 6.583e-06, 'freq': 1841.3703074},
    full_index[45-1]+'-'+full_index[38-1]: {'A_ij': 5.154e-07, 'freq': 1281.1916573},
    full_index[45-1]+'-'+full_index[40-1]: {'A_ij': 7.811e-08, 'freq': 1106.1348591},
    full_index[45-1]+'-'+full_index[41-1]: {'A_ij': 1.407e-07, 'freq': 678.1700097},
    full_index[45-1]+'-'+full_index[42-1]: {'A_ij': 3.532e-03, 'freq': 644.3789180},
    full_index[45-1]+'-'+full_index[43-1]: {'A_ij': 1.013e-05, 'freq': 503.8418290},
    full_index[46-1]+'-'+full_index[38-1]: {'A_ij': 1.786e-07, 'freq': 1422.6047695},
    full_index[46-1]+'-'+full_index[41-1]: {'A_ij': 2.516e-05, 'freq': 819.5831218},
    full_index[46-1]+'-'+full_index[43-1]: {'A_ij': 3.550e-03, 'freq': 645.2549330},
    full_index[46-1]+'-'+full_index[44-1]: {'A_ij': 2.074e-08, 'freq': 173.7072018},
    full_index[46-1]+'-'+full_index[45-1]: {'A_ij': 1.115e-08, 'freq': 141.4131122},
    full_index[47-1]+'-'+full_index[44-1]: {'A_ij': 4.346e-03, 'freq': 688.7357000},
    full_index[48-1]+'-'+full_index[38-1]: {'A_ij': 6.186e-06, 'freq': 1968.6493628},
    full_index[48-1]+'-'+full_index[41-1]: {'A_ij': 5.494e-07, 'freq': 1365.6277152},
    full_index[48-1]+'-'+full_index[43-1]: {'A_ij': 8.619e-08, 'freq': 1191.2995345},
    full_index[48-1]+'-'+full_index[44-1]: {'A_ij': 1.311e-07, 'freq': 719.7517951},
    full_index[48-1]+'-'+full_index[45-1]: {'A_ij': 4.300e-03, 'freq': 687.4576940},
    full_index[48-1]+'-'+full_index[46-1]: {'A_ij': 1.116e-05, 'freq': 546.0445933},
    full_index[49-1]+'-'+full_index[41-1]: {'A_ij': 1.869e-07, 'freq': 1507.7877617},
    full_index[49-1]+'-'+full_index[44-1]: {'A_ij': 2.619e-05, 'freq': 861.9118416},
    full_index[49-1]+'-'+full_index[46-1]: {'A_ij': 4.317e-03, 'freq': 688.2046300},
    full_index[49-1]+'-'+full_index[47-1]: {'A_ij': 2.058e-08, 'freq': 173.1761453},
    full_index[49-1]+'-'+full_index[48-1]: {'A_ij': 1.135e-08, 'freq': 142.1600465},
    full_index[50-1]+'-'+full_index[47-1]: {'A_ij': 5.217e-03, 'freq': 731.5964800},
    full_index[51-1]+'-'+full_index[41-1]: {'A_ij': 5.833e-06, 'freq': 2096.1285099},
    full_index[51-1]+'-'+full_index[44-1]: {'A_ij': 5.836e-07, 'freq': 1450.2525899},
    full_index[51-1]+'-'+full_index[46-1]: {'A_ij': 9.433e-08, 'freq': 1276.5453881},
    full_index[51-1]+'-'+full_index[47-1]: {'A_ij': 1.226e-07, 'freq': 761.5168935},
    full_index[51-1]+'-'+full_index[48-1]: {'A_ij': 5.171e-03, 'freq': 730.5007947},
    full_index[51-1]+'-'+full_index[49-1]: {'A_ij': 1.220e-05, 'freq': 588.3407482},
    full_index[52-1]+'-'+full_index[44-1]: {'A_ij': 1.952e-07, 'freq': 1593.0531532},
    full_index[52-1]+'-'+full_index[47-1]: {'A_ij': 2.723e-05, 'freq': 904.3174568},
    full_index[52-1]+'-'+full_index[49-1]: {'A_ij': 5.188e-03, 'freq': 731.1413115},
    full_index[52-1]+'-'+full_index[50-1]: {'A_ij': 2.044e-08, 'freq': 172.7209769},
    full_index[52-1]+'-'+full_index[51-1]: {'A_ij': 1.152e-08, 'freq': 142.8005633},
    full_index[53-1]+'-'+full_index[50-1]: {'A_ij': 6.199e-03, 'freq': 774.4546775},
    full_index[54-1]+'-'+full_index[47-1]: {'A_ij': 6.177e-07, 'freq': 1535.0292195},
    full_index[54-1]+'-'+full_index[49-1]: {'A_ij': 1.025e-07, 'freq': 1361.8530742},
    full_index[54-1]+'-'+full_index[50-1]: {'A_ij': 1.152e-07, 'freq': 803.4327395},
    full_index[54-1]+'-'+full_index[51-1]: {'A_ij': 6.152e-03, 'freq': 773.5123260},
    full_index[54-1]+'-'+full_index[52-1]: {'A_ij': 1.325e-05, 'freq': 630.7117627},
    full_index[55-1]+'-'+full_index[47-1]: {'A_ij': 2.036e-07, 'freq': 1678.3815983},
    full_index[55-1]+'-'+full_index[50-1]: {'A_ij': 2.829e-05, 'freq': 946.7851183},
    full_index[55-1]+'-'+full_index[52-1]: {'A_ij': 6.169e-03, 'freq': 774.0641414},
    full_index[55-1]+'-'+full_index[53-1]: {'A_ij': 2.033e-08, 'freq': 172.3304408},
    full_index[55-1]+'-'+full_index[54-1]: {'A_ij': 1.167e-08, 'freq': 143.3523788},
    full_index[56-1]+'-'+full_index[53-1]: {'A_ij': 7.297e-03, 'freq': 817.3073600},
    full_index[57-1]+'-'+full_index[47-1]: {'A_ij': 5.229e-06, 'freq': 2351.5243857},
    full_index[57-1]+'-'+full_index[50-1]: {'A_ij': 6.521e-07, 'freq': 1619.9279057},
    full_index[57-1]+'-'+full_index[52-1]: {'A_ij': 1.108e-07, 'freq': 1447.2069289},
    full_index[57-1]+'-'+full_index[53-1]: {'A_ij': 1.086e-07, 'freq': 845.4732283},
    full_index[57-1]+'-'+full_index[54-1]: {'A_ij': 7.249e-03, 'freq': 816.4951840},
    full_index[57-1]+'-'+full_index[55-1]: {'A_ij': 1.430e-05, 'freq': 673.1427874},
    full_index[58-1]+'-'+full_index[50-1]: {'A_ij': 2.120e-07, 'freq': 1763.7574331},
    full_index[58-1]+'-'+full_index[53-1]: {'A_ij': 2.933e-05, 'freq': 989.3027556},
    full_index[58-1]+'-'+full_index[55-1]: {'A_ij': 7.265e-03, 'freq': 816.9723280},
    full_index[58-1]+'-'+full_index[56-1]: {'A_ij': 2.023e-08, 'freq': 171.9954092},
    full_index[58-1]+'-'+full_index[57-1]: {'A_ij': 1.181e-08, 'freq': 143.8295274},
    full_index[59-1]+'-'+full_index[56-1]: {'A_ij': 8.515e-03, 'freq': 860.1520215},
    full_index[60-1]+'-'+full_index[50-1]: {'A_ij': 4.969e-06, 'freq': 2479.3791851},
    full_index[60-1]+'-'+full_index[53-1]: {'A_ij': 6.864e-07, 'freq': 1704.9245077},
    full_index[60-1]+'-'+full_index[56-1]: {'A_ij': 1.028e-07, 'freq': 887.6171613},
    full_index[60-1]+'-'+full_index[57-1]: {'A_ij': 8.467e-03, 'freq': 859.4512794},
    full_index[60-1]+'-'+full_index[58-1]: {'A_ij': 1.535e-05, 'freq': 715.6217520},
    full_index[61-1]+'-'+full_index[53-1]: {'A_ij': 2.204e-07, 'freq': 1849.1677723},
    full_index[61-1]+'-'+full_index[56-1]: {'A_ij': 3.039e-05, 'freq': 1031.8604259},
    full_index[61-1]+'-'+full_index[58-1]: {'A_ij': 8.485e-03, 'freq': 859.8650167},
    full_index[61-1]+'-'+full_index[59-1]: {'A_ij': 2.014e-08, 'freq': 171.7084044},
    full_index[61-1]+'-'+full_index[60-1]: {'A_ij': 1.192e-08, 'freq': 144.2432647},
    full_index[62-1]+'-'+full_index[59-1]: {'A_ij': 9.864e-03, 'freq': 902.9865820},
    full_index[63-1]+'-'+full_index[53-1]: {'A_ij': 4.733e-06, 'freq': 2607.3064857},
    full_index[63-1]+'-'+full_index[56-1]: {'A_ij': 7.206e-07, 'freq': 1789.9991393},
    full_index[63-1]+'-'+full_index[59-1]: {'A_ij': 9.748e-08, 'freq': 929.8471178},
    full_index[63-1]+'-'+full_index[60-1]: {'A_ij': 9.817e-03, 'freq': 902.3819640},
    full_index[63-1]+'-'+full_index[61-1]: {'A_ij': 1.640e-05, 'freq': 758.1387134},
    full_index[64-1]+'-'+full_index[56-1]: {'A_ij': 2.289e-07, 'freq': 1934.6018580},
    full_index[64-1]+'-'+full_index[59-1]: {'A_ij': 3.144e-05, 'freq': 1074.4498364},
    full_index[64-1]+'-'+full_index[61-1]: {'A_ij': 9.831e-03, 'freq': 902.7414190},
    full_index[64-1]+'-'+full_index[62-1]: {'A_ij': 2.007e-08, 'freq': 171.4632440},
    full_index[64-1]+'-'+full_index[63-1]: {'A_ij': 1.202e-08, 'freq': 144.6027186},
    full_index[66-1]+'-'+full_index[56-1]: {'A_ij': 4.518e-06, 'freq': 2735.2872367},
    full_index[66-1]+'-'+full_index[59-1]: {'A_ij': 7.550e-07, 'freq': 1875.1352152},
    full_index[66-1]+'-'+full_index[62-1]: {'A_ij': 9.271e-08, 'freq': 972.1486228},
    full_index[66-1]+'-'+full_index[63-1]: {'A_ij': 1.130e-02, 'freq': 945.2880974},
    full_index[66-1]+'-'+full_index[64-1]: {'A_ij': 1.746e-05, 'freq': 800.6853788},
    full_index[67-1]+'-'+full_index[62-1]: {'A_ij': 3.250e-05, 'freq': 1117.0639897},
    full_index[67-1]+'-'+full_index[64-1]: {'A_ij': 1.131e-02, 'freq': 945.6007457},
    full_index[67-1]+'-'+full_index[65-1]: {'A_ij': 2.001e-08, 'freq': 171.2547735},
    full_index[67-1]+'-'+full_index[66-1]: {'A_ij': 1.211e-08, 'freq': 144.9153669},
    full_index[68-1]+'-'+full_index[65-1]: {'A_ij': 1.297e-02, 'freq': 988.6182539},
    full_index[69-1]+'-'+full_index[59-1]: {'A_ij': 4.320e-06, 'freq': 2863.3053327},
    full_index[69-1]+'-'+full_index[62-1]: {'A_ij': 7.895e-07, 'freq': 1960.3187402},
    full_index[69-1]+'-'+full_index[66-1]: {'A_ij': 1.292e-02, 'freq': 988.1701175},
    full_index[69-1]+'-'+full_index[67-1]: {'A_ij': 1.852e-05, 'freq': 843.2547505},
    full_index[70-1]+'-'+full_index[65-1]: {'A_ij': 3.356e-05, 'freq': 1159.6969159},
    full_index[70-1]+'-'+full_index[67-1]: {'A_ij': 1.293e-02, 'freq': 988.4421425},
    full_index[70-1]+'-'+full_index[68-1]: {'A_ij': 1.996e-08, 'freq': 171.0786621},
    full_index[70-1]+'-'+full_index[69-1]: {'A_ij': 1.219e-08, 'freq': 145.1873920},
    full_index[71-1]+'-'+full_index[68-1]: {'A_ij': 1.474e-02, 'freq': 1031.4122130},
    full_index[72-1]+'-'+full_index[62-1]: {'A_ij': 4.139e-06, 'freq': 2991.3469909},
    full_index[72-1]+'-'+full_index[65-1]: {'A_ij': 8.239e-07, 'freq': 2045.5377746},
    full_index[72-1]+'-'+full_index[69-1]: {'A_ij': 1.469e-02, 'freq': 1031.0282490},
    full_index[72-1]+'-'+full_index[70-1]: {'A_ij': 1.958e-05, 'freq': 885.8408550},
    full_index[73-1]+'-'+full_index[68-1]: {'A_ij': 3.462e-05, 'freq': 1202.3434690},
    full_index[73-1]+'-'+full_index[70-1]: {'A_ij': 1.471e-02, 'freq': 1031.2648040},
    full_index[73-1]+'-'+full_index[71-1]: {'A_ij': 1.992e-08, 'freq': 170.9312452},
    full_index[73-1]+'-'+full_index[72-1]: {'A_ij': 1.226e-08, 'freq': 145.4239482},
    full_index[74-1]+'-'+full_index[71-1]: {'A_ij': 1.666e-02, 'freq': 1074.1897671},
    full_index[75-1]+'-'+full_index[65-1]: {'A_ij': 3.971e-06, 'freq': 3119.4002799},
    full_index[75-1]+'-'+full_index[68-1]: {'A_ij': 8.584e-07, 'freq': 2130.7820260},
    full_index[75-1]+'-'+full_index[72-1]: {'A_ij': 1.662e-02, 'freq': 1073.8625052},
    full_index[75-1]+'-'+full_index[73-1]: {'A_ij': 2.064e-05, 'freq': 928.4385570},
    full_index[76-1]+'-'+full_index[71-1]: {'A_ij': 3.568e-05, 'freq': 1244.9991686},
    full_index[76-1]+'-'+full_index[73-1]: {'A_ij': 1.663e-02, 'freq': 1074.0679234},
    full_index[76-1]+'-'+full_index[74-1]: {'A_ij': 1.989e-08, 'freq': 170.8094016},
    full_index[77-1]+'-'+full_index[74-1]: {'A_ij': 1.875e-02, 'freq': 1116.9496215},
    full_index[78-1]+'-'+full_index[68-1]: {'A_ij': 3.815e-06, 'freq': 3247.4547576},
    full_index[78-1]+'-'+full_index[71-1]: {'A_ij': 8.929e-07, 'freq': 2216.0425339},
    full_index[78-1]+'-'+full_index[75-1]: {'A_ij': 1.870e-02, 'freq': 1116.6727316},
    full_index[78-1]+'-'+full_index[76-1]: {'A_ij': 2.170e-05, 'freq': 971.0433652},
    full_index[79-1]+'-'+full_index[74-1]: {'A_ij': 3.674e-05, 'freq': 1287.6600780},
    full_index[79-1]+'-'+full_index[76-1]: {'A_ij': 1.872e-02, 'freq': 1116.8506765},
    full_index[79-1]+'-'+full_index[77-1]: {'A_ij': 1.986e-08, 'freq': 170.7104566},
    full_index[80-1]+'-'+full_index[77-1]: {'A_ij': 2.100e-02, 'freq': 1159.6906013},
    full_index[81-1]+'-'+full_index[71-1]: {'A_ij': 3.671e-06, 'freq': 3375.5011913},
    full_index[81-1]+'-'+full_index[74-1]: {'A_ij': 9.275e-07, 'freq': 2301.3114243},
    full_index[81-1]+'-'+full_index[78-1]: {'A_ij': 2.095e-02, 'freq': 1159.4586575},
    full_index[81-1]+'-'+full_index[79-1]: {'A_ij': 2.277e-05, 'freq': 1013.6513462},
    full_index[82-1]+'-'+full_index[77-1]: {'A_ij': 3.780e-05, 'freq': 1330.3227067},
    full_index[82-1]+'-'+full_index[79-1]: {'A_ij': 2.096e-02, 'freq': 1159.6122502},
    full_index[83-1]+'-'+full_index[80-1]: {'A_ij': 2.342e-02, 'freq': 1202.4115823},
    full_index[84-1]+'-'+full_index[74-1]: {'A_ij': 3.537e-06, 'freq': 3503.5313377},
    full_index[84-1]+'-'+full_index[77-1]: {'A_ij': 9.620e-07, 'freq': 2386.5817162},
    full_index[84-1]+'-'+full_index[81-1]: {'A_ij': 2.338e-02, 'freq': 1202.2199134},
    full_index[84-1]+'-'+full_index[82-1]: {'A_ij': 2.383e-05, 'freq': 1056.2590095},
    full_index[85-1]+'-'+full_index[80-1]: {'A_ij': 3.887e-05, 'freq': 1372.9839340},
    full_index[85-1]+'-'+full_index[82-1]: {'A_ij': 2.339e-02, 'freq': 1202.3518286},
    full_index[86-1]+'-'+full_index[83-1]: {'A_ij': 2.602e-02, 'freq': 1245.1111280},
    full_index[87-1]+'-'+full_index[84-1]: {'A_ij': 2.598e-02, 'freq': 1244.9555080},
    full_index[87-1]+'-'+full_index[85-1]: {'A_ij': 2.489e-05, 'freq': 1098.8632340},
    full_index[88-1]+'-'+full_index[83-1]: {'A_ij': 3.993e-05, 'freq': 1415.6409473},
    full_index[88-1]+'-'+full_index[85-1]: {'A_ij': 2.599e-02, 'freq': 1245.0678590},
    full_index[89-1]+'-'+full_index[86-1]: {'A_ij': 2.881e-02, 'freq': 1287.7892740},
    full_index[90-1]+'-'+full_index[80-1]: {'A_ij': 3.295e-06, 'freq': 3759.5137368},
    full_index[90-1]+'-'+full_index[87-1]: {'A_ij': 2.876e-02, 'freq': 1287.6670620},
    full_index[90-1]+'-'+full_index[88-1]: {'A_ij': 2.596e-05, 'freq': 1141.4612072},
    full_index[91-1]+'-'+full_index[86-1]: {'A_ij': 4.098e-05, 'freq': 1458.2911923},
    full_index[91-1]+'-'+full_index[88-1]: {'A_ij': 2.877e-02, 'freq': 1287.7617510},
}
line_index = [key for key in line_list]


def Q_SO_i(index, Tex=5*u.K) -> float:
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
    if type(index) == str:
        return (SO_levels[index]['g_u'] * np.exp(-SO_levels[index]['Eup'] / Tex)).value
    else:
        Q_SO_index = np.zeros(len(index))
        for i, index_i in enumerate(index):
            Q_SO_index[i] = Q_SO_i(index_i, Tex=Tex)
        return Q_SO_index


@u.quantity_input
def check_Tex(Tex: u.K) -> bool:
    if np.isnan(Tex):
        print('The excitation temperature is lower than the CMB temperature of 2.73 K')
        return True
    if Tex <= 0.0*u.K:
        print('The excitation temperature is lower than 0 K')
        return True
    else:
        return False


@u.quantity_input
def Q_SO(Tex: u.K = 5*u.K) -> float:
    """
    It returns the particion function for SO with an excitation 
    temperature.
    It uses the first 91-energy levels.
    If an array of temperatures is given, then it returns an array 
    with the corresponding partition function for the requested Tex.

    Parameters
    ----------
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    float
        The partition function.
    """
    if Tex.size == 1:
        if check_Tex(Tex):
            return np.nan
        else:
            return np.sum(Q_SO_i(full_index, Tex=Tex))
    else:
        Q_SO_all = np.zeros_like(Tex.value)
        # for i in range(Tex.size):
        for i, Trot_i in np.ndenumerate(Tex):
            if np.isnan(Tex[i]):
                Q_SO_all[i] = np.nan
            else:
                Q_SO_all[i] = np.sum(Q_SO_i(full_index, Tex=Tex[i]))
        return Q_SO_all


@u.quantity_input
# -> u.cm**-2 | tuple[u.cm**-2, u.K]:
def SO_thin_Nu_Rot(N_J_up: str = '2_1', N_J_low: str = '1_1', TdV: u.K*u.km/u.s = 1.0*u.K*u.km/u.s, give_Eup: bool = False):
    """
    Calculation of the Nu/g_u term, which is used for the rotational diagram.
    The column density determination is from the SO N_J_up - N_J_low transition, 
    where N_J_up and N_J_low are the upper and lower N_J levels, 
    e.g., '2_1' and '1_1' to make the '2_1-1_1' transition.
    The A_ul, frequency and Einstein coefficient are obtained from LAMBDA database.
    If give_Eup is True, then the E_up (in K) is also returned.

    Parameters
    ----------
    N_J_up : str
        The upper level of the transition.
    N_J_low : str
        The lower level of the transition.
    TdV : u.K*u.km/u.s
        The integrated intensity of the transition, in units of K km/s.
    give_Eup : bool
        If True, then the energy of the upper level is also returned.

    Returns
    -------
    u.cm**-2 | tuple[u.cm**-2, u.K]
        The column density/(level degeneracy) of the upper energy level, and the upper energy level if requested.
    """
    N_J = N_J_up + '-' + N_J_low
    if N_J in line_index:
        freq = line_list[N_J]['freq'] * u.GHz
        A_ul = line_list[N_J]['A_ij'] / u.s
        E_up = SO_levels[N_J_up]['Eup'] * u.K
        g_up = SO_levels[N_J_up]['g_u']
    else:
        print('Transition {0} is not available'.format(N_J))
        return np.nan * u.cm**-2
    Ncol = (8*np.pi*k_B*freq**2/c**3) * TdV / A_ul / g_up / h
    if give_Eup:
        return Ncol.to(u.cm**-2), E_up
    else:
        return Ncol.to(u.cm**-2)


@u.quantity_input
def SO_thin(N_J_up: str = '2_1', N_J_low: str = '1_1', Tex: u.K = 5*u.K, TdV=1.0*u.K*u.km/u.s, T_bg: u.K = 2.73 * u.K) -> u.cm**-2:
    """
    Total column density determination from the SO N_J_up - N_J_low transition, 
    where N_J_up and N_J_low are the upper and lower N_J levels, 
    e.g., '0_1' and '1_0' to make the '0_1-1_0' transition.
    The A_ul, frequency and Einstein coefficient are obtained from LAMBDA database.

    Parameters
    ----------
    N_J_up : str
        The upper level of the transition.
    N_J_low : str
        The lower level of the transition.
    Tex : u.K
        The excitation temperature.
    TdV : u.K*u.km/u.s
        The integrated intensity of the transition, in units of K km/s.
    T_bg : u.K
        The background temperature.

    Returns
    -------
    u.cm**-2
        The total column density in units of cm^-2.
    """
    N_J = N_J_up + '-' + N_J_low
    if N_J in line_index:
        freq = line_list[N_J]['freq'] * u.GHz
        A_ul = line_list[N_J]['A_ij'] / u.s
    else:
        print('Transition {0} is not available'.format(N_J))
        return np.nan * u.cm**-2
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)
    # J_up = 2
    Ncol = (8*np.pi*freq**3/c**3) * Q_SO(Tex=Tex) \
        / A_ul / Q_SO_i(N_J_up, Tex=Tex) \
        / (np.exp((h*freq/k_B/Tex).decompose().value) - 1) * TdV / (Jex - Jbg)
    return Ncol
