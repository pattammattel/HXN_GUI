import numpy as np

elem_K_list = np.array(['Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn',
                        'Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
                        'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I',
                        'Xe','Cs','Ba','La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl',
                        'Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Ce','Pr','Nd','Pm','Sm','Eu',
                        'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf'])

energy_K_list = np.array([1040,1254,1487,1740,2011,2310,2622,2958,3314,3692,4093,4512,4953,5415,5900,6405,6931,
                          7480,8046,8637,9251,9886,10543,11224,11924,12648,13396,14165,14958,15775,16615,17480,
                          18367,19279,20216,21177,22163,23173,24210,25271,26359,27473,28612,29775,30973,32194,
                          33442,55790,57535,59318,61141,63000,64896,66831,68806,70818,72872,74970,77107,79291,
                          81516,83785,86106,88478,90884,34720,36027,37361,38725,40118,41542,42996,44482,45999,
                          47547,49128,50742,52388,54070,93351,95868,98440,101059,103734,106472,109271,112121,115032])

elem_K_tooltip = list(elem_K_list)
for i in range(len(elem_K_list)):
    elem_K_tooltip[i] = elem_K_tooltip[i] + ':%d'%(energy_K_list[i])

elem_L_list = np.array(['Zn_L','Ga_L','Ge_L','AS_L','Se_L','Br_L','Kr_L','Rb_L','Sr_L','Y_L','Zr_L','Nb_L',
                        'Mo_L','Tc_L','Ru_L','Rh_L','Pd_L','Ag_L','Cd_L','In_L','Sn_L','Sb_L','Te_L','I_L',
                        'Xe_L','Cs_L','Ba_L','La_L','Hf_L','Ta_L','W_L','Re_L','Os_L','Ir_L','Pt_L','Au_L',
                        'Hg_L','Tl_L','Pb_L','Bi_L','Po_L','At_L','Rn_L','Fr_L','Ra_L','Ac_L','Ce_L','Pr_L',
                        'Nd_L','Pm_L','Sm_L','Eu_L','Gd_L','Tb_L','Dy_L','Ho_L','Er_L','Tm_L','Yb_L','Lu_L',
                        'Th_L','Pa_L','U_L','Np_L','Pu_L','Am_L','Cm_L','Bk_L','Cf_L'])

energy_L_list = np.array([1012,1098,1186,1282,1379,1481,1585,1692,1806,1924,2044,2169,2292,2423,2558,2697,
                          2838,2983,3133,3280,3444,3604,3768,3938,4110,4285,4467,4647,7899,8146,8398,8652,
                          8911,9175,9442,9713,9989,10269,10551,10839,11131,11427,11727,12031,12339,12652,
                          4839,5035,5228,5432,5633,5850,6053,6273,6498,6720,6949,7180,7416,7655,12968,13291,
                          13614,13946,14282,14620,14961,15308,15660])

elem_L_tooltip = list(elem_L_list)
for i in range(len(elem_L_list)):
    elem_L_tooltip[i] = elem_L_tooltip[i] + ':%d'%(energy_L_list[i])

elem_M_list = np.array(['Hf_M','Ta_M','W_M','Re_M','Os_M','Ir_M','Pt_M','Au_M','Hg_M','Tl_M',
                        'Pb_M','Bi_M','Po_M','At_M','Rn_M','Fr_M','Ra_M','Ac_M','Ce_M','Pr_M',
                        'Nd_M','Pm_M','Sm_M','Eu_M','Gd_M','Tb_M','Dy_M','Ho_M','Er_M','Tm_M',
                        'Yb_M','Lu_M','Th_M','Pa_M','U_M','Np_M','Pu_M','Am_M','Cm_M','Bk_M','Cf_M'])

energy_M_list = np.array([1646,1712,1775,1840,1907,1976,2048,2118,2191,2267,2342,2418,2499,2577,
                          2654,2732,2806,2900,884,927,979,1023,1078,1122,1181,1233,1284,1342,1404,
                          1463,1526,1580,2990,3071,3164,3250,3339,3429,3525,3616,3709])

elem_M_tooltip = list(elem_M_list)
for i in range(len(elem_M_list)):
    elem_M_tooltip[i] = elem_M_tooltip[i] + ':%d'%(energy_M_list[i])

all_elem_lines = np.concatenate((elem_K_tooltip,elem_L_tooltip,elem_M_tooltip))
