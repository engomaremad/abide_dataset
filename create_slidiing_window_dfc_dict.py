# Todo: c the sliding window dynamic functional connectivity dictionary

import os
import numpy as np
import scipy.stats as sc

os.chdir('Data_dictionaries')
ids = np.load('used_names.npy')
id_ados_mod  = np.load('id_ados_mod.npy').item()
id_ados_tot  = np.load('id_ados_tot.npy').item()
id_ados_comm  = np.load('id_ados_comm.npy').item()
id_ados_soc = np.load('id_ados_soc.npy').item()
id_ados_beh = np.load('id_ados_beh.npy').item()
id_ados_rel  = np.load('id_ados_rel.npy').item()
id_tcs = np.load('id_tcs.npy').item()
num_areas = 116

id_dfc = dict()
cnt = 0
for id in ids:
    cnt+=1
    print(cnt)
    tc = id_tcs[id]


    # dfc matrix shift
    # size of window = 30 TR
    # window is shifted by 5 TR
    #number of window = (lenght of signal - win_size)/shift
    win_size = 30
    shift = 5
    num_wins =int(np.ceil((np.shape(tc)[0]-win_size)/shift))
    dfc_mat = np.ones([num_areas,num_areas,num_wins])
    for i in range(num_areas):
        for j in range(i):
            curr = 0
            for w in range(num_wins):
                if curr+win_size <= np.shape(tc)[0]:
                    dfc_mat[i,j,w],p = sc.pearsonr(tc[curr:curr+win_size,i],tc[curr:curr+win_size,j])
                    curr = curr + shift
                else:
                    dfc_mat[i, j, w], p = sc.pearsonr(tc[curr:np.shape(tc)[0], i], tc[curr:np.shape(tc)[0], j])
                dfc_mat[j, i, w] = dfc_mat[i, j, w]
    id_dfc[id] =dfc_mat
np.save('id_dfc',id_dfc)