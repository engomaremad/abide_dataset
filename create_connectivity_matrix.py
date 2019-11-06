#Todo create the connectivity matrix (stationary) for each subject and save it in dictionary
import os
import numpy as np
import scipy.stats as sc

os.chdir('Data_dictionaries')
ids = np.load('used_names.npy')
ids_tcs = np.load('id_tcs.npy').item()
ids_conn_mats = dict()
num_areas = 116
for id in ids:
    conn_mat = np.ones([num_areas,num_areas])
    tcs = ids_tcs[id]
    for i in range(num_areas):
        for j in range(i):
            conn_mat[i,j],p = sc.pearsonr(tcs[:,i],tcs[:,j])
            conn_mat[j,i] =conn_mat[i,j]
    ids_conn_mats[id] = conn_mat
np.save('ids_conns_mats',ids_conn_mats)