#Todo create the connectivity matrix (stationary) for each subject and save it in dictionary
import os
import numpy as np
import scipy.stats as sc
from nilearn.connectome import ConnectivityMeasure
os.chdir('Data_dictionaries')
ids = np.load('used_names.npy')
ids_tcs = np.load('id_tcs_craddock.npy').item()
ids_conn_mats = dict()
tcs= []
num_areas = 116
for id in ids:
    conn_mat = np.ones([num_areas,num_areas])
    tcs.append(ids_tcs[id])
meas = ConnectivityMeasure(kind='tangent')
conn_mat = meas.fit_transform(tcs)
cnt =0
for id in ids:
    ids_conn_mats[id] = conn_mat[cnt,:,:]
    cnt+=1
np.save('ids_conns_mats_aal_tangent',ids_conn_mats)