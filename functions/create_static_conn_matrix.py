#TODO: This funcition is to create the stationary functional connectivity matrix

import numpy as np
import scipy.stats as sc
def create_sfc(num_areas,tcs):
    conn_mat = np.ones([num_areas,num_areas])
    for i in range(num_areas):
        for j in range(i):
            conn_mat[i,j],p = sc.pearsonr(tcs[:,i],tcs[:,j])
            conn_mat[j,i] =conn_mat[i,j]
    return conn_mat