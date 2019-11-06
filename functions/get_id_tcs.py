import numpy as np
import os

def ids_tcs(ids):
    path = os.path.join(os.getcwd(), 'Data_dictionaries')
    ids_tcs = np.load(os.path.join(path,'id_tcs.npy')).item()
    ids_tcs_ret = dict()
    for id in ids:
        ids_tcs_ret[id] = ids_tcs[id]
    return ids_tcs_ret