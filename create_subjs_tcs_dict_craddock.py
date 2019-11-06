#TODO create a dictionarty with subject ID as key and time courses at each AAL atlas area as value

import numpy as np
import os


tcs_paths = '/media/biolab/easystore/abide_dataset_fmri/craddock_atlas'
os.chdir('Data_dictionaries')
ids = np.load('used_names.npy')
id_tcs = dict()
for id in ids:
    tc = np.loadtxt(os.path.join(tcs_paths,id+'.txt'),skiprows=1)
    id_tcs[id]= tc
np.save('id_tcs_craddock',id_tcs)
