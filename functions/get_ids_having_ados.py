#TODO return list of IDS having all ADOS reports avaialble for ados module sent to the function

import os
import numpy as np



def get_ids_with_ados(ados_module,path):
    if os.path.exists(os.path.join(path,'have_all_ados_'+str(ados_module)+'.npy')):
        have_all_ados = np.load(os.path.join(path,'have_all_ados_'+str(ados_module)+'.npy'))
    else:
        ids = np.load('used_names.npy')
        ids_conns = np.load('id_dfc_tapered_shift_1.npy').item()
        id_ados_mod  = np.load('id_ados_mod.npy').item()
        id_ados_tot  = np.load('id_ados_tot.npy').item()
        id_ados_comm  = np.load('id_ados_comm.npy').item()
        id_ados_soc = np.load('id_ados_soc.npy').item()
        id_ados_beh = np.load('id_ados_beh.npy').item()
        id_ados_rel  = np.load('id_ados_rel.npy').item()
        #used ADOS module
        used_ados = ados_module
        ids_with_ados_total = []
        ados_total = []
        ids_with_ados_beh = []
        ados_beh = []
        ids_with_ados_comm = []
        ados_comm = []
        ids_with_ados_soc = []
        ados_soc = []
        #get the ids that have ados total
        for id in ids:
            if id_ados_mod[id] == used_ados  and id_ados_rel[id]==1 and id_ados_tot[id] >= 0:
                ids_with_ados_total.append(id)
                ados_total.append(id_ados_tot[id])


        #get the ids that have ados behaviour
        for id in ids:
            if id_ados_mod[id] == used_ados  and id_ados_rel[id]==1 and id_ados_beh[id] >= 0:
                ids_with_ados_beh.append(id)
                ados_beh.append(id_ados_beh[id])

        #get the ids that have ados communication
        for id in ids:
            if id_ados_mod[id] == used_ados  and id_ados_rel[id]==1 and id_ados_comm[id] >= 0:
                ids_with_ados_comm.append(id)
                ados_comm.append(id_ados_comm[id])

        #get the ids that have ados social
        for id in ids:
            if id_ados_mod[id] == used_ados  and id_ados_rel[id]==1 and id_ados_soc[id] >= 0:
                ids_with_ados_soc.append(id)
                ados_soc.append(id_ados_soc[id])

        have_all_ados = list(set(ids_with_ados_soc) & set(ids_with_ados_comm) & set(ids_with_ados_beh) & set(ids_with_ados_total))
        np.save(os.path.join(path,'have_all_ados_'+str(ados_module)+'.npy'),have_all_ados)
    return have_all_ados
