#Todo calculate the correlation between the stationary connectivity and ADOS reports
# this code will calculate the correlation for each ados module separitly

import os
import numpy as np
import scipy.stats as sc

os.chdir('Data_dictionaries')
ids = np.load('used_names.npy')
ids_conns = np.load('ids_conns_mats.npy').item()
id_ados_mod  = np.load('id_ados_mod.npy').item()
id_ados_tot  = np.load('id_ados_tot.npy').item()
id_ados_comm  = np.load('id_ados_comm.npy').item()
id_ados_soc = np.load('id_ados_soc.npy').item()
id_ados_beh = np.load('id_ados_beh.npy').item()
id_ados_rel  = np.load('id_ados_rel.npy').item()
#used ADOS module
used_ados = 4
x = 0
cnt = 0
num_areas = 116
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

cnt = 0
for i in range(num_areas):
    for j in range(i):
        conn_vect = []
        for cnt in range(len(ids_with_ados_total)):
            conn_mat = ids_conns[ids_with_ados_total[cnt]]
            conn_vect.append(conn_mat[i,j])
        r,p =sc.pearsonr(conn_vect,ados_total)
        if p <0.001:
            print(i+1,j+1,r,p,'total')
x =0

#get the ids that have ados behaviour
for id in ids:
    if id_ados_mod[id] == used_ados  and id_ados_rel[id]==1 and id_ados_beh[id] >= 0:
        ids_with_ados_beh.append(id)
        ados_beh.append(id_ados_beh[id])

cnt = 0
for i in range(num_areas):
    for j in range(i):
        conn_vect = []
        for cnt in range(len(ids_with_ados_beh)):
            conn_mat = ids_conns[ids_with_ados_beh[cnt]]
            conn_vect.append(conn_mat[i,j])
        r,p =sc.pearsonr(conn_vect,ados_beh)
        if p<0.001:
            print(i+1,j+1,r,p,'beh')
x =0

#get the ids that have ados communication
for id in ids:
    if id_ados_mod[id] == used_ados  and id_ados_rel[id]==1 and id_ados_comm[id] >= 0:
        ids_with_ados_comm.append(id)
        ados_comm.append(id_ados_comm[id])

cnt = 0
for i in range(num_areas):
    for j in range(i):
        conn_vect = []
        for cnt in range(len(ids_with_ados_comm)):
            conn_mat = ids_conns[ids_with_ados_comm[cnt]]
            conn_vect.append(conn_mat[i,j])
        r,p =sc.pearsonr(conn_vect,ados_comm)
        if p<0.001:
            print(i+1,j+1,r,p,'comm')
x =0


#get the ids that have ados social
for id in ids:
    if id_ados_mod[id] == used_ados  and id_ados_rel[id]==1 and id_ados_soc[id] >= 0:
        ids_with_ados_soc.append(id)
        ados_soc.append(id_ados_soc[id])

cnt = 0
for i in range(num_areas):
    for j in range(i):
        conn_vect = []
        for cnt in range(len(ids_with_ados_soc)):
            conn_mat = ids_conns[ids_with_ados_soc[cnt]]
            conn_vect.append(conn_mat[i,j])
        r,p =sc.pearsonr(conn_vect,ados_soc)
        if p<0.001:
            print(i+1,j+1,r,p,'soc')
x =0