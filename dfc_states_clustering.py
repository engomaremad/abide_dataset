#Todo calculate the correlation between the dynamic connectivity states occurenace frequency and ADOS reports
# this code will calculate the correlation for each ados module separitly

import os
import numpy as np
import scipy.stats as sc
from sklearn.cluster import KMeans

def cluster(ids):
    # concatenate all subjs for the clustering step
    concatenated_mat = ids_conns[ids[0]]
    for i in range(len(ids)):
        concatenated_mat = np.concatenate((concatenated_mat, ids_conns[ids[i]]), axis=2)
    concatenated_mat_reshaped = np.transpose(
        np.reshape(concatenated_mat, [num_areas * num_areas, np.shape(concatenated_mat)[2]]))
    concatenated_mat_reshaped[np.isnan(concatenated_mat_reshaped)] = 0
    kmeans = KMeans()
    kmeans.fit(concatenated_mat_reshaped)
    centers = kmeans.cluster_centers_
    centers = np.reshape(centers, [np.shape(centers)[0], num_areas, num_areas])
    return kmeans,centers


def get_labels(kmeans,ids):
    id_labels_vect = dict()
    for id in ids:
        vect = []
        mat = concatenated_mat = ids_conns[id]
        for i in range(np.shape(mat)[2]):
            reshaped = np.reshape(mat[:,:,i],[num_areas*num_areas,1])
            reshaped[np.isnan(reshaped)]=0
            vect.append(kmeans.predict(np.transpose(reshaped)))
        id_labels_vect[id] = vect
    return(id_labels_vect)



def corr(id_labels_dict,ados):

    for i in range(num_clusters):
        cnt_vect = []
        for key in id_labels_dict:
            cnt_vect.append(id_labels_dict[key].count(i))
        r, p = sc.pearsonr(cnt_vect, ados)
        x = 0





os.chdir('Data_dictionaries')
ids = np.load('used_names.npy')
ids_conns = np.load('id_dfc.npy').item()
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
num_clusters = 8
#get the ids that have ados total




for id in ids:
    if id_ados_mod[id] == used_ados  and id_ados_rel[id]==1 and id_ados_tot[id] >= 0:
        ids_with_ados_total.append(id)
        ados_total.append(id_ados_tot[id])



kmeans_total,centers_total  =cluster(ids_with_ados_total)
ids_labels_total = get_labels(kmeans_total,ids_with_ados_total)
corr(ids_labels_total,ados_total)
x = 0


cnt = 0
for i in range(num_areas):
    for j in range(i):
        conn_vect = []
        for cnt in range(len(ids_with_ados_total)):
            conn_mat = ids_conns[ids_with_ados_total[cnt]]
            conn_vect.append(np.std(conn_mat[i,j,:]))
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
            conn_vect.append(np.std(conn_mat[i, j, :]))
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
            conn_vect.append(np.std(conn_mat[i, j, :]))
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
            conn_vect.append(np.std(conn_mat[i, j, :]))
        r,p =sc.pearsonr(conn_vect,ados_soc)
        if p<0.001:
            print(i+1,j+1,r,p,'soc')
x =0