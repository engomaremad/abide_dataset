#TODO get ids,SFC,DFC for a study, use study_id = 'ALL' to get all subjects in ABIDE dataset
import os
import numpy as np
def get_dfc_sfc(study_id):
    dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'

    sfc = np.load(os.path.join(dict_path, 'ids_conns_mats.npy')).item()
    dfc = np.load(os.path.join(dict_path, 'id_dfc_tapered.npy')).item()
    labels_dict = np.load(os.path.join(dict_path, 'id_label.npy')).item()
    ages_dict = np.load(os.path.join(dict_path, 'id_age.npy')).item()
    good_idx = np.load(os.path.join(dict_path, 'good_idx.npy'))
    ids = np.load(os.path.join(dict_path, 'all_ids.npy'))
    ids_used = []
    if study_id == 'ALL':
        ids_used = ids
    else:
        for id in ids:
            if study_id in id:
                ids_used.append(id)
    ids = ids_used
    num_subjs = len(ids)
    num_areas = 116
    conn_mat = np.zeros([num_subjs, num_areas, num_areas])
    conn_mat_d = np.zeros([num_subjs, num_areas, num_areas])
    cnt = 0
    labels = np.zeros([num_subjs, 1])
    ages = np.zeros([num_subjs, 1])
    for key in ids:
        conn_mat[cnt, :, :] = sfc[key]
        conn_mat_d[cnt, :, :] = np.std(dfc[key], 2)
        labels[cnt] = labels_dict[key]
        cnt += 1
    return conn_mat,conn_mat_d,labels

conn_mat,conn_mat_d,labels = get_dfc_sfc('UCLA')
x=0