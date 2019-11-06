#TODO calculte conditioanal funcitonal connectivity as explained in
# "Dynamic fluctuations coincide with periods of high and low modularity in resting-state functional brain networks"
import numpy as np
from functions.get_ids_having_ados import get_ids_with_ados
from functions.get_id_tcs import ids_tcs
from functions.generate_surrogates import gen_surrogate
from functions.create_static_conn_matrix import create_sfc
from functions.dfc_tappered_win import  dfc_with_win
from functions.dfc_to_percntile import get_percentile
from functions.get_percntile_ados_corr import calc_corr
from functions.get_percntile_feats import get_percentile_feats
from functions.classify_ados_total import classify_ados_tot
from functions.classify_asd_td_cdfc import classify_asd_td
from timeit import default_timer as timer


import os
dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'
surrogates_path = '/media/biolab/easystore/abide_dataset_fmri/dfc_surrogates'
ados_module = 3
#ids = get_ids_with_ados(ados_module,dict_path)
ids = np.load(os.path.join(dict_path,'all_ids.npy'))

tcs_subjs = ids_tcs(ids)
n_surrogates = 50
num_areas = 116
num_subjs = np.size(ids)
win_size = 30
shift = 3
window = np.load(os.path.join(dict_path,'hamming.npy'))
cnt = 0
id_ados_mod  = np.load(os.path.join(dict_path,'id_ados_mod.npy')).item()
id_ados_tot  = np.load(os.path.join(dict_path,'id_ados_tot.npy')).item()
id_ados_comm  = np.load(os.path.join(dict_path,'id_ados_comm.npy')).item()
id_ados_soc = np.load(os.path.join(dict_path,'id_ados_soc.npy')).item()
id_ados_beh = np.load(os.path.join(dict_path,'id_ados_beh.npy')).item()
id_ados_rel  = np.load(os.path.join(dict_path,'id_ados_rel.npy')).item()
#labels
labels_dict = np.load(os.path.join(dict_path,'id_label.npy')).item()
labels = np.zeros([len(ids),1])
cnt = 0
for id in ids:
    labels[cnt]= labels_dict[id]
    cnt+=1




used_ados = 4
ids_with_ados_total = []
ados_total = []

cnt = 0
# correlation analysis
percentiles = np.zeros([num_areas,num_areas,num_subjs])
percentile_feats_dict = dict()
percentile_feats = np.zeros([num_areas,num_areas,num_subjs,4])
if not os.path.isfile(os.path.join(surrogates_path,'percentiles_dict.npy')):
    for id in ids:
        tcs = tcs_subjs[id]
        stationary_conn = create_sfc(num_areas,tcs)
        tcs_with_surrogates = np.zeros([np.shape(tcs)[0],np.shape(tcs)[1],n_surrogates+1])
        #get surrogates for each area
        for i in range(np.shape(tcs)[1]):
            surrogates = gen_surrogate(tcs[:,i],n_surrogates)
            tcs_with_surrogates[:,i,0] = tcs[:,i]
            tcs_with_surrogates[:, i, 1:] = surrogates
        if not os.path.isfile(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy')):
            surrogate_dfcs = dfc_with_win(num_areas,win_size,shift,window,tcs_with_surrogates[:,:,1:])
            np.save(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy'),surrogate_dfcs)
        else:
            surrogate_dfcs = np.load(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy'))

        original_dfc  = dfc_with_win(num_areas,win_size,shift,window,tcs_with_surrogates[:,:,0:1])[0]
        percentile_feats[:,:,cnt,:] = get_percentile_feats(num_areas,original_dfc,surrogate_dfcs)
        cnt = cnt+1

        print(cnt)
    for id in ids:
        percentile_feats_dict[id] = percentiles[:,:,cnt,:]
        cnt+=1
    np.save(os.path.join(surrogates_path, 'percentiles_dict'), percentile_feats_dict)
else:
    percentiles_dict = np.load(os.path.join(surrogates_path, 'percentiles_dict.npy')).item()
    ados_soc = []
    # get the ids that have ados total

    for id in ids:
        if id_ados_mod[id] == used_ados and id_ados_rel[id] == 1 and id_ados_tot[id] >= 0:
            ids_with_ados_total.append(id)
            ados_total.append(id_ados_tot[id])
    used_percntiles = np.zeros((num_areas,num_areas,len(ados_total),4))
    cnt =0
    for id in ids:
        if id in ids_with_ados_total:
            used_percntiles[:,:,cnt,:] = percentiles_dict[id]
            cnt+=1
    ados_total = np.asarray(ados_total)
    classify_ados_tot(ados_total, 'tot', num_areas, used_percntiles)

