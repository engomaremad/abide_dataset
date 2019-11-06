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
from functions.calc_FDR import calc_FDR
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from functions.get_percntile_ados_GLM import  calc_glm





import os
dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'
surrogates_path = '/media/biolab/easystore1/abide_dataset_fmri/dfc_surrogates'

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
id_age = np.load(os.path.join(dict_path,'id_age.npy')).item()
viq_dict = np.load(os.path.join(dict_path, 'id_viq.npy')).item()
fiq_dict = np.load(os.path.join(dict_path, 'id_fiq.npy')).item()
piq_dict = np.load(os.path.join(dict_path, 'id_piq.npy')).item()
#labelsnp.load(os.path.join(dict_path,'id_ados_rel.npy')).item()
labels_dict = np.load(os.path.join(dict_path,'id_label.npy')).item()
labels = np.zeros([len(ids),1])
ages = []
piq = []
viq =[]
fiq =[]
cnt = 0
for id in ids:
    labels[cnt]= labels_dict[id]
    ages.append(id_age[id])
    viq.append(viq_dict[id])
    piq.append(piq_dict[id])
    fiq.append(fiq_dict[id])
    cnt+=1
with open('AAL_regions.txt','r') as f:
    aal = f.readlines()




ids_with_ados_total = []
ados_total = []

cnt = 0
# correlation analysis
percentiles = np.zeros([num_areas,num_areas,num_subjs,4])
def write2file(corr_res,type,cor_p,rej, feat_num):
    p = corr_res[:, 3]
    n_tests = len(p)
    idx = np.argsort(p)
    corr_res = corr_res[idx, :]
    for i in range(n_tests):
        if cor_p[i]<1:
            with open('res_corrected.txt', 'a') as f:
                f.writelines(str(corr_res[i, 0]) + '\t' + str(corr_res[i, 1]) + '\t' + str(corr_res[i, 2]) + '\t' + str(
                    corr_res[i, 3])+'\t'+str(corr_res[i, 4]) + '\t'+str(corr_res[i, 5])  +'\t'+str(cor_p[i])+ '\t'+ str(rej[i])+'\t'+type + '\t'+str(feat_num)+'\n')


if True:
    if not os.path.isfile(os.path.join(surrogates_path,'percentiles.npy')):
        for id in ids:
            tcs = tcs_subjs[id]
            stationary_conn = create_sfc(num_areas,tcs)
            tcs_with_surrogates = np.zeros([np.shape(tcs)[0],np.shape(tcs)[1],n_surrogates+1])
            #get surrogates for each area
            for i in range(np.shape(tcs)[1]):
                surrogates = gen_surrogate(tcs[:,i],n_surrogates)
                tcs_with_surrogates[:,i,0] = tcs[:,i]
                tcs_with_surrogates[:, i, 1:] = surrogates
                tmp =(tcs_with_surrogates[:, i,0]-np.min(tcs_with_surrogates[:, i,0]))/(np.max(tcs_with_surrogates[:, i,0])-np.min(tcs_with_surrogates[:, i,0]))
                #fig = plt.figure()

                plt.plot(range(len(tcs[:,i])), tmp,linewidth=3.0)
                plt.plot(range(len(tcs[:,i])), tcs_with_surrogates[:, i,1],linewidth=3.0)
                plt.plot(range(len(tcs[:,i])), tcs_with_surrogates[:, i,15],linewidth=3.0)
                plt.plot(range(len(tcs[:, i])), tcs_with_surrogates[:, i,30],linewidth=3.0)
                plt.plot(range(len(tcs[:, i])), tcs_with_surrogates[:, i, 45], linewidth=3.0)
                plt.plot(range(len(tcs[:, i])), tcs_with_surrogates[:, i, 49], linewidth=3.0)
                plt.legend(['Normaized initail TC', 'Surrogate 1','Surrogate 15','Surrogate 30','Surrogate 45','Surrogate 50'],prop={'size': 22})
                plt.xlabel('Time',fontsize=38)
                plt.ylabel('Normalized BOLD Signal Value',fontsize=38)
                plt.xticks(fontsize= 30)
                plt.yticks(fontsize=30)
                plt.show()
                x = 0
            if not os.path.isfile(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy')):
                surrogate_dfcs = dfc_with_win(num_areas,win_size,shift,window,tcs_with_surrogates[:,:,1:])
                np.save(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy'),surrogate_dfcs)
            else:
                surrogate_dfcs = np.load(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy'))
                original_dfc  = dfc_with_win(num_areas,win_size,shift,window,tcs_with_surrogates[:,:,0:1])[0]
            # plt.plot(range(len(original_dfc[5,3,:])), original_dfc[5,3,:],linewidth=3.0)
            # plt.plot(range(len(original_dfc[5,3,:])), surrogate_dfcs[1, 5,3, :],linewidth=3.0)
            # plt.plot(range(len(original_dfc[5, 3, :])), surrogate_dfcs[15, 5, 3, :], linewidth=3.0)
            # plt.plot(range(len(original_dfc[5,3,:])), surrogate_dfcs[30, 5,3, :],linewidth=3.0)
            # plt.plot(range(len(original_dfc[5,3,:])), surrogate_dfcs[45, 5,3, :],linewidth=3.0)
            # plt.plot(range(len(original_dfc[5, 3, :])), surrogate_dfcs[49, 5, 3, :], linewidth=3.0)
            # plt.legend(['Original DFC', 'Surrogate 1 DFC', 'Surrogate 15 DFC', 'Surrogate 30 DFC'
            #                , 'Surrogate 45 DFC', 'Surrogate 50 DFC'], prop={'size': 22})
            # plt.xlabel('DFC Window Steps', fontsize=38)
            # plt.xticks(fontsize=30)
            # plt.yticks(fontsize=30)
            # plt.ylabel('DFC Values', fontsize=38)
            # plt.show()
            x = 0
            percentiles[:,:,cnt,:] = get_percentile(num_areas,original_dfc,surrogate_dfcs)
            cnt = cnt+1

            print(cnt)


        np.save(os.path.join(surrogates_path,'percentiles'),percentiles)
    else:
        percentile_feats_dict = dict()
        percentiles =np.load(os.path.join(surrogates_path,'percentiles.npy'))
        cnt = 0
        for id in ids:
            percentile_feats_dict[id] = percentiles[:,:,cnt,:]
            cnt+=1
        np.save(os.path.join(surrogates_path, 'percentiles_dict'), percentile_feats_dict)
        ados_comm_used =[]
        ados_tot_used =[]
        ados_soc_used =[]
        ados_beh_used =[]
        used_ids = []
        ados_age= []
        used_viq =[]
        used_piq = []
        used_fiq = []
        used_feat = 3
        used_ados = [4]
        fdr= 0.1
        print('comm')
        for i in  range(num_subjs):
            if id_ados_mod[ids[i]] in used_ados and id_ados_rel[ids[i]] == 1 and id_ados_comm[ids[i]] > 0:
                ados_comm_used.append(id_ados_comm[ids[i]])
                ados_age.append(ages[i])
                used_ids.append(i)
                used_viq.append(viq[i])

                used_piq.append(piq[i])
                used_fiq.append(fiq[i])
        #corr_res= calc_corr(ados_age,'comm',num_areas,percentiles[:,:,used_ids,:],used_feat)
        corr_res= calc_corr(ados_comm_used,'comm',num_areas,percentiles[:,:,used_ids,:],used_feat,ados_age,used_viq,used_piq,used_fiq)
        #corr_res= calc_glm(ados_comm_used,'comm',num_areas,percentiles[:,:,used_ids,:],used_feat,ados_age,used_viq,used_piq,used_fiq)

        cor_p, rej = calc_FDR(corr_res, fdr)
        write2file(corr_res, 'communication', cor_p, rej, used_feat)
        used_ids = []
        ados_age = []
        used_viq = []
        used_piq = []
        used_fiq = []
        print('tot')
        for i in range(num_subjs):
            if id_ados_mod[ids[i]] in used_ados and id_ados_rel[ids[i]] == 1 and id_ados_tot[ids[i]] > 0:
                ados_tot_used.append(id_ados_tot[ids[i]])
                used_ids.append(i)
                ados_age.append(ages[i])
                used_viq.append(viq[i])
                used_piq.append(fiq[i])
                used_fiq.append(fiq[i])
        corr_res = calc_corr(ados_tot_used, 'tot', num_areas, percentiles[:, :, used_ids, :], used_feat,ados_age,used_viq,used_piq,used_fiq)
        #corr_res= calc_glm(ados_tot_used,'comm',num_areas,percentiles[:,:,used_ids,:],used_feat,ados_age,used_viq,used_piq,used_fiq)

        cor_p,rej=calc_FDR(corr_res, fdr)
        write2file(corr_res, 'total',cor_p,rej,used_feat)
        used_ids = []
        ados_age = []
        used_viq = []
        used_piq = []
        used_fiq = []
        print('soc')
        for i in range(num_subjs):
            if id_ados_mod[ids[i]] in used_ados and id_ados_rel[ids[i]] == 1 and id_ados_soc[ids[i]] > 0:
                ados_soc_used.append(id_ados_soc[ids[i]])
                used_ids.append(i)
                ados_age.append(ages[i])
                used_viq.append(viq[i])
                used_piq.append(fiq[i])
                used_fiq.append(fiq[i])
        corr_res = calc_corr(ados_soc_used, 'soc', num_areas, percentiles[:, :, used_ids, :], used_feat,ados_age,used_viq,used_piq,used_fiq)
        #corr_res = calc_glm(ados_soc_used, 'soc', num_areas, percentiles[:, :, used_ids, :], used_feat,ados_age,used_viq,used_piq,used_fiq)

        cor_p, rej = calc_FDR(corr_res, fdr)
        write2file(corr_res, 'social', cor_p, rej, used_feat)
        used_ids = []
        ados_age = []
        used_viq = []
        used_piq = []
        used_fiq = []
        print('beh')
        for i in range(num_subjs):
            if id_ados_mod[ids[i]] in used_ados and id_ados_rel[ids[i]] == 1 and id_ados_beh[ids[i]] > 0:
                ados_beh_used.append(id_ados_beh[ids[i]])
                used_ids.append(i)
                ados_age.append(ages[i])
                used_viq.append(viq[i])
                used_piq.append(fiq[i])
                used_fiq.append(fiq[i])
        corr_res = calc_corr(ados_beh_used, 'beh', num_areas, percentiles[:, :, used_ids, :], used_feat,ados_age,used_viq,used_piq,used_fiq)
        #corr_res = calc_glm(ados_beh_used, 'beh', num_areas, percentiles[:, :, used_ids, :], used_feat,ados_age,used_viq,used_piq,used_fiq)

        cor_p, rej = calc_FDR(corr_res, fdr)
        write2file(corr_res, 'behavior', cor_p, rej, used_feat)
        x=0







# #classification analysis
# percentile_feats = np.zeros([num_areas,num_areas,num_subjs,4])
# if not os.path.isfile(os.path.join(surrogates_path,'percentiles_dict.npy')):
#     for id in ids:
#         tcs = tcs_subjs[id]
#         stationary_conn = create_sfc(num_areas,tcs)
#         tcs_with_surrogates = np.zeros([np.shape(tcs)[0],np.shape(tcs)[1],n_surrogates+1])
#         #get surrogates for each area
#         for i in range(np.shape(tcs)[1]):
#             surrogates = gen_surrogate(tcs[:,i],n_surrogates)
#             tcs_with_surrogates[:,i,0] = tcs[:,i]
#             tcs_with_surrogates[:, i, 1:] = surrogates
#         if not os.path.isfile(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy')):
#             surrogate_dfcs = dfc_with_win(num_areas,win_size,shift,window,tcs_with_surrogates[:,:,1:])
#             np.save(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy'),surrogate_dfcs)
#         else:
#             surrogate_dfcs = np.load(os.path.join(surrogates_path,'dfcs_surrogates_'+id+'.npy'))
#
#         original_dfc  = dfc_with_win(num_areas,win_size,shift,window,tcs_with_surrogates[:,:,0:1])[0]
#         percentile_feats[:,:,cnt,:] = get_percentile_feats(num_areas,original_dfc,surrogate_dfcs)
#         cnt = cnt+1
#
#         print(cnt)
#     for id in ids:
#
#         percentile_feats_dict[id] = percentiles[:,:,cnt,:]
#         cnt+=1
#     np.save(os.path.join(surrogates_path, 'percentiles_dict'), percentile_feats_dict)
# else:
#     percentiles_dict = np.load(os.path.join(surrogates_path, 'percentiles_dict.npy')).item()
#     ados_soc = []
#     # get the ids that have ados total
#
#     for id in ids:
#         if id_ados_mod[id] in used_ados and id_ados_rel[id] == 1 and id_ados_tot[id] >= 0:
#             ids_with_ados_total.append(id)
#             ados_total.append(id_ados_tot[id])
#     used_percntiles = np.zeros((num_areas,num_areas,len(ados_total),4))
#     cnt =0
#     for id in ids:
#         if id in ids_with_ados_total:
#             used_percntiles[:,:,cnt,:] = percentiles_dict[id]
#             cnt+=1
#
#     #classify_ados_tot(ados_total, 'tot', num_areas, used_percntiles)
#     corr_res = calc_corr(ados_total, 'tot', num_areas, used_percntiles,3)
#     calc_FDR(corr_res,0.2)
#     x = 0
#
#
