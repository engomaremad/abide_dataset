#TODO: get cohort summary stats

import numpy as np
import os
import pandas as pd
import scipy.stats as sc
dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'

ages_dict = np.load(os.path.join(dict_path, 'id_age.npy')).item()
sex_dict = np.load(os.path.join(dict_path, 'id_sex.npy')).item()
viq_dict = np.load(os.path.join(dict_path, 'id_viq.npy')).item()
fiq_dict = np.load(os.path.join(dict_path, 'id_fiq.npy')).item()
piq_dict = np.load(os.path.join(dict_path, 'id_piq.npy')).item()
hand_dict = np.load(os.path.join(dict_path, 'id_hand.npy')).item()
ids = np.load(os.path.join(dict_path, 'all_ids.npy'))
labels_dict = np.load(os.path.join(dict_path, 'id_label.npy')).item()

def save_stats(site):


    age_asd =[]
    sex_asd =[]
    viq_asd = []
    fiq_asd = []
    piq_asd = []
    hand_asd = []
    age_td = []
    sex_td = []
    viq_td = []
    fiq_td = []
    piq_td = []
    hand_td = []
    for id in ids:
        if site in id:
            if labels_dict[id] == 1:
                age_asd.append(ages_dict[id])
                sex_asd.append(sex_dict[id])
                viq_asd.append(viq_dict[id])
                fiq_asd.append(fiq_dict[id])
                piq_asd.append(piq_dict[id])
                hand_asd.append(hand_dict[id])
                age_asd[:]  = [item for item in age_asd if item >= 0]
                viq_asd[:]  = [item for item in viq_asd if item >= 0]
                piq_asd[:]  = [item for item in piq_asd if item >= 0]
                fiq_asd[:]  = [item for item in fiq_asd if item >= 0]
                if len(viq_asd) ==0 :
                    viq_asd=[0]
                if len(piq_asd) ==0 :
                    piq_asd=[0]
                if len(fiq_td) == 0:
                    fiq_asd = [0]
            else:
                age_td.append(ages_dict[id])
                sex_td.append(sex_dict[id])
                viq_td.append(viq_dict[id])
                fiq_td.append(fiq_dict[id])
                piq_td.append(piq_dict[id])
                hand_td.append(hand_dict[id])
                age_td[:] = [item for item in age_td if item >= 0]
                viq_td[:] = [item for item in viq_td if item >= 0]
                piq_td[:] = [item for item in piq_td if item >= 0]
                fiq_td[:] = [item for item in fiq_td if item >= 0]
                if len(viq_td) ==0 :
                    viq_td=[0]
                if len(piq_td) ==0 :
                    piq_td=[0]
                if len(fiq_td) == 0:
                    fiq_td = [0]
    t,p_age = sc.ttest_ind(age_asd,age_td)
    t, p_viq = sc.ttest_ind(viq_asd, viq_td)
    t, p_piq = sc.ttest_ind(piq_asd, piq_td)
    t, p_fiq = sc.ttest_ind(fiq_asd, fiq_td)
    with open('stats.txt','a') as f :
        f.writelines('\n\t\t'+site+ '\n')
        f.writelines('\t\t ASD \n')
        f.writelines('males \t '+  str(sex_asd.count(1)) +'\t females \t ' + str(sex_asd.count(2)))
        f.writelines('\n\tmin\tmax\tmean\tstd\n')
        f.writelines('age\t'+str(np.min(age_asd)) +'\t'+ str(np.max(age_asd))+'\t'+str(np.mean(age_asd))+'\t'+str(np.std(age_asd)))
        f.writelines(
            '\nviq\t' + str(np.min(viq_asd)) + '\t' + str(np.max(viq_asd)) + '\t' + str(np.mean(viq_asd)) + '\t' + str(
                np.std(viq_asd)))
        f.writelines(
            '\nfiq\t' + str(np.min(fiq_asd)) + '\t' + str(np.max(fiq_asd)) + '\t' + str(np.mean(fiq_asd)) + '\t' + str(
                np.std(fiq_asd)))
        f.writelines(
            '\nviq\t' + str(np.min(piq_asd)) + '\t' + str(np.max(piq_asd)) + '\t' + str(np.mean(piq_asd)) + '\t' + str(
                np.std(piq_asd)))
        f.writelines('\n\t\t TD \n')
        f.writelines('males \t ' + str(sex_td.count(1)) + '\t females \t ' + str(sex_td.count(2)))
        f.writelines('\n\tmin\tmax\tmean\tstd\n')
        f.writelines(
            'age\t' + str(np.min(age_td)) + '\t' + str(np.max(age_td)) + '\t' + str(np.mean(age_td)) + '\t' + str(
                np.std(age_td)))
        f.writelines(
            '\nviq\t' + str(np.min(viq_td)) + '\t' + str(np.max(viq_td)) + '\t' + str(np.mean(viq_td)) + '\t' + str(
                np.std(viq_td)))
        f.writelines(
            '\nfiq\t' + str(np.min(fiq_td)) + '\t' + str(np.max(fiq_td)) + '\t' + str(np.mean(fiq_td)) + '\t' + str(
                np.std(fiq_td)))
        f.writelines(
            '\npiq\t' + str(np.min(piq_td)) + '\t' + str(np.max(piq_td)) + '\t' + str(np.mean(piq_td)) + '\t' + str(
                np.std(piq_td)))

sites = ['UCLA','NYU','Leuven','Caltech','MaxMun','Olin','KKI','OHSU','Pitt','SBL','SDSU','Stanford','Trinity','UM_','USM','Yale']
for site in sites:
    save_stats(site)
    x = 0