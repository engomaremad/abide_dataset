#TODO : Read the excel cohort excel file and create the corresponding data dictionaries
# For full legend please visit : http://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Data_Legend.pdf
import xlrd
import pandas as pd
import numpy as np
import os

a,b = [3,4]
sheet = pd.read_excel('abide_phenotype.xlsx')
cols = [ 'FILE_ID','AGE_AT_SCAN','DX_GROUP','SEX','HANDEDNESS_CATEGORY','FIQ','VIQ','PIQ', 'ADOS_MODULE', 'ADOS_TOTAL','ADOS_COMM', \
    'ADOS_SOCIAL','ADOS_STEREO_BEHAV','ADOS_RSRCH_RELIABLE']



mat = sheet[cols].as_matrix()
[file_id, age,label, sex, hand, fiq,viq,piq,ados_module,ados_total,ados_comm,ados_social,ados_behav ,ados_reliable] = np.split(mat,np.shape(mat)[1],1)

fiq = fiq.astype(float)
viq = viq.astype(float)
piq = piq.astype(float)
ados_module = ados_module.astype(float)
ados_total = ados_total.astype(float)
ados_comm = ados_comm.astype(float)
ados_social = ados_social.astype(float)
ados_behav = ados_behav.astype(float)
ados_reliable = ados_reliable.astype(float)

fiq[np.isnan(fiq)]= -8888
viq[np.isnan(viq)]= -8888
piq[np.isnan(piq)]= -8888
ados_module[np.isnan(ados_module)]= -8888
ados_total[np.isnan(ados_total)]= -8888
ados_comm[np.isnan(ados_comm)]= -8888
ados_social[np.isnan(ados_social)]= -8888
ados_behav[np.isnan(ados_behav)]= -8888
ados_reliable[np.isnan(ados_reliable)]= -8888
used_names = []
id_age = dict()
# 1 for ASD 2 for TD
id_label = dict()
# 1 male 2 female
id_sex = dict()
# 1 L 2 R 3 Mixed 4 unknown
id_hand = dict()
# -9999 unknown -8888 missing
id_fiq = dict()
id_viq = dict()
id_piq = dict()
# -9999 unknown -8888 missing
id_ados_mod = dict()
id_ados_tot = dict()
id_ados_comm = dict()
id_ados_soc = dict()
id_ados_beh = dict()
# 1 yes 0 no
id_ados_rel = dict()
for i in range(len(file_id)):
    if file_id[i] !=  'no_filename':
        id_age[file_id[i][0]] = float(age[i])
        id_label[file_id[i][0]] = int(label[i])
        id_sex[file_id[i][0]] =  int(sex[i])
        if hand[i] == 'L':
            id_hand[file_id[i][0]] = 1
        elif hand[i] =='R':
            id_hand[file_id[i][0]] = 2
        elif hand[i] == 'mixed':
            id_hand[file_id[i][0]] = 3
        else:
            id_hand[file_id[i][0]] =4
        id_fiq[file_id[i][0]]  = float(fiq[i])
        id_viq[file_id[i][0]] = float(viq[i])
        id_piq[file_id[i][0]] = float(piq[i])
        id_ados_mod[file_id[i][0]] = float(ados_module[i])
        id_ados_beh[file_id[i][0]] = float(ados_behav[i])
        id_ados_mod[file_id[i][0]] = float(ados_module[i])
        id_ados_comm[file_id[i][0]] = float(ados_comm[i])
        id_ados_tot[file_id[i][0]] = float(ados_total[i])
        id_ados_rel[file_id[i][0]] = float(ados_reliable[i])
        id_ados_soc[file_id[i][0]] = float(ados_social[i])
        used_names.append(file_id[i][0])
os.chdir('Data_dictionaries')
np.save('id_age',id_age)
np.save('id_label',id_label)
np.save('id_sex',id_sex)
np.save('id_hand',id_hand)
np.save('id_fiq',id_fiq)
np.save('id_viq',id_viq)
np.save('id_piq',id_piq)
np.save('id_ados_mod',id_ados_mod)
np.save('id_ados_comm',id_ados_comm)
np.save('id_ados_tot',id_ados_tot)
np.save('id_ados_beh',id_ados_beh)
np.save('id_ados_rel',id_ados_rel)
np.save('id_age',id_age)
np.save('id_ados_soc',id_ados_soc)
np.save('used_names',used_names)


x = 0
