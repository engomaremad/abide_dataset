#TODO convert the AAL atlas labels to YEO functional atlas

import os
import numpy as np
import nibabel as nib
import nilearn as ni
dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'


aal_atlas = ni.datasets.fetch_atlas_aal(version='SPM12', data_dir=None, url=None, resume=True, verbose=1)
yeo_labels = dict()
yeo_labels =['visual','somatomotor','dorsal_attention','ventral_attention','limbic','frontoparietal','default']
np.save(os.path.join('yeo_labels'),yeo_labels)

num_yeo_areas = 7
num_aal_areas = 116
yeo_img = nib.load(os.path.join(dict_path,'YEO_7_nws_FSL.nii.gz'))
yeo = yeo_img.get_data()
aal_img = nib.load(aal_atlas.maps)
aal = aal_img.get_data()
intersection =np.zeros([num_yeo_areas,num_aal_areas])
for i in range(np.shape(aal)[0]):
    for j in range(np.shape(aal)[1]):
        for k in range(np.shape(aal)[2]):
            a = yeo[i,j,k]
            if str(aal[i,j,k]) in aal_atlas.indices:
                b = aal_atlas.indices.index(str(aal[i,j,k]))
            else:
                b= 0
            if a>0 and b>0:
                intersection[a-1,b-1]+=1

aal_2_yeo = np.zeros([num_aal_areas,1])
for i in range(num_aal_areas):
    tmp = intersection[:,i]
    print(i)
    if np.sum(tmp)>0:
        aal_2_yeo[i] = np.where(tmp==np.max(tmp))[0][0]+1
        aal_2_yeo_f = open('dk_2yeo.txt','a')
        aal_2_yeo_f.writelines(yeo_labels[int(aal_2_yeo[i]-1)]+'\n')
        aal_2_yeo_f .close()
np.save(os.path.join(dict_path,'aal_2_yeo'),aal_2_yeo)

x=0