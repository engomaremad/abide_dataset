import pandas as pd
import requests
import os

save_dir = '/media/biolab/easystore/abide_dataset_fmri/craddock_atlas/'
data = pd.read_excel('abide_phenotype.xlsx')
subject_Id = data['subject'].tolist()

pre_processing = 'ccs'
global_filter = 'filt_global'
dervative = 'rois_cc200'
for s in subject_Id:
    fid = data.loc[data['subject']==s]['FILE_ID'].tolist()[0]
    x =0
    myurl = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/'+pre_processing+'/'+global_filter+'/'+dervative+'/'+fid+'_'+dervative+'.1D'
   # myurl = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/OHSU_0050147_func_minimal.nii.gz'

    f = open('log_dr.txt', 'a')
    try:
        r =requests.get(myurl, allow_redirects=True)
        path = os.path.join(save_dir,fid+'.txt')
        open(path, 'wb').write(r.content)
        f.writelines(fid +'\t\t'+'Passed\n')
    except:
        f.writelines(fid + '\t\t' + 'Failed\n')
    f.close()
x=0