#Todo classify all abide dataset into ASD- TD based on stationary functional connectivity on the same way as Frontiers papers
import numpy as np
import os
import matplotlib.pyplot as py
from sklearn.ensemble import  RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from functions.visualize_tsne import visualize
import scipy.stats as sc
def feats_fusion(feats,labels):

    kf = KFold(n_splits=4)
    labels = np.ravel(labels)
    res_final = np.zeros([len(labels)])
    for train_index, test_index in kf.split(feats):
        #clf_cal = MLPClassifier(solver='sgd',alpha=1e-6, hidden_layer_sizes=(20), random_state=1)
        #clf_cal = GaussianNB()
        clf_cal =RandomForestClassifier(n_estimators=200,max_depth=4,random_state=0)
        clf_cal.fit(feats[train_index], labels[train_index])
        res= clf_cal.predict_proba(feats[test_index])
        res_final[test_index] = res[:,1]

       # res_final[test_index] =clf_cal.predict(feats[test_index])
    auc = roc_auc_score(labels-1, res_final)
    res_final = np.round(res_final)+1
    tn, fp, fn, tp = confusion_matrix(res_final, labels).ravel()
    sens = (tp)/(fn+tp)
    spec = ((tn)/(tn+fp))
    overall_acc = accuracy_score(res_final,labels)
    return overall_acc,sens,spec,auc



def local_classify_kfold(feats,labels):
    # split to 4 folds in order of data to make sure same split is applied in fusion
    # this prevent any possible data leakage
    kf = KFold(n_splits=4)
    feats = feats.reshape(-1,1)
    labels = np.ravel(labels)
    res_all_subj = np.zeros(np.shape(labels))

    #initialize  KNN
    clf_cal = KNeighborsClassifier(n_neighbors=5)
    #clf_cal = SVC()
    # Fit on training and predict on testing
    for train_index, test_index in kf.split(feats):
        clf_cal.fit(feats[train_index],labels[train_index])
        res = clf_cal.predict(feats[test_index])
        res_all_subj[test_index]= res
    x=0
    acc_local = accuracy_score(res_all_subj,labels)
    t,p = sc.ttest_ind(feats[np.where(labels==1)],feats[np.where(labels==2)])
    return acc_local,res_all_subj

















dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'


sfc = np.load(os.path.join(dict_path,'ids_conns_mats.npy')).item()
dfc = np.load(os.path.join(dict_path,'id_dfc_tapered_shift_1.npy')).item()
labels_dict = np.load(os.path.join(dict_path,'id_label.npy')).item()
ages_dict = np.load(os.path.join(dict_path,'id_age.npy')).item()
good_idx = np.load(os.path.join(dict_path,'good_idx.npy'))
key_words= ['UCLA','NYU','Leuven','CMU','Caltech','MaxMun',
'Olin','KKI','OHSU','Pitt','SBL','SDSU','Stanford','Trinity','UM_','USM','Yale']

for key_word in key_words:
    ids_used = []
    ids = np.load(os.path.join(dict_path, 'all_ids.npy'))
    print(key_word)
    for id in ids:
        if key_word in id :
            ids_used.append(id)
    ids = ids_used
    num_subjs = len(ids)
    num_areas =116
    conn_mat = np.zeros([num_subjs,num_areas,num_areas])
    conn_mat_d = np.zeros([num_subjs,num_areas,num_areas])
    cnt = 0
    labels =np.zeros([num_subjs,1])
    ages =np.zeros([num_subjs,1])
    for key in ids:
        conn_mat[cnt, :, :] = sfc[key]
        conn_mat_d[cnt, :, :] = np.std(dfc[key],2)
        labels[cnt] = labels_dict[key]
        cnt+=1

    #conn_mat_d = np.random.random(np.shape(conn_mat))
    global_feats = np.zeros([num_subjs,])
    np.random.shuffle(labels)
    local_accs= [0]
    used_areas = []
    if not os.path.exists(os.path.join(dict_path,'fmri_sorted.npy')) or not os.path.exists(os.path.join(dict_path,'fmri_global_props.npy')) or True :
        for i in range(num_areas):
            for j in range(i):
                feats  = conn_mat_d[:,i,j]
                feats[np.isnan(feats)] = 0
                acc_local , probs = local_classify_kfold(feats, labels)
                global_feats = np.column_stack((global_feats,feats))
                local_accs.append(acc_local)
                used_areas.append((i,j))
        py.plot(range(len((local_accs))),local_accs)
        sorted = np.argsort(local_accs)
        sorted =sorted[::-1]
        x=0
        np.save(os.path.join(dict_path, 'fmri_sorted.npy'), sorted)
        np.save(os.path.join(dict_path, 'fmri_global_props.npy'), global_feats)
    else:
        sorted = np.load(os.path.join(dict_path,'fmri_sorted.npy'))
        global_feats = np.load(os.path.join(dict_path,'fmri_global_props.npy'))



    for i in range(10):
        f = open(os.path.join(dict_path,key_word+'.txt'),'a')
        f.writelines(str(local_accs[sorted[i]])+str(used_areas[sorted[i]]) +'\n')
        f.close()


    x=0
    overall_acc=np.zeros([1,100])
    overall_sens=np.zeros([1,100])
    overall_spec=np.zeros([1,100])
    overall_auc = np.zeros([1,100])
    acc =0
    best_idx =0
    for i in range(1,101):
        global_feats_selected   = global_feats[:,sorted[0:i]]
        overall_acc_tmp ,sens_tmp,spec_tmp,auc_tmp= feats_fusion(global_feats_selected, labels)
        overall_acc[0, i - 1] = overall_acc_tmp
        overall_sens[0, i - 1] = sens_tmp
        overall_spec[0, i - 1] = spec_tmp
        overall_auc[0, i - 1] = auc_tmp
        if overall_acc_tmp > acc:
            best_idx = i-1
            #print(i,overall_acc[0,i-1],overall_sens[0,i-1],overall_spec[0,i-1],overall_auc[0,i-1])
            x = 0
            acc = overall_acc[0,i-1]
        x =0
    print(best_idx, overall_acc[0, best_idx], overall_sens[0, best_idx], overall_spec[0, best_idx], overall_auc[0, best_idx])

    # visualize(global_feats[:,sorted[0:best_idx]],labels)
    # py.plot(range(100),overall_acc[0,:],)
    # py.plot(range(100), overall_sens[0,:])
    # py.plot(range(100), overall_spec[0,:])
    # py.plot(range(100), overall_auc[0,:])
    # py.ylabel('result')
    # py.xlabel('number of features')
    # py.legend(['accuracy','sens','spec','auc'])
    # x=0
    # py.show()
    #











    x= 0