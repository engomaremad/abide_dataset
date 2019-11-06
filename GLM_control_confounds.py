'''
TODO: Create GLM matrix for all confounds to controll them while studying label- connecitivity matrix
'''
import numpy as np
import statsmodels.api as sm
import os
from sklearn.model_selection import KFold
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

def feats_fusion(feats,labels):

    kf = KFold(n_splits=4)
    labels = np.ravel(labels)
    res_final = np.zeros([len(labels)])
    for train_index, test_index in kf.split(feats):
        #clf_cal = MLPClassifier(solver='sgd',alpha=1e-6, hidden_layer_sizes=(50), random_state=1)
        #clf_cal = SVC(kernel='linear',probability=True,random_state=0)
        clf_cal =RandomForestClassifier(n_estimators=200,max_depth=4,random_state=0)
        clf_cal.fit(feats[train_index], labels[train_index])
        res= clf_cal.predict_proba(feats[test_index])
        res_final[test_index] = res[:,1]

       # res_final[test_index] =clf_cal.predict(feats[test_index])
    res_final = np.round(res_final)+1
    tn, fp, fn, tp = confusion_matrix(res_final, labels).ravel()
    sens = (tp)/(fn+tp)
    spec = ((tn)/(tn+fp))
    overall_acc = accuracy_score(res_final,labels)
    return overall_acc,sens,spec








def local_classify_kfold(feats,labels):
    # split to 4 folds in order of data to make sure same split is applied in fusion
    # this prevent any possible data leakage
    kf = KFold(n_splits=4)
    #feats = feats.reshape(-1,1)
    labels = np.ravel(labels)
    res_all_subj = np.zeros(np.shape(labels))

    #initialize  KNN
    #clf_cal = KNeighborsClassifier(n_neighbors=5)
    clf_cal = SVC()
    #clf_cal = GaussianNB()
    #clf_cal = MLPClassifier(solver='sgd',alpha=1e-6, hidden_layer_sizes=(5), random_state=1)

    # Fit on training and predict on testing
    for train_index, test_index in kf.split(feats):
        clf_cal.fit(feats[train_index,:],labels[train_index])
        res = clf_cal.predict(feats[test_index,:])
        res_all_subj[test_index]= res
    x=0
    acc_local = accuracy_score(res_all_subj,labels)
    return acc_local












def get_dfc_sfc(study_id):
    dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'

    sfc = np.load(os.path.join(dict_path, 'ids_conns_mats.npy')).item()
    dfc = np.load(os.path.join(dict_path, 'id_dfc_tapered.npy')).item()
    labels_dict = np.load(os.path.join(dict_path, 'id_label.npy')).item()
    ages_dict = np.load(os.path.join(dict_path, 'id_age.npy')).item()
    sex_dict = np.load(os.path.join(dict_path, 'id_sex.npy')).item()
    viq_dict = np.load(os.path.join(dict_path, 'id_viq.npy')).item()
    fiq_dict = np.load(os.path.join(dict_path, 'id_fiq.npy')).item()
    piq_dict = np.load(os.path.join(dict_path, 'id_piq.npy')).item()
    hand_dict = np.load(os.path.join(dict_path, 'id_hand.npy')).item()
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
    sex = np.zeros([num_subjs, 1])
    piq = np.zeros([num_subjs, 1])
    viq = np.zeros([num_subjs, 1])
    fiq = np.zeros([num_subjs, 1])
    hand = np.zeros([num_subjs, 1])
    for key in ids:
        conn_mat[cnt, :, :] = sfc[key]
        conn_mat_d[cnt, :, :] = np.std(dfc[key], 2)
        labels[cnt] = labels_dict[key]
        ages[cnt] = ages_dict[key]
        fiq[cnt] = fiq_dict[key]
        piq[cnt] = piq_dict[key]
        viq[cnt] = viq_dict[key]
        sex[cnt] = sex_dict[key]
        hand[cnt] = hand_dict[key]
        cnt += 1
    return conn_mat,conn_mat_d,labels,ages,fiq,piq,viq,sex,hand
conn_mat,conn_mat_d,labels,ages,fiq,piq,viq,sex,hand = get_dfc_sfc('ALL')
#conn_mat = np.random.random(np.shape(conn_mat))
#np.random.shuffle(labels)
const = np.ones([len(labels),1])
num_subjs = len(labels)
global_feats = np.zeros([num_subjs,])
num_areas = 116
cnt =0
n_feats = int(num_areas*(num_areas-1)*0.5)
alpha = 0.1
for i in range(num_areas):
    for j in range(i):
        y = labels
        connt_vect = np.zeros(np.shape(ages))
        connt_vect[:,0] = conn_mat[:, i, j]
        connt_vect[np.isnan(connt_vect)] = 0
        data_abide = np.concatenate((const, ages, fiq, piq, viq,sex,connt_vect), 1)
        gamma_model = sm.GLM(y, data_abide[:,:],family=sm.families.Gaussian())
        gamma_results = gamma_model.fit()
        if gamma_results.pvalues[6] <alpha/n_feats:
            cnt+=1
            print(str(cnt)+'\t'+str(i)+'\t'+ str(j))
            coef = gamma_results.params

            labels_pred  = coef[0] + coef[1]*ages + coef[2]*fiq +coef[3]*piq +coef[4]*viq + coef[5]*sex + coef[6]* connt_vect

            #feats = np.concatenate((ages,fiq,piq,viq,sex,connt_vect),1)
            #global_feats = np.column_stack((global_feats, feats))
            #print(gamma_results.summary())
            x =0

'''
x=0
overall_acc=np.zeros([1,cnt])
overall_sens=np.zeros([1,cnt])
overall_spec=np.zeros([1,cnt])
overall_auc = np.zeros([1,cnt])
acc =0
best_idx =0
for i in range(1,cnt):
    global_feats_selected   = global_feats[:,0:i]
    overall_acc_tmp ,sens_tmp,spec_tmp= feats_fusion(global_feats_selected, labels)
    overall_acc[0, i - 1] = overall_acc_tmp
    overall_sens[0, i - 1] = sens_tmp
    overall_spec[0, i - 1] = spec_tmp
    if overall_acc_tmp > acc:
        best_idx = i-1
        print(i,overall_acc[0,i-1],overall_sens[0,i-1],overall_spec[0,i-1])
        x = 0
        acc = overall_acc[0,i-1]
    x =0
'''