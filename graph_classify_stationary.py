# todo: classify asd vs td using stationary graph feats :degree , closeness_centrality, betweenness_centrality,
#   eigenvector_centrality

import os
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as py
from functions.plt_conf_mat import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from functions.accuracy_curve_fitting import fit
import scipy.stats as sc
import matplotlib.pyplot as plt
def classify_single_entry (feats,labels,n_est=10 ,depth=2):
    n_folds = 4
    kf = KFold(n_splits=n_folds)
    labels = np.ravel(labels)
    res_final = np.zeros([len(labels)])
    for train_index, test_index in kf.split(feats):
        #clf_cal = SVC(random_state=0)
        #clf_cal = LinearSVC(C=1.0, class_weight='balanced', dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge',
        #                 max_iter=1000,multi_class='crammer_singer', penalty='l2', random_state=0, tol=1e-05, verbose=0)
        #clf_cal = KNeighborsClassifier(n_neighbors=9)
        #clf_cal = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (4,2), random_state = 1)
        clf_cal = RandomForestClassifier(n_estimators=20,random_state=0)
        clf_cal.fit(feats[train_index,:], labels[train_index])
        res_final[test_index] =clf_cal.predict(feats[test_index,:])
    acc = accuracy_score(res_final,labels)
    x=0
    return acc
from functions.get_graph_measures import get_graph_measures
dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'
conn_matts = np.load(os.path.join(dict_path,'ids_conns_mats.npy')).item()
ids = np.load(os.path.join(dict_path,'all_ids.npy'))
th = 0.3
num_areas = 116
labels_dict = np.load(os.path.join(dict_path,'id_label.npy')).item()
cnt = 0
used_ids =[]
for id in ids:
    used_ids.append(id)
labels = np.zeros([len(used_ids),1])
feats = np.zeros([len(used_ids),num_areas,3])
if not os.path.exists(os.path.join(dict_path,'graph_feats_stationary.npy')):
    for id  in used_ids:
        print(cnt)
        G,cs,bs,d = get_graph_measures(conn_matts[id],th)
        labels[cnt] = labels_dict[id]
        feats[cnt,:,0]= cs.ravel()
        feats[cnt,:, 1] = bs.ravel()
        #feats[cnt,:,2] = es.ravel()
        feats[cnt,:, 2] = d.ravel()
        cnt+=1
    np.save(os.path.join(dict_path, 'graph_feats_stationary.npy'),feats)
else:
    for id in ids:
        labels[cnt] = labels_dict[id]
        cnt += 1
    feats = np.load(os.path.join(dict_path,'graph_feats_stationary.npy'))
asd_idx = np.where(labels==2)[0]
td_idx =np.where(labels==1)[0]
fs = np.zeros([num_areas,1])
for i in range(num_areas):
    asd_feats = feats[asd_idx,i,1]
    td_feats = feats[td_idx, i, 1]
    m = np.mean(feats[:,i,1])
    n1 =len(asd_idx)
    n2 = len(td_idx)
    m1 = np.mean(asd_feats)
    m2 =np.mean(td_feats)
    v1 = np.var(asd_feats)
    v2= np.var(td_feats)
    fs[i] = ((n1*(m1-m)**2)+(n2*(m2-m)**2))/(n1*v1+n2*v2)
fs_idx = np.argsort(fs[:,0])
f= np.zeros([len(labels),1])
for i in range(num_areas):
    f = np.column_stack((f,feats[:,fs_idx[i],2]))
    acc = classify_single_entry(f[:,1:],labels)
    x =0
x=0
x =0
x = 0