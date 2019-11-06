#TODO use a fraction of the selected feats using RFE for K-fold classification
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score

import warnings

import scipy.stats as sc


def classify(x,y,l1,l2,l3,a):
    #visualize(x,y)
    kf = KFold(n_splits=4)
    kf.get_n_splits(x)
    res = np.zeros([len(labels),1])
    res_p = np.zeros([len(labels), 1])
    for  train_index, test_index in kf.split(x):
        clf = MLPClassifier(hidden_layer_sizes=[l1,l2],alpha=a,random_state=0)
        clf.fit(x[train_index,:],y[train_index])
        res[test_index,0] = clf.predict(x[test_index,:])
        res_p[test_index, 0] = clf.predict_proba(x[test_index, :])[:,1]
    acc = accuracy_score(res,labels)
    conf = confusion_matrix(res,labels)
    roc = roc_auc_score(y,res_p)
    #plot_confusion_matrix(conf, ['ASD', 'TD'])
    return acc,conf

#np.random.seed(0)
dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'
sfc = np.load(os.path.join(dict_path,'ids_conns_mats.npy')).item()
labels_dict = np.load(os.path.join(dict_path,'id_label.npy')).item()
ids = np.load(os.path.join(dict_path,'all_ids.npy'))
feat_history =  np.load(os.path.join(dict_path,'feat_history.npy'))
percntiles = np.zeros(np.shape(feat_history))
for i in range(len(feat_history)):
    percntiles[i] = sc.percentileofscore(feat_history,feat_history[i])

num_subjs = len(ids)
num_areas =116
conn_mat = np.zeros([num_subjs,num_areas,num_areas])
labels =np.zeros([num_subjs,1])
cnt = 0
for key in ids:
    conn_mat[cnt, :, :] = sfc[key]
    labels[cnt] = labels_dict[key]
    cnt+=1
conn_mat[np.isnan(conn_mat)]=0
conn_vect = np.zeros([num_subjs,int(0.5*num_areas*(num_areas-1))])
cnt = 0
for i in range(num_areas):
    for j in range(i):
        conn_vect[:,cnt] = conn_mat[:,i,j]

        cnt+=1
acc = 0
all_accs =[]
for i in range(50):
    feats_th = 85
    l1 = np.int(np.random.random()*500)+5
    l2 = np.int(np.random.random() * 150) + 1
    l3 = np.int(np.random.random() * 75) + 1
    alpha = (np.random.random() * 1000+1)*(10**-6)
    # l1 =499
    # l2 =150
    # alpha = 0.0004892428036557068
    used_feats = np.where(percntiles >= feats_th)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val_acc,conf = classify(conn_vect[:,used_feats[0]], labels, l1, l2,l3,alpha)
       # plot_confusion_matrix(conf,['ASD','TD'])
        x =0
    if val_acc>acc:
        acc = val_acc
        all_accs.append(acc)
        print(feats_th,acc,l1,l2,l3,alpha)
        # f = open('train_test_selected_feats.txt','a')
        # f.writelines(str(feats_th)+'\t'+str(acc)+'\t'+str(l1)+'\t'+str(l2)+'\t'+str(l2)+'\t'+str(alpha)+'\n')
        # f.close()
    x = 0