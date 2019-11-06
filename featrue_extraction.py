#Todo: Feature extraction for the entire dataset using RFE with multiple sampling instansces
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import  RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
# Import smtplib for the actual sending function
import smtplib
import warnings
from sklearn.model_selection import cross_validate
from email.mime.text import MIMEText

# Import the email modules we'll need
from sklearn.feature_selection import RFE,RFECV



def feat_select(x,y,n_est,max_d):
    estimator = RandomForestClassifier(n_estimators=n_est,max_depth=max_d,random_state=0)
    selector = RFECV(estimator, step=0.10,verbose=False,cv=3,n_jobs=-1)
    selector = selector.fit(x, y)
    x =0
    return selector.support_
def classify(X_train, X_test, y_train, y_test,l1,l2):
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    #clf = RandomForestClassifier(n_estimators=n_est,max_depth=max_d,random_state=0)
    clf = MLPClassifier(hidden_layer_sizes=[l1,l2],random_state=0)
    #clf = SVC()

    clf.fit(X_train,y_train)
    res = clf.predict(X_test)
    acc = accuracy_score(y_test,res)
    return acc



dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'
sfc = np.load(os.path.join(dict_path,'ids_conns_mats.npy')).item()
labels_dict = np.load(os.path.join(dict_path,'id_label.npy')).item()
ids = np.load(os.path.join(dict_path,'all_ids.npy'))
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

feats_history = np.zeros([np.shape(conn_vect)[1],1])
n_trials = 1000
accs =np.zeros([n_trials,1])
np.random.seed(1111)
for n in range(n_trials):
    print(n)
    X_train, X_test, y_train, y_test = train_test_split(conn_vect, labels, test_size=0.20)
    used_feats = feat_select(X_train,y_train.ravel(),500,20)
    feats_history[used_feats] = feats_history[used_feats]+1
    x =0
    np.save('feat_history',feats_history)
    #np.save('accs', accs)