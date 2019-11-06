#todo classify ados total severity based on the percntile feats obtained
import numpy as np
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import  svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as py
from functions.plt_conf_mat import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from functions.accuracy_curve_fitting import fit
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.feature_selection import RFE,RFECV
from sklearn.model_selection import train_test_split

def feats_fusion(percntile,labels,n_est,depth,sorted_idx,i_j):
    n_folds = 4
    kf = KFold(n_splits=n_folds)
    res_final = np.zeros([len(labels)])
    acc = np.zeros([100,1])
    feats =np.zeros(np.size(labels))
    overall_acc = 0
    y_pred = np.zeros(np.size(res_final))
    for i in range(100):
        a1 = int(i_j[sorted_idx[i]][0,0])
        a2 = int(i_j[sorted_idx[i]][0,1])
        feats = np.column_stack((feats,percntile[a1,a2,:,:]))
        used_feats = feats[:,1:]
        for train_index, test_index in kf.split(used_feats):
            #clf_cal = RandomForestClassifier(n_estimators=n_est,max_depth=depth,random_state=0)
            clf_cal = MLPClassifier(hidden_layer_sizes=(15),random_state=0 )
            clf_cal.fit(used_feats[train_index, :], labels[train_index])
            res_final[test_index] = clf_cal.predict(used_feats[test_index])
        acc[i] = accuracy_score(res_final, labels)
        if acc[i]>overall_acc:
            print(i+1,acc[i])
            overall_acc=acc[i]
            target_names = ['class 0', 'class 1', 'class 2']
            y_pred = np.copy(res_final)
            print(classification_report(res_final, labels, target_names=target_names))
    return acc,y_pred, labels




def feat_select(x,y,n_est,max_d):
    estimator = RandomForestClassifier(n_estimators=n_est,max_depth=max_d,random_state=0)
    selector = RFECV(estimator, step=0.10,verbose=False,cv=3,n_jobs=-1)
    selector = selector.fit(x, y)
    x =0
    return selector.support_

def classify_single_entry (feats,labels,n_est ,depth):
    n_folds = 4
    kf = KFold(n_splits=n_folds)
    res_final = np.zeros([len(labels)])
    for train_index, test_index in kf.split(feats):
        #clf_cal = svm.SVC(class_weight='balanced')
        #clf_cal = LinearSVC(C=1.0, class_weight='balanced', dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge',
        #                   max_iter=1000,multi_class='crammer_singer', penalty='l2', random_state=0, tol=1e-05, verbose=0)
        #clf_cal = RandomForestClassifier(n_estimators=n_est,max_depth=depth,random_state=0,class_weight='balanced')
        clf_cal = KNeighborsClassifier(n_neighbors=9)
        #clf_cal = LogisticRegression(random_state=0,solver='lbfgs',multi_class='multinomial',class_weight='balanced')
        clf_cal.fit(feats[train_index,:], labels[train_index])
        res_final[test_index] =clf_cal.predict(feats[test_index,:])
    acc = accuracy_score(res_final,labels)
    return acc
def classify_ados_tot(ados,name,num_areas,pernctiles):
    ados = np.asarray(ados)
    #define boundries
    b1,b2 = 10,14
    classes = np.zeros(np.size(ados))
    classes[np.where(ados<b1)[0]]=1
    classes[np.where(np.logical_and(ados>=b1,ados<b2))[0]] = 2
    classes[np.where(ados >= b2)[0]] = 3
    cnt = 0
    #accuracies for entries
    acc = np.zeros([int(0.5*num_areas*(num_areas-1)),1])
    area_idx = np.zeros([int(0.5*num_areas*(num_areas-1)),2])
    n_folds = 4
    kf = KFold(n_splits=n_folds)
    feats = np.zeros((len(ados),1))
    for i in range(num_areas):
        for j in range(i):
            feats = np.hstack((feats,pernctiles[i,j,:,:]))

    feats = feats[:,1:]
    x = 0

    feats_history = np.zeros([np.shape(feats)[1], 1])
    n_trials = 1000
    accs = np.zeros([n_trials, 1])
    np.random.seed(1111)
    for n in range(n_trials):
        print(n)
        X_train, X_test, y_train, y_test = train_test_split(feats, classes, test_size=0.20)
        used_feats = feat_select(X_train, y_train.ravel(), 500, 20)
        feats_history[used_feats] = feats_history[used_feats] + 1
        x = 0


        np.save('feat_history_percntiles', feats_history)
        # np.save('accs', accs)


















    sorted_idx = np.argsort(acc,0)
    sorted_idx = sorted_idx[::-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc, y_pred, labels = feats_fusion(pernctiles, classes, 50, 2, sorted_idx,area_idx)
    cnf_matrix = confusion_matrix(labels, y_pred)
    np.set_printoptions(precision=2)
    class_names = ['mild','moderate','severe']
    # Plot non-normalized confusion matrix
    py.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    fit(range(100),acc)
    x=0