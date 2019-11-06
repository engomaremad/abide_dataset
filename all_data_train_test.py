#Todo: train test split experiment for the entire abide dataset

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import  RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
# Import smtplib for the actual sending function
import smtplib
import warnings
from sklearn.model_selection import cross_validate
from email.mime.text import MIMEText

# Import the email modules we'll need
from sklearn.feature_selection import RFE

def send_email(str):
    msg = MIMEText(str)

    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = 'new result achieved'
    msg['From'] = 'engomaremad@gmail.com'
    msg['To'] = 'engomaremad@gmail.com,bioeng.m.tarek@hotmail.com'

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP('localhost')
    s.sendmail('engomaremad@gmail.com', ['engomaremad@gmail.com,bioeng.m.tarek@hotmail.com'], msg.as_string())
    s.quit()
def autoencoder(x,encoding_dim):
    X_train, X_test, y_train, y_test = train_test_split(x, labels[:np.shape(x)[0]], test_size=0.2)
    x_size = np.shape(x)[1]
    input_img = Input(shape=(x_size,))
    # add a Dense layer with a L1 activity regularizer
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(input_img)
    decoded = Dense(x_size, activation='sigmoid')(encoded)


    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test))
    return autoencoder

def get_fisher_score(x,y):
    num_feats = np.shape(x)[1]
    fs = np.zeros([num_feats,1])
    asd_idx = np.where(y == 2)[0]
    td_idx = np.where(y == 1)[0]
    for i in range(np.shape(x)[1]):
        asd_feats = x[asd_idx, i]
        td_feats = x[td_idx, i]
        m = np.mean(x[:, i])
        n1 = len(asd_idx)
        n2 = len(td_idx)
        m1 = np.mean(asd_feats)
        m2 = np.mean(td_feats)
        v1 = np.var(asd_feats)
        v2 = np.var(td_feats)
        fs[i] = ((n1 * (m1 - m) ** 2) + (n2 * (m2 - m) ** 2)) / ((n1 * v1)   +( n2 * v2))
    sorted = np.argsort(fs.ravel())
    return(sorted)

def feat_select(x,y,n_feats,n_est,max_d):
    estimator = RandomForestClassifier(n_estimators=n_est,max_depth=max_d,random_state=0)
    selector = RFE(estimator, n_feats, step=0.10,verbose=False)
    selector = selector.fit(x, y)
    x =0
    return selector.support_
def classify(X_train, X_test, y_train, y_test,l1,l2,n_feats,n_est,max_d):
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    used_feats = feat_select(X_train,y_train,n_feats,n_est,max_d)
    #clf = RandomForestClassifier(n_estimators=n_est,max_depth=max_d,random_state=0)
    clf = MLPClassifier(hidden_layer_sizes=[l1,l2],random_state=0)
    X_train, X_val, y_train, y_val =   train_test_split(X_train, y_train, test_size=0.1)
    #clf = SVC()

    clf.fit(X_train[:,used_feats],y_train)
    res = clf.predict(X_test[:,used_feats])
    acc = accuracy_score(y_test,res)
    res_val = clf.predict(X_val[:,used_feats])
    acc_val = accuracy_score(y_val, res_val)
    return acc,acc_val



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
acc = 0

np.random.seed(1111)
X_train, X_test, y_train, y_test = train_test_split(conn_vect, labels, test_size=0.20)
for i in range(100000):
    l1 = np.int(np.random.random()*500)+5
    l2 = np.int(np.random.random() * 50) + 1
    n_feats = np.int(np.random.random()*4000)+5
    n_est = np.int(np.random.random()*500)+5
    max_d = np.int(np.random.random() * 50) + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_acc , val_acc  = classify(X_train, X_test, y_train, y_test, l1, l2, n_feats, n_est, max_d)
    if val_acc>acc or True:
        acc = val_acc
        print(test_acc,acc,l1,l2,n_feats,n_est,max_d)
        f = open('train_test.txt','a')
        f.writelines(str(test_acc)+'\t'+str(acc)+'\t'+str(l1)+'\t'+str(l2)+'\t'+str(n_feats)+'\t'+str(n_est)+'\t'+str(max_d)+'\n')
        #send_email(str(acc)+'\t'+str(l1)+'\t'+str(l2)+'\t'+str(n_feats)+'\t'+str(n_est)+'\t'+str(max_d)+'\n')
        f.close()
    x = 0