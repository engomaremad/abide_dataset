import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import  RandomForestClassifier
from sklearn.svm import SVC
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#num_sample = mnist.train.num_examples
#input_dim = mnist.train.images[0].shape[0]
#w = h = int(np.sqrt(input_dim))

def classify(X_train, X_test, y_train, y_test,n_est=100,max_dep = 4):
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    clf = MLPClassifier(hidden_layer_sizes = [100,50,10])
    #clf = RandomForestClassifier(n_estimators=400,max_depth=4)
    #clf =SVC()
    clf.fit(X_train,y_train)
    res = clf.predict(X_test)
    acc = accuracy_score(y_test,res)
    return acc
def test_transformation(model_2d, X_test, batch_size=3000):
    # Test the trained model: transformation
    assert model_2d.n_z == 2
    batch = X_test
    z = model_2d.transformer(batch)
    plt.figure(figsize=(10, 8))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1), s=20)
    plt.colorbar()
def trainer(model_object, learning_rate=1e-4,
            batch_size=64, num_epoch=20, n_z=16, log_step=5,x_train= None):
    model = model_object(
        learning_rate=learning_rate, batch_size=batch_size, n_z=n_z)

    for epoch in range(num_epoch):
        start_time = time.time()
        for iter in range(num_sample // batch_size):
            # Get a batch
            #batch = mnist.train.next_batch(batch_size)
            batch = x_train[iter*batch_size:(iter*batch_size)+batch_size,: ]
            # Execute the forward and backward pass
            # Report computed losses
            losses = model.run_single_step(batch)
        end_time = time.time()

        if epoch % log_step == 0:
            log_str = '[Epoch {}] '.format(epoch)
            for k, v in losses.items():
                log_str += '{}: {:.3f}  '.format(k, v)
            log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
            #print(log_str)

    print('Done!')
    return model
def test_reconstruction(model, x_test, h=28, w=28, batch_size=100):
    # Test the trained model: reconstruction

    batch = x_test
    x_reconstructed = model.reconstructor(batch)

    n = np.sqrt(batch_size).astype(np.int32)
    I_reconstructed = np.empty((h*n, 2*w*n))
    for i in range(n):
        for j in range(n):
            x = np.concatenate(
                (x_reconstructed[i*n+j, :].reshape(h, w),
                 batch[i*n+j, :].reshape(h, w)),
                axis=1
            )
            I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x

    plt.figure(figsize=(10, 20))
    plt.imshow(I_reconstructed, cmap='gray')
class VariantionalAutoencoder(object):

    def __init__(self, learning_rate=1e-4, batch_size=64, n_z=16):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z

        tf.reset_default_graph()
        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(
            name='x', dtype=tf.float32, shape=[None, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, 1024, scope='enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, 512, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 64, scope='enc_fc3', activation_fn=tf.nn.relu)
        self.z_mu = fc(f3, self.n_z, scope='enc_fc4_mu',
                       activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma',
                                 activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(self.z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 1024, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 512, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 64, scope='dec_fc3', activation_fn=tf.nn.relu)
        self.x_hat = fc(g3, input_dim, scope='dec_fc4',
                        activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon + self.x_hat) +
            (1 - self.x) * tf.log(epsilon + 1 - self.x_hat),
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between
        # the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) -
            tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = self.recon_loss + self.latent_loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)

        self.losses = {
            'recon_loss': self.recon_loss,
            'latent_loss': self.latent_loss,
            'total_loss': self.total_loss,
        }
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, losses = self.sess.run(
            [self.train_op, self.losses],
            feed_dict={self.x: x}
        )
        return losses

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z





dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'
sfc = np.load(os.path.join(dict_path,'ids_conns_mats.npy')).item()
labels_dict = np.load(os.path.join(dict_path,'id_label.npy')).item()
ids = np.load(os.path.join(dict_path,'all_ids.npy'))
num_subjs = len(ids)
num_areas =116
used_subjs = 1000
conn_mat = np.zeros([num_subjs,num_areas,num_areas])
labels =np.zeros([num_subjs,1])
cnt = 0
for key in ids:
    conn_mat[cnt, :, :] = sfc[key]
    labels[cnt] = labels_dict[key]
    cnt+=1
conn_mat[np.isnan(conn_mat)]=0
conn_vect = np.zeros([num_subjs,int(num_areas**2)])
cnt = 0

#
used_idx = []
for i in range(num_areas):
    for j in range(num_areas):
        conn_vect[:,cnt] = np.abs(conn_mat[:,i,j])
        if j<i:
            used_idx.append(cnt)
        cnt+=1
conn_vect[np.isnan(conn_vect)] = 0
X_train, X_test, y_train, y_test = train_test_split(conn_vect[:used_subjs,:], labels[:used_subjs], test_size=0.2)

asd_idx = np.where(y_train==2)[0]
td_idx = np.where(y_train==1)[0]
input_dim = np.shape(conn_vect)[1]
w = h = num_areas
num_sample = 380
generated_samples = 5
best_acc=0
for i in range(5000):
    latent_size = np.int(np.random.random()*5000)+2
    n_est = np.int(np.random.random()*500)+2
    max_dep = 4
    model = trainer(VariantionalAutoencoder,1e-4,20, 100, latent_size, 5,X_train[asd_idx[:num_sample],:])

    test_reconstruction(model, X_test,116,116,100)

# Test the trained model: generation
# Sample noise vectors from N(0, 1)
    z = np.random.normal(size=[generated_samples, model.n_z])
    x_generated_asd = model.generator(z)




    model = trainer(VariantionalAutoencoder,1e-4,20, 100, latent_size, 5,X_train[td_idx[:num_sample],:])

    test_reconstruction(model, X_test,116,116,100)

    # Test the trained model: generation
    # Sample noise vectors from generated_samples(0, 1)
    z = np.random.normal(size=[generated_samples, model.n_z])
    x_generated_td = model.generator(z)


    asd_data = np.concatenate((X_train[asd_idx[:num_sample]],x_generated_asd))
    td_data =  np.concatenate((X_train[td_idx[:num_sample]],x_generated_td))
    new_data  = np.concatenate((asd_data,td_data))
    new_labels =2*np.ones([np.shape(new_data)[0]])
    new_labels[num_sample+generated_samples:]=1
    new_data[np.isnan(new_data)] = 0
    acc = classify(new_data[:,used_idx],X_test[:,used_idx],new_labels,y_test,n_est,max_dep)
    if acc>best_acc:
        print(latent_size,acc)
        best_acc = acc
        x = 0
    plt.close()