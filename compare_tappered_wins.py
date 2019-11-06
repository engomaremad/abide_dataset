# Todo: calculate the tapered sliding window dynamic functional connectivity dictionary
# check this paper for more details : https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002469

import os
import numpy as np
import scipy.stats as sc
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
def calc_dfc(id):
    print(np.where(ids==id)[0])
    tc = id_tcs[id]

    # dfc matrix shift
    # size of window = 30 TR
    # window is shifted by 5 TR
    # number of window = (lenght of signal - win_size)/shift

    num_wins = int(np.ceil((np.shape(tc)[0] - win_size) / shift))
    dfc_mat = np.ones([num_areas, num_areas, num_wins])
    for i in range(num_areas):
        for j in range(i):
            curr = 0
            for w in range(num_wins):
                if curr + win_size <= np.shape(tc)[0]:
                    sig1 = tc[curr:curr + win_size, i]
                    sig2 = tc[curr:curr + win_size, j]
                    if np.isnan(sig1).any():
                        sig1[np.isnan(sig1)]= 0
                    if np.isnan(sig2).any():
                        sig2[np.isnan(sig2)]= 0
                    sig1_zero_mean = sig1 - np.mean(sig1)
                    sig2_zero_mean = sig2 - np.mean(sig2)
                    #numerator
                    num_t1 = np.multiply(sig1_zero_mean,sig2_zero_mean)
                    num = np.sum(np.multiply(tappered_filter,num_t1))
                    # denominator
                    sig1_zero_mean_sq = np.multiply(sig1_zero_mean,sig1_zero_mean)
                    sig2_zero_mean_sq = np.multiply(sig2_zero_mean, sig2_zero_mean)
                    den_t1 = np.sqrt(np.sum(np.multiply(tappered_filter,sig1_zero_mean_sq)))
                    den_t2 = np.sqrt(np.sum(np.multiply(tappered_filter, sig2_zero_mean_sq)))
                    r =num/(den_t1*den_t2)
                    dfc_mat[i, j, w] = r
                    curr = curr + shift
                else:
                    sig1 = tc[curr:np.shape(tc)[0], i]
                    sig2 = tc[curr:curr + win_size, j]
                    if np.isnan(sig1).any():
                        sig1[np.isnan(sig1)]= 0
                    if np.isnan(sig2).any():
                        sig2[np.isnan(sig2)]= 0
                    sig1_zero_mean = sig1 - np.mean(sig1)
                    sig2_zero_mean = sig2 - np.mean(sig2)
                    # numerator
                    num_t1 = np.multiply(sig1_zero_mean, sig2_zero_mean)
                    num = np.sum(np.multiply(tappered_filter, num_t1))
                    # denominator
                    sig1_zero_mean_sq = np.multiply(sig1_zero_mean, sig1_zero_mean)
                    sig2_zero_mean_sq = np.multiply(sig2_zero_mean, sig2_zero_mean)
                    den_t1 = np.sqrt(np.sum(np.multiply(tappered_filter, sig1_zero_mean_sq)))
                    den_t2 = np.sqrt(np.sum(np.multiply(tappered_filter, sig2_zero_mean_sq)))
                    r = num / (den_t1 * den_t2)
                    dfc_mat[i, j, w] = r
                dfc_mat[j, i, w] = dfc_mat[i, j, w]
    return id,dfc_mat






os.chdir('Data_dictionaries')
ids = np.load('used_names.npy')
id_ados_mod  = np.load('id_ados_mod.npy').item()
id_ados_tot  = np.load('id_ados_tot.npy').item()
id_ados_comm  = np.load('id_ados_comm.npy').item()
id_ados_soc = np.load('id_ados_soc.npy').item()
id_ados_beh = np.load('id_ados_beh.npy').item()
id_ados_rel  = np.load('id_ados_rel.npy').item()
id_tcs = np.load('id_tcs.npy').item()
cnt = 0
num_areas = 116

win_size = 30
shift = 1

# hamming window function
a0_hamming= 0.53836
hamming = a0_hamming- (1-a0_hamming)*np.cos(2*np.pi*np.arange(win_size)/win_size)
tappered_filter = a0_hamming

try:
    dfc_mat_hamming = np.load('hamming_sample.npy')
except:
    id, dfc_mat_hamming = calc_dfc(ids[0])
    id_dfc = dict()
try:
    dfc_mat_hanning = np.load('hanning_sample.npy')
except:
    hanning = np.hanning(win_size)
    tappered_filter = hanning
    num_cores = multiprocessing.cpu_count()
    id, dfc_mat_hanniing = calc_dfc(ids[0])

r = []
for i in range(num_areas):
    for j in range(i):
        tmp,p = sc.pearsonr(dfc_mat_hamming[i,j,:],dfc_mat_hanning[i,j,:])
        r.append(tmp)
print(np.mean(r))
plt.plot(range(len(dfc_mat_hamming[i,j,:])),dfc_mat_hamming[4,1,:])
plt.plot(range(len(dfc_mat_hamming[i,j,:])),dfc_mat_hanning[4,1,:])
plt.legend(('Hamming','Hanning'))
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.show()
