# Todo: calculate the tapered sliding window dynamic functional connectivity dictionary
# check this paper for more details : https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002469

import numpy as np
import multiprocessing
from timeit import default_timer as timer
from numba import vectorize,autojit,jit,cuda
from joblib import Parallel, delayed

@jit()
def calc_corr(sig1,sig2,hamming):
    # hamming window function
    sig1_zero_mean = sig1 - np.mean(sig1)
    sig2_zero_mean = sig2 - np.mean(sig2)
    # numerator
    num_t1 = np.multiply(sig1_zero_mean, sig2_zero_mean)
    num = np.sum(np.multiply(hamming, num_t1))
    # denominator
    sig1_zero_mean_sq = np.multiply(sig1_zero_mean, sig1_zero_mean)
    sig2_zero_mean_sq = np.multiply(sig2_zero_mean, sig2_zero_mean)
    den_t1 = np.sqrt(np.sum(np.multiply(hamming, sig1_zero_mean_sq)))
    den_t2 = np.sqrt(np.sum(np.multiply(hamming, sig2_zero_mean_sq)))
    if den_t1 ==0 or den_t2==0:
        r = 0
    else:
        r = num / (den_t1 * den_t2)
    return r


def calc_dfc(tc,num_areas,win_size,shift,window):
    num_wins = int(np.ceil((np.shape(tc)[0] - win_size) / shift))
    dfc_mat = np.ones([num_areas, num_areas, num_wins])
    for i in range(num_areas):
        for j in range(i):
            curr = 0
            for w in range(num_wins):
                if curr + win_size <= np.shape(tc)[0]:
                    sig1 = tc[curr:curr + win_size, i]
                    sig2 = tc[curr:curr + win_size, j]
                else:
                    sig1 = tc[curr:np.shape(tc)[0], i]
                    sig2 = tc[curr:curr + win_size, j]
                if np.isnan(sig1).any():
                    sig1[np.isnan(sig1)]= 0
                if np.isnan(sig2).any():
                    sig2[np.isnan(sig2)]= 0
                r = calc_corr(sig1, sig2,window)
                dfc_mat[i, j, w] = r
                curr = curr + shift
                dfc_mat[i, j, w] = r
                dfc_mat[j, i, w] = dfc_mat[i, j, w]
    return dfc_mat




def dfc_with_win(num_areas,win_size,shift,window,tcs):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(calc_dfc)(tcs[:,:,i],num_areas,win_size,shift,window) for i  in range(np.shape(tcs)[2]))
    ret = np.zeros([len(results),np.shape(results[0])[0],np.shape(results[0])[1],np.shape(results[0])[2]])
    for cnt in range(len(results)):
        ret[cnt,:,:,:] = results[cnt]
    return ret