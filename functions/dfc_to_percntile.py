#TODO this function is to get the number percntile of each DFC using above 97.5% and less than 2.5%
# the null distribution generated from the surrogates
import numpy as np
import scipy.stats as sc
from numba import vectorize,autojit,jit,cuda
@autojit()
def get_percentile(num_areas,original_dfc,surrogate_dfc):
    percentile = np.zeros([num_areas,num_areas,4])
    for i in range(np.shape(original_dfc)[2]):
        for j in range(num_areas):
            for k in range(j):
                p =sc.percentileofscore(surrogate_dfc[:,j,k,i],original_dfc[j,k,i])
                if p >95:
                    percentile[j,k,0] = percentile[j,k,0] + 1
                    percentile[k, j,0] = percentile[j,k,0]
                elif p >90:
                    percentile[j,k,1] = percentile[j,k,1] + 1
                    percentile[k, j,1] = percentile[j,k,1]
                elif p <5:
                    percentile[j,k,2] = percentile[j,k,2] + 1
                    percentile[k, j,2] = percentile[j,k,2]
                elif p <10:
                    percentile[j,k,3] = percentile[j,k,3] + 1
                    percentile[k, j,3] = percentile[j,k,3]
    return percentile/ np.shape(original_dfc)[2]