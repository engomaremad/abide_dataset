#TODO calculate FDR for corrlation multiple testing

import numpy as np
import statsmodels.stats.multitest as ml
def calc_FDR(results,fdr=0):
    print(fdr)
    #results order : area 1, area 2 , r, p
    # FDR is the allowed number of false positive ratio
    p = results[:,3]
    n_tests = len(p)
    idx = np.argsort(p)
    results = results[idx,:]
    rej,p_cor = ml.fdrcorrection(results[:,3],is_sorted=True,alpha=fdr)
    return p_cor, rej
    x =0

    # for i in range(n_tests):
    #    if results[i,3] <= fdr * (i+1) /n_tests:
    #        print(results[i,0],results[i,1],results[i,2],results[i,3])
    #    else:
    #         break
    x=0