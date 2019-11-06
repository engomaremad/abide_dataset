#TODO calculate the correlation between percntiles of DFC outliers and ADOS modules
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plot
import os
from functions.accuracy_curve_fitting import fit


def plot_corr(x,y,a1,a2,feat,s,i,r,p,name):
    output_dir = 'correlation_ados_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot.scatter(x,y)
    plot.hold(True)
    plot.plot(x,np.multiply(x,s)+i,'r')
    plot.title('%.1f %.1f '%(a1,a2) +feat+' r = %0.2f  p = %0.6f ' %(r,p) )
    plot.savefig(os.path.join(output_dir,str(a1)+'_'+str(a2)+'_'+feat+'_'+name))
    plot.close()
    plot.show()
    x =0

def calc_corr(ados,name,num_areas,pernctiles,perc_idx,ages,used_viq,used_piq,used_fiq):
    #perc_idx : 0 top 5%, 1 top 10%,2 lower 10%,3 lower 5%
    num_feats = int(0.5*num_areas*(num_areas-1))
    reuslts = np.zeros((num_feats,6))
    cnt_res = 0
    for i in range(num_areas):
        for j in range(i):
            conn_vect = []
            for cnt in range(np.shape(pernctiles)[2]):
                conn_vect.append(pernctiles[i,j,cnt,perc_idx])
            # slope, intercept, r_value, p_value, std_err = sc.linregress(ages, conn_vect)
            # res = conn_vect -  (np.multiply(ages,slope)+intercept)
            #
            # slope, intercept, r_value, p_value, std_err = sc.linregress(used_viq, res)
            # res = res - (np.multiply(used_viq, slope) + intercept)
            # slope, intercept, r_value, p_value, std_err = sc.linregress(used_piq, res)
            # res = res - (np.multiply(used_piq, slope) + intercept)
            # slope, intercept, r_value, p_value, std_err = sc.linregress(used_fiq, res)
            # res = res - (np.multiply(used_fiq, slope) + intercept)
            sl, intr, r, p, err = sc.linregress(conn_vect, ados)
            r_,p_ = sc.spearmanr(conn_vect,ados)
            sl_age, intr_age, r_age, p_age, err_age  = sc.linregress(ages, ados)
            sl_viq, intr_viq, r_viq, p_viq, err_viq = sc.linregress(used_viq, ados)
            sl_piq, intr_piq, r_piq, p_piq, err_piq = sc.linregress(used_piq, ados)
            sl_fiq, intr_fiq, r_fiq, p_fiq, err_fiq = sc.linregress(used_fiq, ados)

            if perc_idx == 0:
                feat = '95-100'
            elif perc_idx ==1:
                feat = '90-95'
            elif perc_idx ==2:
                feat = '0-5'
            else:
                feat = '5-10'
            if np.abs(p)<=4e-4:
                plot_corr(conn_vect, ados, i, j,feat , sl, intr, r, p,name)
                #fit(conn_vect,ados)
                x=0
            reuslts[cnt_res,0] =i
            reuslts[cnt_res, 1] = j
            reuslts[cnt_res, 2] = r
            reuslts[cnt_res, 3] = p
            reuslts[cnt_res, 4] = r_
            reuslts[cnt_res, 5] = p_
            cnt_res+=1
    return reuslts