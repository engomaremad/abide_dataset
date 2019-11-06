#TODO calculate the GLM between percntiles of DFC outliers, age and ADOS modules
import numpy as np
import scipy.stats as sc
import statsmodels.api as sm
import matplotlib.pyplot as plot
import os
from sklearn.metrics import r2_score
def plot_corr(x,y,a1,a2,feat,s,i,r,p):
    output_dir = 'correlation_ados_delta'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot.scatter(x,y)
    plot.hold(True)
    plot.plot(x,np.multiply(x,s)+i,'r')
    plot.title('%.1f %.1f '%(a1,a2) +feat+' r = %0.2f  p = %0.4f ' %(r,p) )
   # plot.savefig(os.path.join(output_dir,str(a1)+str(a2)+feat))
    #plot.close()
    plot.show()
    x =0



def calc_glm(ados,name,num_areas,pernctiles,perc_idx,ages,used_viq,used_piq,used_fiq):
    #perc_idx : 0 top 5%, 1 top 10%,2 lower 10%,3 lower 5%
    num_feats = int(0.5*num_areas*(num_areas-1))
    reuslts = np.zeros((num_feats,4))
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
            const = np.ones([len(conn_vect), 1])
            used_viq = np.asarray(used_viq)
            used_piq = np.asarray(used_piq)
            used_fiq = np.asarray(used_fiq)
            used_viq[np.where(used_viq<0)] = 0
            used_piq[np.where(used_piq < 0)] = 0
            used_fiq[np.where(used_fiq < 0)] = 0
            conn_vect = np.asarray(conn_vect)
            ages =np.asarray(ages)
            ados = np.asarray(ados)
            idx = np.where(conn_vect > 0)
            x_data = np.column_stack((const[idx[0]],conn_vect[idx[0]],ages[idx[0]],used_viq[idx[0]],used_piq[idx[0]],used_fiq[idx[0]]))
            glm = sm.GLM(ados[idx[0]], x_data[:,:],family=sm.families.Poisson())
            res  = glm.fit()
            p_vals = res.pvalues
            coef = res.params

            y_pred = glm.predict(coef,x_data)
            #r2 =r2_score(ados[idx[0]],y_pred)
            x = 0
            #reuslts =0
            sl,intr,r, p,err = sc.linregress(conn_vect[idx[0]], ados[idx[0]])
            y_pred_r = np.multiply(conn_vect[idx[0]],sl)+intr
            r2 = r2_score(ados[idx[0]], y_pred_r)
            x =0
            # if np.abs(r)>0.5:
            #     plot_corr(conn_vect[idx[0]], ados[idx[0]],i,j,'feat',sl,intr,r, p)
            #     x =0
            # reuslts[cnt_res,0] =i
            # reuslts[cnt_res, 1] = j
            # reuslts[cnt_res, 2] = r
            # reuslts[cnt_res, 3] = p
            # cnt_res+=1

    return reuslts