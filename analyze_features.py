#TODO: analyze the extracted features and get the most common ones
import numpy as np
import os
from scipy import stats as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from functions.calc_FDR import calc_FDR
dict_path = '/home/biolab/PycharmProjects/abide_dataset/Data_dictionaries'
#feat_history =  np.load(os.path.join(dict_path,'feat_history.npy'))
feat_history =  np.load(os.path.join(dict_path,'accs_ados_4.npy'))
sfc = np.load(os.path.join(dict_path,'ids_conns_mats.npy')).item()
aal2yeo = np.load(os.path.join(dict_path,'aal_2_yeo.npy'))
percntiles = np.zeros(np.shape(feat_history))
ids = np.load(os.path.join(dict_path,'all_ids.npy'))
labels_dict = np.load(os.path.join(dict_path,'id_label.npy')).item()
n_feats = 116
num_subjs = len(ids)
yeo_areas_num = 7
yeo_labels = np.load(os.path.join(dict_path,'yeo_labels.npy'))

yeo_labels = ['visual','somatomotor','dorsal_attention','ventral_attention','limbic','frontoparietal','default']
percntiles_mat = np.zeros([n_feats,n_feats])
p_mat = np.zeros([n_feats,n_feats])
for i in range(len(feat_history)):
    percntiles[i] = sc.percentileofscore(feat_history,feat_history[i])
num_areas =116
conn_mat = np.zeros([num_subjs,num_areas,num_areas])
labels =np.zeros([num_subjs,1])
cnt = 0
for key in ids:
    conn_mat[cnt, :, :] = sfc[key]
    labels[cnt] = labels_dict[key]
    cnt+=1
conn_mat[np.isnan(conn_mat)]=0
networks_cnt = np.zeros(np.size(yeo_labels))
cnt = 0
high_freq_feats = 0
used =[]
asd_idx = np.where(labels==1)[0]
td_idx = np.where(labels==2)[0]
stat_res = np.zeros([len(feat_history),4])
used_cnt = []
for i in range(n_feats):
    for j in range(i):
        percntiles_mat[i,j] = percntiles[cnt]
        percntiles_mat[j, i] = percntiles_mat[i,j]
        t, p = sc.ttest_ind(conn_mat[asd_idx,i,j],conn_mat[td_idx,i,j])
        p_mat[i, j] = p
        stat_res[cnt,:]=[i,j,t,p]
        if percntiles[cnt]>=85:
            used_cnt.append(cnt)
            high_freq_feats+=1
            #print(high_freq_feats,i+1,j+1,yeo_labels[int(aal2yeo[i]-1)],yeo_labels[int(aal2yeo[j]-1)])
            if not i in used or True:
                used.append(i)
                networks_cnt[int(aal2yeo[i]-1)] = networks_cnt[int(aal2yeo[i]-1)] +1
            if not j in used or True:
                used.append(j)
                networks_cnt[int(aal2yeo[j]-1)] = networks_cnt[int(aal2yeo[j]-1)] + 1
        cnt+=1
p_cor, rej = calc_FDR(stat_res[used_cnt,:])
cnt =0
for i in range(len(used_cnt)):
    if percntiles[i]>=85 and rej[i] ==True:
        cnt+=1
print(cnt)

has_yeo_labels_num = len(np.where(aal2yeo>0)[0])
displayed_res  = np.zeros([has_yeo_labels_num,has_yeo_labels_num])
tmp  = []
tmp2 =[]
boundary=[0]
for i in range(1,yeo_areas_num+1):
    tmp = np.where(aal2yeo==i)[0]
    boundary.append(boundary[i-1]+len(tmp))
    tmp2 = np.hstack([tmp,tmp2])
tmp2 = tmp2.astype(int)
displayed_res[:,:]= percntiles_mat[tmp2,:][:,tmp2]
fig, ax = plt.subplots()
displayed_res[displayed_res<85]=85
#displayed_res[displayed_res<0.3]=0.3





cax = ax.imshow(displayed_res, interpolation='nearest', cmap=cm.jet)
plt.title('Map of features that appeared more than 85% of times',y=-0.08)




# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax, ticks=np.round(np.linspace(85,100,16),decimals=2))
cbar.ax.set_yticklabels(np.round(np.linspace(85,100,16),decimals=2),verticalalignment='center')  # vertically oriented colorbar
ax.xaxis.tick_top()
       # inc. width of y-axis and color it red

plt.xticks(boundary,yeo_labels,rotation =90,fontsize =18)
plt.yticks(boundary,yeo_labels,rotation =0,fontsize =18 )
ax.set_yticks(boundary[1:], minor=True)
ax.set_yticks(boundary[1:], minor=False)
ax.yaxis.grid(True, which='major',color='w', linestyle='-', linewidth=2)
ax.xaxis.grid(True, which='major',color='w', linestyle='-', linewidth=2)
# Make plot with horizontal colorbar

plt.show()

x=0
x=0