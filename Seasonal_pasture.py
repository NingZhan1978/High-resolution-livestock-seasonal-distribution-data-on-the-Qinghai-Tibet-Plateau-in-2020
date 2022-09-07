import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import xlrd
import xlwt
import xlsxwriter
import scipy.stats as stats
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import seaborn as sns
from sklearn import ensemble
from sklearn import  metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import roc_curve, auc
from IPython.display import HTML
from sklearn.inspection import partial_dependence 
from scipy.interpolate import splev, splrep 
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interp


def jud_array(x):
    print(np.isnan(x).any())

df = pd.read_csv('F:\\snowcal\\pasture\model_wintersummer_all_ratio_1km_0726.csv') 
df = df.dropna(axis=0, how='any') 
x_variable=pd.read_csv('F:\\snowcal\\pasture\\x_relative value\\model_x_relative value_0802.csv',header = None)
col=['ndvi','dem','summer_tem','winter_tem','summer_pre','winter_pre','Travel time']
col2 = ['pasture']

###Import pasture data
X = df[col].values
###Import predictors
Y = df[col2].values

x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.3,random_state = 2)

#### RFC training
forest = ensemble.RandomForestClassifier(n_estimators = 200, 
                              #max_depth= 5,      
                              oob_score=True,
                              #class_weight = "balanced",
                              random_state=2)
forest.fit(x_train, y_train)

### Visualization of training results
result = forest.predict(x_test)
score = forest.score(x_test , y_test)
OOB_score=forest.oob_score_
result1 = result.reshape(-1,1)
y_test1 = y_test.flatten()
#R = np.corrcoef(y_test,result)
F = stats.pearsonr(y_test1,result)
F1 = np.array(F)
R2=F1[0]*F1[0]
print(F1[0]*F1[0])
#R2 = metrics.r2_score(y_test,result)
mse=metrics.mean_squared_error(y_test,result)
variance_score=metrics.explained_variance_score(result, y_test)


### feature importances
importances = forest.feature_importances_
print("feature importance", importances)
#'Gross value of animal husbandry','AFHF personnel','per capital GDP','Highland barley sown area',
x_columns=['ndvi','dem','GStem','Wtem','GSpre','Wpre','Travel time']
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, col[indices[f]], importances[indices[f]]))
    
### Visualization of importance results
plt.figure(figsize=(8,12))
plt.xlabel("feature importance", fontsize=15, rotation=0)

x_columns1 = [x_columns[i] for i in indices]
x_columns2 =  list(reversed(x_columns1)) 
colors=['cornflowerblue','lightpink','darkseagreen','bisque','#EB8E55','lightpink','darkseagreen','royalblue','royalblue']
#del (dict)
for i in range(len(x_columns)):
    plt.barh(len(x_columns)-1-i,importances[indices[i]],color=[colors[i]])
    plt.yticks(np.arange(len(x_columns)), x_columns2, fontsize=16, rotation =0)
    y_num=np.arange(len(x_columns)-1-i)
    #plt.ylim(min(y_num)-1,len(x_columns))
    plt.xlim(0,0.30)
    plt.text(importances[indices[i]]+0.0005,len(x_columns)-1-i-0.25,round(importances[indices[i]],3), fontsize=16)
    #plt.xticks(i, importances[indices[i]], color='blue', align='center')
    #plt.bar(np.arange(len(x_columns)), x_columns1, fontsize=5, rotation =45)
    plt.legend(['Precipitation','Tempreture','Vegetation','Topography','Socioeconomic'],fontsize=16,
                  loc="lower right",
                )

plt.show()
threshold = 0.15
x_selected = x_train[:, importances > threshold]
rfr_s = cross_val_score(forest, X, Y, cv=10,
                scoring = "r2"
                )
validationscore = np.mean(rfr_s)
print(rfr_s)

##### 10-fold cross-validation of livestock densities
c=[]
d=[]
e=[]
ff=[]
xx=[]
xxx=[]
kmse=[]
kmae=[]
kevs=[]
z=[]
aa=[]
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)
jj=0
plt.figure(dpi=600,figsize=(10,8))
#plt.plot([0, 1], [0, 1], 'k--')

folder = KFold(n_splits=10,random_state=20,shuffle=True)
for K_train_index,K_test_index in folder.split(X,Y):
    print("TRAIN:", K_train_index, "TEST:", K_test_index)
    K_X_train, K_X_test = X[K_train_index], X[K_test_index]
    K_Y_train, K_Y_test = Y[K_train_index], Y[K_test_index]
    K_Y_test1 = K_Y_test.flatten()
    K_Y_train1 = K_Y_train.flatten()
    forest.fit(K_X_train, K_Y_train1)
    K_resultloo = forest.predict(K_X_test)
    ll=np.size(K_resultloo)
    K_resultloo1=np.reshape(K_resultloo,(ll,1))
    KR2 = metrics.r2_score(K_Y_test1,K_resultloo)
    MAE=mean_absolute_error(K_Y_test1,K_resultloo)
    MSE=metrics.mean_squared_error(K_Y_test1,K_resultloo)
    EVS=metrics.explained_variance_score(K_Y_test1,K_resultloo)
    #accuracy=forest.score(K_Y_test1,K_resultloo)
    accuracy=forest.score(K_X_test,K_Y_test)
    J = stats.pearsonr(K_Y_test1,K_resultloo)
    J1 = np.array(J)
    K_R2=J1[0]*J1[0]
    c.append(KR2)
    d.append(K_R2)
    e.extend(K_resultloo)
    ff.extend(K_Y_test)
    xx.extend(K_X_test)
    ###plot
    pre_y1 = forest.predict_proba(K_X_test)[:, 1]
    y_test = np.array(y_test)
    pre_y1 = np.array(pre_y1)
    y_test = y_test.flatten()
    fpr_Nb, tpr_Nb, _ = roc_curve(K_Y_test, pre_y1)
    #interp
    tprs.append(interp(mean_fpr,fpr_Nb,tpr_Nb))
    tprs[-1][0]=0.0
    aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值
    aucs.append(aucval)
    plt.plot(fpr_Nb, tpr_Nb,linewidth = 1.5,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (jj,aucval))
    ####
    kmae.append(MAE)
    kmse.append(MSE)
    kevs.append(EVS)
    aa.append(accuracy)
    path = r"F:\\snowcal\\pasture\\txt"
    filename = os.listdir(path)
    #predict pasture
    x_predict=forest.predict(x_variable)
    
    np.savetxt('F:\\snowcal\\pasture\\txt\\'+str(filename[jj]), x_predict, delimiter = ',')
    jj=jj+1
    print(K_R2)
    print(KR2)
    print(MAE)
    print(MSE)
    print(EVS)
    print(aa)

e=np.array(e)
ff=np.array(ff)
aa=np.array(aa)
ff=ff.flatten()    
z= np.vstack((ff,e)).T
#np.savetxt('F:\\livestock\\Winter and summer pastures\\z_wintersummer_rfc.csv', z, delimiter = ',')
df1 = pd.read_csv('F:\\snowcal\\pasture\\z_wintersummer_rfc.csv') 
#z = z.astype('int32')  
K_adj_r2=np.array(c)
K_r2=np.array(d)
K_adj_r2=K_adj_r2.flatten()
K_r2=K_r2.flatten()
k_aa=aa.flatten()
K_r2_mean = np.mean(K_r2)
MAE_mean = np.mean(kmae)
MSE_mean=np.mean(kmse)
EVS_mean=np.mean(kevs)
K_adj_r2_mean = np.mean(K_adj_r2)
k_aa_mean=np.mean(k_aa)
# plt.scatter(e, ff, alpha=0.5, marker=(9, 3, 30))
# plt.show()
print('*****')
print(K_r2_mean)
print(K_adj_r2_mean)
print(MAE_mean)
print(MSE_mean)
print(EVS_mean)
print(k_aa_mean)

###ROC curve
#ff1=np.reshape(ff,(114776,1))
pre_yall = forest.predict_proba(xx)[:, 1]
y_test = np.array(y_test)
pre_yall = np.array(pre_yall)
y_test = y_test.flatten()
fpr_Nb, tpr_Nb, _ = roc_curve(ff, pre_yall)

aucval = auc(fpr_Nb, tpr_Nb)    # calculate auc
#plt.figure(dpi=600,figsize=(10,8))
plt.plot([0, 1], [0, 1], 'k--')
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr) # calculate auc mean
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC', fontsize=16)
plt.legend(loc='lower right')
plt.rcParams['figure.dpi'] = 600 #分辨率
plt.show()    
