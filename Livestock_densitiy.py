import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import xlrd
import xlwt
import xlsxwriter
import scipy.stats as stats
from scipy.interpolate import splev, splrep
from sklearn import ensemble
from sklearn import  metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error 
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import seaborn as sns
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay



def jud_array(x):
    print(np.isnan(x).any())

df = pd.read_excel('D:\\RF\\1117\\1117_QTP_xiangzhen.xlsx') 
df = df.dropna(axis=0, how='any') # Deletes rows containing Nan
col=['Winter accumulated precipitation','Travel time','NPP','DEM','Snow cover days','Alpine grass and carex steppe ratio','Alpine kobresia meadow ratio']
#Import predictors
col2 = ['ln_all']
#Import livestock data
X = df[col].values
Y = df[col2].values

##### Random forest training
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.3,random_state = 2)
forest = ensemble.RandomForestRegressor(
    n_estimators=200,
    random_state=2,
    n_jobs=-1)
forest.fit(x_train, y_train)

### Visualization of training results
result = forest.predict(x_test)
score = forest.score(x_test , y_test)
result1 = result.reshape(-1,1)
y_test1 = y_test.flatten()
F = stats.pearsonr(y_test1,result)
F1 = np.array(F)
R2=F1[0]*F1[0]
print(F1[0]*F1[0])
mse=metrics.mean_squared_error(y_test,result)
variance_score=metrics.explained_variance_score(result, y_test)

### featureimportances
importances = forest.feature_importances_
print("Importance:", importances)
x_columns=['Winter accumulated precipitation','Travel time','NPP','DEM','Snow cover days','Alpine grass and carex steppe ratio','Alpine kobresia meadow ratio']
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, col[indices[f]], importances[indices[f]]))
    
### Visualization of importance results
plt.figure(figsize=(8,12))
plt.xlabel("feature importance", fontsize=15, rotation=0)
x_columns1 = [x_columns[i] for i in indices]
x_columns2 =  list(reversed(x_columns1)) 
colors=['darkseagreen','darkseagreen','bisque','lightpink','cornflowerblue','darkseagreen','darkseagreen','cornflowerblue','royalblue','darkseagreen','darkseagreen']
for i in range(len(x_columns)):
    plt.barh(len(x_columns)-1-i,importances[indices[i]],color=[colors[i]])
    plt.yticks(np.arange(len(x_columns)), x_columns2, fontsize=16, rotation =0)
    y_num=np.arange(len(x_columns)-1-i)
    plt.xlim(0,0.56)
    plt.text(importances[indices[i]]+0.0005,len(x_columns)-1-i-0.10,round(importances[indices[i]],3), fontsize=16)
    plt.legend(['Vegetation','Socioeconomic','Topography','Climate'],fontsize=16,loc="lower right")
plt.show()

threshold = 0.15
x_selected = x_train[:, importances > threshold]
rfr_s = cross_val_score(forest, X, Y, cv=10,
                scoring = "r2"
                )
validationscore = np.mean(rfr_s)
print(rfr_s)



##### 10-fold cross-validation of livestock densities
Adjusted_R_squared=[]
R_squared=[]
e=[]
ff=[]
kmse=[]
kmae=[]
kevs=[]
z=[]
jj=0
folder = KFold(n_splits=10,random_state=20,shuffle=True)
for K_train_index,K_test_index in folder.split(X,Y):
    print("TRAIN:", K_train_index, "TEST:", K_test_index)
    K_X_train, K_X_test = X[K_train_index], X[K_test_index]
    K_Y_train, K_Y_test = Y[K_train_index], Y[K_test_index]
    K_Y_test1 = K_Y_test.flatten()
    K_Y_train1 = K_Y_train.flatten()
    forest.fit(K_X_train, K_Y_train1)
    K_resultloo = forest.predict(K_X_test)
    KR2 = metrics.r2_score(K_Y_test1,K_resultloo)
    MAE=mean_absolute_error(K_Y_test1,K_resultloo)
    MSE=metrics.mean_squared_error(K_Y_test1,K_resultloo)
    EVS=metrics.explained_variance_score(K_Y_test1,K_resultloo)
    J = stats.pearsonr(K_Y_test1,K_resultloo)
    J1 = np.array(J)
    K_R2=J1[0]*J1[0]
    Adjusted_R_squared.append(KR2)
    R_squared.append(K_R2)
    e.extend(K_resultloo)
    ff.extend(K_Y_test)
    kmae.append(MAE)
    kmse.append(MSE)
    kevs.append(EVS)
    #path = r"F:\\snowcal\\1001QTP\\txt"
    #filename = os.listdir(path)
    #x_predict=forest.predict(x_variable)
    #np.savetxt('F:\\snowcal\\1001QTP\\txt\\'+str(filename[jj]), x_predict, delimiter = ',')
    jj=jj+1
    print(K_R2)
    print(KR2)
    print(MAE)
    print(MSE)
    print(EVS)
    
e=np.array(e)
ff=np.array(ff)
ff=ff.flatten()    
z= np.vstack((ff,e)).T
#.savetxt('F:\\snowcal\\0823xiangzhen\\z_xiangzhen.csv', z, delimiter = ',')
df1 = pd.read_csv('D:\\RF\\z.csv') 
#z = z.astype('int32')  
K_adj_r2=np.array(Adjusted_R_squared)
K_r2=np.array(R_squared)
K_adj_r2=K_adj_r2.flatten()
K_r2=K_r2.flatten()
K_r2_mean = np.mean(K_r2)
MAE_mean = np.mean(kmae)
MSE_mean=np.mean(kmse)
EVS_mean=np.mean(kevs)
K_adj_r2_mean = np.mean(K_adj_r2)
# plt.scatter(e, ff, alpha=0.5, marker=(9, 3, 30))
# plt.show()
print('*****')
print(K_r2_mean)
print(K_adj_r2_mean)
print(MAE_mean)
print(MSE_mean)
print(EVS_mean)


##### the partial dependence plot
col1=['"pre_winter"','"travel time"','"NPP"','"vegetation coverage"','"dem"','"days of snow cover2000_2017"','"Alpine grass and carex grassland ratio"','"Alpine kobresia meadow ratio"']

#features = list(x_columns.index)
for i, feat in enumerate(col):
    
   # PartialDependenceDisplay(forest,x_train, col1)

    pdp = partial_dependence(forest,X, [i], kind="both",grid_resolution=25) 
    sns.set_theme(style="ticks", palette="muted", font_scale = 1.1)
    sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    fig = plt.figure(figsize=(6, 5), dpi=500)
    ax=plt.subplot(111)
    
    cd=X[:,i]
    plot_x = pd.Series(pdp['values'][0]).rename('x')
    plot_i = pdp['individual'] 
    plot_y = pdp['average'][0]
    tck = splrep(plot_x, plot_y, s=30)
    xnew = np.linspace(plot_x.min(),plot_x.max(),300)
    ynew = splev(xnew, tck, der=0)
    plot_df = pd.DataFrame(columns=['x','y']) 
    for a in plot_i[0]: 
        a2 = pd.Series(a)
        df_i = pd.concat([plot_x, a2.rename('y')], axis=1) 
        plot_df = plot_df.append(df_i)

    sns.lineplot(data=plot_df, x="x", y="y", color='k', linewidth = 1.5, linestyle='--', alpha=0.5)
    plt.plot(xnew, ynew, linewidth=2)  
    sns.rugplot(cd, height=.05, color='k', alpha = 0.3)
    x_min = plot_x.min()-(plot_x.max()-plot_x.min())*0.1
    x_max = plot_x.max()+(plot_x.max()-plot_x.min())*0.1
    plt.ylabel('Ln(Livestock density')
    plt.xlabel(col[i])
    plt.xlim(x_min,x_max)
    plt.show()

##### Correlation between livestock density and covariates
# sns.set_context(font_scale=2)
data = pd.read_excel(r'D:\RF\1117\1117_QTP_cor.xlsx')
data=data.dropna(how='any',axis=0)
data = data.dropna(axis=0, how='any')
fig, ax = plt.subplots(figsize=(12,12))
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 1, top -1)
sns.heatmap(data.corr(),annot=True,annot_kws={'size':9},vmin=-1,vmax=1,cmap=sns.color_palette('RdBu',n_colors=128))
data_p=[]
for i in range(0,30):
    datap=stats.pearsonr(data[2],data[i])
    data_p.append(datap[2])
plt.show()