#impoer library
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#LOADING DATASET
train = pd.read_csv('Training_Bivariate_Data.csv')

test=pd.read_csv("Testing_Data.csv")
test.drop("Date_Time",inplace=True, axis=1)

#CONCAT
df=pd.concat([train,test]).reset_index()
df.drop(['index'], inplace=True, axis=1)
df.describe()

df.shape

#CHOOSING DATE TIME RANDOM
new_date = pd.DataFrame(pd.date_range(start='1979-6-1', periods=len(df), freq='D'), columns=['load_date'])

new_date = pd.to_datetime(new_date['load_date']).dt.date
len(new_date)

df['load_date'] = new_date
df.head()



X = df.iloc[:,0:2].values
 



ifr = IsolationForest(contamination=0.1, random_state=42) 
y_pred = ifr.fit_predict(X)
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

 # 1 means normal value, -1 means anomalous value
y_pred 

pred_ifr=pd.DataFrame(y_pred, columns=['anomaly'])

print(pred_ifr.loc[pred_ifr['anomaly']==-1])

#Finding the ROC Accuracy score for the prediction label.

sklearn_score_anomalies = abs(ifr.score_samples(X))
sklearn_score_anomalies = ifr.decision_function(X)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]





------------------------>

#KNN Classifier (Proximity-Based)¶

from pyod.models.knn import KNN   # kNN detector

# train kNN detector
clf_name = 'KNN'
clf = KNN()
clf.fit(X)


pred_knn = clf.predict(X)  # outlier labels (0 or 1)
scores_knn = clf.decision_function(X)  # outlier scores

pred_knn





# 0 means normal value while 1 means anomalous value.

#plotting normal vs anomaly
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred - 1) // 2])


#Finding the ROC Accuracy score for the prediction label.
clf.fit_predict_score(X[:, 0].reshape(-1,1), y_pred, scoring='roc_auc_score')



-------------------------->


#Linear Model PCA

from pyod.models.pca import PCA

clf_pca = PCA()
clf_pca.fit(X)


pred_pca = clf_pca.predict(X)  # outlier labels (0 or 1)
scores_pca = clf_pca.decision_function(X)  # outlier scores

y_pred

pred_df=pd.DataFrame(y_pred, columns=['anomaly'])

print(pred_df.loc[pred_df['anomaly']==1])

# 0 means normal value while 1 means anomalous value.

#plotting normal vs anomaly
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred - 1) // 2])

#Finding the ROC Accuracy score for the prediction label.

clf_pca.fit_predict_score(X[:, 0].reshape(-1,1), y_pred, scoring='roc_auc_score')




#final model

------------>

#impoer library
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#LOADING DATASET
train = pd.read_csv('Training_Bivariate_Data.csv')

test=pd.read_csv("Testing_Data.csv")
test.drop("Date_Time",inplace=True, axis=1)

#CONCAT
df=pd.concat([train,test]).reset_index()
df.drop(['index'], inplace=True, axis=1)
df.describe()

df.shape

#CHOOSING DATE TIME RANDOM
new_date = pd.DataFrame(pd.date_range(start='1979-6-1', periods=len(df), freq='D'), columns=['load_date'])

new_date = pd.to_datetime(new_date['load_date']).dt.date
len(new_date)

df['load_date'] = new_date
df.head()



X = df.iloc[:,0:2].values
final_data=pd.DataFrame(X, columns=['Moisture','humidity'])


#isolation forest model
RANDOM_STATE = 123
outlier_fraction = 0.028


model =  IsolationForest(n_jobs=-1, n_estimators=200, max_features=2, random_state=RANDOM_STATE, contamination=outlier_fraction)

model.fit(X)

#prediction
pred_isf=model.predict(X)

# add the data to the main  
final_data['anomaly_isolated'] = pd.Series(model.predict(X))

#0 means normal and -1 means anomaly
anomaly = final_data.loc[final_data['anomaly_isolated']== -1]

#0 means normal and 1 means anomaly

#final_data['anomaly_isolated'] = final_data['anomaly_isolated'].map( {1: 0, -1: 1} )
#anomaly_re=final_data.loc[final_data['anomaly_isolated']== 1]

final_data['anomaly_isolated'].value_counts()

for idx,i in enumerate(final_data['anomaly_isolated']):
    if i== -1:
        print(idx,i)


#plotting normal vs anomaly
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(pred_isf - 1) // 2])


import joblib 
  
# # Save the model as a pickle in a file 
joblib.dump(model, 'bivariate_anomaly.pkl')


model_anomaly = joblib.load('bivariate_anomaly.pkl')


#validate
valid=[[355,70]]
valid=pd.DataFrame(valid)

pred_valid=model_anomaly.predict(valid)



-------------->


#One Class SVM
import pandas as pd
import numpy as np
import datetime as dt
import random

import matplotlib.pyplot as plt
%matplotlib inline

# Set some parameters to get good visuals - style to ggplot and size to 15,10
plt.style.use('ggplot')
import matplotlib.style as style
style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)

import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

RANDOM_STATE = 123
outlier_fraction = 0.028

model =  OneClassSVM(nu=outlier_fraction, degree=2, kernel='rbf')


model.fit(X)


final_data['anomaly_svm'] = pd.Series(model.predict(X))

final_data['anomaly_svm'] = final_data['anomaly_svm'].map( {1: 0, -1: 1} )

final_data['anomaly_svm'].value_counts()

final_data.loc[final_data['anomaly_svm']==1]


#acuarcy of all model

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

