#impoer library
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#LOADING DATASET
train = pd.read_csv('Training_Data.csv')

test=pd.read_csv("Testing_Data_Engine_Speed.csv")

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



X = df.iloc[:,0].values
final_data=pd.DataFrame(X, columns=['RPM'])


#isolation forest model
RANDOM_STATE = 123
outlier_fraction = 0.028


model =  IsolationForest(n_jobs=-1, n_estimators=200, max_features=1, random_state=RANDOM_STATE, contamination=outlier_fraction)

model.fit(final_data)

#prediction

# add the data to the main  
final_data['anomaly_isolated'] = pd.Series(model.predict(final_data))

#0 means normal and -1 means anomaly
anomaly = final_data.loc[final_data['anomaly_isolated']== -1]

#0 means normal and 1 means anomaly

#final_data['anomaly_isolated'] = final_data['anomaly_isolated'].map( {1: 0, -1: 1} )
#anomaly_re=final_data.loc[final_data['anomaly_isolated']== 1]

final_data['anomaly_isolated'].value_counts()


for idx,i in enumerate(final_data['anomaly_isolated']):
    if i== -1:
        print(idx,i)




import joblib 
  
# # Save the model as a pickle in a file 
joblib.dump(model, 'Univariate_anomaly.pkl')


model_anomaly = joblib.load('Univariate_anomaly.pkl')


#validate
valid=[[1353]]
valid=pd.DataFrame(valid)

pred_valid=model_anomaly.predict(valid)





def normal_accuracy(values):
    
    tp=list(values).count(1)
    total=values.shape[0]
    accuracy=np.round(tp/total,4)
    
    return accuracy



normal_isf = model.predict(final_data)


in_accuracy_isf=normal_accuracy(normal_isf)

print("Accuracy in Detecting isolation forest Cases:", in_accuracy_isf)

#ACCURACY = 97%