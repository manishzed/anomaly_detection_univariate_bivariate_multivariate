###type-1----------------
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Read data
data_path = r"D:/softweb/Anomaly-Detection-main/Anomaly-Detection-main/Lokesh_Nahar_IITG_Anomaly_Detection/Dataset/ambient_temperature_system_failure.csv"
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {data_path}")
except pd.errors.EmptyDataError:
    raise ValueError("The dataset is empty.")

df_ =df.copy()
# Preprocess data
# change the type of timestamp column for plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])
# change fahrenheit to °C (temperature mean= 71 -> fahrenheit)
df['value'] = (df['value'] - 32) * 5/9
# the hours and if it's night or day (7:00-22:00)
df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
# the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)

# Plot data
try:
    df.plot(x='timestamp', y='value', color='orange')
    plt.title("Data stream plot")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.show()
except Exception as e:
    print(f"Error plotting data: {e}")

# Train model
data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)

model = IsolationForest(
    n_estimators=150,
    max_samples='auto',
    contamination=0.01,
    max_features=1
)
model.fit(data)
df['anomaly'] = pd.Series(model.predict(data))
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Visualize anomalies over time
try:
    fig, ax = plt.subplots()
    a = df.loc[df['anomaly'] == 1, ['time_epoch', 'value']]
    ax.plot(df['time_epoch'], df['value'], color='orange')
    ax.scatter(a['time_epoch'], a['value'], color='red')
    plt.title("Anomalies over time plot")
    plt.xlabel("Time Epoch")
    plt.ylabel("Value")
    plt.show()
except Exception as e:
    print(f"Error during visualization: {e}")

# Visualize anomalies distribution
try:
    a = df.loc[df['anomaly'] == 0, 'value']
    b = df.loc[df['anomaly'] == 1, 'value']

    fig, axs = plt.subplots()
    axs.hist([a, b], bins=32, stacked=True, color=['yellow', 'red'], label=['normal', 'anomaly'])
    plt.title("Anomalies Distribution Plot")
    plt.legend()
    plt.show()
except Exception as e:
    print(f"Error during visualization: {e}")




###type-2--------------------------------------------------------------------------------------


# libraries
#%matplotlib notebook

import pandas as pd
import numpy as np

import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


data_path = r"D:/softweb/Anomaly-Detection-main/Anomaly-Detection-main/Lokesh_Nahar_IITG_Anomaly_Detection/Dataset/ambient_temperature_system_failure.csv"
df = pd.read_csv(data_path)
# change the type of timestamp column for plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])
# change fahrenheit to °C (temperature mean= 71 -> fahrenheit)
df['value'] = (df['value'] - 32) * 5/9
# plot the data
df.plot(x='timestamp', y='value')



# the hours and if it's night or day (7:00-22:00)
df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
# the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
# An estimation of anomly population of the dataset (necessary for several algorithm)
outliers_fraction = 0.01
# time with int to plot easily
df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)



#algo-1-----------
#Isolation Forest¶
#Use for collective anomalies (unordered).
#Simple, works well with different data repartition and efficient with high dimention data.
#IsolationForest
# Take useful feature and standardize them 
data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# train isolation forest 
model =  IsolationForest(contamination = outliers_fraction)
model.fit(data)
# add the data to the main  
df['anomalyIsolationForest'] = pd.Series(model.predict(data))
df['anomalyIsolationForest'] = df['anomalyIsolationForest'].map( {1: 0, -1: 1} )
print(df['anomalyIsolationForest'].value_counts())



#plotting
# visualisation of anomaly throughout time (viz 1)
fig, ax = plt.subplots()

a = df.loc[df['anomalyIsolationForest'] == 1, ['time_epoch', 'value']] #anomaly

ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'],a['value'], color='red')
plt.show()


# visualisation of anomaly with temperature repartition (viz 2)
a = df.loc[df['anomalyIsolationForest'] == 0, 'value']
b = df.loc[df['anomalyIsolationForest'] == 1, 'value']

fig, axs = plt.subplots()
axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'], label = ['normal', 'anomaly'])
plt.legend()
plt.show()



#algo-2-----------------------

#One class SVM
#Use for collective anomalies (unordered).
#Good for novelty detection (no anomalies in the train set). This algorithm performs well for multimodal data.

# Take useful feature and standardize them 
data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
# train one class SVM 
model =  OneClassSVM(nu=0.95 * outliers_fraction) #nu=0.95 * outliers_fraction  + 0.05
data = pd.DataFrame(np_scaled)
model.fit(data)
# add the data to the main  
df['anomalyOneClassSVM'] = pd.Series(model.predict(data))
df['anomalyOneClassSVM'] = df['anomalyOneClassSVM'].map( {1: 0, -1: 1} )
print(df['anomalyOneClassSVM'].value_counts())



#plotting

# visualisation of anomaly throughout time (viz 1)
fig, ax = plt.subplots()

a = df.loc[df['anomalyOneClassSVM'] == 1, ['time_epoch', 'value']] #anomaly

ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'],a['value'], color='red')
plt.show()









