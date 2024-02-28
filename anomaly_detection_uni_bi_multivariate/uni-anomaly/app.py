import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#impoer library
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
#from pyemma import msm
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

app = Flask(__name__)


# returns a compiled model
model_anomaly = joblib.load('univariate_anomaly.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_data = pd.DataFrame(final_features)

    
    #prediction

    # add the data to the main  
    final_data['anomaly_isolated'] = pd.Series(model_anomaly.predict(final_data))
    
    #0 means normal and -1 means anomaly
    anomaly = final_data.loc[final_data['anomaly_isolated']== -1]
    
    #0 means normal and 1 means anomaly
    
    #final_data['anomaly_isolated'] = final_data['anomaly_isolated'].map( {1: 0, -1: 1} )
    #anomaly_re=final_data.loc[final_data['anomaly_isolated']== 1]
    
    final_data['anomaly_isolated'].value_counts()

    
    #return anomaly
    for anomaly in final_data['anomaly_isolated']:
        if anomaly== -1:
            text="anomaly"
            print(text)
            return render_template('index.html', prediction_text='Customer will have {}'.format(text))

        if anomaly==1:
            text="normal"
            print(text)
            return render_template('index.html', prediction_text='Customer will have {}'.format(text))


if __name__ == "__main__":
    app.run(debug=False)