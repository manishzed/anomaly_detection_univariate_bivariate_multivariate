import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class AnomalyDetection:
    def __init__(self, data_path):
        '''
        param: data_path
        '''
        self.data_path = data_path
        self.df = None
        self.model = None

    def read_data(self):
        '''
        function to read data from the given path
        '''
        try:
            self.df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.data_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("The dataset is empty.")

    def preprocess_data(self):
        '''
        preprocessing data & feature Engineering
        '''
        if self.df is None:
            raise ValueError("Data not loaded. Call read_data() first.")
        
        # change the type of timestamp column for plotting
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        # change fahrenheit to °C (temperature mean= 71 -> fahrenheit)
        self.df['value'] = (self.df['value'] - 32) * 5/9
        # the hours and if it's night or day (7:00-22:00)
        self.df['hours'] = self.df['timestamp'].dt.hour
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 22)).astype(int)
        # the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
        self.df['DayOfTheWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayOfTheWeek'] < 5).astype(int)
        # time with int to plot easily
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)

    def train_model(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call read_data() first.")
        
        # Take useful feature and standardize them 
        data = self.df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        # train isolation forest 
        self.model = IsolationForest(
            n_estimators=150,
            max_samples='auto',
            contamination=0.01,
            max_features=1
        )
        self.model.fit(data)
        self.df['anomaly'] = pd.Series(self.model.predict(data))
        self.df['anomaly'] = self.df['anomaly'].map({1: 0, -1: 1})

    def plot_data(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call read_data() first.")

        try:
            self.df.plot(x='timestamp', y='value', color='orange')
            plt.title("Data stream plot")
            plt.xlabel("Timestamp")
            plt.ylabel("Value")
            plt.show()
        except Exception as e:
            print(f"Error plotting data: {e}")

    def visualize_anomalies_over_time(self):
        '''
        visualisation of anomaly throughout time
        '''
        try:
            if self.df is None or self.model is None:
                raise ValueError("Data or model not ready. Call read_data() and train_model() first.")

            fig, ax = plt.subplots()
            a = self.df.loc[self.df['anomaly'] == 1, ['time_epoch', 'value']]
            ax.plot(self.df['time_epoch'], self.df['value'], color='orange')
            ax.scatter(a['time_epoch'], a['value'], color='red')
            plt.title("Anomalies over time plot")
            plt.xlabel("Time Epoch")
            plt.ylabel("Value")
            plt.show()
        except Exception as e:
            print(f"Error during visualization: {e}")

    def visualize_anomalies_distribution(self):
        '''
        visualisation of anomaly with temperature repartition
        '''
        try:
            if self.df is None or self.model is None:
                raise ValueError("Data or model not ready. Call read_data() and train_model() first.")

            a = self.df.loc[self.df['anomaly'] == 0, 'value']
            b = self.df.loc[self.df['anomaly'] == 1, 'value']

            fig, axs = plt.subplots()
            axs.hist([a, b], bins=32, stacked=True, color=['yellow', 'red'], label=['normal', 'anomaly'])
            plt.title("Anomalies Distribution Plot")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error during visualization: {e}")


if __name__ == "__main__":
    data_path = "Dataset/ambient_temperature_system_failure.csv"
    anomaly_detector = AnomalyDetection(data_path)
    anomaly_detector.read_data()
    anomaly_detector.preprocess_data()
    anomaly_detector.plot_data()
    anomaly_detector.train_model()
    anomaly_detector.visualize_anomalies_over_time()
    anomaly_detector.visualize_anomalies_distribution()
