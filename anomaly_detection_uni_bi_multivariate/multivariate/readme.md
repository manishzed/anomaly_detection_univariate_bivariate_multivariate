'*'Anomaly Detection with Isolation Forest'*'

This repository contains Python code for anomaly detection using the Isolation Forest algorithm.
The code is organized into a class named AnomalyDetection to encapsulate various functionalities such as data loading, preprocessing, model training, and visualization.

Navigate to project directory:
cd <project_directory>

Install the required dependencies using pip3 and the provided requirements.txt file.
pip3 install -r requirements.txt

Run the main.py script.:
python3 main.py

This script demonstrates the usage of the AnomalyDetection class on a sample dataset (ambient_temperature_system_failure.csv). It reads the data, preprocesses it, plots the data stream, trains the Isolation Forest model, and visualizes anomalies over time and their distribution.



About Algorithm Isolation Forest:
Isolation Concept:
The Isolation Forest works on the principle that anomalies are easier to isolate than normal instances. It builds an ensemble of isolation trees, where each tree is constructed by recursively partitioning the dataset. Anomalies are expected to be isolated closer to the root of the trees, requiring fewer splits.

Random Partitioning:
The algorithm randomly selects a feature and a random split value for each partition, leading to a tree structure. Multiple trees are built independently. Randomization is a key element of the algorithm, making it efficient and able to handle high-dimensional data without the need for extensive parameter tuning.

Anomaly Score:
The anomaly score is determined by the average depth of the isolation trees in which a data point resides. Anomalies, being isolated early, have lower average depths, resulting in higher anomaly scores.

Scalability:
Isolation Forest is efficient and scalable, especially in high-dimensional spaces. The average time complexity for building an isolation tree is O(log N), where N is the number of instances in the dataset.

Handling High-Dimensional Data:
Traditional methods may struggle with high-dimensional data due to the curse of dimensionality. Isolation Forest, by design, is less affected by the curse of dimensionality and can effectively handle datasets with many features.

Outlier Detection:
Isolation Forest is particularly well-suited for outlier detection tasks. Its ability to isolate anomalies efficiently makes it useful in scenarios where identifying rare events or unusual patterns is crucial.

One-Class Learning:
Isolation Forest is a one-class learning algorithm, making it suitable for unsupervised tasks where only normal instances are available during training. It does not require labeled anomaly examples for training.

While Isolation Forest has shown good performance in various scenarios, it's important to note that no single algorithm is universally optimal for all types of data



Class Methods:
AnomalyDetection(data_path)
Constructor method to initialize the AnomalyDetection object with the path to the dataset.

read_data()
Reads data from the specified path and handles file not found or empty dataset errors.

preprocess_data()
Performs data preprocessing and feature engineering, converting temperature values to Celsius and extracting additional features.

train_model()
Trains the Isolation Forest model on the preprocessed data.

plot_data()
Plots the data stream over time.

visualize_anomalies_over_time()
Visualizes anomalies throughout time using a line plot.

visualize_anomalies_distribution()
Visualizes anomalies with temperature distribution using a histogram.


Requirements:
The code relies on the following Python libraries:

1. pandas
2. numpy
3. scikit-learn
4. matplotlib


Ensure these dependencies are installed before running the code.

