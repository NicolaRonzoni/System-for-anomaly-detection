# System for Anomaly detection 
Analysis of 3 public dataset:
- Water pump system https://www.kaggle.com/datasets/nphantawee/pump-sensor-data
- Machine predictive maintenance classification https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
- Electrical fault classification https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification?select=classData.csv
### Pandas-InfluxDB pipelines 
Workflow  for time series analysis in real time. 
Jupyter for Data Ingestion on Influxdb from Pandas :
1. Generate API token from UserInterface and create a bucket (copy the ID) where store the time series. 
2. Load a csv file in Pandas. The Pandas dataframe should have as index the timestamp, it must contain a column that indicates the origin* of the data (by default creates a column that contains the same value in each row).
3. Despite InfluxDB is robust to events coming at different intervals, make sense to make the data regular (model development). For this reason using Pandas add rows of missing values if jump in the series occur. 
4. Make a connection with the server
5. Send data, specifying the name of the dataframe and the origin*.
6. Close the connection

It is always possible to restore data in Pandas once manipulation are computed on InfluxDB using the same scheme showed above.  

### Anomaly detection pipelines 
Functions and main jupyter for Clustering task and detection models of failures in pump water system. 
Unsupervised techniques for univariate sensors time series: 
1. Clustering a period based approach with DTW, soft-DTW and euclidean metrics to detect anomalies in a large window of time (12h)
2. Zoom in analysis to detect anomalies in point granularity and restricted window of time. 3 approaches are used: (DynamicBaseline) comparison with historical data, (Prophet) additive forecasting model adopted for anomaly detection problem, (RandomCutForest) tree based model that isolate anomaly by binary decisions. 
3. Interpret anomalies as z-score, (possibility to reduce false alarms)
4. Point Adjusted Evaluation, Revised Point Adjusted Evaluation in different scenarios. 

### Predictive maintenance fault classification pipelines 
Functions and jupyters of supervised techniques on detection and failure type classification. 
Workflow for Machine predictive maintenance and electrical fault classification:
1. Data strategy 
2. Bayesian optimization on a specific target 
3. Ensamble learning boosting (Gradient Boost classifier, Adaptive boost classifier) and bagging (Random Forest). 
4. Neural Network (MLP classifier)
5. Evaluation on confusion matrices, balanced accuracy, micro/macro metrics. 



