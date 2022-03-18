{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

# import the libraries 
import pandas as pd
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
import sys
import calplot 
from toolz.itertoolz import sliding_window, partition 
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.metrics import gamma_soft_dtw
from tslearn.metrics import soft_dtw, gamma_soft_dtw,dtw
from sklearn.model_selection import train_test_split
import datetime
from io import StringIO # python3; python2: BytesIO 
from datetime import date
from datetime import datetime, timedelta
import math 
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, mean_squared_error,adjusted_rand_score
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from merlion.utils import TimeSeries
from merlion.plot import plot_anoms
from merlion.models.defaults import DefaultDetectorConfig, DefaultDetector
from merlion.models.anomaly.dbl import DynamicBaseline, DynamicBaselineConfig
from merlion.models.anomaly.windstats import WindStats, WindStatsConfig
from merlion.models.anomaly.forecast_based.prophet import ProphetDetector, ProphetDetectorConfig
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.models.anomaly.random_cut_forest import RandomCutForest, RandomCutForestConfig
from merlion.transform.moving_average import DifferenceTransform
from merlion.evaluate.anomaly import TSADMetric
from merlion.post_process.threshold import AggregateAlarms
import merlion.plot
from merlion.plot import plot_anoms, Figure
import matplotlib.pyplot as plt

# data: pd dataframe containing 52 sensors, Unnamed: 0, timestamp and the machine status
def imputation(data):
    # set as index the timestamp and check for jump in the series 
    # delete Unnamed: 0
    data.index = pd.to_datetime(data['timestamp'])
    data.drop(['timestamp','Unnamed: 0'], axis=1, inplace=True)
    data = data.asfreq('1Min')
    data.info()
    #remove sensors: 00,15,50,51
    data = data.drop(['sensor_15','sensor_50','sensor_51'], axis = 1)
    # 1) For sensor_00, sensor_06, sensor_07, sensor_08, sensor_09 all missing values are replaced with 0. 
    # Missing occur when the machine is in manteinance phase 
    #  the trajectories during these times are flat near to 0 value.  
    data['sensor_00']=data['sensor_00'].fillna(value=0)
    data['sensor_06']=data['sensor_06'].fillna(value=0)
    data['sensor_07']=data['sensor_07'].fillna(value=0)
    data['sensor_08']=data['sensor_08'].fillna(value=0)
    data['sensor_09']=data['sensor_09'].fillna(value=0)
    #2) For all other sensors propagate the last valid observation forward up to a limit of 1 hour (60 entries), 
    # for the remaining missing use the median over all the period. 
    data=data.fillna(method='ffill', limit=60)
    data1 = data.fillna(data.median())
    print('Is there any missing values:',data1.isna().any().any())
    return(data1)
# output: pd dataframe with 50 columns, 49 sensors and the machine status. 

#define a function to create the daily time series  without the scaling 
def daily_series_pred(data,n):
    #normalization of the data 
    data=np.array(data)
    data=data.reshape((len(data), 1))
    #from array to list 
    series=data.tolist()
    len(series)
    #create daily time series 
    time_series=list(partition(n,series))
    #from list to multidimensional array 
    time_series=np.asarray(time_series)
    #create univariate series for normalized observations 
    daily_time_series = to_time_series(time_series)
    return daily_time_series
# define a function that compute the clustering with soft-dtw
# data = dataframe.namecolumn (univariate time series)
# k is the number of clusters (default =2)
# aggregate='1min' '3min' '6min'
### hyperparameter for 1 minute 
# split= 131040/220320 (train: April-May-June)
# split= 87840/220320 (train: April-May)
### hyperparameter for 3 minute 
# train_split= 43680/73440 (train: April-May-June) 
# train_split= 29280/73440 (train: April-May)
### hyperparameter for 6 minute 
# train_split= 21840/36720 (train: April-May-June) 
# train_split= 14640/36720 (train: April-May)
# n  number of observations in a series: 
# daily series 1440 ('1minute'), 480 ('3minute'), 240 ('6minute')
# cycle series 12hours:  720 ('1minute'), 240 ('3minute'), 120 ('6minute')
def clustering(data,train_split,n,k,aggregate):
    data1= data.resample(aggregate).median()
    #train-test split of the data
    train,test=train_test_split(data1,train_size=train_split, shuffle=False)
    # create daily series with the function (daily_series_pred)
    train_series=daily_series_pred(train,n)
    test_series=daily_series_pred(test,n)
    #CLUSTERING with soft-DTW
    #fit the model on train data 
    km_dba = TimeSeriesKMeans(n_clusters=k, metric="softdtw",metric_params={"gamma":gamma_soft_dtw(dataset=train_series,         n_samples=200,random_state=0) }, max_iter=5,max_iter_barycenter=5, random_state=0).fit(train_series)
    centroids=km_dba.cluster_centers_
    #predict train (label)
    prediction_train=km_dba.predict(train_series)
    #prediction test (label)
    prediction_test=km_dba.predict(test_series)
    # days in clusters for visualization (k=2)
    cluster1=test_series[prediction_test==0]
    cluster2=test_series[prediction_test==1]
    # if the number of observations in the cluster k=0 < obs in cluster k=1 switch the labels both in train and test set 
    if len(cluster1)<len(cluster2): 
        # set 0 to 2
        np.place(prediction_test, prediction_test<1, [2])
        np.place(prediction_train, prediction_train<1, [2])
        # set 1 to 0
        np.place(prediction_test, prediction_test<2, [0])
        np.place(prediction_train, prediction_train<2, [0])
        # set 2 to 1
        np.place(prediction_test, prediction_test>1, [1])
        np.place(prediction_train, prediction_train>1, [1])
    return (prediction_train,prediction_test,centroids,cluster1,cluster2,data.name)
# prediction_train= array with cluster labels
# prediction_test= array with cluster labels
# centroids 3d-array of shape (k,n,1)
# cluster1 3d-array of shape (p1,n,1) where p1 is the number of period (12h/day) in cluster1
# cluster2 3d-array of shape (p2,n,1) where p2 is the number of period (12h/day) in cluster2
#data.name name of the column dataframe
# define a function that compute the clustering with euclidean distance
# data = dataframe.namecolumn (univariate time series)
# k is the number of clusters (default =2)
# aggregate='1min' '3min' '6min'
### hyperparameter for 1 minute 
# split= 131040/220320 (train: April-May-June)
# split= 87840/220320 (train: April-May)
### hyperparameter for 3 minute 
# train_split= 43680/73440 (train: April-May-June) 
# train_split= 29280/73440 (train: April-May)
### hyperparameter for 6 minute 
# train_split= 21840/36720 (train: April-May-June) 
# train_split= 14640/36720 (train: April-May)
# n  number of observations in a series: 
# daily series 1440 ('1minute'), 480 ('3minute'), 240 ('6minute')
# cycle series 12hours:  720 ('1minute'), 240 ('3minute'), 120 ('6minute')
def clustering_euclidean(data,train_split,n,k,aggregate):
    data1= data.resample(aggregate).median()
    #train-test split of the data
    train,test=train_test_split(data1,train_size=train_split, shuffle=False)
    # create daily series with the function (daily_series_pred)
    train_series=daily_series_pred(train,n)
    test_series=daily_series_pred(test,n)
    #CLUSTERING with soft-DTW
    #fit the model on train data 
    km_dba = TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=5,max_iter_barycenter=5, random_state=0).fit(train_series)
    centroids=km_dba.cluster_centers_
    #predict train (label)
    prediction_train=km_dba.predict(train_series)
    #prediction test (label)
    prediction_test=km_dba.predict(test_series)
    # days in clusters for visualization (k=2)
    cluster1=test_series[prediction_test==0]
    cluster2=test_series[prediction_test==1]
    # if the number of observations in the cluster k=0 < obs in cluster k=1 switch the labels both in train and test set 
    if len(cluster1)<len(cluster2): 
        # set 0 to 2
        np.place(prediction_test, prediction_test<1, [2])
        np.place(prediction_train, prediction_train<1, [2])
        # set 1 to 0
        np.place(prediction_test, prediction_test<2, [0])
        np.place(prediction_train, prediction_train<2, [0])
        # set 2 to 1
        np.place(prediction_test, prediction_test>1, [1])
        np.place(prediction_train, prediction_train>1, [1])
    return (prediction_train,prediction_test,centroids,cluster1,cluster2,data.name)
# prediction_train= array with cluster labels
# prediction_test= array with cluster labels
# centroids 3d-array of shape (k,n,1)
# cluster1 3d-array of shape (p1,n,1) where p1 is the number of period (12h/day) in cluster1
# cluster2 3d-array of shape (p2,n,1) where p2 is the number of period (12h/day) in cluster2
#data.name name of the column dataframe
# define a function that compute the clustering with soft-dtw metrics
# data = dataframe.namecolumn (univariate time series)
# k is the number of clusters (default =2)
# aggregate='1min' '3min' '6min'
### hyperparameter for 1 minute 
# split= 131040/220320 (train: April-May-June)
# split= 87840/220320 (train: April-May)
### hyperparameter for 3 minute 
# train_split= 43680/73440 (train: April-May-June) 
# train_split= 29280/73440 (train: April-May)
### hyperparameter for 6 minute 
# train_split= 21840/36720 (train: April-May-June) 
# train_split= 14640/36720 (train: April-May)
# n  number of observations in a series: 
# daily series 1440 ('1minute'), 480 ('3minute'), 240 ('6minute')
# cycle series 12hours:  720 ('1minute'), 240 ('3minute'), 120 ('6minute')
def clustering_dtw(data,train_split,n,k,aggregate):
    data1= data.resample(aggregate).median()
    #train-test split of the data
    train,test=train_test_split(data1,train_size=train_split, shuffle=False)
    # create daily series with the function (daily_series_pred)
    train_series=daily_series_pred(train,n)
    test_series=daily_series_pred(test,n)
    #CLUSTERING with soft-DTW
    #fit the model on train data 
    km_dba = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=5,max_iter_barycenter=5, random_state=0).fit(train_series)
    centroids=km_dba.cluster_centers_
    #predict train (label)
    prediction_train=km_dba.predict(train_series)
    #prediction test (label)
    prediction_test=km_dba.predict(test_series)
    # days in clusters for visualization (k=2)
    cluster1=test_series[prediction_test==0]
    cluster2=test_series[prediction_test==1]
    # if the number of observations in the cluster k=0 < obs in cluster k=1 switch the labels both in train and test set 
    if len(cluster1)<len(cluster2): 
        # set 0 to 2
        np.place(prediction_test, prediction_test<1, [2])
        np.place(prediction_train, prediction_train<1, [2])
        # set 1 to 0
        np.place(prediction_test, prediction_test<2, [0])
        np.place(prediction_train, prediction_train<2, [0])
        # set 2 to 1
        np.place(prediction_test, prediction_test>1, [1])
        np.place(prediction_train, prediction_train>1, [1])
    return (prediction_train,prediction_test,centroids,cluster1,cluster2,data.name)
# prediction_train= array with cluster labels
# prediction_test= array with cluster labels
# centroids 3d-array of shape (k,n,1)
# cluster1 3d-array of shape (p1,n,1) where p1 is the number of period (12h/day) in cluster1
# cluster2 3d-array of shape (p2,n,1) where p2 is the number of period (12h/day) in cluster2
#data.name name of the column dataframe
def plot(centroids,days1,days2,k,start,end,granularity,dataname): 
    # centroids (k 3d-arrays )
    # days (3d-arrays belonging in each cluster )
    # k number of cluster (default=2) 
    # start:0
    # stop: 12 or 24 (start and stop define the number of hours in the series)
    # granularity: 0.05 (3min) or 0.1 (6min) or  0.01667 (1min)
# plot centroids with random subset of the series 
    centroid1=centroids[k-2]
    centroid1=centroid1.reshape((len(centroid1), 1))
    centroid2=centroids[k-1]
    centroid2=centroid2.reshape((len(centroid2), 1))
    x=np.arange(start,end,granularity)
    len(x)
    #plt.figure(figsize=(35,30))
    plt.subplot(1,2,1)
    for i in range(len(days1)):
        plt.plot(x,days1[i],'k-',alpha=0.3)
    plt.plot(x,centroid1,'r',label =dataname,linewidth=3)
    plt.xlabel('number of hours in the series',fontsize=10)
    plt.title('cluster 1',fontsize=18)
    plt.subplot(1,2,2)
    for i in range(len(days2)):
        plt.plot(x,days2[i],'k-',alpha=0.3)
    plt.plot(x,centroid2,'r',label =dataname,linewidth=3)
    plt.xlabel('number of hours in the series',fontsize=10)
    plt.title('cluster 2',fontsize=18)
    return (plt.show())

# function for creating intervals in a day 
# it returns a list of even lenght containing start and end time of each cycle.
def between (n): #number of cycle in a day
    dt = date.today()
    curr=datetime.combine(dt, datetime.min.time())
    seq1 = []
    seq1.append(curr.strftime("%H:%M"))
    for x in range(n):
         curr = curr + timedelta(hours = 24/n) - timedelta(minutes=1)
         seq1.append(curr.strftime("%H:%M"))
         curr = curr + timedelta(minutes=1)
         seq1.append(curr.strftime("%H:%M"))
    del seq1[-1]
    return(seq1)

 # n: number of series to be created (number of cycle in a day)
# suggested lenght for a cycle (12h,8h,6h,4h)
# data: series containing only timestamp for which the value is 1 : 1= machine is stopped
def ground_truth(n,data):
    # freq in hours 
    converted_freq = f'{int(24/n)}'+'h'
    # initialize the final list
    series_list = []
    # create intervals in the day with between function
    intervals=between(n)
    # initialize the  first identificator to 00:00 
    h=0
    for x in range(n):
        temporary=data.between_time(intervals[x*2],intervals[x*2+1])
        # select the first entries
        temporary1 = temporary.groupby(temporary.index.date).apply(lambda x: x.iloc[[0]])
        # keep only the first entry of each day
        temporary1.index = temporary1.index.droplevel(0)
        # delete the time specification (the interval is already set) of each day
        temporary1.index =  temporary1.index.date
        if isinstance(temporary1.index, pd.core.indexes.base.Index):
            temporary1.index = pd.to_datetime(temporary1.index, utc=True)
        # add a unique identificator at each day
        temporary1.index = temporary1.index.normalize() + timedelta(hours = h) 
        # update interval identificator
        h=h+24/n
        # append series  
        series_list.append(temporary1)
    #concatenate series
    result = pd.concat(series_list)
    # reorder the series based on index
    result=result.sort_index()
    # fill jumps in the series
    result= result.asfreq(freq=converted_freq)
    result.index=result.index.tz_localize(None)
    return (result)
# result : series that start with the first cycle in which occur a stop and end with the last cycle in which a stop occur. 
# 1 indicates the presence of a problem in the interval, NaN no problem.

# change labels for calplot representation 
# label resulting label array 
# starting: starting date ex. '7/1/2018'
# periods: number of cycle 
#freq: duration of each cycle
def calplot_index(label,starting,periods,freq):
    new=[]
    for i in range(periods):
        if label[i] == 0:
            y=0.05
        elif label[i] !=0: 
            y=label[i]
        new.append(y)
    test_index=pd.date_range(starting, periods=periods, freq=freq)
    test_periods = pd.Series(new,index=test_index)
    return (test_periods)
# data: univariate time series (train set ) 131040 
# periodicity: list of varying cycle length [1440,720,480,360] 1 minute aggregation
# seasonality_label: corresponding string of cycle lengthin hours ['24h','12h','8h','6h']
# anomalies_label: univariate series, for each timestamp 1 if NORMAL 0 if BROKEN/MANTEINANCE (train set) 131040
def cycle1(data,periodicity,seasonality_label,anomalies_label):
    # initialize a list for store precision,recall,f1 scores
    PRECISION=[]
    RECALL=[]
    F1=[]
    #AVG_PRECISION=[]
    # initialize a dataframe where store the result 
    result=pd.DataFrame(data=None, columns=['precision','recall','f1'],index=seasonality_label)
    # compute a grid-search 
    for i in range(len(periodicity)):
        # decompose the series with an additive model and variable seasonality length that simulates the cycle
        model = sm.tsa.seasonal_decompose(data,period=periodicity[i],two_sided=False)
        # select the error series of the model
        error1=model.resid
        # delete Nan from the series --> due to moving average 
        error1.dropna(axis=0, inplace=True)
        # compute quantile 
        q1, q3 = error1.quantile([0.25, 0.75])
        # compute interquartile range 
        iqr = q3-q1
        #lower and upper bound for outliers 
        lower = q1 - (3*iqr) #-2*error1.std()
        upper = q3 + (3*iqr) # +2*error1.std()#
        # if the condition is not respected imput other value, otherwise keep the existing one
        error1.where(error1>lower,other=1, inplace=True)
        error1.where(error1<upper,other=1, inplace=True)
        error1.where(error1==1,other=0, inplace=True)
        # make of the same size of error anomalies_label based on Moving Average of the decomposition
        anomaly1=anomalies_label.drop(anomalies_label.head(int(periodicity[i])).index, inplace=False)
        #anomaly1=anomaly1.drop(anomaly1.tail(int(periodicity[i]/2)).index, inplace=False)
        # count anomalies 
        true =anomaly1.cumsum().max()
        #print('number of true anomalies',true)
        predicted=error1.cumsum().max()
        #print('number of predicted anomalies',predicted)
        # append metrics to the list
        RECALL.append(recall_score(anomaly1.values,error1.values))
        PRECISION.append(precision_score(anomaly1.values,error1.values))
        F1.append(f1_score(anomaly1.values,error1.values))
        #AVG_PRECISION.append(average_precision_score(anomaly1.values,error1.values))
    #assign RMSE to the dataframe 
    result['recall']=RECALL
    result['precision']=PRECISION
    result['f1']=F1
    #result['avg_precision']=AVG_PRECISION
    result1=pd.DataFrame(result.nlargest(1, 'precision'))
    return (result1)
# data: univariate time series 1 minute cadence
# split: starting point of the prediction 
# window_past: how many observations in the past to be cosidered 
# window_future: how many steps ahead forecast 
# ground_truth: univariate time series 0/1 (normal/anomaly)
# activate or not daily seasonality True/False
def prophet_anomaly(data,split,window_past, window_future, ground_truth,Daily):
    print(data.name)
    # initial dataframe for prophet
    input_prophet=pd.DataFrame(data=None, columns=['ds','y'])
    input_prophet['y']=data.iloc[split-window_past:split].values
    input_prophet['ds']=data.iloc[split-window_past:split].index  
    # model for prophet 
    m = Prophet(yearly_seasonality=False,weekly_seasonality=True,daily_seasonality=Daily).fit(input_prophet)
    # declare step for the forecasting 
    future = m.make_future_dataframe(periods=window_future, freq='min')
    # make the prediction store train and prediction in fcst dataframe
    fcst=m.predict(future)
    # change the index of the dataframe 
    fcst.index=fcst.ds
    # extract series for evaluations
    y_pred_train=fcst.yhat.iloc[0:window_past]
    y_pred_test=fcst.yhat.iloc[window_past:window_past+window_future]
    # initialize the error
    error_train=data.iloc[split-window_past:split]
    error_test=data.iloc[split:split+window_future]
    # compute the error
    error_train1=error_train.subtract(y_pred_train, axis = 0)
    error_test1=error_test.subtract(y_pred_test,axis=0)
    print('RMSE',mean_squared_error(error_test.values,y_pred_test.values))
    # 
    anomaly_test=ground_truth.iloc[split:split+window_future]
    # rule for anomaly detection 
    q1, q3 = error_train1.quantile([0.25, 0.75])
    # compute interquartile range 
    iqr = q3-q1
    #lower and upper bound for outliers 
    lower = q1 - (3*iqr) #-2*error1.std()
    upper = q3 + (3*iqr) # +2*error1.std()#
    # if the condition is not respected imput other value, otherwise keep the existing one
    error_test1.where(error_test1>lower,other=1, inplace=True)
    error_test1.where(error_test1<upper,other=1, inplace=True)
    error_test1.where(error_test1==1,other=0, inplace=True)
    precision=precision_score(anomaly_test.values,error_test1.values)
    recall=recall_score(anomaly_test.values,error_test1.values)
    f1=f1_score(anomaly_test.values,error_test1.values)
    print('precision:',precision)
    print('recall:',recall)
    print('f1:',f1)
    return (precision,recall,f1)
# walkforward dynamic baseline models that retrain the model every week (by default with most recent data), by move forward the split between train and test
# the models simulate the prediction of the anomalies in one week ahead for four times (total length of the prediction horizon equal to  28 days)
# data: univariate pd.series() contaning data from the sensor 
# split initial train/test division
# ground truth: univariate pd.series() containing labels 0/1
# window_future = how many steps in the future we would like to predict.
def dynamicbaseline (data,split,window_future,ground_truth):
    # initialize the dataframe where store z-scores from the sensor
    result=pd.DataFrame(data=None)
    for i in range (4): 
        # train test split (test set as always the same size, while train increases at each iteration )
        train_data = TimeSeries.from_pd(data.iloc[0:split+window_future*i])
        train_labels = TimeSeries.from_pd(ground_truth.iloc[0:split+window_future*i])
        test_data=TimeSeries.from_pd(data.iloc[split+window_future*i:split+window_future*i+window_future])
        # model configuration the model considers only most recent data (4 weeks in the past)
        config=DynamicBaselineConfig(trend='daily',train_window='4w',window_sz='1min')
        model = DynamicBaseline(config)
        print(f"Training {type(model).__name__}...")
        train_scores = model.train(train_data=train_data, anomaly_labels=train_labels)
        # get the anomalies scores converted in z-scores
        labels =model.get_anomaly_label(test_data)
        # return a Pandas dataframe with z-scores and timestamp as index
        labels1=labels.to_pd()
        result=result.append(labels1)
    return (result) 
# walkforward prophet models that retrain the model every week (with most recent data), by move forward the split between train and test
# the models simulate the prediction of the anomalies in 28 days, by retraining the model every week only with most recent data.
# data: univariate pd.series() contaning data from the sensor 
# split initial train/test division
# window_future how many steps ahead we would like to forecast each time 
# window_past how many historical data consider
# ground truth: univariate pd.series() containing labels 0/1
def prophetAD (data,split,window_future,window_past,ground_truth):
    # initialize the dataframe where store z-scores from the sensor
    result=pd.DataFrame(data=None)
    for i in range (4): 
        # train test split (test set as always the same size, as the train size)
        train_data = TimeSeries.from_pd(data.iloc[split+window_future*i-window_past:split-1+window_future*i])
        train_labels = TimeSeries.from_pd(ground_truth.iloc[split+window_future*i-window_past:split-1+window_future*i])
        test_data=TimeSeries.from_pd(data.iloc[split-1+window_future*i:split+window_future*i+window_future])
        # default configuration of prophet
        config = ProphetDetectorConfig(yearly_seasonality=False,weekly_seasonality=True,daily_seasonality=True,changepoint_prior_scale=0.05,changepoint_range=0.8)
        model= ProphetDetector(config)
        print(f"Training {type(model).__name__}...")
        train_scores = model.train(train_data=train_data, anomaly_labels=train_labels)
        # get z-scores anomaly values
        labels =model.get_anomaly_label(test_data)
        # store result in a pandas serie
        labels1=labels.to_pd()
        result=result.append(labels1)
    return (result) 
# walkforward  random cut forest models that retrain the model every week, by move forward the split between train and test
# the models simulate the prediction of the anomalies in 28 days retraining the model four times adding each time new data to the train set
# data: univariate pd.series() contaning data from the sensor 
# split initial train/test division
# window_future how many steps forward we would like to predict each time. 
# ground truth: univariate pd.series() containing labels 0/1
def RCFad2 (data,split,window_future,window_past,ground_truth):
    # set the number of trees in the forest 
    num_trees=128
    # initialize the dataframe where store z-scores from the sensor
    result=pd.DataFrame(data=None)
    for i in range (4): 
        # train test split (test set as always the same size, while train )
        train_data = TimeSeries.from_pd(data.iloc[split+window_future*i-window_past:split-1+window_future*i])
        # compute median
        median=data.iloc[split+window_future*i-window_past:split-1+window_future*i].quantile(q=0.5)
        # compute Median Absolute Deviation (MAD)
        mad=data.iloc[split+window_future*i-window_past:split-1+window_future*i].mad()
        # lower and upper bound for outliers 
        lower= median -3*mad
        upper=median + 3*mad
        # count observations lower than lower bound
        lower_count=(train_data.to_pd()<lower).sum()
        # count observations greater than upper bound 
        upper_count=(train_data.to_pd()>upper).sum()
        far_out=lower_count+upper_count+1
        print('number of far out observations in the train set',far_out)
        num_samples_per_tree= int(1/(far_out/(len(train_data.to_pd()))))
        # number of observations in the train set >= than num_samples_per_tree*n_estimators
        if num_samples_per_tree>int(len(train_data.to_pd())/num_trees):
            num_samples_per_tree=256
        print('number of sample per trees:',num_samples_per_tree)
        test_data=TimeSeries.from_pd(data.iloc[split-1+window_future*i:split+window_future*i+window_future])
        model = RandomCutForest(RandomCutForestConfig(n_estimators=num_trees, max_n_samples=num_samples_per_tree, dimensions=1))
        model.train(train_data=train_data)
        print(f"Training {type(model).__name__}...")
        labels =model.get_anomaly_label(test_data)
        labels1=labels.to_pd()
        result=result.append(labels1)
    return (result) 
# Point Adjusted evaluation
# input:  pandas dataframe containing the result for the scenario of interest and labels as last columns df.iloc[0:10080,]
def evaluation_PA(model1,model2,model3): 
    model1.name='DynamicBaseline'
    model2.name='Prophet'
    model3.name='RandomCutForest'
    # define ground truth labels from one of the model (all models have the same ground truth labels)
    test_labels = TimeSeries.from_pd(model1.labels)
    # for every sensor in the models
    for i in range(19):
        print()
        print(model1.iloc[:,i].name)
        print(model1.name)
        precision_model1 = TSADMetric.PointAdjustedPrecision.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model1.iloc[:,i]))
        #recall_mode1 = TSADMetric.PointAdjustedRecall.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model1.iloc[:,i]))
        #f1_model1 = TSADMetric.PointAdjustedF1.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model1.iloc[:,i]))
        mttd_model1 = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model1.iloc[:,i]))
        print(f"  Precision: {precision_model1:.4f}")
        #print(f" Recall:    {recall_mode1:.4f}")
        #print(f"  F1:        {f1_model1:.4f}")
        print(f"  MTTD:      {mttd_model1}")
        print(model2.name)
        precision_model2 = TSADMetric.PointAdjustedPrecision.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model2.iloc[:,i]))
        #recall_mode2 = TSADMetric.PointAdjustedRecall.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model2.iloc[:,i]))
        #f1_model2 = TSADMetric.PointAdjustedF1.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model2.iloc[:,i]))
        mttd_model2 = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model2.iloc[:,i]))
        print(f"  Precision: {precision_model2:.4f}")
        #print(f"  Recall:    {recall_mode2:.4f}")
        #print(f"  F1:        {f1_model2:.4f}")
        print(f"  MTTD:      {mttd_model2}")
        print(model3.name)
        precision_model3 = TSADMetric.PointAdjustedPrecision.value(ground_truth=test_labels, predict=TimeSeries.from_pd(model3.iloc[:,i]))
        #recall_mode3 = TSADMetric.PointAdjustedRecall.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model3.iloc[:,i]))
        #f1_model3 = TSADMetric.PointAdjustedF1.value(ground_truth=test_labels, predict=TimeSeries.from_pd(model3.iloc[:,i]))
        mttd_model3 = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=TimeSeries.from_pd(model3.iloc[:,i]))
        print(f"  Precision: {precision_model3:.4f}")
        #print(f"  Recall:    {recall_mode3:.4f}")
        #print(f"  F1:        {f1_model3:.4f}")
        print(f"  MTTD:      {mttd_model3}")
        print()
    return ()
# Revised point adjusted evaluation
# input: pandas dataframe containing the result for the scenario of interest and labels as last columns df.iloc[10080:20160,]
# df.iloc[30240:40320,]
def evaluation_RPA(model1,model2,model3): 
    model1.name='DynamicBaseline'
    model2.name='Prophet'
    model3.name='RandomCutForest'
    # define ground truth labels from one of the model (all models have the same ground truth labels)
    test_labels = TimeSeries.from_pd(model1.labels)
    # for every sensor in the models
    for i in range(19):
        print()
        print(model1.iloc[:,i].name)
        print(model1.name)
        precision_model1 = TSADMetric.Precision.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model1.iloc[:,i]), max_early_sec=600)
        recall_mode1 = TSADMetric.Recall.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model1.iloc[:,i]), max_early_sec=600)
        f1_model1 = TSADMetric.F1.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model1.iloc[:,i]), max_early_sec=600)
        mttd_model1 = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model1.iloc[:,i]))
        print(f"  Precision: {precision_model1:.4f}")
        print(f"  Recall:    {recall_mode1:.4f}")
        #print(f"  F1:        {f1_model1:.4f}")
        print(f"  MTTD:      {mttd_model1}")
        print(model2.name)
        precision_model2 = TSADMetric.Precision.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model2.iloc[:,i]), max_early_sec=600)
        recall_mode2 = TSADMetric.Recall.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model2.iloc[:,i]), max_early_sec=600)
        f1_model2 = TSADMetric.F1.value(ground_truth=test_labels, predict=TimeSeries.from_pd(model2.iloc[:,i]), max_early_sec=600)
        mttd_model2 = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict= TimeSeries.from_pd(model2.iloc[:,i]))
        print(f"  Precision: {precision_model2:.4f}")
        print(f"  Recall:    {recall_mode2:.4f}")
        #print(f"  F1:        {f1_model2:.4f}")
        print(f"  MTTD:      {mttd_model2}")
        print()
        print(model3.name)
        precision_model3 = TSADMetric.Precision.value(ground_truth=test_labels, predict=TimeSeries.from_pd(model3.iloc[:,i]), max_early_sec=600)
        recall_mode3 = TSADMetric.Recall.value(ground_truth=test_labels, predict=TimeSeries.from_pd(model3.iloc[:,i]), max_early_sec=600)
        f1_model3 = TSADMetric.F1.value(ground_truth=test_labels, predict=TimeSeries.from_pd(model3.iloc[:,i]), max_early_sec=600)
        mttd_model3 = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=TimeSeries.from_pd(model3.iloc[:,i]))
        print(f"  Precision: {precision_model3:.4f}")
        print(f"  Recall:    {recall_mode3:.4f}")
        #print(f"  F1:        {f1_model3:.4f}")
        print(f"  MTTD:      {mttd_model3}")
        print()
    return ()
# count the number of z-scores different from 0 df.iloc[20160:30240,]
def count_anomalies(model1,model2,model3): 
    model1.name='DynamicBaseline'
    model2.name='Prophet'
    model3.name='RandomCutForest'
    # for every sensor in the models
    for i in range(19):
        print()
        # print the name of the sensor
        print(model1.iloc[:,i].name)
        # count the number of anomalies 
        anomalies_model1 = (model1.iloc[:,i]!=0).sum()
        anomalies_model2 = (model2.iloc[:,i]!=0).sum()
        anomalies_model3 = (model3.iloc[:,i]!=0).sum()
        #print name of the model and number of false anomalies 
        if (anomalies_model1==0):
            print(model1.name,': ', anomalies_model1,'predicted false anomaly')
        else :
             print(model1.name,': ', anomalies_model1/len(model1.iloc[:,i])*100,'%  of false anomalies')
        if (anomalies_model2==0):    
            print(model2.name,': ', anomalies_model2,'predicted false anomaly')
        else:
            print(model2.name,': ', anomalies_model2/len(model2.iloc[:,i])*100,'%  of false anomalies')
        if (anomalies_model3==0):
            print(model3.name,': ', anomalies_model3,'predicted false anomaly')
        else:
            print(model3.name,': ', anomalies_model3/len(model3.iloc[:,i])*100,'%  of false anomalies')
        print()
    return ()
# functions for visualizations
# prophet
def prophetAD_visual (data,split,window_future,window_past,ground_truth):
    # train test split (test set as always the same size, as the train size)
    train_data = TimeSeries.from_pd(data.iloc[split-window_past:split-1])
    train_labels = TimeSeries.from_pd(ground_truth.iloc[split-window_past:split-1])
    test_data=TimeSeries.from_pd(data.iloc[split-1:split+window_future])
    test_labels=TimeSeries.from_pd(ground_truth.iloc[split-1:split+window_future])
    # default configuration of prophet
    config = ProphetDetectorConfig(yearly_seasonality=False,weekly_seasonality=True,daily_seasonality=True,changepoint_prior_scale=0.05,changepoint_range=0.8)
    model= ProphetDetector(config)
    train_scores = model.train(train_data=train_data, anomaly_labels=train_labels)
    # get z-scores anomaly values
    labels =model.get_anomaly_label(test_data)
    print(type(model).__name__)
    fig, ax = model.plot_anomaly(
    time_series=test_data, time_series_prev=train_data,
    filter_scores=True, plot_time_series_prev=True)
    plot_anoms(ax=ax, anomaly_labels=test_labels)
    plt.show()
    print()
    return () 
# dynamic baseline
def dynamicbaseline_visual (data,split,window_future,window_past,ground_truth): 
        # train test split (test set as always the same size, while train increases at each iteration )
        train_data = TimeSeries.from_pd(data.iloc[0:split])
        train_labels = TimeSeries.from_pd(ground_truth.iloc[0:split])
        test_data=TimeSeries.from_pd(data.iloc[split:split+window_future])
        test_labels=TimeSeries.from_pd(anomaly.iloc[split:split+window_future])
        # model configuration the model considers only most recent data (4 weeks in the past)
        config=DynamicBaselineConfig(trend='daily',train_window='4w',window_sz='1min')
        model = DynamicBaseline(config)
        train_scores = model.train(train_data=train_data, anomaly_labels=train_labels)
        # get the anomalies scores converted in z-scores
        labels =model.get_anomaly_label(test_data)
        print(type(model).__name__)
        fig, ax = model.plot_anomaly(
        time_series=test_data, time_series_prev=TimeSeries.from_pd(data.iloc[split-window_past:split]),
        filter_scores=True, plot_time_series_prev=True)
        plot_anoms(ax=ax, anomaly_labels=test_labels)
        plt.show()
        print()
        return ()
# random cut forest
def RCFad2_visual(data,split,window_future,window_past,ground_truth):
    # set the number of trees in the forest 
    num_trees=128
    # train test split (test set as always the same size, while train )
    train_data = TimeSeries.from_pd(data.iloc[split-window_past:split-1])
    # compute median
    median=data.iloc[split-window_past:split-1].quantile(q=0.5)
    # compute Median Absolute Deviation (MAD)
    mad=data.iloc[split-window_past:split-1].mad()
    # lower and upper bound for outliers 
    lower= median -3*mad
    upper=median + 3*mad
    # count observations lower than lower bound
    lower_count=(train_data.to_pd()<lower).sum()
    # count observations greater than upper bound 
    upper_count=(train_data.to_pd()>upper).sum()
    far_out=lower_count+upper_count+1
    print('number of far out observations in the train set',far_out)
    num_samples_per_tree= int(1/(far_out/(len(train_data.to_pd()))))
    # number of observations in the train set >= than num_samples_per_tree*n_estimators
    if num_samples_per_tree>int(len(train_data.to_pd())/num_trees):
        num_samples_per_tree=256
    print('number of sample per trees:',num_samples_per_tree)
    test_data=TimeSeries.from_pd(data.iloc[split-1:split+window_future])
    test_labels=TimeSeries.from_pd(ground_truth.iloc[split-1:split+window_future])
    model = RandomCutForest(RandomCutForestConfig(n_estimators=num_trees, max_n_samples=num_samples_per_tree, dimensions=1))
    model.train(train_data=train_data)
    labels =model.get_anomaly_label(test_data)
    print(type(model).__name__)
    fig, ax = model.plot_anomaly(
    time_series=test_data, time_series_prev=TimeSeries.from_pd(data.iloc[split-window_past:split]),
    filter_scores=True, plot_time_series_prev=True)
    plot_anoms(ax=ax, anomaly_labels=test_labels)
    plt.show()
    print()
    return () 


       
    
