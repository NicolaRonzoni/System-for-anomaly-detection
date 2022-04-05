{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score 
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import time
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
# function for pre-processing thee data 
def pre_process_PM(data):
    # mark Random Failure as O in Target 
    data.loc[data['Failure Type']=='Random Failures', 'Target'] = 0 
    # mark Random Failures as No Failure 
    data.loc[data['Failure Type']=='Random Failures', 'Failure Type'] = 'No Failure'
    #label encoder of Type and Failure Type
    le = LabelEncoder()
    # categories follow the alphabetic order
    # high low medium replaced by 0,1,2
    data['Type']= le.fit_transform(data.loc[:,["Type"]].values)
    #No failure (1), Tool wear failure(4), Heat dissipation failure (0), Power failure (2), Overstrain failure (3)  
    data['Failure Type'] = le.fit_transform(data.loc[:,["Failure Type"]].values)
    # create the new variable Heat dissipation
    data['Heat dissipation [K]'] = data['Air temperature [K]'] - data['Process temperature [K]']
    data['Heat dissipation [K]'] = data['Heat dissipation [K]'].abs()
    # create the new variable Power 
    data['Power [W x sec]'] = data['Torque [Nm]'] *( data['Rotational speed [rpm]']* 0.10472)
    # create the new variable Overstrain
    data['Overstrain [min x Nm]']=data['Tool wear [min]']* data['Torque [Nm]']
    # drop useless features 
    data = data.drop(["UDI","Product ID", "Air temperature [K]","Torque [Nm]"],axis = 1)
    #reorder the variable
    data=data[['Type','Power [W x sec]','Process temperature [K]','Heat dissipation [K]','Tool wear [min]','Overstrain [min x Nm]','Rotational speed [rpm]','Target','Failure Type']]
    return (data)
# plausability of fault 
def feat_prob(feature,data):
    x,y = [],[]
    for j in data[feature].unique():
        temp = data
        temp = temp[temp[feature]>=j]
        y.append(round((temp.Target.mean()*100),2))
        x.append(j)
    return(x,y)
# Split training and testing function
# take a dataframe and test_size (0.33, or 0.2 for example)
def split_training_testing(data,test_size):
    X  = data.iloc[:, :-2].values
    y  = data.loc[:,['Target','Failure Type']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
# return 6 arrays:
# 0: X_train, 1: X_test, 2: X_train_scaled, 3:X_test_scaled, 4:y_train, 5: y_test
# Gradient Boosting Machine
# regressors:  split[2] for scaled train regressors
# labels: split[4] for labels of the train set
# avg_type: 'micro' or 'macro'
def grid_gbm (regressors,labels,type_avg):
    # function for the maximization of the target
    def gbm_cl_bo(max_depth, max_features, learning_rate, n_estimators, subsample):
        params_gbm = {}
        params_gbm['max_depth'] = int(max_depth)
        params_gbm['max_features'] = max_features
        params_gbm['learning_rate'] = learning_rate
        params_gbm['n_estimators'] = int(n_estimators)
        params_gbm['subsample'] = subsample
        classifier = GradientBoostingClassifier(random_state=123, **params_gbm)
        multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
        multi_target_classifier.fit(regressors,labels)
        y_pred=multi_target_classifier.predict(regressors)
        #target=precision_score(labels[:,1], y_pred[:,1],average=type_avg)
        target=balanced_accuracy_score(labels[:,1], y_pred[:,1])
        #target=recall_score(labels[:,1], y_pred[:,1],average=type_avg)
        #target=f1_score(labels[:,1], y_pred[:,1],average=type_avg)
        return (target)
    # Candidates
    params_gbm ={
        'max_depth':(2, 10),
        'max_features':(0.8, 1),
        'learning_rate':(0.01, 1),
        'n_estimators':(80, 150),
        'subsample': (0.8, 1)
    }
    gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm, random_state=111)
    gbm_bo.maximize(init_points=20, n_iter=5)
    params_gbm = gbm_bo.max['params']
    params_gbm['max_depth'] = int(params_gbm['max_depth'])
    params_gbm['n_estimators'] = int(params_gbm['n_estimators'])
    print(params_gbm)
    return(params_gbm)
#Function for Performance Metrics micro
# predicted 2D-array containing predicted classes (machine status / Failure type)
# ground_truth 2D-array containing true classes (machine status /Failure type)
# type_avg 'micro' or 'macro'
def performance_failure_type(ground_truth,predicted,type_avg):
    print("Test Precision (Failure Type)      : ",round(precision_score(ground_truth[:,1], predicted[:,1], average=type_avg)*100,2),"%")
    print("Test Recall (Failure Type)         : ",round(recall_score(ground_truth[:,1], predicted[:,1],average=type_avg)*100,2),"%")
    print("Test F1-Score (Failure Type) : ",round(f1_score(ground_truth[:,1], predicted[:,1],average=type_avg)*100,2),"%")
    print("Test Balanced Accuracy Score (Failure Type):",round(balanced_accuracy_score(ground_truth[:,1], predicted[:,1])*100,2),"%")
    return ()
# Performance Metrics Machine Status
# predicted 2D-array containing predicted classes (machine status / Failure type)
# ground_truth 2D-array containing true classes (machine status /Failure type)
def performance_machine_status (ground_truth,predicted):
    print("Test Precision (Machine Status)      : ",round(precision_score(ground_truth[:,0], predicted[:,0])*100,2),"%")
    print("Test Recall (Machine Status)         : ",round(recall_score(ground_truth[:,0], predicted[:,0])*100,2),"%")
    print("Test F1-score (Machine Status) : ",round(f1_score(ground_truth[:,0], predicted[:,0])*100,2),"%")
    return ()
# Confusion Matrix function for machine status and failure type 
# predicted 2D-array containing predicted classes (machine status / Failure type)
# ground_truth 2D-array containing true classes (machine status /Failure type)
# labels : list of labels 
def confusion_machine_status(ground_truth,predicted,labels=['No Failure','Failure']):
    cm = confusion_matrix(ground_truth[:,0],predicted[:,0])
    disp    = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels)
    fig, ax = plt.subplots(figsize = (14,14))
    plt.rcParams.update({'font.size': 14})
    disp.plot(cmap = plt.cm.Greys, ax   = ax)
    plt.xticks(rotation=30, ha='right')
    plt.plot()
    return ()
def confusion_failure_type(ground_truth,predicted,labels=['Heat Dissipation Failure','No Failure','Overstrain Failure','Power Failure','Tool Wear Failure']):
    cm = confusion_matrix(ground_truth[:,1],predicted[:,1])
    disp    = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels)
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize = (28,22))
    disp.plot(cmap = plt.cm.Greys, ax   = ax)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=30,ha='right')
    plt.plot()
    return ()
# Adam Boosting Machine grid_search function
# regressors:  split[2] for scaled train regressors
# labels: split[4] for labels of the train set
# avg_type: 'micro' or 'macro'
def grid_abm(regressors,labels,avg_type):
    # function for the maximization of the target
    def abm_cl_bo(learning_rate, n_estimators):
        params_abm = {}
        params_abm['learning_rate'] = learning_rate
        params_abm['n_estimators'] = int(n_estimators)
        classifier = AdaBoostClassifier(random_state=123, **params_abm)
        multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
        multi_target_classifier.fit(regressors, labels)
        y_pred=multi_target_classifier.predict(regressors)
        #target=precision_score(labels[:,1], y_pred[:,1],average=avg_type)
        #target=f1_score(labels[:,1], y_pred[:,1],average=avg_type)
        target=balanced_accuracy_score(labels[:,1], y_pred[:,1])
        #target=recall_score(labels[:,1], y_pred[:,1],average=avg_type)
        return target
    # set of possible candidates
    params_abm ={
        'learning_rate':(0.01, 1),
        'n_estimators':(80, 150),
    }
    abm_bo = BayesianOptimization(abm_cl_bo, params_abm, random_state=111)
    abm_bo.maximize(init_points=20, n_iter=5)
    params_abm = abm_bo.max['params']
    params_abm['n_estimators'] = int(params_abm['n_estimators'])
    print(params_abm)
    return(params_abm)
# Random Forest Machine
def grid_rfm(regressors,labels,avg_type):
    weights = [None, "balanced_subsample"]
    def rfm_cl_bo(max_depth, max_features, n_estimators):
        params_rfm = {}
        params_rfm['max_depth'] = int(max_depth)
        params_rfm['max_features'] = max_features
        params_rfm['n_estimators'] = int(n_estimators)
        classifier = RandomForestClassifier(random_state=123, **params_rfm)
        multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
        multi_target_classifier.fit(regressors, labels)
        y_pred=multi_target_classifier.predict(regressors)
        #target=precision_score(labels[:,1], y_pred[:,1],average=avg_type)
        target=balanced_accuracy_score(labels[:,1], y_pred[:,1])
        #target=recall_score(labels[:,1], y_pred[:,1],average=avg_type)
        #target=f1_score(labels[:,1], y_pred[:,1],average=avg_type)
        return target
    #set of possible candidates
    params_rfm ={
        'max_depth':(3, 10),
        'max_features':(0.8, 1),
        'n_estimators':(80, 150),
    }
    # bayes optimization
    rfm_bo = BayesianOptimization(rfm_cl_bo, params_rfm, random_state=111)
    rfm_bo.maximize(init_points=20, n_iter=5)
    # return the best hyperparameter
    params_rfm = {}
    params_rfm['max_depth'] = int(rfm_bo.max["params"]["max_depth"])
    params_rfm['max_features'] = rfm_bo.max["params"]["max_features"]
    params_rfm['n_estimators'] = int(rfm_bo.max["params"]["n_estimators"])
    print (params_rfm)
    return (params_rfm)
# Create function grid for neural network
# regressors: train regressors split[2]
# labels: train labels split[4]
# avg_type: "micro" or "macro"
def grid_nn(regressors,labels,avg_type):
    solver=['lbfgs', 'sgd', 'adam']
    activation=['identity', 'logistic', 'tanh', 'relu']
    learning_rate=['constant', 'invscaling', 'adaptive']
    early_stopping=[True,False]
    # maximize the target
    def nn_cl_bo(neurons1, neurons2, neurons3, layers, activation_function, optimizer, initial_learning_rate, batch_size,
               momentum, validation_fraction, early_stop,alpha,max_iter):
        params_nn={}
        params_nn['batch_size'] = int(batch_size)
        params_nn['validation_fraction']=validation_fraction
        params_nn['momentum']=momentum
        params_nn['solver']=solver[int(optimizer)]
        params_nn['activation']=activation[int(activation_function)]
        params_nn['learning_rate']=learning_rate[int(initial_learning_rate)]
        params_nn['early_stopping']=early_stopping[int(early_stop)]
        params_nn['max_iter']=int(max_iter)
        params_nn['alpha']=alpha
        if int(layers)==3:
            params_nn['hidden_layer_sizes']=(int(neurons1),int(neurons2),int(neurons3),)
        elif int(layers)==2:
            params_nn['hidden_layer_sizes']=(int(neurons1),int(neurons2),)
        else :
            params_nn['hidden_layer_sizes']=(int(neurons1),)
        classifier = MLPClassifier(random_state=123, **params_nn)
        multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
        multi_target_classifier.fit(regressors,labels)
        y_pred=multi_target_classifier.predict(regressors)
        #target=precision_score(labels[:,1], y_pred[:,1],average=avg_type)
        target=balanced_accuracy_score(labels[:,1], y_pred[:,1])
        #target=recall_score(labels[:,1], y_pred[:,1],average=avg_type)
        #target=f1_score(labels[:,1], y_pred[:,1],average=avg_type)
        return (target)
    # set of possible candidates
    params_nn ={
        'validation_fraction':(0,0.99),
        'momentum':(0,1),
        'batch_size':(16,256),
        'early_stop':(0,1.99), # int 0,1
        'initial_learning_rate':(0,2.99) ,# int 0,1,2,
        'activation_function':(0,3.99), # int 0,1,2,3
        'optimizer':(0,2.99), # int 0,1,2
        'neurons1':(int(regressors.shape[1]),int(5/2*regressors.shape[1])),
        'neurons2':(int(2/3*regressors.shape[1]),int(3/2*regressors.shape[1])),
        'neurons3':(int(1/2*regressors.shape[1]),int(regressors.shape[1])), 
        'layers':(1,3.99), #int 1,2,3 
        'max_iter':(2000,5000),
        'alpha':(0.00001,0.01)
    }
    # bayes optimization
    nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=111)
    nn_bo.maximize(init_points=20, n_iter=5)
    params_nn={}
    params_nn["activation"]= activation[int(nn_bo.max["params"]["activation_function"])]
    params_nn["momentum"] = nn_bo.max["params"]["momentum"]
    params_nn["validation_fraction"]=nn_bo.max["params"]["validation_fraction"]
    params_nn["batch_size"] = int(nn_bo.max["params"]["batch_size"])
    params_nn["learning_rate"]= learning_rate[int(nn_bo.max["params"]["initial_learning_rate"])]
    params_nn["solver"]= solver[int(nn_bo.max["params"]["optimizer"])]
    params_nn["early_stopping"]= early_stopping[int(nn_bo.max["params"]["early_stop"])]
    params_nn["max_iter"]= int(nn_bo.max["params"]["max_iter"])
    params_nn["alpha"]= nn_bo.max["params"]["alpha"]
    if int(nn_bo.max["params"]["layers"])==3:
        params_nn["hidden_layer_sizes"]=(int(nn_bo.max["params"]["neurons1"]),int(nn_bo.max["params"]["neurons2"]),int(nn_bo.max["params"]["neurons3"]),)
    elif int(nn_bo.max["params"]["layers"])==2:
        params_nn["hidden_layer_sizes"]=(int(nn_bo.max["params"]["neurons1"]),int(nn_bo.max["params"]["neurons2"]),)
    else:
        params_nn["hidden_layer_sizes"]=(int(nn_bo.max["params"]["neurons1"]),)
    print(params_nn)
    return (params_nn)
# pre process function for electrical fault 
# definition of failure type variable and machine status variable. 
def pre_process_EF(data):
# define a function for failure type classification in a unique variable    
    def conditions_FT(s):
        if (s['G']== 1) and (s['C'] == 0) and (s['B'] == 0) and (s['A']==1):
            return "Fault between Phase A and ground"
        elif (s['G']== 0) and (s['C'] == 0) and (s['B'] == 1) and (s['A']==1):
            return "Fault between Phase A and Phase B"
        elif (s['G']== 1) and (s['C'] == 0) and (s['B'] == 1) and (s['A']==1):
            return "Fault between Phase A,B and ground"
        elif (s['G']== 0) and (s['C'] == 1) and (s['B'] == 1) and (s['A']==1):
            return "Fault between all three phases"
        elif (s['G']== 1) and (s['C'] == 1) and (s['B'] == 1) and (s['A']==1):
            return "Three phase symmetrical fault"
        else: 
            return "No failure"
    data['Failure Type'] = data.apply(conditions_FT, axis=1)
# define a function for machibe status classification 
    def conditions_MS(s):
        if (s['Failure Type']== "No failure"):
            return 0
        else: 
            return 1
    data['Target'] = data.apply(conditions_MS, axis=1)    
    data= data.drop(["G","C", "B","A"],axis = 1)
    data=data[["Ia","Ib","Ic","Va","Vb","Vc","Target","Failure Type"]]
    return (data) 
def grid_nn_ef(regressors,labels,avg_type):
    solver=['lbfgs','sgd', 'adam']
    activation=['identity', 'logistic', 'tanh', 'relu']
    learning_rate=['constant', 'invscaling', 'adaptive']
    early_stopping=[True,False]
    # maximize the target
    def nn_cl_bo(neurons1, neurons2, layers, activation_function, optimizer, initial_learning_rate, batch_size,
               momentum, validation_fraction, early_stop,alpha,max_iter):
        params_nn={}
        params_nn['batch_size'] = int(batch_size)
        params_nn['validation_fraction']=validation_fraction
        params_nn['momentum']=momentum
        params_nn['solver']=solver[int(optimizer)]
        params_nn['activation']=activation[int(activation_function)]
        params_nn['learning_rate']=learning_rate[int(initial_learning_rate)]
        params_nn['early_stopping']=early_stopping[int(early_stop)]
        params_nn['max_iter']=int(max_iter)
        params_nn['alpha']=alpha
        if int(layers)==2:
            params_nn['hidden_layer_sizes']=(int(neurons1),int(neurons2),)
        else :
            params_nn['hidden_layer_sizes']=(int(neurons1),)
        classifier = MLPClassifier(random_state=123, **params_nn)
        multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
        multi_target_classifier.fit(regressors,labels)
        y_pred=multi_target_classifier.predict(regressors)
        #target=precision_score(labels[:,1], y_pred[:,1],average=avg_type)
        target=balanced_accuracy_score(labels[:,1], y_pred[:,1])
        #target=recall_score(labels[:,1], y_pred[:,1],average=avg_type)
        #target=f1_score(labels[:,1], y_pred[:,1],average=avg_type)
        return (target)
    # set of possible candidates
    params_nn ={
        'validation_fraction':(0,0.99),
        'momentum':(0,1),
        'batch_size':(16,128),
        'early_stop':(0,1.99), # int 0,1
        'initial_learning_rate':(0,2.99) ,# int 0,1,2,
        'activation_function':(0,3.99), # int 0,1,2,3
        'optimizer':(0,2.99), # int 0,1
        'neurons1':(int(regressors.shape[1]),int(5/2*regressors.shape[1])),
        'neurons2':(int(2/3*regressors.shape[1]),int(3/2*regressors.shape[1])),
        'layers':(1,2.99), #int 1,2
        'max_iter':(80000,90000),
        'alpha':(0.00001,0.01)
    }
    # bayes optimization
    nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=111)
    nn_bo.maximize(init_points=20, n_iter=5)
    params_nn={}
    params_nn["activation"]= activation[int(nn_bo.max["params"]["activation_function"])]
    params_nn["momentum"] = nn_bo.max["params"]["momentum"]
    params_nn["validation_fraction"]=nn_bo.max["params"]["validation_fraction"]
    params_nn["batch_size"] = int(nn_bo.max["params"]["batch_size"])
    params_nn["learning_rate"]= learning_rate[int(nn_bo.max["params"]["initial_learning_rate"])]
    params_nn["solver"]= solver[int(nn_bo.max["params"]["optimizer"])]
    params_nn["early_stopping"]= early_stopping[int(nn_bo.max["params"]["early_stop"])]
    params_nn["max_iter"]= int(nn_bo.max["params"]["max_iter"])
    params_nn["alpha"]= nn_bo.max["params"]["alpha"]
    if int(nn_bo.max["params"]["layers"])==2:
        params_nn["hidden_layer_sizes"]=(int(nn_bo.max["params"]["neurons1"]),int(nn_bo.max["params"]["neurons2"]),)
    else:
        params_nn["hidden_layer_sizes"]=(int(nn_bo.max["params"]["neurons1"]),)
    print(params_nn)
    return (params_nn)   
    
    