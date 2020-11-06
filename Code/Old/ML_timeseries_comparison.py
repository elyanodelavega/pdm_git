# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:18:27 2020

@author: Yann
"""


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


import df_prepare as dfp

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

data = dfp.df_from_time_series('Timeseries_PV.csv')

#min_temp = np.min(data_TS.T2m)



def regression_results(y_true, y_pred):
# Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


data.loc[:,'Hour_m1'] = data.loc[:,'G_i'].shift()

data.loc[:,'Hour_m1_Diff'] = data.loc[:,'Hour_m1'].diff()

data['Hour_m2'] = data['Hour_m1'].shift()

data['Hour_m2_Diff'] = data['Hour_m2'].diff()

data['last_6'] = np.zeros(len(data))
data['last_6'].iloc[:5] = [np.mean(data.G_i[i-6:i]) for i in range(6, len(data.G_i))]





data = data.dropna()


X_train = data['2010':'2015'].drop(['G_i'], axis = 1)
y_train = data.loc['2010':'2015', 'G_i']
X_test = data['2016'].drop(['G_i'], axis = 1)
y_test = data.loc['2016', 'G_i']

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
models.append(('KNN', KNeighborsRegressor())) 
models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees

# Evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     # TimeSeries Cross validation
#  tscv = TimeSeriesSplit(n_splits=10)
    
#  cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
#  results.append(cv_results)
#  names.append(name)
#  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
# # Compare Algorithms
# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# plt.show()


