# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:20:54 2020

@author: Yann
"""

import numpy as np
import pandas as pd
from df_prepare import data_EV_csv
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
 
EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(EV_csv)

import warnings

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 
#%%
def evaluate_arima_model(X, arima_order):
    
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    
    predictions = list()
    for t in range(len(test)):
        arima = ARIMA(history, order=arima_order)
        model = arima.fit(disp=0)
        yhat = arima.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
 

def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_conf = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_conf = mse, order
                    print(f'ARIMA{order} MSE=(int{mse})')
                except:
                    continue

    print(f'Best: ARIMA{order} MSE=(int{mse})')

    return best_conf


#%%

series = data_EV.load

p_values = np.arange(4)
d_values = np.arange(2)
q_values =  np.arange(3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)



#%%
best_parameters = (2, 0, 1)
for e in range(5,15):
    previous = list(data_EV[data_EV.episode < e].load)
    episode = list(data_EV[data_EV.episode == e].load)
    
    history = [l for l in previous]
    
    predictions = []
    predictions_low = []
    predictions_high = []
    x = range(len(episode))
    
    for t in range(len(episode)):
        arima = ARIMA(history, order=best_parameters)
        model = arima.fit(disp=0)
        yhat = model.forecast()[0]
        yhat_low = model.forecast()[2][0][0]
        yhat_high = model.forecast()[2][0][1]
        predictions.append(yhat)
        predictions_low.append(yhat_low)
        predictions_high.append(yhat_high)
        
        history.append(episode[t])
    
    plt.figure(figsize = (10,7))
    plt.plot(x, episode, label = 'expected')
    plt.plot(x, predictions, label = 'predicted')
    plt.fill_between(x, predictions_low, predictions_high, color = 'grey', alpha = 0.2)
    plt.legend()
    plt.title(f'Episode {e}')
    plt.show()

#%%

