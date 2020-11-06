# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:31:59 2020

@author: Yann
"""

from Forecast_Class import Forecast


import numpy as np
import pandas as pd
import random
import os

from to_video import to_video
from plot_res import plot_dropout_results, plot_MPC
from df_prepare import data_PV_csv, data_EV_csv

img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM_git/Images'
# pickle_file_name = 'df_PV_load_EV_big'

# data = df_from_pickle(pickle_file_name)

PV_csv = 'pv_simulation_evolene.csv'
data_PV = data_PV_csv(PV_csv)


#%% PV Forecast 
n_hour_future = 24
pred_variable = 'DCPmp'
ratio = 2

PV_model_forecast = Forecast(pred_variable = pred_variable, data = data_PV,  n_hour_future = n_hour_future, ratio = ratio)

PV_model_forecast.build_LSTM(epochs = 50, dropout = 0.15, plot_results=True)
PV_LSTM = PV_model_forecast.LSTM_model

#%% Dropout evalutation

PV_model_forecast = Forecast(pred_variable = pred_variable, data = data_PV,  n_hour_future = n_hour_future, ratio = ratio)
test_days = 1
threshold = int(len(data_PV)/1.5)
idx_start = random.randint(n_hour_future * ratio + 1, threshold)
dropouts = np.arange(0.2, 0.55, step = 0.05)
n_iterations = 20
predictions, predictions_dropout = PV_model_forecast.uncertainty_evaluation(PV_LSTM, test_days, idx_start, 
                                                                  dropouts, n_iterations)                                                                       



#%% Plot dropout results
# plot_dropout_results(data_full = data_PV,pred_variable = pred_variable, 
#                      predictions = predictions, predictions_dropout = predictions_dropout,
#                      n_hour_future = n_hour_future, plot_cumulative = True)

#%%
from Forecast_Class import df_predictions_unique, df_predictions_uncertainty

model_number = 1
name = 'model'

#%%
EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(EV_csv)

#%%
from Model_Class import EV, House, Model
from plot_res import plot_MPC_uncertainty

t_res = PV_model_forecast.t_res
save = 0

MPC_days = 3
threshold = int(len(data_EV)/1.5)
idx_start = random.randint(n_hour_future * ratio + 1, threshold)


idx_end = int(idx_start + MPC_days * 24)
t_MPC_start = data_EV.index[idx_start]

EV1 = EV()
House1 = House()
model = Model(name = name + str(model_number), data_full = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)

img_path = img_folder_path+name+str(model_number)+'/'

if save:
    os.mkdir(img_path)

figname = None
for i in range(idx_start, idx_end):
    t_decision = data_EV.index[i]
    t_forecast = data_EV.index[i+1]
    
    
    PV_predictions = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.5, dataframe = True)
    # df = df_predictions_uncertainty(data_EV = data_EV, time = t_forecast, n_hour_future = n_hour_future, 
    #              forecast_array = PV_predictions, pred_variable = 'PV', scale_PV = House1.scale_PV)
    
    model.optimize(t_decision = t_decision, pred_variable = 'PV', predictions = PV_predictions) 
    #results = model.results()
    
    decisions = model.decisions
    
    SOC = model.predictions_SOC()
    if save:
        figname = img_path+name + str(i)
    
    plot_MPC_uncertainty(decisions, SOC, PV_predictions, model.load)

if save:
    to_video(img_path)

model_number += 1

#%%
import matplotlib.pyplot as plt
def plot_interval(df):
    arr = np.array(df.values).astype('float64')
    med = np.median(arr, axis = 1)
    low = np.quantile(arr, 0.2, axis = 1)
    
    high = np.quantile(arr, 0.8, axis = 1)
    
    x = np.arange(df.shape[0])
    
    plt.fill_between(x, low, high,color = 'red', alpha = 0.2)
    plt.plot(med, color = 'red')
    plt.show()

plot_interval(PV_predictions)
plot_interval(SOC)

# plt.plot(pred_high)

    

















