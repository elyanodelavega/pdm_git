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

img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'
# pickle_file_name = 'df_PV_load_EV_big'

# data = df_from_pickle(pickle_file_name)

PV_csv = 'pv_simulation_evolene.csv'
data_PV = data_PV_csv(PV_csv)


#%% PV Forecast 
n_hour_future = 23
pred_variable = 'DCPmp'
ratio = 2

PV_model_forecast = Forecast(pred_variable = pred_variable, data = data_PV,  n_hour_future = n_hour_future, ratio = ratio)

PV_model_forecast.build_LSTM(epochs = 50, dropout = 0.1, plot_results=True)
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
name = 'model_uncertainty'

#%%
EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(EV_csv)

#%%
# from Model_Class import EV, House, Model
# from plot_res import plot_MPC_uncertainty

# t_res = PV_model_forecast.t_res
# save = 0

# MPC_days = 1
# threshold = int(len(data_EV)/1.5)
# idx_start = random.randint(n_hour_future * ratio + 1, threshold)


# idx_end = int(idx_start + MPC_days * 24)
# idx_end = idx_start + 1
# t_MPC_start = data_EV.index[idx_start]



# EV1 = EV()
# House1 = House()
# model = Model(name = name + str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
#               t_res = t_res,  EV = EV1, House = House1)

# img_path = img_folder_path+name+'_'+str(model_number)+'/'

# if save:
#     os.mkdir(img_path)

# figname = None
# for i in range(idx_start, idx_end):
#     t_decision = data_EV.index[i]
#     t_forecast = data_EV.index[i+1]
    
    
    
#     PV_predictions = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
#                                                             dropout = 0.35, dataframe = True)

#     predictions = data_EV.loc[t_decision:t_decision + pd.Timedelta(hours = 23), 'PV']
    
#     model.optimize(t_decision, 'PV', predictions, forecasting = False)
#     #model.optimize(t_decision = t_decision, pred_variable = 'PV', predictions = PV_predictions) 
#     results = model.results_determinitic()
    
#     decisions = model.decisions
    
#     SOC = model.predictions_SOC()
#     if save:
#         figname = img_path+name + str(i)
    
#     plot_MPC_uncertainty(decisions, SOC, model.predictions, model.load, figname)

# if save:
#     to_video(img_path)

# model_number += 1

#%%
    
from Model_Class import EV, House, Model
from Forecast_Class import Forecast

model_number = 1
name = 'model'

t_res = PV_model_forecast.t_res
save = 0

MPC_days = 3
threshold = int(len(data_EV)-(MPC_days + 1)*24)

idx_start = random.randint(n_hour_future * ratio + 1, threshold)
t_MPC_start = data_EV.index[idx_start]

idx_end = int(idx_start + MPC_days * 24)

days_range = np.arange(idx_start, idx_end, 24)

EV1 = EV()
House1 = House()
model = Model(name = name + str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)

if save:
    os.mkdir(img_folder_path)

methods = ['deterministic', 'day_ahead', 'MPC_deterministic', 
           'MPC_stochastic_Exp', 'MPC_stochastic_CVaR']
costs = {m:[] for m in methods}
alpha = 0.8

figname = None
for start in days_range:
    t_day_start = data_EV.index[start]
    t_day_end = data_EV.index[start+23]
    
    # Full deterministic
    model_0 = Model(name = name + methods[0] + str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    t_decision = t_day_start
    t_forecast = data_EV.index[start+1]
    PV_predictions_0 = data_EV.loc[t_forecast:t_day_end, 'PV']
    model_0.optimize(t_decision, 'PV', PV_predictions_0, forecasting = False)
    costs['deterministic'].append(model_0.cost)
    
    # Day ahead dispatch plan
    model_1 = Model(name = name + methods[1]+ str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    PV_predictions_1 = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
    model_1.optimize(t_decision, 'PV', PV_predictions_1, forecasting = True)
    costs['day_ahead'].append(model_1.cost)
    
    
    #MPC deterministic
    model_2 = Model(name = name + methods[2] + str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    for t in range(start,start+23):
        t_decision = data_EV.index[t]
        t_forecast = data_EV.index[t+1]
        
        PV_predictions_2 = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
        model_2.optimize(t_decision, 'PV', PV_predictions_2, forecasting = True, method = 'deterministic')
    
    decisions_2 = model_2.decisions
    cost = np.sum(decisions_2['pv'])
    costs['MPC_deterministic'].append(cost)
    
    
    #MPC stochastic Expected
    model_3 = Model(name = name + methods[3]+str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    for t in range(start,start+23):
        t_decision = data_EV.index[t]
        t_forecast = data_EV.index[t+1]
        
        PV_predictions_3 = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.35, dataframe = True)

        model_3.optimize(t_decision, 'PV', PV_predictions_3, forecasting = True, method = 'expected value')
    
    decisions_3 = model_3.decisions
    cost = np.sum(decisions_3['pv'])
    costs['MPC_stochastic_Exp'].append(cost)
    
    
    #MPC stochastic CVaR
    model_4 = Model(name = name + methods[4]+ str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    
    for t in range(start,start+23):
        t_decision = data_EV.index[t]
        t_forecast = data_EV.index[t+1]
        
        PV_predictions_4 = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.35, dataframe = True)

        model_4.optimize(t_decision, 'PV', PV_predictions_4, forecasting = True, method = 'CVaR', parameters = {'alpha': alpha})
    
    decisions_4 = model_4.decisions
    cost = np.sum(decisions_4['pv'])
    costs['MPC_stochastic_CVaR'].append(cost)


    model_number += 1

#%%
import matplotlib.pyplot as plt
for m in costs.keys():
    plt.plot(costs[m], label = m)

plt.legend()