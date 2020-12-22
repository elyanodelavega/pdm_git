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

model_number = 2
name = 'model_uncertainty'

#%%
EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(EV_csv)



#%%
    
from Model_Class import EV, House, Model
from Forecast_Class import Forecast
from plot_res import plot_MPC, plot_results_day_ahead, plot_results_deterministic
model_number = model_number + 1
name = 'model'

t_res = PV_model_forecast.t_res
save = 1

MPC_days = 1
threshold = int(len(data_EV)-(MPC_days + 1)*24)

idx_start = random.randint(n_hour_future * ratio + 1, threshold)
t_MPC_start = data_EV.index[idx_start]

idx_end = int(idx_start + MPC_days * 24)

days_range = np.arange(idx_start, idx_end, 24)

EV1 = EV()
House1 = House()
model = Model(name = name + str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)


methods = ['1. Fully deterministic', '2. Day ahead derministic','3. Day ahead stochastic', '4. MPC deterministic', 
           '5. MPC stochastic']
costs = {m:[] for m in methods}
results = {m:[] for m in methods}
alpha = 0.75

img_paths = {m:[] for m in methods}

if save:
    os.mkdir(img_folder_path+name+'_'+str(model_number))
    for m in methods:
        img_path = img_folder_path+name+'_'+str(model_number)+'/'+m+ '/'
        img_paths[m] = img_path
        os.mkdir(img_path)

figname = None

for start in days_range:
    t_day_start = data_EV.index[start]
    t_day_end = data_EV.index[start+23]
    
    # Full deterministic
    model_0 = Model(name = name + methods[0] + str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    t_decision = t_day_start
    t_forecast = data_EV.index[start+1]
    PV_predictions_0 = data_EV.loc[t_forecast:t_day_end, 'PV'].to_frame()
    model_0.optimize(t_decision, 'PV', PV_predictions_0, forecasting = False, method = 'deterministic')
    results_0 = model_0.results_deterministic()
    costs['1. Fully deterministic'].append(model_0.cost)
    results['1. Fully deterministic'].append(results_0)
    
    plot_results_deterministic(results_0, figname = str(start), img_path = img_paths['1. Fully deterministic'])
    
    # Day ahead dispatch plan
    model_1 = Model(name = name + methods[1]+ str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    PV_predictions_1 = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
    model_1.optimize(t_decision, 'PV', PV_predictions_1, forecasting = True, method = 'day_ahead')
    cost_1, results_1 = model_1.day_ahead_update()
    costs['2. Day ahead derministic'].append(cost_1)
    results['2. Day ahead derministic'].append(results_1)
    plot_results_day_ahead(results_1, figname = str(start), img_path = img_paths['2. Day ahead derministic'])
    
    # Stochastic Day ahead
    model_2 = Model(name = name + methods[2]+ str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    PV_predictions_2 = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.35, dataframe = True)
    model_2.optimize(t_decision, 'PV', PV_predictions_2, forecasting = True, method = 'day_ahead')
    cost_2, results_2 = model_2.day_ahead_update()
    costs['3. Day ahead stochastic'].append(cost_2)
    results['3. Day ahead stochastic'].append(results_2)
    plot_results_day_ahead(results_2, PV_predictions_2, figname = str(start), img_path = img_paths['3. Day ahead stochastic'])
    
    
    # #MPC deterministic
    model_3 = Model(name = name + methods[2] + str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    k = start
    k_path = img_paths['4. MPC deterministic'] + str(k) + '/' 
    os.mkdir(k_path)
    for t in range(start,start+24):
        k += 1
        t_decision = data_EV.index[t]
        print(t_decision)
        t_forecast = data_EV.index[t+1]
        
        PV_predictions_3 = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
        model_3.optimize(t_decision, 'PV', PV_predictions_3, forecasting = True, method = 'deterministic' )
        decisions_3 = model_3.model_decisions()
        results_3 = model_3.results_stochastic()
        plot_MPC(decisions_3, results_3, figname = str(k), img_path = k_path)
    
    decisions_3 = model_3.model_decisions()
    results_3 = model_3.results_stochastic()
    cost = decisions_3.loc[:,['pv_ev','grid_ev']].sum(axis = 0).sum()
    costs['4. MPC deterministic'].append(cost)
    results['4. MPC deterministic'].append(decisions_3)
    to_video(k_path)
    
    
    #MPC stochastic Expected
    model_4 = Model(name = name + methods[3]+str(model_number), data_EV = data_EV, t_start = t_MPC_start, 
              t_res = t_res,  EV = EV1, House = House1)
    k = start
    k_path = img_paths['5. MPC stochastic'] + str(k) +'/'
    os.mkdir(k_path)
    for t in range(start,start+24):
        k += 1
        t_decision = data_EV.index[t]
        t_forecast = data_EV.index[t+1]
        
        PV_predictions_4 = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.35, dataframe = True)

        model_4.optimize(t_decision, 'PV', PV_predictions_4, forecasting = True, method = 'expected value')
        decisions_4 = model_4.model_decisions()
        results_4 = model_4.results_stochastic()
        SOC_4 = model_4.predictions_SOC()
        plot_MPC(decisions_4, results_4, SOC_4, PV_predictions_4, figname = str(k), img_path = k_path)

    
    decisions_4 = model_4.model_decisions()
    
    results_4 = model_3.results_stochastic()
    cost = decisions_4.loc[:,['pv_ev','grid_ev']].sum(axis = 0).sum()
    costs['5. MPC stochastic'].append(cost)
    results['5. MPC stochastic'].append(decisions_4)
    to_video(k_path)

    model_number += 1

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))

cost2 = {m: np.cumsum([c/1000 for c in costs[m]]) for m in costs}
for m in costs.keys():
    plt.plot(cost2[m], label = m)

plt.legend(ncol = 2)
plt.xlabel('Days')
plt.ylabel('Electricity bought [kWh]')
plt.title('Cumulative electricity bought comparison')

#%%
import math
import seaborn as sns

plt.figure(figsize=(10,7))
actual = list(PV_predictions_0.values/1000)
predict_no_drop = list(PV_predictions_1.values*4/1000)
predictions_drop = list(PV_predictions_4.values*4/1000)
sns.boxplot(data = predictions_drop)
plt.plot(actual, color = 'black', label = 'Actual')
ticks = np.arange(0, 24+1, step=6)
labels = [str(ticks[i]) for i in range(len(ticks))]
    
plt.xticks(ticks = ticks, labels = labels)
plt.xlabel('Forecast window')
plt.ylabel('[kW]')
plt.title('PV Distribution prediction, dropout: 0.35')
plt.plot(predict_no_drop, linestyle = '--', color = 'black', label = 'Prediction no dropout')
plt.annotate('Model Settings: ' + 'a', xy=(0.06, .015),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='bottom', fontsize=10)
plt.legend()
plt.show()
