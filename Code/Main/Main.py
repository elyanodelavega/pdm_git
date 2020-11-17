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
from keras.models import load_model


img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'


PV_csv = 'pv_simulation_evolene.csv'
data_PV = data_PV_csv(PV_csv)


#%% PV Forecast 
n_hour_future = 23
pred_variable = 'DCPmp'
ratio = 2

PV_model_forecast = Forecast(pred_variable = pred_variable, data = data_PV,  n_hour_future = n_hour_future, ratio = ratio)

PV_model_forecast.build_LSTM(epochs = 50, dropout = 0.1, plot_results=True)
PV_LSTM = PV_model_forecast.LSTM_model


#%%
EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(EV_csv)
model_number = 1
name = 'model'


#%% Optimization
    
from Model_Class import EV, House, Model
from Forecast_Class import Forecast
from plot_res import plot_MPC, plot_results_day_ahead, plot_results_deterministic



t_res = PV_model_forecast.t_res
save = 0
columns = ['pv', 'load', 'pv_ev', 'pv_load', 'grid_ev', 'grid_load', 'soc',
       'avail', 'episode']


EV1 = EV(SOC_min_departure = 0.95)
House1 = House()
episode_start = 10
n_episodes = 18

range_episodes = range(episode_start, episode_start + n_episodes)


methods = ['1. Fully deterministic',  '4. MPC deterministic', 
           '5. MPC stochastic']
costs = {m:[] for m in methods}
results = {m: pd.DataFrame(index = data_EV.index, columns = columns) for m in methods}
constraints_violation = {m:[] for m in methods}
alpha = 0.75

img_paths = {m:[] for m in methods}

if save:
    os.mkdir(img_folder_path+name+'_'+str(model_number))
    for m in methods:
        img_path = img_folder_path+name+'_'+str(model_number)+'/'+m+ '/'
        img_paths[m] = img_path
        os.mkdir(img_path)

figname = None

for e in range_episodes:
    episode = data_EV[data_EV.episode == e]
    t_start_episode = episode.index[0]
    t_end_episode = episode.index[-1]
    episode_length = episode.shape[0]
    
    # Full deterministic
    model_0 = Model(name = name + methods[0] + str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1)
    t_decision = t_start_episode
    t_forecast = episode.index[1]
    PV_predictions_0 = episode.loc[t_forecast:t_end_episode,'PV'].to_frame()
    model_0.optimize(t_decision, t_end_episode, 'PV', PV_predictions_0, forecasting = False, method = 'deterministic')
    decisions_0 = model_0.results_deterministic()[:-1]
    cost = decisions_0.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum()
    costs['1. Fully deterministic'].append(cost)
    results['1. Fully deterministic'].loc[t_start_episode:t_end_episode] = decisions_0

    constraints_violation['1. Fully deterministic'].append(model_0.constraints_violation)
    
    #plot_results_deterministic(results_0, figname = str(start), img_path = img_paths['1. Fully deterministic'])
    
    
    # #MPC deterministic
    model_3 = Model(name = name + methods[1] + str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1)
    # k = start
    # k_path = img_paths['4. MPC deterministic'] + str(k) + '/' 
    # os.mkdir(k_path)
    for t in range(episode_length-1):
        #k += 1
        t_decision = episode.index[t]
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        
        PV_predictions_3 = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
        model_3.optimize(t_decision, t_end, 'PV', PV_predictions_3, forecasting = True, method = 'deterministic' )
        
        #results_3 = model_3.results_stochastic()
        #plot_MPC(decisions_3, results_3, figname = str(k), img_path = k_path)
    
    decisions_3 = model_3.decisions[:-1]
    #results_3 = model_3.results_stochastic()
    cost = decisions_3.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum()
    costs['4. MPC deterministic'].append(cost)
    results['4. MPC deterministic'].loc[t_start_episode:t_end_episode] = decisions_3
    constraints_violation['4. MPC deterministic'].append(model_3.constraints_violation)
    #to_video(k_path)
    
    
    #MPC stochastic Expected
    model_4 = Model(name = name + methods[2]+str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1)
    # k = start
    # k_path = img_paths['5. MPC stochastic'] + str(k) +'/'
    # os.mkdir(k_path)
    for t in range(episode_length-1):
        # k += 1
        t_decision = episode.index[t]
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        
        PV_predictions_4 = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.35, dataframe = True)

        model_4.optimize(t_decision, t_end, 'PV', PV_predictions_4, forecasting = True, method = 'expected value')
        #decisions_4 = model_4.decisions
        #results_4 = model_4.results_stochastic()
        #SOC_4 = model_4.predictions_SOC()
        #plot_MPC(decisions_4, results_4, SOC_4, PV_predictions_4, figname = str(k), img_path = k_path)
    
    
    decisions_4 = model_4.decisions[:-1]
    
    #results_4 = model_3.results_stochastic()
    cost = decisions_4.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum()
    costs['5. MPC stochastic'].append(cost)
    results['5. MPC stochastic'].loc[t_start_episode:t_end_episode] = decisions_3
    constraints_violation['5. MPC stochastic'].append(model_4.constraints_violation)
    #to_video(k_path)

    model_number += 1

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))

cost2 = {m: np.cumsum([c/1000 for c in costs[m]]) for m in costs}
for m in costs.keys():
    plt.plot(cost2[m], label = m)

plt.legend(ncol = 2)
plt.xlabel('Episodes')
plt.ylabel('Electricity bought [kWh]')
plt.title('Cumulative electricity bought comparison')

#%%
for e in range(12,18):
    
    for m in results.keys():
        fig, ax1 = plt.subplots(figsize=(10,7)) 
        episode = results[m][results[m].episode == e]
        grid_ev_bought = episode.loc[:,['grid_ev']].sum(axis = 0).sum()
        grid_load_bought = episode.loc[:,['grid_load']].sum(axis = 0).sum()
        total = grid_ev_bought + grid_load_bought
        
        x = np.arange(len(episode))
        ax1.plot(x, list(episode.grid_ev/1000), label = f'grid to ev: {int(grid_ev_bought/1000)} kWh')
        ax1.plot(x, list(episode.grid_load/1000), label = f'grid to load: {int(grid_load_bought/1000)} kWh')
        ax1.plot(x, list(episode.pv/1000), color = 'green', label = 'PV')
        ax1.fill_between(x, list(episode.pv/1000), color = 'green', alpha = 0.3)
        plt.legend()
        ax2 = ax1.twinx() # ax for plotting EV SOC evolution
        ax2.plot(x,list(episode.soc*100),'--', color = 'grey')
        ax2.set_ylabel('EV SoC [%]', color='black')
        ax2.set_ylim([0,100])
        ax2.grid(False)
        
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Electricity bought [kWh]')
        plt.title(f'{m}, Episode {e}, Power bought: {int(total/1000)} kWh ')
#%%



