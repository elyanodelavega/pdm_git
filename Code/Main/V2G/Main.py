# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:31:59 2020

@author: Yann
"""

from Forecast_Class import Forecast_LSTM, Forecast_ARIMA


import numpy as np
import pandas as pd
import random
import os

from to_video import to_video
from plot_res import plot_dropout_results, plot_MPC
from df_prepare import data_PV_csv, data_EV_csv, data_spot_market_csv
from keras.models import load_model


img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'
folder_path = 'C:/Users/Yann/Documents/EPFL/pdm_git/pdm_git/Code/Main/V2G/'

PV_csv = 'pv_simulation_evolene.csv'
data_PV = data_PV_csv(PV_csv)

EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(EV_csv)

SM_csv = 'spot_market.csv'
spot_prices = data_spot_market_csv(SM_csv)

#%% PV Forecast 
n_hour_future = 23
pred_variable = 'DCPmp'
ratio = 2

PV_model_forecast = Forecast_LSTM(pred_variable = pred_variable, data = data_PV,  n_hour_future = n_hour_future, ratio = ratio)

PV_model_forecast.build_LSTM(epochs = 50, dropout = 0.1, plot_results=True)
PV_LSTM = PV_model_forecast.LSTM_model

#%%

pred_variable = 'load'

Load_model_forecast = Forecast_ARIMA(data_EV, pred_variable = pred_variable)

#%%

model_number = 1
name = 'model'

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

warnings.filterwarnings("ignore")
#%% Optimization
    
from Model_Class import EV, House, Model
from plot_res import plot_MPC, plot_results_day_ahead, plot_results_deterministic
import time



t_res = PV_model_forecast.t_res
save = 0
columns = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load','ev_load','ev_grid ','y_buy',
           'y_sell','y_ch','y_dis','soc',
       'avail', 'episode']


EV1 = EV()
House1 = House()
episode_start = 1
n_episodes = 5

range_episodes = range(episode_start, episode_start + n_episodes)

methods = ['1. Fully deterministic',  '4. MPC deterministic', 
           '5. MPC stochastic']
costs = {m:[] for m in methods}

stats = {m:{e: None for e in range_episodes} for m in methods}

results = {m: pd.DataFrame(index = data_EV.index, columns = columns) for m in methods}

predictions_load = {m:{} for m in methods}

predictions_PV = {m:{} for m in methods}

MPC_results = {m:{} for m in methods}

time_algo = {m:[] for m in methods}

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
    start_0 = time.time()
    model_0 = Model(name = name + methods[0] + str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1, spot_prices = spot_prices)
    t_decision = t_start_episode
    t_forecast = episode.index[1]
    PV_predictions_0 = episode.loc[t_forecast:t_end_episode,'PV'].to_frame()
    Load_predictions_0 = episode.loc[t_forecast:t_end_episode,'load'].to_frame()
    model_0.optimize(t_decision, t_end_episode, PV_predictions_0,Load_predictions_0, forecasting = False, method = 'deterministic')
    decisions_0 = model_0.results_deterministic()
    cost = decisions_0.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum()
    costs['1. Fully deterministic'].append(cost)
    results['1. Fully deterministic'].loc[t_start_episode:t_end_episode] = decisions_0
    
    predictions_load['1. Fully deterministic'][e] = Load_predictions_0
    predictions_PV['1. Fully deterministic'][e] = PV_predictions_0
    end_0 = time.time()
    total_ep_0 = end_0 -start_0
    time_algo['1. Fully deterministic'].append(total_ep_0)

    
    #plot_results_deterministic(results_0, figname = str(start), img_path = img_paths['1. Fully deterministic'])
    
    start_3 = time.time()
    predictions_load['4. MPC deterministic'][e] = {}
    predictions_PV['4. MPC deterministic'][e] = {}
    MPC_results['4. MPC deterministic'][e] = {}
    # #MPC deterministic
    model_3 = Model(name = name + methods[1] + str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1, spot_prices = spot_prices)
    # k = start
    # k_path = img_paths['4. MPC deterministic'] + str(k) + '/' 
    # os.mkdir(k_path)
    for t in range(episode_length-1):
        #k += 1
        t_decision = episode.index[t]
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        
        Load_predictions_3 = Load_model_forecast.predict(t_forecast, t_end)
        PV_predictions_3 = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
        model_3.optimize(t_decision, t_end, PV_predictions_3, Load_predictions_3, forecasting = True, method = 'deterministic' )
        predictions_load['4. MPC deterministic'][e][t_decision]  = Load_predictions_3
        predictions_PV['4. MPC deterministic'][e] = PV_predictions_3
        results_3 = model_3.results_stochastic()
        MPC_results['4. MPC deterministic'][e][t_decision] = results_3
        #plot_MPC(decisions_3, results_3, figname = str(k), img_path = k_path)
    
    decisions_3 = model_3.decisions[:-1]
    decisions_3 = decisions_3.append(decisions_0.tail(1))
    decisions_3.loc[t_end_episode,'soc'] = model_3.decisions.loc[t_end,'soc']

    cost = decisions_3.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum()
    costs['4. MPC deterministic'].append(cost)
    results['4. MPC deterministic'].loc[t_start_episode:t_end_episode] = decisions_3
    
    end_3 = time.time()
    total_ep_3 = end_3 -start_3
    time_algo['4. MPC deterministic'].append(total_ep_3)
    #to_video(k_path)
    
    
    #MPC stochastic Expected
    start_4 = time.time()
    predictions_load['5. MPC stochastic'][e] = {}
    predictions_PV['5. MPC stochastic'][e] = {}
    MPC_results['5. MPC stochastic'][e] = {'results': {}, 'soc': {}}
    model_4 = Model(name = name + methods[2]+str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1, spot_prices = spot_prices)
    # k = start
    # k_path = img_paths['5. MPC stochastic'] + str(k) +'/'
    # os.mkdir(k_path)

    for t in range(episode_length-1):
        # k += 1
        t_decision = episode.index[t]
        
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        
        Load_predictions_4 = Load_model_forecast.predict(t_forecast, t_end)
        PV_predictions_4 = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.35, dataframe = True)

        model_4.optimize(t_decision, t_end, PV_predictions_4, Load_predictions_4, forecasting = True, method = 'expected value')
        decisions_4 = model_4.decisions
        results_4 = model_4.results_stochastic()
        MPC_results['5. MPC stochastic'][e]['results'][t_decision] = results_4
        
        SOC_4 = model_4.predictions_SOC()
        MPC_results['5. MPC stochastic'][e]['soc'][t_decision] = SOC_4
        
        #plot_MPC(decisions_4, results_4, SOC_4, PV_predictions_4, figname = str(k), img_path = k_path)
        predictions_load['5. MPC stochastic'][e][t_decision]  = Load_predictions_4
        predictions_PV['5. MPC stochastic'][e] = PV_predictions_4
    
    
    decisions_4 = model_4.decisions[:-1]
    decisions_4 = decisions_4.append(decisions_0.tail(1))
    decisions_4.loc[t_end_episode,'soc'] = model_4.decisions.loc[t_end,'soc']
    #results_4 = model_3.results_stochastic()
    cost = decisions_4.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum() + model_4.extra_cost
    costs['5. MPC stochastic'].append(cost)
    results['5. MPC stochastic'].loc[t_start_episode:t_end_episode] = decisions_4
    
    end_4 = time.time()
    total_ep_4 = end_4 -start_4
    time_algo['5. MPC stochastic'].append(total_ep_4)
    #to_video(k_path)

    model_number += 1

#%%
import pickle
names = ['opti', 'mpc_d', 'mpc_s']
for i,m in enumerate(methods):
    res = results[m].dropna()
    res.to_csv(f'results_{names[i]}_{n_episodes}.csv')
    file_res = open(f'results_{names[i]}_{n_episodes}.pickle', 'wb') 
    pickle.dump(res, file_res)
    
    if i != 0:
        pred_load = predictions_load[m]
        file_pred_load = open(f'load_pred_{names[i]}_{n_episodes}.pickle', 'wb') 
        pickle.dump(pred_load, file_pred_load)
        
        pred_pv = predictions_PV[m]
        file_pred_pv = open(f'pv_pred_{names[i]}_{n_episodes}.pickle', 'wb') 
        pickle.dump(pred_pv, file_pred_pv)
        
        full_res = MPC_results[m]
        file_full_res = open(f'full_res_{names[i]}_{n_episodes}.pickle', 'wb') 
        pickle.dump(full_res, file_full_res)


