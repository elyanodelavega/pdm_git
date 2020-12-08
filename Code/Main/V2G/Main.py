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
from df_prepare import data_PV_csv, data_EV_csv, data_spot_market_csv, prices_romande_energie
from keras.models import load_model


img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'
folder_path = 'C:/Users/Yann/Documents/EPFL/pdm_git/pdm_git/Code/Main/V2G/'

PV_csv = 'pv_simulation_evolene.csv'
data_PV = data_PV_csv(folder_path + PV_csv)

EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(folder_path +EV_csv)

SM_csv = 'spot_market.csv'
spot_prices = data_spot_market_csv(folder_path +SM_csv)



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
from MPC import run_MPC, run_MPC_save

import time

t_res = PV_model_forecast.t_res
save = 0
columns = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load','ev_load','ev_grid','y_buy',
           'y_sell','y_ch','y_dis','soc',
       'avail', 'episode']

V2X = 1
semi_dynamic_pricing = True
save_mode = False

if semi_dynamic_pricing:
    spot_prices = prices_romande_energie(spot_prices)
    
EV1 = EV(eta_EV_ch = 0.95)
House1 = House()


episode_start = 1
n_episodes = 30


range_episodes = range(episode_start, episode_start + n_episodes)

names = ['v2g_opti', 'v2g_mpc_d', 'v2g_mpc_s', 'v2g_mpc_s_cvar','v2g_mpc_s_cvar_cost', 'v2g_mpc_s_cvar_soc']


methods = ['1. Fully deterministic',  '4. MPC deterministic', 
           '5. MPC stochastic', '6. MPC stochastic CVaR',
           '7. MPC stochastic CVaR cost', '8. MPC stochastic CVaR soc']

algorithms = {methods[i]:names[i]  for i in range(len(names))}

MPC_methods = ['4. MPC deterministic', 
               '5. MPC stochastic', '6. MPC stochastic CVaR',
               '7. MPC stochastic CVaR cost', '8. MPC stochastic CVaR soc']

MPC_opti_methods = {'4. MPC deterministic': 'deterministic', '5. MPC stochastic': 'expected value',
                  '6. MPC stochastic CVaR': 'CVaR','7. MPC stochastic CVaR cost': 'Markowitz', 
                  '8. MPC stochastic CVaR soc': 'Markowitz' }

MPC_parameters = {'4. MPC deterministic': None, '5. MPC stochastic': None,
                  '6. MPC stochastic CVaR': {'alpha': 0.75},
                  '7. MPC stochastic CVaR cost':{'alpha_cost': 0.75,
                                                 'alpha_soc': 0},
                  '8. MPC stochastic CVaR soc':{'alpha_cost': 0,
                                                 'alpha_soc': 0.75}}

results = {m: pd.DataFrame(index = data_EV.index, columns = columns) for m in methods}

predictions_load = {m:{} for m in methods}

predictions_PV = {m:{} for m in methods}

MPC_results = {m:{} for m in methods}

time_algo = {m:{} for m in methods}

img_paths = {m:[] for m in methods}

if save:
    os.mkdir(img_folder_path+name+'_'+str(model_number))
    for m in methods:
        img_path = img_folder_path+name+'_'+str(model_number)+'/'+m+ '/'
        img_paths[m] = img_path
        os.mkdir(img_path)

figname = None

parameters = {'alpha': 0.75}

for e in range_episodes:
    episode = data_EV[data_EV.episode == e]
    t_start_episode = episode.index[0]
    t_end_episode = episode.index[-1]
    episode_length = episode.shape[0]
    
    
    # Full deterministic
    start_0 = time.time()
    model_0 = Model( data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1, spot_prices = spot_prices, V2X = V2X)
    t_decision = t_start_episode
    t_forecast = episode.index[1]
    PV_predictions_0 = episode.loc[t_forecast:t_end_episode,'PV'].to_frame()
    Load_predictions_0 = episode.loc[t_forecast:t_end_episode,'load'].to_frame()
    model_0.optimize(t_decision, t_end_episode, PV_predictions_0,Load_predictions_0, forecasting = False, method = 'deterministic')
    decisions_0 = model_0.results_deterministic()
    
    results['1. Fully deterministic'].loc[t_start_episode:t_end_episode] = decisions_0
    
    predictions_load['1. Fully deterministic'][e] = Load_predictions_0
    predictions_PV['1. Fully deterministic'][e] = PV_predictions_0
    end_0 = time.time()
    total_ep_0 = end_0 -start_0
    time_algo['1. Fully deterministic'][e] = total_ep_0
    
    for m in MPC_methods:
        model = Model(data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1, spot_prices = spot_prices, V2X = V2X)
        
        opti_method = MPC_opti_methods[m]
        opti_parameters = MPC_parameters[m]
        
        if save_mode:
            results[m], MPC_results[m][e], predictions_load[m][e], predictions_PV[m][e], time_algo[m][e] = run_MPC_save(m, episode, model, decisions_0,results[m],
                                                                                                                        Load_model_forecast, PV_model_forecast, PV_LSTM,
                                                                                                                        opti_method, opti_parameters,
                                                                                                                        n_hour_future)
                                                                                             
        else:
            results[m] = run_MPC(m, episode, model, decisions_0,results[m],Load_model_forecast, PV_model_forecast, PV_LSTM,
                                                                                                    opti_method, opti_parameters,
                                                                                                    n_hour_future)
                                                                                                    
    model_number += 1

#%%
import pickle
res_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/'

for i,m in enumerate(methods):
    res = results[m].dropna()
    res = res[res.episode < e]
    res.to_csv(res_folder_path+f'results_{names[i]}_{n_episodes}.csv')
    file_res = open(res_folder_path+f'results_{names[i]}_{n_episodes}.pickle', 'wb') 
    pickle.dump(res, file_res)
    
    if i != 0 and save:
        pred_load = predictions_load[m]
        file_pred_load = open(res_folder_path+f'load_pred_{names[i]}_{n_episodes}.pickle', 'wb') 
        pickle.dump(pred_load, file_pred_load)
        
        pred_pv = predictions_PV[m]
        file_pred_pv = open(res_folder_path+f'pv_pred_{names[i]}_{n_episodes}.pickle', 'wb') 
        pickle.dump(pred_pv, file_pred_pv)
        
        full_res = MPC_results[m]
        file_full_res = open(res_folder_path+f'full_res_{names[i]}_{n_episodes}.pickle', 'wb') 
        pickle.dump(full_res, file_full_res)


