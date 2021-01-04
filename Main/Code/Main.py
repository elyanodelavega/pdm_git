# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:31:59 2020

@author: Yann
"""

import Forecast_Class
from Forecast_Class import Forecast_LSTM, Forecast_ARIMA

from Model_Class import EV, House, Model
from MPC import run_MPC, run_MPC_save
import pickle
import time


import numpy as np
import pandas as pd
import random
import os

from to_video import to_video
from df_prepare import data_PV_csv, data_EV_csv, data_spot_market_csv, prices_romande_energie
from keras.models import load_model

#%%
img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'
folder_path = 'C:/Users/Yann/Documents/EPFL/pdm_git/pdm_git/Main/Code/'

PV_csv = 'pv_simulation_evolene.csv'
data_PV = data_PV_csv(folder_path + PV_csv)

EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(folder_path +EV_csv)

SM_csv = 'spot_market.csv'
spot_prices = data_spot_market_csv(folder_path +SM_csv)

def time_prevision(time_algo, methods_left, episodes_left):
    average_time_per_episode_all = np.sum([np.mean(time_algo[k]) for k in time_algo.keys()])
    
    rest_episodes_full = average_time_per_episode_all*episodes_left
    
    average_time_per_episode_methods = np.sum(np.mean(time_algo[k]) for k in methods_left)
    
    time_left = rest_episodes_full + average_time_per_episode_all
    
    return int(time_left)

#%%
def time_prevision_method(time_algo, method, episodes_left):
    average_time_per_episode_all = np.mean(time_algo[method]) 
    
    time_left = average_time_per_episode_all*episodes_left
    
    
    return int(time_left)
#%%
plot_split = True

if plot_split:
    df = data_PV['DCPmp'][:data_EV.index[-1]].resample('D').sum()
    t_test = pd.Timestamp(year = 2020, month = 9, day = 9)
    t_val = pd.Timestamp(year = 2020, month = 8, day = 19)
    idx_test = list(df.index).index(t_test)
    idx_val = list(df.index).index(t_val)
    
    ticks = [0,idx_val, idx_test, len(df)-1]
    ticks = [df.index[i] for i in ticks]
    labels = [f'{i.year}-{i.month}-{i.day}' for i in ticks]
    import matplotlib.pyplot as plt
    plt.figure(figsize = (16,9),dpi = 600)
    plt.bar(df.index[:idx_val],df[df.index[:idx_val]]/1000, label = 'train')
    plt.bar(df.index[idx_val:idx_test],df[df.index[idx_val:idx_test]]/1000, label = 'validation')
    plt.bar(df.index[idx_test:],df[df.index[idx_test:]]/1000, label = 'test')
    plt.xticks(ticks = ticks, labels = labels, rotation = 45)
    plt.legend(fontsize = 15)
    plt.title('DCPmp grouped by day',fontsize = 16)
    plt.ylabel('kWh',fontsize = 15)
    plt.grid(True)
    plt.show()
    

#%% PV Forecast 
n_hour_future = 23
pred_variable = 'DCPmp'
ratio = 2

data_PV_train = data_PV[:data_EV.index[0]]
PV_model_forecast = Forecast_LSTM(pred_variable = pred_variable, data = data_PV,  n_hour_future = n_hour_future, ratio = ratio)

PV_model_forecast.build_LSTM(epochs = 50, dropout = 0.1, plot_results=True)
PV_LSTM = PV_model_forecast.LSTM_model

#%% Evaluate dropout
from plot_res import plot_dropout_results
evaluate = False
dropouts = [0.25,0.3,0.35,0.4,0.5, 0.6, 0.7, 0.8]
model = PV_LSTM
test_days = 10
idx = 130
n_iter = 20

if evaluate:
    pred,pred_dropout = PV_model_forecast.uncertainty_evaluation(model,test_days,idx,dropouts,n_iter)
    
    plot_dropout_results(data_PV_train,pred_variable, pred, pred_dropout, n_hour_future, plot_cumulative = True, plot_boxplot = True, boxplot_dropouts = [0.35])
#%%
idx_start = 129
t_start = data_EV.index[idx_start]
s = 3
ticks = [0,s,2*s, 22,s+22, 2*s+22]
labels = ['k', 'k+1','k+2', 'k+23', 'k+1+23', 'k+2+23']
plt.figure(figsize = (16,9), dpi = 600)
plt.grid(True)
linewidth = 4
color_1 = 'red'
color_2 = 'cyan'
color_3 = 'dimgray'
PV_predictions = PV_model_forecast.predict(model = PV_LSTM, time = data_EV.index[idx_start], dataframe = False)
plt.plot(range(len(PV_predictions)),PV_predictions/1000, label = 'step k', linewidth = linewidth, color = color_1)
PV_predictions = PV_model_forecast.predict(model = PV_LSTM, time = data_EV.index[idx_start+ s], dataframe = False)
plt.plot(range(s,len(PV_predictions)+s),PV_predictions/1000,'--', label = 'step k + 1', linewidth = linewidth + 1, color = color_2)
PV_predictions = PV_model_forecast.predict(model = PV_LSTM, time = data_EV.index[idx_start+ 2*s], dataframe = False)
plt.plot(range(2*s,len(PV_predictions)+2*s),PV_predictions/1000,'-.', label = 'step k + 2', linewidth = linewidth + 1, color = color_3)
plt.xticks(ticks,labels)
plt.legend()
plt.show()
#%%
pred_variable = 'load'

Load_model_forecast = Forecast_ARIMA(data_EV, pred_variable = pred_variable)

#%%
plot_load_forecast = True
idx = 500
t_forecast = data_EV.index[idx]
t_d = data_EV.index[idx-1]
t_end = data_EV.index[idx+23]
if plot_load_forecast:
    load_values = list(data_EV.loc[t_d:t_end,'load'])
    load_decision = load_values[0]
    plt.figure(figsize = (16,9), dpi = 500)
    plt.grid(True)
    plt.plot(load_values,label = 'Actual',marker = 'o', linewidth = 3)
    
    load_mean = Load_model_forecast.predict_with_stats(t_forecast, t_end)
    load_mean.insert(0, load_decision)
    plt.plot(load_mean, label ='Predicted on mean', marker = 'o')
    
    load_median = Load_model_forecast.predict_with_stats(t_forecast, t_end, method = 'median')
    load_median.insert(0, load_decision)
    plt.plot(load_median, label ='Predicted on median', marker = 'o')

    arima_simple = Load_model_forecast.predict(t_forecast, t_end, dataframe = False, iterative = True)
    arima_simple.insert(0,load_decision)
    plt.plot(arima_simple, label ='ARMA, order = (2,1)', marker = 'o')

    plt.xlabel('Hours ahead', fontsize = 14)
    plt.ylabel('kW')
    plt.title('Load forecast comparison')
    plt.legend(fontsize = 15)
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
    


res_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/'

t_res = PV_model_forecast.t_res

columns = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load','ev_load','ev_grid','y_buy',
            'y_sell','y_ch','y_dis','soc',
        'avail', 'episode']

lambda_soc = {'cost': 0.44,
              'pv': 0.02,
              'peak': 0.23}
################################### PARAMETERS ########################################
V2X = 1
semi_dynamic_pricing = True
save_mode = True
objective_1 = 'pv'
speciality = '_'+objective_1
soc_penalty = 1.2



if semi_dynamic_pricing:
    spot_prices = prices_romande_energie(data_EV)[['buy','sell']]
    
EV1 = EV(eta_EV_ch = 0.95,SOC_min_departure = 1)
House1 = House()


episode_start = 1
n_episodes = 60

range_episodes = range(episode_start, episode_start + n_episodes)

MPC_methods = ['4. MPC deterministic',
                '5. MPC stochastic',
                  '8. MPC stochastic CVaR soc']

MPC_methods = ['4. MPC deterministic']

          
################################## NAMES #############################################
names = ['opti', 'mpc_d', 'mpc_s', 'mpc_s_cvar_cost', 'mpc_s_cvar_soc']

if V2X:
    names = ['v2g_'+n for n in names]

else:
    prefix = ''

methods = ['1. Fully deterministic',  '4. MPC deterministic', 
            '5. MPC stochastic', '7. MPC stochastic CVaR obj1',
            '8. MPC stochastic CVaR soc']

algorithms = {methods[i]:names[i]  for i in range(len(names))}



MPC_opti_methods = {'4. MPC deterministic': 'deterministic', '5. MPC stochastic': 'expected value',
                  '6. MPC stochastic CVaR': 'CVaR','7. MPC stochastic CVaR obj1': 'Markowitz', 
                  '8. MPC stochastic CVaR soc': 'Markowitz' }

MPC_parameters = {'4. MPC deterministic': None, '5. MPC stochastic': None,
                  '6. MPC stochastic CVaR': {'alpha': 0.75},
                  '7. MPC stochastic CVaR obj1':{'alpha_obj1': 0.75,
                                                  'alpha_soc': 0},
                  '8. MPC stochastic CVaR soc':{'alpha_obj1': 0,
                                                  'alpha_soc': 0.75}}

time_algo = {m:[] for m in methods}


for i,m in enumerate(MPC_methods):
    
    name = algorithms[m]
    
    predictions_load = {e: None for e in range_episodes}
    
    predictions_PV = {e: None for e in range_episodes}
    
    MPC_results = {e: None for e in range_episodes}


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
        model_0.optimize(t_decision, t_end_episode, PV_predictions_0,Load_predictions_0, 
                          objective_1 = objective_1, forecasting = False, method = 'deterministic',
                          soc_penalty = soc_penalty, lambda_soc = lambda_soc[objective_1])
        decisions_0 = model_0.results_deterministic()
        

        end_0 = time.time()
        total_ep_0 = end_0 -start_0
        time_algo['1. Fully deterministic'].append(total_ep_0)
        
        
        if e > episode_start:
            t_left = time_prevision_method(time_algo, m, episode_start + n_episodes -e)
        else:
            t_left = 'unknown'
        start = time.time()
        model = Model(data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1, spot_prices = spot_prices, V2X = V2X)
        
        opti_method = MPC_opti_methods[m]
        opti_parameters = MPC_parameters[m]
        
        if save_mode:
            decisions, MPC_results[e], predictions_load[e], predictions_PV[e] = run_MPC_save(m, episode, model, decisions_0,
                                                                                                Load_model_forecast, PV_model_forecast, PV_LSTM,
                                                                                                opti_method, opti_parameters,
                                                                                                n_hour_future,t_left)
                                                                                             
        else:
            decisions = run_MPC(m, episode, model, decisions_0, Load_model_forecast, PV_model_forecast, PV_LSTM,
                                                                                                    opti_method, opti_parameters, objective_1, 
                                                                                                    n_hour_future,t_left, soc_penalty, lambda_soc = lambda_soc[objective_1])
            
            
            
        end = time.time()
        total_ep = end - start
        time_algo[m].append(total_ep)                                                                                         


        if e == episode_start:
            
            results = decisions.copy()

        else:
            
            results = results.append(decisions)
        
        if e == 25 or e == 50:
            results.to_csv(res_folder_path+f'backup_{e}_{name}{speciality}.csv')
    
    
    results.to_csv(res_folder_path+f'results_{name}{speciality}.csv')
    
    if save_mode:
        intermediate_results = {'PV forecast': predictions_PV,
                                'Load forecast': predictions_load,
                                'Decisions': MPC_results}
    
        file_inter = open(res_folder_path+f'Intermediate_results_{name}_{e}.pickle', 'wb') 
        pickle.dump(intermediate_results, file_inter)
        file_inter.close()

#%% Optimization Fully Deterministic
    
from Model_Class import EV, House, Model
from MPC import run_MPC, run_MPC_save
import pickle
import time

res_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/'

t_res = PV_model_forecast.t_res
save = 0
columns = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load','ev_load','ev_grid','y_buy',
            'y_sell','y_ch','y_dis','soc',
        'avail', 'episode']

lambda_soc = {'cost': 0.6,
              'pv': 0.5,
              'peak': 0.8}
################################### PARAMETERS ########################################
V2X = 1
semi_dynamic_pricing = True
save_mode = False
objective_1 = 'peak'
speciality = '_'+objective_1
soc_penalty = 1.2
lambda_soc = {objective_1: np.arange(0.25,0.3,0.02)}


if semi_dynamic_pricing:
    spot_prices = prices_romande_energie(data_EV)[['buy','sell']]
    
EV1 = EV(eta_EV_ch = 0.95,SOC_min_departure = 1)
House1 = House()


episode_start = 1
n_episodes = 60

range_episodes = range(episode_start, episode_start + n_episodes)

MPC_methods = ['4. MPC deterministic',
                '5. MPC stochastic',
                  '8. MPC stochastic CVaR soc']


MPC_methods = ['4. MPC deterministic']
               
################################## NAMES #############################################
names = ['opti', 'mpc_d', 'mpc_s', 'mpc_s_cvar_cost', 'mpc_s_cvar_soc']

if V2X:
    names = ['v2g_'+n for n in names]

else:
    prefix = ''

methods = ['1. Fully deterministic',  '4. MPC deterministic', 
            '5. MPC stochastic', '7. MPC stochastic CVaR obj1',
            '8. MPC stochastic CVaR soc']

algorithms = {methods[i]:names[i]  for i in range(len(names))}



MPC_opti_methods = {'4. MPC deterministic': 'deterministic', '5. MPC stochastic': 'expected value',
                  '6. MPC stochastic CVaR': 'CVaR','7. MPC stochastic CVaR obj1': 'Markowitz', 
                  '8. MPC stochastic CVaR soc': 'Markowitz' }

MPC_parameters = {'4. MPC deterministic': None, '5. MPC stochastic': None,
                  '6. MPC stochastic CVaR': {'alpha': 0.75},
                  '7. MPC stochastic CVaR obj1':{'alpha_obj1': 0.75,
                                                  'alpha_soc': 0},
                  '8. MPC stochastic CVaR soc':{'alpha_obj1': 0,
                                                  'alpha_soc': 0.75}}

time_algo = {m:[] for m in methods}

for l in lambda_soc[objective_1]:
    for i,m in enumerate(MPC_methods):
        
        
        predictions_load = {e: None for e in range_episodes}
        
        predictions_PV = {e: None for e in range_episodes}
        
        MPC_results = {e: None for e in range_episodes}
    
    
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
            model_0.optimize(t_decision, t_end_episode, PV_predictions_0,Load_predictions_0, 
                              objective_1 = objective_1, forecasting = False, 
                              method = 'deterministic', soc_penalty = soc_penalty, lambda_soc = l)
            decisions_0 = model_0.results_deterministic()
            
    
            end_0 = time.time()
            total_ep_0 = end_0 -start_0
            time_algo['1. Fully deterministic'].append(total_ep_0)
            
                                                                                      
    
    
            if e == episode_start:
                
                results = decisions_0.copy()
    
            else:
                
                results = results.append(decisions_0)

        results.to_csv(res_folder_path+f'results_opti_{objective_1}_{np.round(l,5)}.csv')
        
        if save_mode:
            intermediate_results = {'PV forecast': predictions_PV,
                                    'Load forecast': predictions_load,
                                    'Decisions': MPC_results}
        
            file_inter = open(res_folder_path+f'Intermediate_results_{name}_{e}.pickle', 'wb') 
            pickle.dump(intermediate_results, file_inter)
            file_inter.close()
#%%