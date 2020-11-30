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
from df_prepare import data_PV_csv, data_EV_csv
from keras.models import load_model


img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'


PV_csv = 'pv_simulation_evolene.csv'
data_PV = data_PV_csv(PV_csv)

EV_csv = 'ev_simulation_evolene.csv'
data_EV = data_EV_csv(EV_csv)

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
    
from Model_Class import EV, House, Model, quick_stats
from plot_res import plot_MPC, plot_results_day_ahead, plot_results_deterministic
from Forecast_Class import predict_dummy


t_res = PV_model_forecast.t_res
save = 0
columns = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load', 'soc',
       'avail', 'episode']


EV1 = EV()
House1 = House()
episode_start = 5
n_episodes = 1

range_episodes = range(episode_start, episode_start + n_episodes)


methods = ['1. Fully deterministic','det', 'batch', 'iterative', 'dummy']
costs = {m:[] for m in methods}
results = {m: pd.DataFrame(columns = columns) for m in methods}
stats = {m:{e: None for e in range_episodes} for m in methods}

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
    Load_predictions_0 = episode.loc[t_forecast:t_end_episode,'load'].to_frame()
    model_0.optimize(t_decision, t_end_episode, PV_predictions_0,Load_predictions_0, forecasting = False, method = 'deterministic')
    decisions_0 = model_0.results_deterministic()
    cost = decisions_0.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum()
    costs['1. Fully deterministic'].append(cost)
    results['1. Fully deterministic'].loc[t_start_episode:t_end_episode] = decisions_0
    stats['1. Fully deterministic'][e] = quick_stats(decisions_0)

    #plot_results_deterministic(results_0, figname = str(start), img_path = img_paths['1. Fully deterministic'])
    
    
    # #MPC deterministic
    model_3_batch = Model(name = name + methods[1] + str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1)
    model_3_det = Model(name = name + methods[1] + str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1)
    model_3_dummy = Model(name = name + methods[1] + str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1)
    model_3_iterative = Model(name = name + methods[1] + str(model_number), data_EV = data_EV, t_start = t_start_episode, 
              t_res = t_res,  EV = EV1, House = House1)
    # k = start
    # k_path = img_paths['4. MPC deterministic'] + str(k) + '/' 
    # os.mkdir(k_path)
    for t in range(episode_length-1):
        #k += 1
        t_decision = episode.index[t]
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        
        Load_predictions_3 = Load_model_forecast.predict(t_forecast, t_end)
        Load_predictions_dummy = predict_dummy(data_EV, t_forecast, t_end, 'load')
        Load_predictions_3_iterative = Load_model_forecast.predict(t_forecast, t_end, iterative = True)
        PV_predictions_3 = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
        model_3_batch.optimize(t_decision, t_end, PV_predictions_3, Load_predictions_3, forecasting = True, method = 'deterministic' )
        model_3_det.optimize(t_decision, t_end, PV_predictions_3, Load_predictions_0, forecasting = True, method = 'deterministic' )
        model_3_dummy.optimize(t_decision, t_end, PV_predictions_3, Load_predictions_dummy, forecasting = True, method = 'deterministic' )
        model_3_iterative.optimize(t_decision, t_end, PV_predictions_3, Load_predictions_3_iterative, forecasting = True, method = 'deterministic' )
        
        #results_3 = model_3.results_stochastic()
        #plot_MPC(decisions_3, results_3, figname = str(k), img_path = k_path)
    
    decisions_3 = model_3_det.decisions[:-1]
    decisions_3 = decisions_3.append(decisions_0.tail(1))
    decisions_3.loc[t_end_episode,'soc'] = model_3_det.decisions.loc[t_end,'soc']
    #results_3 = model_3.results_stochastic()
    cost = decisions_3.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum() 
    costs['det'].append(cost)
    results['det'].loc[t_start_episode:t_end_episode] = decisions_3
    stats['det'][e] = quick_stats(decisions_3)
    
    decisions_3 = model_3_batch.decisions[:-1]
    decisions_3 = decisions_3.append(decisions_0.tail(1))
    decisions_3.loc[t_end_episode,'soc'] = model_3_batch.decisions.loc[t_end,'soc']
    #results_3 = model_3.results_stochastic()
    cost = decisions_3.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum() 
    costs['batch'].append(cost)
    results['batch'].loc[t_start_episode:t_end_episode] = decisions_3
    stats['batch'][e] = quick_stats(decisions_3)
    
    decisions_3 = model_3_iterative.decisions[:-1]
    decisions_3 = decisions_3.append(decisions_0.tail(1))
    decisions_3.loc[t_end_episode,'soc'] = model_3_iterative.decisions.loc[t_end,'soc']
    #results_3 = model_3.results_stochastic()
    cost = decisions_3.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum()
    costs['iterative'].append(cost)
    results['iterative'].loc[t_start_episode:t_end_episode] = decisions_3
    stats['iterative'][e] = quick_stats(decisions_3)
    
    decisions_3 = model_3_dummy.decisions[:-1]
    decisions_3 = decisions_3.append(decisions_0.tail(1))
    decisions_3.loc[t_end_episode,'soc'] = model_3_dummy.decisions.loc[t_end,'soc']
    #results_3 = model_3.results_stochastic()
    cost = decisions_3.loc[:,['grid_load','grid_ev']].sum(axis = 0).sum() 
    costs['dummy'].append(cost)
    results['dummy'].loc[t_start_episode:t_end_episode] = decisions_3
    stats['dummy'][e] = quick_stats(decisions_3)
    
   


#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))

cost2 = {m: np.cumsum([c/1000 for c in costs[m]]) for m in costs}
for m in costs.keys():
    plt.plot(cost2[m], label = m)

plt.xticks(ticks = range(n_episodes), labels = range_episodes)
plt.legend(ncol = 2)
plt.xlabel('Episode')
plt.ylabel('Electricity bought [kWh]')
plt.title('Cumulative electricity bought comparison')

#%%
for e in range_episodes:
    
    for m in results.keys():
        fig, axes = plt.subplots(4,1, sharex=True, figsize=(16,9))
        episode = results[m][results[m].episode == e]
        grid_ev_bought = episode.loc[:,['grid_ev']].sum(axis = 0).sum()
        grid_load_bought = episode.loc[:,['grid_load']].sum(axis = 0).sum()
        p_grid_bought = episode.loc[:,['grid_ev', 'grid_load']].sum(axis = 1)
        total = grid_ev_bought + grid_load_bought
        plt.suptitle(f'{m}, Episode {e}, Power bought: {int(total/1000)} kWh ')
        x = np.arange(len(episode))
        
        axes[0].plot(x, list(episode.pv/1000), label = 'PV')
        axes[0].plot(x, list(episode.pv_ev/1000), label = 'PV_EV')
        axes[0].plot(x, list(episode.pv_load/1000), label = 'PV_Load')
        axes[0].set_title('PV')
        axes[0].legend(ncol = 3, loc = 'upper left')
        
        
        axes[1].plot(x, list(episode.load/1000), label = 'Load')
        axes[1].plot(x, list(episode.grid_load/1000), label = 'Grid_load')
        axes[1].plot(x, list(episode.pv_load/1000), label = 'PV_Load')
        axes[1].set_title('Load')
        axes[1].legend(ncol = 3, loc = 'upper left')
        
        axes[2].plot(list(p_grid_bought/1000) , label = 'Grid')
        axes[2].plot(list(episode.grid_load/1000), label = 'Grid_load')
        axes[2].plot(list(episode.grid_ev/1000), label = 'Grid_EV')
        axes[2].set_title('Grid')
        axes[2].legend(ncol = 3, loc = 'upper left')
        
        axes[3].plot(list(episode.pv_ev/1000), label = 'pv_ev')
        axes[3].plot(list(episode.grid_ev/1000), label = 'grid_ev')
        ax3 = axes[3].twinx()
        ax3.plot(x,list(episode.soc*100),'--', color = 'grey')
        ax3.set_ylabel('EV SoC [%]', color='black')
        ax3.set_ylim([0,110])
        ax3.set_title('SOC')
        axes[3].set_xlabel('Hours')
        axes[3].legend(ncol = 2, loc = 'upper left')
        
        try:
            t_dep = list(episode.avail).index(0)
        except:
            t_dep = x[-1]
            
        for i in range(3):
            axes[i].axvline(t_dep, color = 'black')
            
            
        axes[3].axvline(t_dep, color = 'black')
        time = episode.index
        ticks = np.arange(0, max(x)+1, step=6)
        ticks = [i for i in range(0, max(x) + 1, 4)]
        labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
        plt.xticks(ticks = ticks, labels = labels)
        
        
        plt.show()
#%%

for e in range_episodes:
    fig, axes = plt.subplots(3,1, sharex=True, figsize=(16,9))
    plt.suptitle(f' Episode {e}')
    for i, m in enumerate(results.keys()):
        
        episode = results[m][results[m].episode == e]
        grid_ev_bought = episode.loc[:,['grid_ev']].sum(axis = 0).sum()
        grid_load_bought = episode.loc[:,['grid_load']].sum(axis = 0).sum()
        p_grid_bought = episode.loc[:,['grid_ev', 'grid_load']].sum(axis = 1)
        total = grid_ev_bought + grid_load_bought
        
        x = np.arange(len(episode))
        
        axes[i].plot(list(episode.pv_ev/1000), label = 'pv_ev')
        axes[i].plot(list(episode.grid_ev/1000), label = 'grid_ev')
        ax1 = axes[i].twinx()
        ax1.plot(x,list(episode.soc*100),'--', color = 'grey')
        ax1.set_ylabel('EV SoC [%]', color='black')
        ax1.set_ylim([0,110])
        axes[i].set_title(f'{m}: cost: {int(costs[m][e - episode_start]/1000)}')
        
        axes[i].legend(ncol = 2, loc = 'upper left')
        
        try:
            t_dep = list(episode.avail).index(0)
        except:
            t_dep = x[-1]
            
        
        axes[i].axvline(t_dep, color = 'black')
            
            
        
    time = episode.index
    ticks = np.arange(0, max(x)+1, step=6)
    ticks = [i for i in range(0, max(x) + 1, 4)]
    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
    plt.xticks(ticks = ticks, labels = labels)
    
    
    plt.show()
    
#%%

import seaborn as sns
fig, axes = plt.subplots(3,3, figsize=(16,9))

for i, m in enumerate(methods):
    r = stats[m]
    soc_last = []
    p_bought = []
    self_cons = []
    
    for e in range_episodes:
        soc_last.append(r[e]['absolut']['SOC_last']*100)
        p_bought.append(r[e]['absolut']['P_G_bought']/1000)
        self_cons.append(r[e]['relative']['Self_consumption']*100)
    
    sns.boxplot(self_cons, ax = axes[i,0], color = 'green')
    sns.boxplot(p_bought, ax = axes[i,1])
    sns.boxplot(soc_last, ax = axes[i,2], color = 'red')
    axes[i,2].set_xlim([70,105])

axes[0,1].set_title('Power Bought')
axes[2,1].set_xlabel('kW')
axes[0,0].set_title('PV self consumption')
axes[2,0].set_xlabel('%')
axes[0,2].set_title('SOC at departure')
axes[2,2].set_xlabel('%')
plt.grid()
plt.show()
    
#%%
import matplotlib.pyplot as plt

def pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    
    return pf_X, pf_Y

power_grid_tot = []
self_cons_tot = []
for m in methods:
    
    r = stats[m]
    
    power_grid = []
    self_cons = []
    for e in range_episodes:
        power_grid.append(r[e]['absolut']['P_G_bought']/1000)
        self_cons.append(r[e]['relative']['Self_consumption']*100)
    
    plt.scatter(self_cons,power_grid, label = m)
    
    power_grid_tot.extend(power_grid)
    self_cons_tot.extend(self_cons)

pf_X, pf_Y = pareto_frontier( self_cons_tot,power_grid_tot, maxY = False)
    
plt.plot(pf_X, pf_Y)
plt.legend()
plt.xlabel("Power from grid")
plt.ylabel("PV self consumption")
plt.show()

