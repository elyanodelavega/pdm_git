# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:26:32 2020

@author: Yann
"""
import time
import pandas as pd
import numpy as np

def run_MPC_save(method_name, episode, model, decisions_0,
            Load_model_forecast, PV_model_forecast, PV_LSTM,
            opti_method, opti_parameters,objective_1,
            n_hour_future, t_left, soc_penalty, lambda_soc = 0.5, num_iter = 20):
    #MPC stochastic Expected
    e = episode.episode[0]
    episode_length = len(episode)

    t_end_episode = episode.index[-1]

    predictions_load = {}
    predictions_PV = {}

    
    MPC_results = {}
    
    stochastic =  'stochastic' in method_name
    
    if stochastic:
        MPC_results = {'results': {t: None for t in episode.index}, 
                       'soc': {t: None for t in episode.index}}
    else:
        MPC_results = {t: None for t in episode.index}
        
    for t in range(episode_length-1):

        t_decision = episode.index[t]
        
        if t_left == 'unknown':
            update = 'unknown'
        else:
            update = np.round(t_left,2)
            
        perc = int(100* t /episode_length)
        print('\n -------------------------')
        print(f'Episode {e}, {method_name}')
        print(f'Estimated time remaining: {update}s')
        print(f'Episode progress: {perc}%')
        print('\n -------------------------')
        
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        
        Load_predictions = Load_model_forecast.predict(t_forecast, t_end)
        PV_predictions = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.35, dataframe = True, num_iter = num_iter)

        model.optimize(t_decision, t_end, PV_predictions, Load_predictions, forecasting = True, 
                       method = opti_method, parameters = opti_parameters,objective_1 = objective_1, 
                       soc_penalty = soc_penalty, lambda_soc = lambda_soc)
        decisions = model.decisions
        results_mpc = model.results_stochastic()
        
        
        if stochastic:
            MPC_results['results'][t_decision] = results_mpc
            SOC = model.predictions_SOC()
            MPC_results['soc'][t_decision] = SOC
        else:
            MPC_results[t_decision] = results_mpc
        
        predictions_load[t_decision]  = Load_predictions
        predictions_PV[t_decision] = PV_predictions
    
    
    decisions = model.decisions[:-1]
    decisions = decisions.append(decisions_0.tail(1))
    decisions.loc[t_end_episode,'soc'] = model.decisions.loc[t_end,'soc']
    

    
    return decisions, MPC_results, predictions_load, predictions_PV

def run_MPC(method_name, episode, model, decisions_0,
            Load_model_forecast, PV_model_forecast, PV_LSTM,
            opti_method, opti_parameters,objective_1,
            n_hour_future, t_left, soc_penalty, lambda_soc = 0.5, num_iter = 20):
    #MPC stochastic Expected

    episode_length = len(episode)

    t_end_episode = episode.index[-1]
    
    e = list(episode.episode)[0]
    stochastic =  'stochastic' in method_name
        
    for t in range(episode_length-1):

        t_decision = episode.index[t]
        if t_left == 'unknown':
            update = 'unknown'
        else:
            update = np.round(t_left,2)
            
        perc = int(100* t /episode_length)
        print('\n -------------------------')
        print(f'Episode {e}, {method_name} {objective_1}')
        print(f'Estimated time remaining: {update}s')
        print(f'Episode progress: {perc}%')
        print('\n -------------------------')
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        
        Load_predictions = Load_model_forecast.predict(t_forecast, t_end)
        if stochastic:
            PV_predictions = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                                dropout = 0.35, dataframe = True, num_iter = num_iter)
        else:
            PV_predictions = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
            
        model.optimize(t_decision, t_end, PV_predictions, Load_predictions, forecasting = True, 
                       method = opti_method, parameters = opti_parameters,objective_1 = objective_1, 
                       soc_penalty = soc_penalty, lambda_soc = lambda_soc)
        

    
    decisions = model.decisions[:-1]
    decisions = decisions.append(decisions_0.tail(1))
    decisions.loc[t_end_episode,'soc'] = model.decisions.loc[t_end,'soc']

    
    return decisions