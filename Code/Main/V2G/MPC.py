# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:26:32 2020

@author: Yann
"""
import time
import pandas as pd

def run_MPC(method_name, episode, model, decisions_0,results,
            Load_model_forecast, PV_model_forecast, PV_LSTM,
            opti_method, opti_parameters,
            n_hour_future):
    #MPC stochastic Expected
    e = episode.episode[0]
    episode_length = len(episode)
    t_start_episode = episode.index[0]
    t_end_episode = episode.index[-1]
    
    start = time.time()
    predictions_load = {}
    predictions_PV = {}

    
    MPC_results = {}
    
    stochastic =  'stochastic' in method_name
    
    if stochastic:
        MPC_results = {'results': {}, 'soc': {}}
    else:
        MPC_results = {}
        
    for t in range(episode_length-1):

        t_decision = episode.index[t]
        
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        
        Load_predictions = Load_model_forecast.predict(t_forecast, t_end)
        PV_predictions = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                            dropout = 0.35, dataframe = True)

        model.optimize(t_decision, t_end, PV_predictions, Load_predictions, forecasting = True, 
                       method = opti_method, parameters = opti_parameters)
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
    
    results.loc[t_start_episode:t_end_episode] = decisions
    
    end = time.time()
    total_ep = end -start
    time_algo = total_ep
    
    return results, MPC_results, predictions_load, predictions_PV, time_algo