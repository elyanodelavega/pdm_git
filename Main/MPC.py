# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:26:32 2020

@author: Yann
"""
# Imports
import time
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers(df, z_value = 1.28):
    ''' Remove values not in confidence interval
        Input:
            df: predictions
            z_value: int, corresponding z_value of a confidence interval when data is standardize
                        1.28 --> 80% conf. interval
        Output:
            df_clean: predictions without oultiers'''
    
    # z_score of each predictions based on mean
    z=np.abs(stats.zscore(df.mean()))
    
    # remove outliers
    df_clean = df.iloc[:, (z < z_value)]
    
    return df_clean


def time_prevision_method(time_algo, method, episodes_left):
    '''Estimate progression and time left
        Input:
            time_algo: dict, execution time up to now
            method: str, which method
            episodes_left: int
        Output:
            estimated time left'''
            
    average_time_per_episode_all = np.mean(time_algo[method]) 
    
    time_left = average_time_per_episode_all*episodes_left
    
    return int(time_left)



def run_MPC(method_name, episode, model, decisions_0,
            Load_model_forecast, PV_model_forecast, PV_LSTM,
            opti_method, opti_parameters,objective_1,
            n_hour_future, t_left, stochastic, lambda_soc , num_iter = 20, z_value = 1.28):
    ''' Run MPC
        Input:
            method_name: str, method used (mpc_d, mpc_sto,...)
            episde: int, number of the episode
            model: Class, Model_class instantiated
            decisions_0: df, results of PF
            Load_model_forecat, PV_model_forecast: Class, used for predictions
            PV_LSTM: LSTM for PV
            opti_method: str, deterministic, expected value, cvar
            opti_parameters: dict, optimization parameters (cf. Main)
            objective_1: str, main objective
            n_hour_future: int, length of predictions
            t_left: int, result from time_prevision_method()
            stocastic: bool, whether stochastic or not
            lamda_soc: int, weight for SOC
            num_iter: int, number of scenario
            z_value: float, for remove_outliers()
        Output:
            decisions: df, results of MPC optimization'''
    
    # initilaization
    episode_length = len(episode)

    t_end_episode = episode.index[-1]
    
    e = list(episode.episode)[0]

    # Run MPC
    for t in range(episode_length-1):
        
        # verbose for update on progress
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
        
        # time initialization
        t_forecast = episode.index[t+1]
        t_end = min(t_decision + pd.Timedelta(hours = n_hour_future), t_end_episode)
        t_start_opti = time.time()
        
        # predictions
        Load_predictions = Load_model_forecast.predict(t_forecast, t_end)
        if stochastic:
            PV_predictions = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                                dropout = 0.2, dataframe = True, num_iter = num_iter)
            PV_predictions = remove_outliers(PV_predictions, z_value)
        else:
            PV_predictions = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
        
        # optmization
        model.optimize(t_decision, t_end, PV_predictions, Load_predictions, forecasting = True, 
                       method = opti_method, parameters = opti_parameters,objective_1 = objective_1, lambda_soc = lambda_soc)
        
        t_end_opti = time.time()
        
        # time is too high, reduce the z_value and thus the number of scenario for next iteration
        if (t_end_opti - t_start_opti) > 30:
            z_value = z_value*0.9
        else:
            z_value = z_value
            
    # results
    decisions = model.decisions[:-1]
    decisions = decisions.append(decisions_0.tail(1))
    decisions.loc[t_end_episode,'soc'] = model.decisions.loc[t_end,'soc']

    
    return decisions

def run_MPC_save(method_name, episode, model, decisions_0,
            Load_model_forecast, PV_model_forecast, PV_LSTM,
            opti_method, opti_parameters,objective_1,
            n_hour_future, t_left, stochastic, lambda_soc , num_iter = 20, z_value = 1.28):
    ''' Same as previous but with intermediate results saving'''
    
    
    episode_length = len(episode)

    t_end_episode = episode.index[-1]
    
    e = list(episode.episode)[0]
    
    predictions_load = {}
    predictions_PV = {}
    
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
        t_start_opti = time.time()
        Load_predictions = Load_model_forecast.predict(t_forecast, t_end)
        if stochastic:
            PV_predictions = PV_model_forecast.predict_distribution(model = PV_LSTM, time = t_forecast,
                                                                dropout = 0.2, dataframe = True, num_iter = num_iter)
            PV_predictions = remove_outliers(PV_predictions, z_value)
        else:
            PV_predictions = PV_model_forecast.predict(model = PV_LSTM, time = t_forecast, dataframe = True)
        
        
        model.optimize(t_decision, t_end, PV_predictions, Load_predictions, forecasting = True, 
                       method = opti_method, parameters = opti_parameters,objective_1 = objective_1, lambda_soc = lambda_soc)
        
        t_end_opti = time.time()
        
        if (t_end_opti - t_start_opti) > 30:
            z_value = z_value*0.9
        else:
            z_value = z_value
        decisions = model.decisions
        results_mpc = model.results_stochastic()
        
        # save predictions of parameters and decision variables
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

