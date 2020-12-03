# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:37:54 2020

@author: Yann
"""
img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'

import pandas as pd
import pickle

columns = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load', 'soc',
       'avail', 'episode']
methods = ['1. Fully deterministic',  '4. MPC deterministic', 
           '5. MPC stochastic']
results = {}

predictions_load = {}

predictions_pv = {}

MPC_results = {}

n_episodes = 60

names = ['opti', 'mpc_d', 'mpc_s']
for i,m in enumerate(methods):
    
    results[m] = pd.read_csv(f'results_{names[i]}_{n_episodes}.csv')
    # file = open(f'results_{names[i]}_{n_episodes}.pickle', 'rb') 
    # pickle.load(res, file_res)
    
    if i != 0:
        
        file_pred_load = open(f'load_pred_{names[i]}_{n_episodes}.pickle', 'rb') 
        predictions_load[m] = pickle.load( file_pred_load)
        
        
        file_pred_pv = open(f'pv_pred_{names[i]}_{n_episodes}.pickle', 'rb') 
        predictions_pv[m] = pickle.load( file_pred_pv)
        
        
        file_full_res = open(f'full_res_{names[i]}_{n_episodes}.pickle', 'rb') 
        MPC_results[m] = pickle.load( file_full_res)