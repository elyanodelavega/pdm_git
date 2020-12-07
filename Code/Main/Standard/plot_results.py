# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:37:54 2020

@author: Yann
"""
img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Standard/Images/'

res_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Standard/Results/'

import pandas as pd
import pickle

columns = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load', 'soc',
       'avail', 'episode']
methods = ['1. Fully deterministic',  '4. MPC deterministic', 
           '5. MPC stochastic']
decisions = {}

predictions_load = {}

predictions_pv = {}

MPC_results = {}

n_episodes = 60

names = ['opti', 'mpc_d', 'mpc_s']
for i,m in enumerate(methods):
    
    df = pd.read_csv(res_folder_path+f'results_{names[i]}_{n_episodes}.csv', index_col=0)
    
    new_index = pd.to_datetime(df.index, dayfirst = True)
    df.index = new_index
    
    decisions[m] = df
    
    if i != 0:
        
        file_pred_load = open(res_folder_path+f'load_pred_{names[i]}_{n_episodes}.pickle', 'rb') 
        predictions_load[m] = pickle.load( file_pred_load)
        
        
        file_pred_pv = open(res_folder_path+f'pv_pred_{names[i]}_{n_episodes}.pickle', 'rb') 
        predictions_pv[m] = pickle.load( file_pred_pv)
        
        
        file_full_res = open(res_folder_path+f'full_res_{names[i]}_{n_episodes}.pickle', 'rb') 
        MPC_results[m] = pickle.load( file_full_res)
        
#%%
img_paths = {m:[] for m in methods}
import os
for m in methods:
    img_path = img_folder_path+m+ '/'
    img_paths[m] = img_path
    os.mkdir(img_path)

#%%
from plot_res import plot_results_deterministic

for ep in range(1,3):
    plot_results_deterministic(decisions[methods[0]], [ep], figname ='Episode_'+str(ep), 
                               img_path = img_paths[methods[0]])
#%%
from to_video import to_video
from plot_res import plot_MPC_det 

for ep in range(1,5):
    
    folder = img_paths[methods[1]]+'Episode '+str(ep)+'/'
    os.mkdir(folder)
    decisions_mpc_det = decisions[methods[1]]
    
    df_dec = decisions_mpc_det[decisions_mpc_det.episode == ep]
    
    df_res = MPC_results[methods[1]][ep]
    
    load_pred = predictions_load[methods[1]][ep]
    
    pv_pred = predictions_pv[methods[1]][ep]
    
    
    for i,t in enumerate(df_dec.index[:-1]):
        
        res = df_res[t]
        load = load_pred[t]
        pv = pv_pred[t]
        
        plot_MPC_det(df_dec,t,res,pv,load,str(i),folder)

    to_video(folder)

from plot_res import plot_MPC_sto 

for ep in range(1,5):
    
    folder = img_paths[methods[2]]+'Episode '+str(ep)+'/'
    os.mkdir(folder)
    
    decisions_mpc_sto = decisions[methods[2]]
    
    df_dec = decisions_mpc_sto[decisions_mpc_sto.episode == ep]
    
    df_res = MPC_results[methods[2]][ep]['results']
    
    soc_pred = MPC_results[methods[2]][ep]['soc']
    
    load_pred = predictions_load[methods[2]][ep]
    
    pv_pred = predictions_pv[methods[2]][ep]
    
    for i,t in enumerate(df_dec.index[:-1]):
        
        res = df_res[t]
        load = load_pred[t]
        pv = pv_pred[t]
        soc = soc_pred[t]
        
        plot_MPC_sto(df_dec,t,res,pv,load,soc,str(i),folder)
        
    to_video(folder)