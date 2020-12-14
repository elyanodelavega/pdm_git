# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:37:54 2020

@author: Yann
"""
img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Images/'

res_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/cost_soc/'

import pandas as pd
import pickle

columns = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load', 'soc',
       'avail', 'episode']
methods = ['Fully deterministic',  'MPC deterministic', 
           'MPC stochastic',
           'MPC stochastic CVaR cost','MPC stochastic CVaR soc']
methods = ['Fully deterministic cost', 'Fully deterministic pv']

names = ['v2g_opti', 'v2g_mpc_d', 'v2g_mpc_s', 'v2g_mpc_s_cvar_cost', 'v2g_mpc_s_cvar_soc']
names = ['v2g_opti', 'v2g_opti_pv']
decisions = {}

predictions_load = {}

predictions_pv = {}

MPC_results = {}

n_episodes = 25

saved = 0

for i,m in enumerate(methods):
    
    df = pd.read_csv(res_folder_path+f'results_{names[i]}.csv', index_col=0)
    
    new_index = pd.to_datetime(df.index, dayfirst = True)
    df.index = new_index
    
    decisions[m] = df
    
    if i != 0 and saved:
        
        file_pred_load = open(res_folder_path+f'load_pred_{names[i]}_{n_episodes}.pickle', 'rb') 
        predictions_load[m] = pickle.load( file_pred_load)
        
        
        file_pred_pv = open(res_folder_path+f'pv_pred_{names[i]}_{n_episodes}.pickle', 'rb') 
        predictions_pv[m] = pickle.load( file_pred_pv)
        
        
        file_full_res = open(res_folder_path+f'full_res_{names[i]}_{n_episodes}.pickle', 'rb') 
        MPC_results[m] = pickle.load( file_full_res)
        
#%%
save = False

figname = None
folder = None
if save:
    img_paths = {m:[] for m in methods}
    import os
    for m in methods:
        img_path = img_folder_path+m+ '/'
        img_paths[m] = img_path
        os.mkdir(img_path)

    
#%%
from plot_res import plot_results_deterministic

for ep in range(1,20):
    if save:
        figname = 'Episode_'+str(ep)
        img_path = img_paths[methods[0]]
    plot_results_deterministic(decisions[methods[0]], [ep], figname = figname, 
                               img_path = folder, method = 'Cost')
    plot_results_deterministic(decisions[methods[1]], [ep], figname = figname, 
                               img_path = folder, method = 'PV')
#%%

from to_video import to_video
from plot_res import plot_MPC_det

decisions_mpc_det = decisions[methods[1]]
for ep in range(3,10):
    
    if save: 
        folder = img_paths[methods[1]]+'Episode '+str(ep)+'/'
        os.mkdir(folder)
        
   
    
    df_dec = decisions_mpc_det[decisions_mpc_det.episode == ep]
    
    df_res = MPC_results[methods[1]][ep]
    
    load_pred = predictions_load[methods[1]][ep]
    
    pv_pred = predictions_pv[methods[1]][ep]
    
    
    for i,t in enumerate(df_dec.index[:-1]):
        
        res = df_res[t]
        load = load_pred[t]
        pv = pv_pred[t]
        
        if save:
            figname = str(i)
            img_path = folder
        plot_MPC_det(df_dec,t,res,pv,load,figname = figname, 
                               img_path = folder)
    if save:
        to_video(folder)

#%%
from plot_res import plot_MPC_sto 

decisions_mpc_sto = decisions[methods[2]]
for ep in range(3,6):
    
    if save:
        folder = img_paths[methods[2]]+'Episode '+str(ep)+'/'
        os.mkdir(folder)
    
    
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
        if save:
            figname = str(i)
            img_path = folder
            
        plot_MPC_sto(df_dec,t,res,pv,load,soc,figname = figname, 
                               img_path = folder)
    
    if save:    
        to_video(folder)
        
#%%
from plot_res import plot_results_comparison
plot_results_comparison(decisions, episodes = [1,2,3])
plot_results_comparison(decisions, episodes = [4,5,6])
plot_results_comparison(decisions, episodes = [7,8,9])
plot_results_comparison(decisions, episodes = [10,11,12])
plot_results_comparison(decisions, episodes = [13,14,15])