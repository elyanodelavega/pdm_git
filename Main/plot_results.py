# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:37:54 2020

@author: Yann
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pickle
import seaborn as sns
import numpy as np

from df_prepare import  prices_romande_energie

res_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/Full/'

sns.set_style('whitegrid')

img_folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'

#%% DEFINITION
objectives = ['cost', 'pv', 'peak']
methods_short = ['opti', 'mpc_d', 'mpc_s', 'mpc_s_cvar']
palette = sns.color_palette(n_colors = len(objectives)*len(methods_short) + 4)

names = []
c = 0
for  o in objectives:
    for m in methods_short:
        
        names.append(f'v2g_{m}_{o}')
        
methods = ['Perfect Foresight,  cost', 
           'MPC deterministic , cost', 
           'MPC stochastic , Exp: cost , Exp: SOC', 
           'MPC stochastic , CVaR: cost, , Exp: SOC',
            'Perfect Foresight,  PVSC ', 
           'MPC deterministic , PVSC', 
           'MPC stochastic , Exp: PV , Exp: SOC', 
           'MPC stochastic , CVaR: PV, , Exp: SOC',
           'Perfect Foresight,  APR', 
           'MPC deterministic , APR ', 
           'MPC stochastic , Exp: APR , Exp: SOC', 
           'MPC stochastic , CVaR: APR, , Exp: SOC']

algorithms = {names[i]: methods[i] for i in range(len(names))}

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
group_code = {'cost': [], 'peak': [], 'pv': [],
          'opti':[], 'mpc_d': [], 'mpc_s': [], 'mpc_s_cvar': []}

group_names = ['Objective: Cost','Objective: APR', 'Objective: PVSC',
               'Perfect Foresight','MPC deterministic',
               'MPC stochastic, Expected', 'MPC stochastic, CVaR']
groups = {}
for n in names:
    
    for i,g in enumerate(group_code.keys()):
        
        if g in n:
            
            group_code[g].append(n)
            
        groups[group_names[i]] = group_code[g]
        
algos_mpc_s =  list(groups['MPC stochastic, Expected'])
for a in algos_mpc_s:    
    if 'cvar' in a:
        algos_mpc_s.remove(a)

groups['MPC stochastic, Expected'] = algos_mpc_s


algos_specs = {n: {'Objective': None, 'Method': None} for n in names}

for g_name in groups.keys():

    algos = groups[g_name]
    
    if 'Objective' in g_name:
        
        for a in algos:
            algos_specs[a]['Objective'] = g_name
            
    else:

        for a in algos:
            algos_specs[a]['Method'] = g_name        
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
from plot_res import plot_results_comparison
n_episodes = [[6,7]]
for e in n_episodes:
    for g in groups:
        if 'Objective' in g:
            s = 'Method'
        else:
            s = 'Objective'
        algos = groups[g]
        dec_g = {algos_specs[a][s]: decisions[algorithms[a]] for a in algos}
        plot_results_comparison(g, dec_g, episodes = e)
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
