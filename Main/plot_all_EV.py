# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:44:59 2021

@author: Yann
"""

"""
Created on Thu Dec  3 09:37:54 2020

@author: Yann
"""

import pandas as pd

def import_decisions(res_folder_path):
    # import the results (v2g)

    objectives = ['cost', 'pv', 'peak']
    methods_short = ['opti', 'mpc_d', 'mpc_s', 'mpc_s_cvar']
    
    # csv codes
    names = []

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
    
    # map csv codes and method names
    algorithms = {names[i]: methods[i] for i in range(len(names))}
    
    decisions = {}
    
    # import + assert datetime index
    for i,m in enumerate(methods):
        
        df = pd.read_csv(res_folder_path+f'results_{names[i]}.csv', index_col=0)
        
        new_index = pd.to_datetime(df.index, dayfirst = True)
        df.index = new_index
        
        decisions[m] = df


    # separation into groups
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
    
    # remove mpc_s_cvar from group mpc_s
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
                
    return decisions, groups, algorithms, algos_specs




def plot_all_EV(res_folder_path, n_episodes):
    
    from plot_res import plot_results_comparison
    
    decisions, groups, algorithms, algos_specs = import_decisions(res_folder_path)
    
    for e in n_episodes:
        for g in groups:
            if 'Objective' in g:
                s = 'Method'
            else:
                s = 'Objective'
                
            algos = groups[g]
            
            # decisions of the group
            dec_g = {algos_specs[a][s]: decisions[algorithms[a]] for a in algos}
            plot_results_comparison(g, dec_g, episodes = e)