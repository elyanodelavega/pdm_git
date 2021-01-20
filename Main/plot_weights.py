# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:36:53 2020

@author: Yann
"""
''' Function to be called in Main to plot the weight sensitivity analysis'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from df_prepare import prices_romande_energie


#%%
def compute_metrics(decisions):
    ''' Function to return metrics per episode of the results (decision)
        Input: 
            decisions: df, results of an algorithm
        Output:
            df: metrics per episode'''
    
    # copy results
    data = decisions.copy()
    
    # add hourly prices
    data = prices_romande_energie(data)

    # compute cost
    data['cost_buy'] = (data['grid_load'] + data['grid_ev'])*data['buy']
    
    data['cost_sell'] = (data['pv_grid'] + data['ev_grid'])*data['buy']
    
    data['cost'] = data['cost_buy'] - data['cost_sell']
    
    # compute power drawn from grid
    data['grid_bought'] = (data['grid_load'] + data['grid_ev'])/1000
    
    d = data.copy()
    
    # groupby
    df = d.groupby('episode').sum()
    
    df_mean = data.groupby('episode').mean()
    
    df_max = data.groupby('episode').max()
    
    df_med = d.groupby('episode').median()
    
    # soc at departure for each episode

    df.drop(['soc'], axis = 1, inplace = True)
    
    soc_dep = [data.soc[i] *100 for i in range(1, len(data)) if data.avail[i] < data.avail[i-1]]
    
    # pv self consumption
        
    PVSC = 100*(df['pv_load'] + df['pv_ev'])/df['pv']
    
    # APR
    
    APR = 100*df_mean.grid_bought / df_max.grid_bought

    df['soc_dep'] = soc_dep
    
    df['PVSC'] = PVSC

    df['Cost'] = df_med['cost']
    
    df['APR'] = APR
    
    return df

    
#%%
def plot_weights(folder_path, obj, method = 'opti', weight_range = np.arange(0,1.1,0.1), V2X = 1):
    ''' Plot weights sensitivity on opti
        Iput:
            folder_path: str, folder path to results
            obj: str, (cost, pv, peak)
            method: str, method used (opti, mpc_d,...), be careful to have the results corresponding
            weight_range: range, weights
            V2X: bool, whether v2x was applied or not'''
            
    # modify names if v2x
    if V2X:
        prefix = 'v2g_'
    else:
        prefix = ''
        
    # csv results name
    names = [f'opti_{obj}_{np.round(i,5)}' for i in weight_range]
    
    # printable names
    weights = [f'{np.round(i,5)}' for i in weight_range]
    
    # map between csv names and printable names
    algorithms = {names[i]: weights[i] for i in range(len(names))}
    
    csv_code = '.csv'
    
    # import results as decisions
    decisions = {n: pd.read_csv(folder_path+'results_'+prefix+n+csv_code, index_col = 0) for n in names}
    
    # convert index to datetime
    for n in names:
        df = decisions[n].copy()
        new_index = pd.to_datetime(df.index, dayfirst = True)
        df.index = new_index
        decisions[n] = df
    
    # add metrics to "stats", where each dict entry is a weight and its corresponding metrics
    stats = {n: compute_metrics(decisions[n]) for n in names}
    
    # metrics name
    metrics = ['PVSC', 'soc_dep', 'Cost', 'APR']
    
    # dict with each entry is a metric, and the values for each weight
    stats_df = {m: pd.DataFrame(data = {n: list(stats[n].loc[:,m] )
                                 for n in names}) for m in metrics}
    
    # plotting names
    metrics_title = ['PVSC','SOC at departure', 'Median Cost', 'APR' ]
    metrics_label = ['%','%', 'CHF','%']
    metrics_props = {metrics[i]: {'title': metrics_title[i],'label': metrics_label[i]} for i in range(len(metrics))}
    
    objectives_metrics = {'pv': 'PVSC',
                          'cost': 'Cost',
                          'peak': 'APR'}
    
    # select corresponding metric for the objective + soc at departure
    objectives = [objectives_metrics[obj], 'soc_dep']
    
    # plot
    fig, axes = plt.subplots(len(objectives),1, sharex = True, figsize=(16,9), dpi = 400)
    
    for i, m in enumerate(objectives):
        
        s_df = stats_df[m]
        
        new_df = {}
        for n in names:
            values = list(s_df[n].values)
            values.remove(max(values))
            new_df[algorithms[n]] = values
        
        df = pd.DataFrame(new_df)
        
        sns.boxplot(data = df, ax = axes[i], palette = 'RdYlGn_r')
    
        axes[i].set_title(metrics_props[m]['title'], fontsize = 18)
        axes[i].set_ylabel(metrics_props[m]['label'], fontsize = 18)
        axes[i].tick_params(axis='both', which='major', labelsize=15)
        axes[i].grid()
    
    axes[-1].set_xlabel(f'SOC weight: {objectives_metrics[obj]}', fontsize = 16)
    
    fig.show()


