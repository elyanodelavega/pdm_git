# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:03:49 2020

@author: Yann
"""
## IMPORTS

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np

from df_prepare import  prices_romande_energie

folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/Full/'

#%% DEFINITION
palette = sns.color_palette()

color_codes = {'v2g_opti_cost':palette[0],
                'v2g_mpc_d_cost':palette[1],
                'v2g_mpc_s_cost':palette[2],
                'v2g_mpc_s_cvar_soc_cost':palette[3],
                'v2g_opti_peak': palette[5],
                'v2g_mpc_d_peak': palette[6],
                'v2g_opti_pv':palette[7],
                'v2g_mpc_d_pv':palette[8],
                'v2g_mpc_s_pv':palette[9],
                'v2g_mpc_s_cvar_soc_pv':palette[4]}

methods = ['Perfect Foresight  \nExp: Cost \nExp: SOC',  'MPC deterministic \nExp: Cost \nExp: SOC', 
           'MPC stochastic \nExp: Cost \nExp: SOC', 
           'MPC stochastic \nExp: Cost, \nCVaR 75%: SOC',
            'Perfect Foresight \nExp: Peak Shaving \nExp: SOC',
            'MPC deterministic \nExp: Peak Shaving \nExp: SOC',
            'Perfect Foresight \nExp: PV \nExp: SOC',
            'MPC deterministic \nExp: PV \nExp: SOC', 
            'MPC stochastic \nExp: PV \nExp: SOC', 
            'MPC stochastic \nExp: PV, \nCVaR 75%: SOC ']

names = list(color_codes.keys())

csv_code = '.csv'
decisions = {n: pd.read_csv(folder_path+'results_'+n+csv_code, index_col = 0) for n in names}

for n in names:
    df = decisions[n][decisions[n].episode < 61]
    new_index = pd.to_datetime(df.index, dayfirst = True)
    df.index = new_index
    decisions[n] = df

algorithms = {names[i]: methods[i] for i in range(len(names))}

#%%
metrics = ['self_cons', 'soc_dep', 'Cost', 'peak_factor']
metrics_title = ['PV self-consumption','SOC at Departure', 'Median Cost', '$PAR^{-1}$' ]
metrics_unit = ['%','%', 'CHF','%']
metrics_props = {metrics[i]: {'title': metrics_title[i],'unit': metrics_unit[i]} for i in range(len(metrics))}

labels_radar = {'self_cons': 'PV',
          'Cost': 'Cost',
          'peak_factor': '$PAR^{-1}$',
          'soc_dep': 'SOC '}
#%%
group_code = {'cost': [], 'peak': [], 'pv': [],
          'opti':[], 'mpc_d': [], 'mpc_s': [], 'mpc_s_cvar': []}

group_names = ['Objective: Cost','Objective: Peak Shaving', 'Objective: PV',
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

#%% Functions
def quick_stats(decisions, spot_prices):
    
    data = decisions.copy()
    
    data = prices_romande_energie(data)

    data['cost_buy'] = (data['grid_load'] + data['grid_ev'])*data['buy']
    
    data['cost_sell'] = (data['pv_grid'] + data['ev_grid'])*data['buy']
    
    data['cost'] = data['cost_buy'] - data['cost_sell']
    
    data['grid_bought'] = (data['grid_load'] + data['grid_ev'])/1000
    
    data['grid_bought'] = (data['grid_load'] + data['grid_ev'])/1000
    
    d = data.copy()
    
    df = d.groupby('episode').sum()
    
    df_mean = data.groupby('episode').mean()
    
    df_max = data.groupby('episode').max()
    
    df_med = d.groupby('episode').median()

    df.drop(['soc'], axis = 1, inplace = True)
    
    soc_arr = [data.soc[0]*100]
    
    soc_arr.extend([data.soc[i]*100 for i in range(1, len(data)) if data.avail[i] > data.avail[i-1]])
    
    soc_dep = [data.soc[i] *100 for i in range(1, len(data)) if data.avail[i] < data.avail[i-1]]
        
    self_cons = 100*(df['pv_load'] + df['pv_ev'])/df['pv']
   
    df['soc_arr'] = soc_arr 
    
    df['soc_dep'] = soc_dep
    
    df['self_cons'] = self_cons

    df['Cost'] = df_med['cost']
    
    df['peak_factor'] = 100*df_mean.grid_bought / df_max.grid_bought
    
    return df

def loss_ratio(stats_loss, quantile_low = 0.1, quantile_high = 0.9):
    
    df = stats_loss.copy()
    df_med = df.median()
    
    best_algo = df_med[df_med == df_med.min()].index.values[0]

    new_df = pd.DataFrame(columns = df.columns)
    for col in new_df.columns:
        values = 100*df[best_algo]/df[col]
        new_df[col] = values
        
    low = new_df.quantile(quantile_low)
    med = new_df.median()
    high = new_df.quantile(quantile_high)
    
    return low,med,high

def adjust(s):
    if ' ' in s:
        s = s.replace(' ', '\n')
    return s    
#%%
stats = {n: quick_stats(decisions[n],prices_romande_energie) for n in names}


           # 'pv_load_perc','pv_ev_perc','pv_grid_perc']

stats_df = {m: pd.DataFrame(data = {n: list(stats[n].loc[:,m] )
                             for n in names}) for m in metrics}

benchmark = 'v2g_opti_cost'
n_episodes = len(stats[benchmark])

range_episodes = range(int(stats[benchmark].index[0]), int(stats[benchmark].index[-1] + 1))


#%% box plot metrics


def ax_boxplot_metrics(ax, metric, g, ylim = None, leg = True):
    # fig, axes = plt.subplots(len(metrics),1, sharex = True, figsize=(25,16))
    algos = groups[g]
    
    if 'Objective' in g:
        s = 'Method'
    else:
        s = 'Objective'

    m = metric
    s_df = stats_df[m]
    
    new_df = {}
    for n in algos:
        values = list(s_df[n].values)
        values.remove(max(values))
        new_df[algos_specs[n][s]] = values
    
    df = pd.DataFrame(new_df)
    # sns.boxplot(data = df, ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    sns.boxplot(data = df, ax = ax, palette = [color_codes[n] for n in algos])
    
    #axes[i, 0].set_ylabel(names_map[n], fontsize = 23)

    ax.set_title(metrics_props[m]['title'])
    ax.set_ylabel(metrics_props[m]['unit'])
    if leg == False:
        ax.set(xticklabels=[])

    ax.grid()
    if ylim is not None:
        ax.set_ylim(ylim)
        
    return ax

#%% AX RADAR

def ax_radar(ax, g, x_legend = 0.5, y_legend = -0.5, start = np.pi/4, leg = True):
    
    algos = groups[g]
    if 'Objective' in g:
        s = 'Method'
    else:
        s = 'Objective'
        
    group_df_low = pd.DataFrame(index = algos)
    group_df_high = pd.DataFrame(index = algos)
    group_df_med = pd.DataFrame(index = algos)


    angles=np.linspace(start, 2*np.pi + start, len(metrics), endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))
    for m in metrics:

        if m == 'Cost':
            low, med, high = loss_ratio(stats_df[m])
            group_df_low[labels_radar[m]] = low[algos]
            group_df_high[labels_radar[m]] = high[algos]
            group_df_med[labels_radar[m]] = med[algos]
        else:
            group_df_low[labels_radar[m]] = stats_df[m].quantile(0.1)[algos]
            group_df_high[labels_radar[m]] = stats_df[m].quantile(0.9)[algos]
            group_df_med[labels_radar[m]] = stats_df[m].median()[algos]
        

    for a in algos:
        vals_low =group_df_low.loc[a,:]
        errors_low = group_df_med.loc[a,:] - vals_low
        
        errors_low=np.concatenate((errors_low,[errors_low[0]]))
        vals_high =group_df_high.loc[a,:]
        errors_high =  vals_high - group_df_med.loc[a,:]
        
        errors_high=np.concatenate((errors_high,[errors_high[0]]))
        vals_med =group_df_med.loc[a,:]
        
        vals_med=np.concatenate((vals_med,[vals_med[0]]))


        ax.plot(angles, vals_med, 'o-',  label = algos_specs[a][s], color = color_codes[a], linewidth = 2)
        ax.errorbar(angles, vals_med, yerr = errors_low,  uplims=True,color = color_codes[a])
        ax.errorbar(angles, vals_med, yerr = errors_high,  lolims=True,color = color_codes[a])
        
    
        ax.set_thetagrids(angles[:-1] * 180/np.pi, group_df_low.columns, fontsize = 16)
        
    
    ax.grid(True)
    if leg == True:
        ax.legend(loc = 'lower center',bbox_to_anchor=(x_legend,y_legend), ncol = 1, fontsize = 14)

    ax.set_xlabel('%', fontsize = 18)
    ax.set_ylim([0,105])
    
    return ax

#%%

def ax_radar_by_metrics(ax, g, x_legend = 0.5, y_legend = -0.2, start = np.pi/4, leg = True):
    algos = groups[g]
    if 'Objective' in g:
        s = 'Method'
    else:
        s = 'Objective'
    angles=np.linspace(start, 2*np.pi+ start, len(algos), endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))
    
    for m in metrics:
    
        if m == 'Cost':
            _, med, _ = loss_ratio(stats_df[m][algos])
            
        else:
            
            med = stats_df[m][algos].median()

        vals_med = med
        
        vals_med=np.concatenate((vals_med,[vals_med[0]]))
    
    
        ax.plot(angles, vals_med, 'o-',  label = labels_radar[m])
        
        ax.set_thetagrids(angles[:-1] * 180/np.pi, [adjust(algos_specs[a][s]) for a in algos],fontsize = 16)
        
    
    ax.grid(True)
    if leg == True:
        ax.legend(loc = 'lower center',bbox_to_anchor=(x_legend,y_legend), ncol = 1, fontsize = 14)
    ax.set_xlabel('%', fontsize = 18)
    ax.set_ylim([0,105])
    ax.set_title(g, fontsize = 18)
    
    return ax
#%%
colors = sns.color_palette()

color_indices = {'Grid': colors[0],
                  'PV': colors[1],
                  'EV': colors[2],
                  'Load': colors[3]}
def plot_repartition(ax, decisions, g, variable, indices, labels, name = None, width = 0.6):
    
    algos = groups[g]
    if 'Objective' in g:
        s = 'Method'
    else:
        s = 'Objective'
    new_df = pd.DataFrame(index = indices, columns=[algorithms[n] for n in algos])

    x = np.arange(len(algos))
    if name == None:
        name = variable[0]
    
    for  n in algos:
        
        s_df = decisions[n]

        tot = s_df[variable].sum().sum()
        
        for ind in indices:
            
            new_df.loc[ind,algorithms[n]] = 100*s_df[ind].sum()/tot

    for k,ind in enumerate(indices):
        if k == 0: 
            ax.bar(x, new_df.loc[ind,:],width=width,  color = color_indices[labels[k]])
        
        elif k== 1:
            ax.bar(x, new_df.loc[ind,:],width=width, bottom=new_df.loc[indices[0],:], color = color_indices[labels[k]])
        
        else:
            ax.bar(x,  new_df.loc[ind,:],width=width, bottom= new_df.loc[:indices[k-1],:].sum().values, color = color_indices[labels[k]])
    
    ax.set_xticks(x)
    ax.set_xticklabels([adjust(algos_specs[n][s]) for n in algos])
    ax.set_ylabel('%')
    ax.set_title(f'{name.upper()}')
    ax.grid(True)

    return ax


#%%
metrics_ylim =  {'self_cons': [0,105],
          'Cost': None,
          'peak_factor': None,
          'soc_dep': [0,105]}

    

    
#%%
labels_tot = ['PV', 'Grid',  'EV', 'Load']

def ax_ev_charge(ax, g):
    new_dec = {}
    for n in names:
        dec = decisions[n].copy()
        dec['p_ev'] = dec['grid_ev'] +  dec['pv_ev']
        new_dec[n] = dec.copy()
    
    indices = ['pv_ev', 'grid_ev']
    
    labels_agent = ['PV', 'Grid']   
    
    variable = ['p_ev']
    
    ax = plot_repartition(ax, new_dec, g, variable, indices, labels_agent,
                      name = 'Agent to EV')

    return ax

def ax_ev_discharge(ax, g):
    new_dec = {}
    for n in names:
        dec = decisions[n].copy()
        dec['p_ev_dis'] = dec['ev_load'] +  dec['ev_grid']
        new_dec[n] = dec.copy()
    
    indices = ['ev_load', 'ev_grid']
    
    labels_agent = ['Load', 'Grid']   
    
    variable = ['p_ev_dis']
    
    ax = plot_repartition(ax, new_dec, g, variable, indices, labels_agent,
                      name = 'EV to Agent')

    return ax

def ax_load_repartition(ax, g):
    variable = ['load']
    indices = ['pv_load', 'grid_load', 'ev_load']
    labels_agent = ['PV', 'Grid', 'EV']
    
    ax = plot_repartition(ax, decisions, g, variable, indices, labels_agent,
                      name = 'Agent to Load')

    return ax

def ax_pv_repartition(ax, g):
    indices = ['pv_load', 'pv_ev', 'pv_grid']

    labels_agent = ['Load', 'EV', 'Grid']   
    
    variable = ['pv']
    
    ax = plot_repartition(ax, decisions, g, variable, indices, labels_agent,
                      name = 'PV to Agent')

    return ax



#%%
for g in groups.keys():
    algos = groups[g]
    fig = plt.figure(figsize=(22, 12))
    plt.suptitle(g, fontsize = 18)
    ax1 = fig.add_subplot(1,4,1, polar = True)
    ax_radar(ax1, g, leg = False)
    
    
    ax3= fig.add_subplot(4,4,2)
    ax4= fig.add_subplot(4,4,6)
    ax5= fig.add_subplot(4,4,10)
    ax6= fig.add_subplot(4,4,14)
    
    ax_box = [ax3, ax4, ax5, ax6]
    
    for i in range(len(ax_box)):
        m = metrics[i]
        ax = ax_box[i]
        ax_boxplot_metrics(ax, m, g, ylim = metrics_ylim[m], leg = False)
        
    patches1 = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in algos]
    ax1.legend(handles=patches1, loc='lower center',ncol=int(len(algos)/2), bbox_to_anchor = (0.5,-0.8)) 
    
    
    ax7= fig.add_subplot(2,4,3)
    ax8= fig.add_subplot(2,4,7)
    ax9= fig.add_subplot(2,4,4)
    ax10= fig.add_subplot(2,4,8)
    
    ax_ev_charge(ax7, g)
        
    ax_ev_discharge(ax9, g)
    
    
    ax_load_repartition(ax8, g)
    
    ax_pv_repartition(ax10, g)
    
    patches = [mpatches.Patch(color=color_indices[labels_tot[i]], label= labels_tot[i]) for i in range(len(labels_tot))]

    fig.legend(handles=patches, loc='upper right', bbox_to_anchor = (0.8, 0.95), ncol = 4)
    
    fig.show()
    
#%%
variables = ['pv_ev','pv_load','grid_ev','grid_load','ev_grid','y_buy','y_sell','y_ch','y_dis','avail','cost','self_cons','peak_factor']

for s in stats:
    c = stats[s][variables].corr()
    plt.figure(figsize = (16,9))
    plt.title(s)
    sns.heatmap(c.loc[:,['cost','self_cons','peak_factor']], annot=True)