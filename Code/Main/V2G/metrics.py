# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:36:53 2020

@author: Yann
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np
import math

palette = sns.color_palette()
color_codes = {'v2g_opti':palette[0],
                'v2g_mpc_d':palette[1],
                'v2g_mpc_s':palette[2],
                'v2g_mpc_s_cvar_cost':palette[4],
                'v2g_mpc_s_cvar_soc':palette[5],
                'v2g_opti_no_pen':palette[3],
                'v2g_mpc_d_no_pen':palette[6],
                'v2g_mpc_s_no_pen':palette[7],
                'v2g_mpc_s_cvar_cost_no_pen':palette[8],
                'v2g_mpc_s_cvar_soc_no_pen':palette[9],
                'v2g_opti_pv':palette[3],
                'v2g_mpc_d_pv':palette[6],
                'v2g_mpc_s_pv':palette[7],
                'v2g_mpc_s_cvar_soc_pv':palette[9]}

# color_codes = {'opti':palette[0],
#                'mpc_d':palette[1],
#                'mpc_s':palette[2],
#                'mpc_s_cvar_cost':palette[4],
#                'mpc_s_cvar_soc':palette[5]}
#                # 'opti_1_5':palette[3],
#                # 'mpc_d_1_5':palette[6],
#                # 'mpc_s_1_5':palette[7],
#                # 'mpc_s_cvar_cost_1_5':palette[8],
#                # 'mpc_s_cvar_soc_1_5':palette[9]}


folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/Full/'

from df_prepare import data_PV_csv, data_EV_csv, data_spot_market_csv, prices_romande_energie


#%%
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
    
    p_ev = (df['pv_ev'] + df['grid_ev'])/1000
    
    self_cons = 100*(df['pv_load'] + df['pv_ev'])/df['pv']
    
    pv_load_perc = 100*df['pv_load'] /df['pv']
    
    pv_ev_perc = 100*df['pv_ev'] /df['pv']
                      
    pv_grid_perc = 100*df['pv_grid'] /df['pv']
    
    df['soc_arr'] = soc_arr 
    
    df['soc_dep'] = soc_dep
    
    df['self_cons'] = self_cons

    df['Loss'] = df_med['cost']
    
    df['p_ev'] = p_ev
    
    df['pv_load_perc'] = pv_load_perc
    
    df['pv_ev_perc'] = pv_ev_perc
    
    df['pv_grid_perc'] = pv_grid_perc
    
    df['peak_factor'] = df_mean.grid_bought / df_max.grid_bought
    
    return df

#%%

methods = ['Fully deterministic',  'MPC deterministic', 
           'MPC stochastic', 
           'MPC stochastic \nExp: SOC, \nCVaR 75%: Cost',
           'MPC stochastic \nExp: Cost, \nCVaR 75%: SOC',
            'Fully deterministic \nSOC No penalty',  
            'MPC deterministic \nSOC No penalty', 
            'MPC stochastic \nSOC No penalty', 
            'MPC stochastic \nExp: SOC, \nCVaR 75%: Cost \nSOC No penalty',
            'MPC stochastic \nExp: Cost, \nCVaR 75%: SOC \nSOC No penalty',
            'Fully deterministic \n PV',
            'MPC deterministic \n PV', 
            'MPC stochastic \n PV', 
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


stats = {n: quick_stats(decisions[n],prices_romande_energie) for n in names}

metrics = ['self_cons', 'soc_dep', 'Loss', 'peak_factor']

stats_df = {m: pd.DataFrame(data = {n: list(stats[n].loc[:,m] )
                             for n in names}) for m in metrics}

benchmark = 'v2g_opti'
n_episodes = len(stats[benchmark])

range_episodes = range(int(stats[benchmark].index[0]), int(stats[benchmark].index[-1] + 1))

metrics_title = ['PV self-consumption','SOC at Departure', 'Median Loss', 'Peak factor' ]
metrics_label = ['%','%', 'CHF','%']
metrics_props = {metrics[i]: {'title': metrics_title[i],'label': metrics_label[i]} for i in range(len(metrics))}
#%% box plot variation

fig, axes = plt.subplots(len(metrics),1, figsize=(int(1.7*len(names)),int(1.2*len(names))), sharex = True)
#plt.suptitle(' Relative comparison with optimal solution', fontsize = 25)


for i,j in zip(metrics,range(len(axes))):
    df = stats_df[i].copy()
    df_dict = {}
    for n in names:
        if n == benchmark :
            continue

        df_dict[algorithms[n]] = 100*(df[n]/df[benchmark])-100

    df = pd.DataFrame(data = df_dict)
    sns.boxplot(data = df, ax = axes[j])
    axes[j].set_title(metrics_props[i]['title'], fontsize = 18)

        

handles, labels= axes[0].get_legend_handles_labels()
fig.legend(handles,labels, loc='lower center',ncol=len(names))

for i in range(len(axes)):
    axes[i].axhline(0, color = 'black')
    axes[i].set_ylabel('%' , fontsize = 20)
    axes[i].grid()


#%% Pareto

submetrics = ['self_cons', 'Loss']
fig, ax = plt.subplots(figsize=(16,9), sharex = True)
#plt.suptitle(' Relative comparison with optimal solution', fontsize = 25)

markers = [f'${m}$' for m in methods[1:]]

df_median = pd.DataFrame(columns = submetrics)
for m in submetrics:
    df = stats_df[m].copy()
    df_dict = {}
    for n in names:
        if n == benchmark :
            continue

        df_dict[algorithms[n]] = 100*(df[n]/df[benchmark])-100
        
    df = pd.DataFrame(data = df_dict)
    df_median[m] = df.median()
    
for i, row in enumerate(df_median.index):
    ax.scatter(df_median.loc[row,'self_cons'], df_median.loc[row,'Loss'], marker = markers[i+1])

#sns.scatterplot(data = scatter, x = scatter['self_cons'], y = scatter['Loss'], hue = scatter.index, markers=markers)
ax.grid()
# patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names[1:]]
# ax.legend(handles=patches, loc='upper left', ncol = 2)
plt.show()
#%% Pareto
submetrics = ['self_cons', 'Loss', 'peak_factor']
fig, ax = plt.subplots(figsize=(16,9), sharex = True)
#plt.suptitle(' Relative comparison with optimal solution', fontsize = 25)

power = []
soc = []
self_cons = []

scatter = pd.DataFrame(columns = submetrics)

markers = [f'${n}$' for n in names[:1]]

for m in submetrics:
    df = stats_df[m].copy()
    df_dict = {m: [], 'name':[]}
    for i,n in enumerate(names):
        if n == benchmark :
            continue

        df_dict[m].extend(list(100*(df[n]/df[benchmark])-100))
        df_dict['name'].extend([n for i in range(len(df))])        
df = pd.DataFrame(data = df_dict)
scatter[m] = df.median()
sns.pairplot(data = df, hue = 'name')

# sns.scatterplot(x = scatter['self_cons'], y = scatter['Loss'])

ax.grid()
# patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names[1:]]
# ax.legend(handles=patches, loc='upper left', ncol = 2)
plt.show()
#%% Box plot self, soc, cons

fig, axes = plt.subplots(len(metrics),1, sharex = True, figsize=(int(2*len(names)),int(1.2*len(names))))

for i, m in enumerate(metrics):
    
    s_df = stats_df[m]
    
    new_df = {}
    for n in names:
        values = list(s_df[n].values)
        values.remove(max(values))
        new_df[algorithms[n]] = values
    
    df = pd.DataFrame(new_df)
    # sns.boxplot(data = df, ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    sns.boxplot(data = df, ax = axes[i])
    
    #axes[i, 0].set_ylabel(names_map[n], fontsize = 23)

    axes[i].set_title(metrics_props[m]['title'], fontsize = 18)
    axes[i].set_ylabel(metrics_props[m]['label'], fontsize = 18)

    axes[i].grid()
    

plt.show()

