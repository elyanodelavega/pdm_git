# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:36:53 2020

@author: Yann
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

palette = sns.color_palette()
color_codes = {'naive':palette[1],
               'rbc':palette[2],
               'v2g_opti':palette[5],
               'v2g_mpc_d':palette[6],
               'v2g_mpc_s':palette[8],
               'v2g_mpc_s_cvar':palette[0],
               'ddpg':palette[3],
               'pdqn':palette[4]}

folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/'
#%%
def quick_stats(decisions):
    data = decisions.copy()
    
    df = data.groupby('episode').sum()

    df.drop(['soc'], axis = 1, inplace = True)
    
    soc_arr = [data.soc[0]*100]
    
    soc_arr.extend([data.soc[i]*100 for i in range(1, len(data)) if data.avail[i] > data.avail[i-1]])
    
    soc_dep = [data.soc[i] *100 for i in range(1, len(data)) if data.avail[i] < data.avail[i-1]]
    
    grid_bought = (df['grid_load'] + df['grid_ev'])/1000
    
    p_ev = (df['pv_ev'] + df['grid_ev'])/1000
    
    self_cons = 100*(df['pv_load'] + df['pv_ev'])/df['pv']
    
    df['soc_arr'] = soc_arr 
    
    df['soc_dep'] = soc_dep
    
    df['self_cons'] = self_cons
    
    df['grid_bought'] = grid_bought
    
    thresholds = data.groupby('episode').quantile([0.9])['grid_load'].values
    
    power_bought_95 = []
    
    for i, e in enumerate(np.unique(data.episode)):
        
        data_episode = data[data.episode == e]
        
        power_bought_95.append(sum(data_episode[data_episode['grid_load'] <= thresholds[i]].sum(), data_episode['grid_ev'].sum())/1000)
    
    df['Energy_bought'] = power_bought_95
    
    df['p_ev'] = grid_bought
    
    return df

#%%



methods = ['Fully deterministic',  'MPC deterministic', 
           'MPC stochastic', 'MPC stochastic CVaR']

names = ['v2g_opti', 'v2g_mpc_d', 'v2g_mpc_s', 'v2g_mpc_s_cvar']

decisions = {n: pd.read_csv(folder_path+'results_'+n+'_10.csv', index_col = 0) for n in names}

names = list(decisions.keys())

for n in names:
    df = decisions[n]
    new_index = pd.to_datetime(df.index, dayfirst = True)
    df.index = new_index

algorithms = {'v2g_opti': 'Fully deterministic', 'v2g_mpc_d': 'MPC deterministic', 
           'v2g_mpc_s': 'MPC stochastic', 'v2g_mpc_s_cvar':'MPC stochastic CVaR'}


stats = {n: quick_stats(decisions[n][decisions[n].episode < 55]) for n in names}

metrics = ['self_cons', 'soc_dep', 'Energy_bought']

stats_df = {m: pd.DataFrame(data = {algorithms[n]: list(stats[n].loc[:,m] )
                             for n in names}) for m in metrics}

n_episodes = len(stats['v2g_opti'])

range_episodes = range(int(stats['v2g_opti'].index[0]), int(stats['v2g_opti'].index[-1] + 1))
#%% bar plot variation

fig, axes = plt.subplots(3,1, figsize=(20,12), sharey = True, sharex = True)
#plt.suptitle(' Relative comparison with optimal solution', fontsize = 25)

benchmark = 'v2g_opti'
power = []
soc = []
self_cons = []


titles = ['PV self-consumption', 'SOC at departure', 'Max Power bought']
for n in list(stats.keys()):
    if n == benchmark:
        continue
    for i,j in zip(metrics,range(len(axes))):
        value = 100*(stats[n][i]/stats[benchmark][i])-100
        axes[j].bar(range(n_episodes), value, color = color_codes[n], label = algorithms[n])
        axes[j].set_title(titles[j], fontsize = 18)
        

handles, labels= axes[0].get_legend_handles_labels()
fig.legend(handles,labels, loc='lower center',ncol=2 , fontsize = 20)


# axes[0].set_ylim([-80, 20])
# axes[1].set_ylim([-80, 20])
# axes[2].set_ylim( [- 20, 80])

for i in range(3):
    axes[i].axhline(0, color = 'black')
    axes[i].set_ylabel('%' , fontsize = 20)
    axes[i].grid()

    
plt.xlabel('Episode', fontsize = 22)

    
    
#%% Box plot self, soc, cons
import matplotlib.patches as mpatches
metrics = ['self_cons', 'soc_dep', 'Energy_bought']
fig, axes = plt.subplots(len(metrics),1, sharex = True, figsize=(20,13))

for i, m in enumerate(metrics):
    
    s_df = stats_df[m]
     
    sns.boxplot(data = stats_df[m], ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    
    #axes[i, 0].set_ylabel(names_map[n], fontsize = 23)

axes[0].set_title('PV self-consumption', fontsize = 20)
axes[0].set_ylabel('%', fontsize = 20)
axes[1].set_title('SOC at departure', fontsize = 22)
axes[1].set_ylabel('%', fontsize = 20)
axes[2].set_title('Energy bought 90%', fontsize = 22)
axes[2].set_ylabel('kWh', fontsize = 20)
for i in range(len(metrics)):
    axes[i].grid()
    

plt.show()

#%% Hist Grid ev

import numpy as np
import math
fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(16,9))
plt.suptitle('Grid charges to EV', fontsize = 22)

metric = ['self_cons', 'soc_dep', 'grid_bought']
maximum = []
bins = [i*0.37 for i in range(math.ceil(3700/370))]
bins.append(3.7)
for i, n in enumerate(names):
     
    #sns.boxplot(data = stats_df[m], ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    grid = decisions[n].grid_ev[decisions[n].grid_ev > 0.001]
    sns.histplot(grid/1000, ax = axes[i], color = color_codes[n], bins = bins)
    axes[i].set_xlim([0,4])
    axes[i].set_title(algorithms[n], fontsize = 20)
    axes[i].set_ylabel('Count')
    axes[i].set_xlim([-0.05,3.75])
    
# ticks = np.arange(0,4,0.25)
plt.xlabel('kW', fontsize = 18)
plt.xticks(bins)
for i in range(3):
    axes[i].grid()
plt.show()

#%%

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(16,9))
plt.suptitle('PV charges to EV', fontsize = 22)

metric = ['self_cons', 'soc_dep', 'grid_bought']
maximum = []
for i, n in enumerate(names):
     
    #sns.boxplot(data = stats_df[m], ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    pv = decisions[n].pv_ev[decisions[n].pv_ev > 0.001]
    sns.histplot(pv/1000, ax = axes[i], color = color_codes[n], bins = bins)
    axes[i].set_xlim([0,4])
    axes[i].set_title(algorithms[n], fontsize = 20)
    axes[i].set_ylabel('Count')
    axes[i].set_xlim([-0.05,3.75])

plt.xlabel('kW', fontsize = 18)
plt.xticks(bins)
for i in range(3):
    axes[i].grid()
plt.show()

#%%

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(16,9))
plt.suptitle('Charges to EV', fontsize = 22)

metric = ['self_cons', 'soc_dep', 'grid_bought']
maximum = []
for i, n in enumerate(names):
     
    #sns.boxplot(data = stats_df[m], ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    pv = decisions[n].pv_ev[decisions[n].pv_ev > 0.001]/1000
    grid = decisions[n].grid_ev[decisions[n].grid_ev > 0.001]/1000
    p_ev = list(pv) + list(grid)
    sns.histplot(p_ev, ax = axes[i], color = color_codes[n], bins = bins)
    axes[i].set_xlim([0,4])
    axes[i].set_title(algorithms[n], fontsize = 20)
    axes[i].set_ylabel('Count')
    axes[i].set_xlim([-0.05,3.75])

plt.xlabel('kW', fontsize = 18)
plt.xticks(bins)
for i in range(3):
    axes[i].grid()
plt.show()

#%%
import numpy as np
soc_arr = np.unique(stats['v2g_opti']['soc_arr'])
fig, axes = plt.subplots(len(soc_arr), len(names), sharex = True, sharey = True, figsize=(16,9))

x = [np.round(0.1*i,2) for i in range(11)]


for i, s_arr in enumerate(soc_arr):
    for j, n in enumerate(names):
        
        decision = decisions[n]
        episodes = stats[n][stats[n].soc_arr == s_arr].index
        soc_array = np.zeros((len(episodes),len(x)))
        for k, e in enumerate(episodes):
            ep = decision[decision.episode == int(e)]
            td = list(ep.avail).index(0)
            soc = ep.soc[:td+1]*100
            quantiles = np.quantile(soc,x)
            soc_array[k,:] = quantiles
        
        df = pd.DataFrame(data = {int(x[q]*100): soc_array[:,q] for q in range(len(x))})

        sns.boxplot(data = df, ax = axes[i,j], orient = 'v')

        axes[i,j].grid()
        
            
for j, n in enumerate(names):
    axes[0,j].set_title(algorithms[n])
    axes[len(soc_arr)-1,j].set_xlabel('Charging time %')

ticks = np.arange(0,11)
labels = [str(t*10) for t in ticks]
labels[0] = labels[0] + '\n' + 'Arrival'
labels[-1] = labels[-1] + '\n' + 'Departure'
     
for i, s_arr in enumerate(soc_arr):
    axes[i,0].set_ylabel(f'SOC arrival: {int(s_arr)}%')
    for col in range(3):
        axes[i,col].set_xticks(ticks)
        axes[len(soc_arr)-1,col].set_xticklabels(labels)
        axes[len(soc_arr)-1,col].set_xlabel('Charging time %', fontsize = 16)

#%%
import numpy as np
soc_arr = np.unique(stats['v2g_opti']['soc_arr'])
fig, axes = plt.subplots(len(soc_arr), sharex = True, sharey = True, figsize=(16,9))

x = [np.round(0.1*i,2) for i in range(11)]


for i, s_arr in enumerate(soc_arr):
    for j, n in enumerate(names):
        
        decision = decisions[n]
        episodes = stats[n][stats[n].soc_arr == s_arr].index
        soc_array = np.zeros((len(episodes),len(x)))
        for k, e in enumerate(episodes):
            ep = decision[decision.episode == int(e)]
            td = list(ep.avail).index(0)
            soc = ep.soc[:td+1]*100
            quantiles = np.quantile(soc,x)
            soc_array[k,:] = quantiles

        axes[i].plot([q*100 for q in x] ,np.median(soc_array, axis = 0), label = algorithms[n], color = color_codes[n])
        axes[i].grid()
handles, labels= axes[0].get_legend_handles_labels()
fig.legend(handles,labels, loc='upper center',ncol=2 , fontsize = 16)
axes[len(soc_arr)-1].set_xlabel('Charging time %',fontsize = 16)

ticks = np.arange(0,110,10)
labels = [str(t) for t in ticks]
labels[0] = labels[0] + '\n' + 'Arrival'
labels[-1] = labels[-1] + '\n' + 'Departure'


     
for i, s_arr in enumerate(soc_arr):
    axes[i].set_title(f'SOC arrival: {int(s_arr)}%')
    axes[i].set_ylabel('SOC [%]')

    for col in range(3):
        axes[i].set_xticks(ticks)
        axes[len(soc_arr)-1].set_xticklabels(labels)
        axes[len(soc_arr)-1].set_xlabel('Charging time %', fontsize = 16)


#%%
import numpy as np
import math
fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(16,9))


for i, n in enumerate(names):
    
    decision = decisions[n][decisions[n].avail > 0]
    episodes = np.unique(decision.episode)
    grid_ev_array = np.zeros((len(episodes),len(x)-1))
    pv_ev_array = np.zeros((len(episodes),len(x)-1))
    ev_array = np.zeros((len(episodes),len(x)-1))
    for e in decision.episode:
        ep = decision[decision.episode == e]
        time_charging = ep.avail.sum()
        pv_ev = list(ep['pv_ev'].values/1000)
        grid_ev = list(ep['grid_ev'].values/1000)
        ev = list((ep['pv_ev'].values+ep['grid_ev'].values) /1000)
        t = np.dot(x,time_charging)
        t = [math.floor(i) for i in t]

        quantiles_ev = [np.median(ev[t[i]:t[i+1]]) for i in range(len(t)-1)]
        ev_array[int(e-1),:] = quantiles_ev

        # quantiles_g = [np.median(grid_ev[t[i]:t[i+1]]) for i in range(len(t)-1)]
        # grid_ev_array[int(e-1),:] = quantiles_g

    # df_pv = pd.DataFrame(data = {int(x[q]*100): pv_ev_array[:,q] for q in range(len(x)-1)})

    # sns.boxplot(data = df_pv, ax = axes[i,0], orient = 'v', color = color_codes[n])
    
    df_grid = pd.DataFrame(data = {int(x[q]*100): ev_array[:,q] for q in range(len(x)-1)})

    sns.boxplot(data = df_grid, ax = axes[i], orient = 'v', color = color_codes[n])
    
    axes[i].grid()
    
    
    
    ticks = np.arange(0,11)
    labels = [str(t*10) for t in ticks]
    labels[0] = labels[0] + '\n' + 'Arrival'
    labels[-1] = labels[-1] + '\n' + 'Departure'
    for row in range(3):
        axes[row].set_ylabel('kW', fontsize = 16)
        axes[row].set_title(algorithms[names[row]], fontsize = 18)

    axes[2].set_xticks(ticks)
    
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel('Charging time %', fontsize = 16)
    
    # axes[0].set_title('Grid', fontsize = 18)
    
    #plt.suptitle('EV Charge by sources', fontsize = 22)
    
        
#%%
import numpy as np
import math
fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(16,9))


for i, n in enumerate(names):
    
    decision = decisions[n][decisions[n].avail > 0]
    episodes = np.unique(decision.episode)
    ev_array = np.zeros((len(episodes),len(x)-1))
    for e in decision.episode:
        ep = decision[decision.episode == e]
        time_charging = ep.avail.sum()
        ev = list((ep['pv_ev'].values+ep['grid_ev'].values) /1000)
        # pv = list(ep['grid_ev'].values/1000)
        # grid = list((ep['grid_ev'].values+ ep['grid_load'].values) /1000)
        t = np.dot(x,time_charging)
        t = [math.floor(i) for i in t]

        quantiles_ev = [sum(p > 0.001 for p in ev[t[i]:t[i+1]]) for i in range(len(t)-1)]
        ev_array[int(e-1),:] = quantiles_ev

    #     quantiles_g = [sum(p > 0.001 for p in grid_ev[t[i]:t[i+1]]) for i in range(len(t)-1)]
    #     grid_ev_array[int(e-1),:] = quantiles_g

    # df_pv = pd.DataFrame(data = {int(x[q]*100): pv_ev_array[:,q] for q in range(len(x)-1)})

    # sns.boxplot(data = df_pv, ax = axes[i,0], orient = 'v', color = color_codes[n])
    
    df_grid = pd.DataFrame(data = {int(x[q]*100): ev_array[:,q] for q in range(len(x)-1)})

    sns.boxplot(data = df_grid, ax = axes[i], orient = 'v', color = color_codes[n])
    
    axes[i].grid()
    
    
    
    ticks = np.arange(0,11)
    labels = [str(t*10) for t in ticks]
    labels[0] = labels[0] + '\n' + 'Arrival'
    labels[-1] = labels[-1] + '\n' + 'Departure'
    for row in range(3):
        axes[row].set_ylabel('N', fontsize = 16)
        axes[row].set_title(algorithms[names[row]], fontsize = 18)

    axes[2].set_xticks(ticks)
    
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel('Charging time %', fontsize = 16)
    
    
    
#%%
import numpy as np
soc_arr = np.unique(stats['v2g_opti']['soc_arr'])
fig, axes = plt.subplots( len(names),len(soc_arr), sharex = True, sharey = True, figsize=(16,9))

x = [np.round(0.1*i,2) for i in range(11)]


for i, s_arr in enumerate(soc_arr):
    for j, n in enumerate(names):
        
        decision = decisions[n]
        episodes = stats[n][stats[n].soc_arr == s_arr].index
        soc_array = np.zeros((len(episodes),len(x)))
        for k, e in enumerate(episodes):
            ep = decision[decision.episode == int(e)]
            td = list(ep.avail).index(0)
            soc = ep.soc[:td+1]*100
            
            time_charging = ep.avail.sum()
            t = np.dot(x,time_charging)
            t = [math.floor(i) for i in t]

            quantiles_soc = soc[t]
            
            soc_array[k,:] = quantiles_soc
        
        df = pd.DataFrame(data = {int(x[q]*100): soc_array[:,q] for q in range(len(x))})

        sns.boxplot(data = df, ax = axes[j,i], orient = 'v', color = color_codes[n])

        axes[j,i].grid()
        
ticks = np.arange(0,11)
labels = [str(t*10) for t in ticks]
labels[0] = labels[0] + '\n' + 'Arrival'
labels[-1] = labels[-1] + '\n' + 'Departure'
           
for j, n in enumerate(names):
    axes[j,0].set_ylabel(algorithms[n]+'\n SOC [%]')
    for col in range(len(soc_arr)):
        axes[0,col].set_title(f'SOC arrival: {int(soc_arr[col])}%')
        axes[j,col].set_xticks(ticks)
        axes[j,col].set_xticklabels(labels)
        axes[2,col].set_xlabel('Charging time %', fontsize = 13)

    
#%%
import numpy as np
import math
import matplotlib.patches as mpatches
#soc_arr = np.unique(stats['opti']['soc_arr'])
avail = np.unique(stats['v2g_opti']['avail'])

bins = list(np.arange(11,30,4))
bins.append(31)
x = [np.round(0.1*i,2) for i in range(11)]

fig, axes = plt.subplots(len(names)+2,len(bins)-1, sharex = True, figsize=(16,11))

ticks = np.arange(0,11,2.5)
labels = [str(int(t)*10)+'%' for t in ticks]
labels[0] = labels[0] + '\n' + 'Arr.'
labels[-1] = labels[-1] + '\n' + 'Dep.'
#plt.suptitle(f'SOC at arrival: {s_arr}%',fontsize = 18)
for i in range(len(bins)-1):
    a_low = int(bins[i])

    a_high = int(bins[i+1])

    for j, n in enumerate(names):
        
        decision = decisions[n]
        lower = stats[n][stats[n].avail < a_high]
        upper = lower[lower.avail >= a_low]
        episodes = upper.index
        
        soc_array = np.zeros((len(episodes),len(x)))
        pv_array = np.zeros((len(episodes),len(x)))
        load_array = np.zeros((len(episodes),len(x)))
        for k, e in enumerate(episodes):
            ep = decision[decision.episode == int(e)]
            td = list(ep.avail).index(0)
            soc = ep.soc[:td+1]*100
            pv = ep.pv[:td+1]/1000
            load = ep.load[:td+1]/1000
            
            time_charging = ep.avail.sum()
            t = np.dot(x,time_charging)
            t = [math.floor(i) for i in t]

            quantiles_soc = soc[t]
            quantiles_pv = pv[t]
            quantiles_load = load[t]
            
            soc_array[k,:] = quantiles_soc
            pv_array[k,:] = quantiles_pv
            load_array[k,:] = quantiles_load

        df = pd.DataFrame(data = {int(x[q]*100): soc_array[:,q] for q in range(len(x))})
    
        sns.boxplot(data = df, ax = axes[j+2,i], orient = 'v', color = color_codes[n])
        
        df = pd.DataFrame(data = {int(x[q]*100): pv_array[:,q] for q in range(len(x))})
    
        sns.boxplot(data = df, ax = axes[0,i], orient = 'v', color = 'grey')
        
        df = pd.DataFrame(data = {int(x[q]*100): load_array[:,q] for q in range(len(x))})
    
        sns.boxplot(data = df, ax = axes[1,i], orient = 'v', color = 'grey')
        
        axes[j+2,i].grid()
        
        axes[j+2,i].set_xticks(ticks)
        
        if i > 0:
            axes[j+2,i].set_yticklabels(' ')
        
        axes[j+2,0].set_ylabel('SOC [%]')
        
        axes[j+2,i].set_ylim([15,105])
        
        
    
    power = ['PV','Load']
    
    for k in range(2):
        axes[k,i].grid()
        axes[k,i].set_ylim([0,31])
        if i > 0:
            axes[k,i].set_yticklabels(' ')
        axes[k,0].set_yticks(np.arange(0,31,10))
        axes[k,0].set_ylabel(f'{power[k]} [kW]')
    
    
    axes[4,i].set_xticklabels(labels)
    axes[0,i].set_title(f'Charging time {a_low} - {a_high-1}h', fontsize = 14)  
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='upper center',ncol=len(names))
           
