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


folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/cost_soc/'

from df_prepare import data_PV_csv, data_EV_csv, data_spot_market_csv, prices_romande_energie


#%%
def quick_stats(decisions, spot_prices):
    
    data = decisions.copy()
    
    data = prices_romande_energie(data)

    data['cost_buy'] = (data['grid_load'] + data['grid_ev'])*data['buy']
    
    data['cost_sell'] = (data['pv_grid'] + data['ev_grid'])*data['buy']
    
    data['cost'] = data['cost_buy'] - data['cost_sell']
    d = data.copy()
    
    df = d.groupby('episode').sum()

    df.drop(['soc'], axis = 1, inplace = True)
    
    soc_arr = [data.soc[0]*100]
    
    soc_arr.extend([data.soc[i]*100 for i in range(1, len(data)) if data.avail[i] > data.avail[i-1]])
    
    soc_dep = [data.soc[i] *100 for i in range(1, len(data)) if data.avail[i] < data.avail[i-1]]
    
    grid_bought = (df['grid_load'] + df['grid_ev'])/1000
    
    p_ev = (df['pv_ev'] + df['grid_ev'])/1000
    
    self_cons = 100*(df['pv_load'] + df['pv_ev'])/df['pv']
    
    pv_load_perc = 100*df['pv_load'] /df['pv']
    
    pv_ev_perc = 100*df['pv_ev'] /df['pv']
                      
    pv_grid_perc = 100*df['pv_grid'] /df['pv']
    
    df['soc_arr'] = soc_arr 
    
    df['soc_dep'] = soc_dep
    
    df['self_cons'] = self_cons
    
    d = data.copy()
    df_med = d.groupby('episode').median()
    df['Loss'] = df_med['cost']
    
    df['p_ev'] = p_ev
    
    
    
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
    df = decisions[n][decisions[n].episode < 26]
    new_index = pd.to_datetime(df.index, dayfirst = True)
    df.index = new_index
    decisions[n] = df

algorithms = {names[i]: methods[i] for i in range(len(names))}


stats = {n: quick_stats(decisions[n],prices_romande_energie) for n in names}

metrics = ['self_cons', 'soc_dep', 'Loss']

stats_df = {m: pd.DataFrame(data = {n: list(stats[n].loc[:,m] )
                             for n in names}) for m in metrics}

benchmark = 'v2g_opti'
n_episodes = len(stats[benchmark])

range_episodes = range(int(stats[benchmark].index[0]), int(stats[benchmark].index[-1] + 1))




#%% box plot variation
submetrics = ['self_cons', 'Loss']
fig, axes = plt.subplots(len(submetrics),1, figsize=(int(2*len(names)),int(1.2*len(names))), sharex = True)
#plt.suptitle(' Relative comparison with optimal solution', fontsize = 25)

power = []
soc = []
self_cons = []


titles = ['PV self-consumption', 'Median Loss', ]
for i,j in zip(submetrics,range(len(axes))):
    df = stats_df[i].copy()
    df_dict = {}
    for n in names:
        if n == benchmark :
            continue

        df_dict[algorithms[n]] = 100*(df[n]/df[benchmark])-100
        print(n)
    df = pd.DataFrame(data = df_dict)
    sns.boxplot(data = df, ax = axes[j])
    axes[j].set_title(titles[j], fontsize = 18)

        

handles, labels= axes[0].get_legend_handles_labels()
fig.legend(handles,labels, loc='lower center',ncol=len(names))


# axes[0].set_ylim([-100, 5])
# axes[1].set_ylim([-100, 5])
# axes[2].set_ylim( [- 20, 80])

for i in range(len(axes)):
    axes[i].axhline(0, color = 'black')
    axes[i].set_ylabel('%' , fontsize = 20)
    axes[i].grid()

    
plt.xlabel('Episode', fontsize = 14)

#%% Pareto
submetrics = ['self_cons', 'Loss']
fig, ax = plt.subplots(figsize=(16,9), sharex = True)
#plt.suptitle(' Relative comparison with optimal solution', fontsize = 25)

power = []
soc = []
self_cons = []

scatter = pd.DataFrame(columns = submetrics, index = names[1:])
for m in submetrics:
    df = stats_df[m].copy()
    df_dict = {}
    for n in names:
        if n == benchmark :
            continue

        df_dict[n] = 100*(df[n]/df[benchmark])-100

    df = pd.DataFrame(data = df_dict)
    scatter[m] = df.median()
    
# sns.scatterplot(x = scatter['self_cons'], y = scatter['Loss'])
sns.scatterplot(data = scatter, x = scatter['self_cons'], y = scatter['Loss'])

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

axes[0].set_title('PV self-consumption', fontsize = 20)
axes[0].set_ylabel('%', fontsize = 20)
axes[1].set_title('SOC at departure', fontsize = 20)
axes[1].set_ylabel('%', fontsize = 20)
axes[1].set_ylim([0,105])
axes[2].set_title('Median Loss per episode', fontsize = 20)
axes[2].set_ylabel('CHF', fontsize = 20)
for i in range(len(metrics)):
    axes[i].grid()
    

plt.show()

#%% Hist Grid ev


fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(18,12))
plt.suptitle('Grid charges to EV', fontsize = 22)

maximum = []
bins = [i*0.37 for i in range(math.ceil(10))]
bins.append(3.7)
for i, n in enumerate(names):
     
    #sns.boxplot(data = stats_df[m], ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    grid = decisions[n].grid_ev[decisions[n].grid_ev > 0.001]
    sns.histplot(grid/1000, ax = axes[i], color = color_codes[n], bins = bins)
    axes[i].set_xlim([0,4])

    axes[i].set_ylabel('Count')
    axes[i].set_xlim([-0.05,3.75])
    
# ticks = np.arange(0,4,0.25)
plt.xlabel('kW', fontsize = 18)
plt.xticks(bins)
for i in range(len(names)):
    axes[i].grid()

patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)))

plt.show()

#%% Hist Ev Load

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(30,17))
plt.suptitle('EV to Load', fontsize = 22)

maximum = []
bins = [i*0.37 for i in range(math.ceil(10))]
bins.append(3.7)
for i, n in enumerate(names):
     
    #sns.boxplot(data = stats_df[m], ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    grid = decisions[n].ev_load[decisions[n].ev_load > 0.001]
    sns.histplot(grid/1000, ax = axes[i], color = color_codes[n], bins = bins)
    axes[i].set_xlim([0,4])
    axes[i].set_ylabel('Count')
    axes[i].set_xlim([-0.05,3.75])
    
# ticks = np.arange(0,4,0.25)
plt.xlabel('kW', fontsize = 18)
plt.xticks(bins)
for i in range(len(names)):
    axes[i].grid()
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)/2))

#%% Hist PV to EV

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(18,12))
plt.suptitle('PV charges to EV', fontsize = 22)


maximum = []
for i, n in enumerate(names):
     
    pv = decisions[n].pv_ev[decisions[n].pv_ev > 0.001]
    sns.histplot(pv/1000, ax = axes[i], color = color_codes[n], bins = bins)
    axes[i].set_xlim([0,4])

    axes[i].set_ylabel('Count')
    axes[i].set_xlim([-0.05,3.75])

plt.xlabel('kW', fontsize = 18)
plt.xticks(bins)
for i in range(len(names)):
    axes[i].grid()
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)/2))


#%% Hist P_EV

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(20,12))
plt.suptitle('Charges to EV', fontsize = 22)

maximum = []
for i, n in enumerate(names):
     
    #sns.boxplot(data = stats_df[m], ax = axes[i], orient = 'v', palette = [color_codes[n] for n in names])
    pv = decisions[n].pv_ev[decisions[n].pv_ev > 0.001]/1000
    grid = decisions[n].grid_ev[decisions[n].grid_ev > 0.001]/1000
    p_ev = list(pv) + list(grid)
    sns.histplot(p_ev, ax = axes[i], color = color_codes[n], bins = bins)
    axes[i].set_xlim([0,4])

    axes[i].set_ylabel('Count')
    axes[i].set_xlim([-0.05,3.75])

plt.xlabel('kW', fontsize = 18)
plt.xticks(bins)
for i in range(len(names)):
    axes[i].grid()
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)/2))



#%% Boxplot charges (kW)

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(20,12))
plt.suptitle('EV charges intensity', fontsize = 22)

x = [np.round(0.1*i,2) for i in range(11)]

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
   
    df_grid = pd.DataFrame(data = {int(x[q]*100): ev_array[:,q] for q in range(len(x)-1)})

    sns.boxplot(data = df_grid, ax = axes[i], orient = 'v', color = color_codes[n])
    
    axes[i].grid()
    
    
    
    ticks = np.arange(0,10)
    labels = [f'{t*10} - {(t+1)*10}%' for t in ticks]
    labels[0] = labels[0] + '\n' + 'Arrival'
    labels[-1] = labels[-1] + '\n' + 'Departure'
    for row in range(len(names)):
        axes[row].set_ylabel('kW', fontsize = 16)
        axes[row].set_xticks(ticks)
    
    
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel('Charging time %', fontsize = 16)
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)/2))

#%% Boxplot discharges (kW)

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(20,12))
plt.suptitle('EV discharges intensity', fontsize = 22)

for i, n in enumerate(names):
    
    decision = decisions[n][decisions[n].avail > 0]
    episodes = np.unique(decision.episode)
    ev_load_array = np.zeros((len(episodes),len(x)-1))
    
    for e in decision.episode:
        ep = decision[decision.episode == e]
        time_charging = ep.avail.sum()
        ev_load = list(ep['ev_load'].values/1000)
        t = np.dot(x,time_charging)
        t = [math.floor(i) for i in t]

        quantiles_ev = [np.median(ev_load[t[i]:t[i+1]]) for i in range(len(t)-1)]
        ev_load_array[int(e-1),:] = quantiles_ev
   
    df_grid = pd.DataFrame(data = {int(x[q]*100): ev_load_array[:,q] for q in range(len(x)-1)})

    sns.boxplot(data = df_grid, ax = axes[i], orient = 'v', color = color_codes[n])
    
    axes[i].grid()
    
    
    
    ticks = np.arange(0,10)
    labels = [f'{t*10} - {(t+1)*10}%' for t in ticks]
    labels[0] = labels[0] + '\n' + 'Arrival'
    labels[-1] = labels[-1] + '\n' + 'Departure'
    for row in range(len(names)):
        axes[row].set_ylabel('kW', fontsize = 16)
        axes[row].set_xticks(ticks)
    
    
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel('Charging time %', fontsize = 16)
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)/2))

 #%% Boxplot discharges (kW)

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(20,12))
plt.suptitle('EV discharges intensity', fontsize = 22)

for i, n in enumerate(names):
    
    decision = decisions[n][decisions[n].avail > 0]
    episodes = np.unique(decision.episode)
    ev_grid_array = np.zeros((len(episodes),len(x)-1))
    
    for e in decision.episode:
        ep = decision[decision.episode == e]
        time_charging = ep.avail.sum()
        ev_grid = list(ep['ev_grid'].values/1000)
        t = np.dot(x,time_charging)
        t = [math.floor(i) for i in t]

        quantiles_ev = [np.median(ev_grid[t[i]:t[i+1]]) for i in range(len(t)-1)]
        ev_grid_array[int(e-1),:] = quantiles_ev
   
    df_grid = pd.DataFrame(data = {int(x[q]*100): ev_grid_array[:,q] for q in range(len(x)-1)})

    sns.boxplot(data = df_grid, ax = axes[i], orient = 'v', color = color_codes[n])
    
    axes[i].grid()
    
    
    
    ticks = np.arange(0,10)
    labels = [f'{t*10} - {(t+1)*10}%' for t in ticks]
    labels[0] = labels[0] + '\n' + 'Arrival'
    labels[-1] = labels[-1] + '\n' + 'Departure'
    for row in range(len(names)):
        axes[row].set_ylabel('kW', fontsize = 16)
        axes[row].set_xticks(ticks)
    
    
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel('Charging time %', fontsize = 16)
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)/2))   
        
#%% Boxplot charge Number
fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(16,9))
plt.suptitle('EV charges Number', fontsize = 22)

for i, n in enumerate(names):
    
    decision = decisions[n][decisions[n].avail > 0]
    episodes = np.unique(decision.episode)
    ev_array = np.zeros((len(episodes),len(x)-1))
    for e in decision.episode:
        ep = decision[decision.episode == e]
        time_charging = ep.avail.sum()
        ev = list((ep['pv_ev'].values+ep['grid_ev'].values) /1000)

        t = np.dot(x,time_charging)
        t = [math.floor(i) for i in t]

        quantiles_ev = [sum(p > 0.001 for p in ev[t[i]:t[i+1]]) for i in range(len(t)-1)]
        ev_array[int(e-1),:] = quantiles_ev


    df_grid = pd.DataFrame(data = {int(x[q]*100): ev_array[:,q] for q in range(len(x)-1)})

    sns.boxplot(data = df_grid, ax = axes[i], orient = 'v', color = color_codes[n])
    
    axes[i].grid()
    
    
    
    ticks = np.arange(0,10)
    labels = [f'{t*10} - {(t+1)*10}%' for t in ticks]
    labels[0] = labels[0] + '\n' + 'Arrival'
    labels[-1] = labels[-1] + '\n' + 'Departure'
    for row in range(len(names)):
        axes[row].set_ylabel('kW', fontsize = 16)

        axes[row].set_xticks(ticks)
    
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel('Charging time %', fontsize = 16)
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)))

#%% Boxplot discharge Number

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(16,9))
plt.suptitle('EV discharges number', fontsize = 22)

for i, n in enumerate(names):
    
    decision = decisions[n][decisions[n].avail > 0]
    episodes = np.unique(decision.episode)
    ev_array = np.zeros((len(episodes),len(x)-1))
    for e in decision.episode:
        ep = decision[decision.episode == e]
        time_charging = ep.avail.sum()
        ev = list((ep['ev_load'].values) /1000)

        t = np.dot(x,time_charging)
        t = [math.floor(i) for i in t]

        quantiles_ev = [sum(p > 0.001 for p in ev[t[i]:t[i+1]]) for i in range(len(t)-1)]
        ev_array[int(e-1),:] = quantiles_ev


    df_grid = pd.DataFrame(data = {int(x[q]*100): ev_array[:,q] for q in range(len(x)-1)})

    sns.boxplot(data = df_grid, ax = axes[i], orient = 'v', color = color_codes[n])
    
    axes[i].grid()
    
    ticks = np.arange(0,10)
    labels = [f'{t*10} - {(t+1)*10}%' for t in ticks]
    labels[0] = labels[0] + '\n' + 'Arrival'
    labels[-1] = labels[-1] + '\n' + 'Departure'
    for row in range(len(names)):
        axes[row].set_ylabel('kW', fontsize = 16)

        axes[row].set_xticks(ticks)
    
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel('Charging time %', fontsize = 16)
    
    
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)))    
    
    
#%%
import numpy as np
soc_arr = np.unique(stats[benchmark]['soc_arr'])
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
    axes[j,0].set_ylabel('SOC [%]')
    for col in range(len(soc_arr)):
        axes[0,col].set_title(f'SOC arrival: {int(soc_arr[col])}%')
        axes[j,col].set_xticks(ticks)
        axes[j,col].set_xticklabels(labels)
        axes[-1,col].set_xlabel('Charging time %', fontsize = 13)

patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)))
   
#%%
import numpy as np
import math
import matplotlib.patches as mpatches
#soc_arr = np.unique(stats['opti']['soc_arr'])
avail = np.unique(stats[benchmark]['avail'])

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
        axes[k,i].set_ylim([0,20])
        if i > 0:
            axes[k,i].set_yticklabels(' ')
        axes[k,0].set_yticks(np.arange(0,21,10))
        axes[k,0].set_ylabel(f'{power[k]} [kW]')
    
    
    axes[4,i].set_xticklabels(labels)
    axes[0,i].set_title(f'Charging time {a_low} - {a_high-1}h', fontsize = 14)  
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)))

#%% Boxplot EV activity

fig, axes = plt.subplots(len(names)+2,1, sharex = True, figsize=(20,12))
plt.suptitle('EV activity by hour', fontsize = 22)



for i, n in enumerate(names):

    decision = decisions[n].copy()
    
    decision['hour'] = [t.hour for t in decision.index]
    
    decision['pv'] = (decision['pv'])/1000
    
    decision['load'] = (decision['load'])/1000
   
    decision['p_ev'] = (decision['pv_ev']+decision['grid_ev'])/1000
    
    decision['ev_load'] = -decision['ev_load']/1000
    
    if i == 0:
        sns.boxplot(x="hour", y="pv", data=decision, ax = axes[0], color = 'green')
        
        sns.boxplot(x="hour", y="load", data=decision, ax = axes[1], color = 'blue')
        
        for j in range(2):
            axes[j].grid()
            axes[j].set_xlabel(None)
            axes[j].set_ylabel('kW')
            axes[j].set_ylim(0,20)
        axes[0].set_title('PV')
        axes[1].set_title('Load')

    sns.boxplot(x="hour", y="p_ev", data=decision, ax = axes[i+2], color = color_codes[n])

    sns.boxplot(x="hour", y="ev_load", data=decision, ax = axes[i+2], color = color_codes[n])

    
    
    axes[i+2].grid()
    axes[i+2].set_xlabel(None)
    axes[i+2].set_ylabel('kW')
    axes[i+2].set_ylim([-5,5])
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
axes[-1].legend(handles=patches, loc='lower center',bbox_to_anchor = (0.5, -1.02),ncol=int(len(names)))
axes[-1].set_xticklabels([str(i)+':00' for i in range(24)])

#%% Boxplot Grid activity



for i, n in enumerate(names):
    
    fig, axes = plt.subplots(3,1, sharex = True, figsize=(20,12))
    plt.suptitle(f'Grid activity by hour: {algorithms[n]}', fontsize = 22)

    decision = decisions[n].copy()
    
    decision['hour'] = [t.hour for t in decision.index]
    
    decision['pv'] = (decision['pv'])/1000
    
    decision['load'] = (decision['load'])/1000
   
    decision['grid_buy'] = (decision['grid_ev']+decision['grid_load'])/1000
    
    decision['grid_sell'] = -decision['pv_grid']/1000
    

    sns.boxplot(x="hour", y="pv", data=decision, ax = axes[0], color = 'green')
    
    sns.boxplot(x="hour", y="load", data=decision, ax = axes[1], color = 'blue')
    
    for j in range(2):
        axes[j].grid()
        axes[j].set_xlabel(None)
        axes[j].set_ylabel('kW')
        axes[j].set_ylim(0,20)
    axes[0].set_title('PV')
    axes[1].set_title('Load')

    sns.boxplot(x='hour', y='grid_buy', data=decision, ax = axes[2], color = color_codes[n])

    sns.boxplot(x='hour', y='grid_sell', data=decision, ax = axes[2], color = color_codes[n])

    
    
    axes[2].grid()
    axes[2].set_xlabel(None)
    axes[2].set_ylabel('<----Selling----> kW <----Buying---->')
    axes[2].set_ylim([-20,20])
    axes[-1].set_xticklabels([str(i)+':00' for i in range(24)])

#%% Boxplot self_cons activity



for i, n in enumerate(names):
    
    fig, axes = plt.subplots(2,1, sharex = True, figsize=(20,12))
    plt.suptitle(f'PV self consumption by hour: {algorithms[n]}', fontsize = 22)

    decision = decisions[n].copy()
    
    decision['hour'] = [t.hour for t in decision.index]
    
    decision['pv_residual'] = (decision['pv'] -decision['pv_load'])/1000

    decision['self_cons'] = 100*(decision['pv_ev']+decision['pv_load'])/1000/(decision['pv'])
    
    # decision['grid_sell'] = -decision['pv_grid']/1000
    
    sns.boxplot(x="hour", y="pv_residual", data=decision, ax = axes[0])
    #sns.boxplot(x="hour", y="pv", data=decision, ax = axes[0], color = 'green')
    
    
    
    for j in range(1):
        axes[j].grid()
        axes[j].set_xlabel(None)
        axes[j].set_ylabel('kW')

    # axes[0].set_title('PV')
    # axes[1].set_title('Load')

    sns.boxplot(x='hour', y='self_cons', data=decision, ax = axes[1], color = color_codes[n])

    #sns.boxplot(x='hour', y='grid_sell', data=decision, ax = axes[2], color = color_codes[n])

    
    
    axes[1].grid()
    axes[1].set_xlabel(None)
    axes[1].set_ylabel('kW')
    #axes[2].set_ylim([0,100])
    axes[-1].set_xticklabels([str(i)+':00' for i in range(24)])

#%% Boxplot discharge Number

fig, axes = plt.subplots(len(names),1, sharex = True, sharey = True, figsize=(16,9))
plt.suptitle('EV charge - discharge switch number', fontsize = 22)

for i, n in enumerate(names):
    
    decision = decisions[n][decisions[n].avail > 0]
    episodes = np.unique(decision.episode)
    ev_array = np.zeros((len(episodes),len(x)-2))
    for e in decision.episode:
        ep = decision[decision.episode == e]
        time_charging = ep.avail.sum()
        ev = list((ep['ev_load'].values) /1000)

        t = np.dot(x,time_charging)
        t = [math.floor(i) for i in t]
        
        soc = list(ep.soc)
        switch_down = {}
        for ind in range(len(soc)-1):
            if np.round(soc[ind+1],6)<np.round(soc[ind],6):
                switch_down[ind] = 1
            else:
                switch_down[ind] = 0
        

        quantiles_ev = [sum([switch_down[k] for k in range(t[i], t[i+1]+1)]) for i in range(len(t)-2)]
        ev_array[int(e-1),:] = quantiles_ev


    df_grid = pd.DataFrame(data = {int(x[q]*100): ev_array[:,q] for q in range(len(x)-2)})

    sns.boxplot(data = df_grid, ax = axes[i], orient = 'v', color = color_codes[n])
    
    axes[i].grid()
    
    
    
    ticks = np.arange(0,10)
    labels = [f'{t*10} - {(t+1)*10}%' for t in ticks]
    labels[0] = labels[0] + '\n' + 'Arrival'
    labels[-1] = labels[-1] + '\n' + 'Departure'
    for row in range(len(names)):
        axes[row].set_ylabel('kW', fontsize = 16)

        axes[row].set_xticks(ticks)
    
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel('Charging time %', fontsize = 16)
    
patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in names]
fig.legend(handles=patches, loc='lower center',ncol=int(len(names)))    
    