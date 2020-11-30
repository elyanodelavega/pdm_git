# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:45:40 2020

@author: Yann
"""

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))

cost2 = {m: np.cumsum([c/1000 for c in costs[m]]) for m in costs}
for m in costs.keys():
    plt.plot(cost2[m], label = m)

plt.xticks(ticks = range(n_episodes), labels = range_episodes)
plt.legend(ncol = 2)
plt.xlabel('Episode')
plt.ylabel('Electricity bought [kWh]')
plt.title('Cumulative electricity bought comparison')

#%% Plot Variantion

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3,1, figsize=(20,12), sharex = True)
#plt.suptitle(' Relative comparison with optimal solution', fontsize = 25)
cost2 = {m: [c/1000 for c in costs[m]] for m in costs}
benchmark_cost = cost2['1. Fully deterministic']
benchmark_stats = stats['1. Fully deterministic']
power = []
soc = []
pv = []

for m in list(costs.keys())[1:]:
    relative_power = [100* cost2[m][i]/benchmark_cost[i] - 100 for i in range(n_episodes)]
    axes[2].bar(range(n_episodes), relative_power, label = m)
    r = stats[m]
    relative_soc = []
    relative_pv = []
    
    for e in range_episodes:
        relative_soc.append(100*r[e]['absolut']['SOC_last']/benchmark_stats[e]['absolut']['SOC_last'] - 100)
        relative_pv.append(100*r[e]['relative']['Self_consumption']/benchmark_stats[e]['relative']['Self_consumption'] - 100)

    axes[0].bar(range(n_episodes), relative_pv, label = m)
    axes[1].bar(range(n_episodes), relative_soc, label = m)
    
    
    axes[0].set_title('PV self-consumption', fontsize = 23)
    axes[1].set_title('SoC at departure', fontsize = 23)
    axes[2].set_title('Power bought', fontsize = 23)
    
    power.extend(relative_power)
    soc.extend(relative_soc)
    pv.extend(relative_pv)

handles, labels= axes[0].get_legend_handles_labels()
fig.legend(handles,labels, loc='lower center',ncol=2 , fontsize = 20)


axes[0].set_ylim([min(min(min(pv, soc)), -max(power))-2, 5])
axes[1].set_ylim([min(min(min(pv,soc)),- max(power))-2, 5])
axes[2].set_ylim( [- 5, max(max(power), - min(min(pv,soc)))+2])

for i in range(3):
    axes[i].axhline(0, color = 'black')
    axes[i].set_ylabel('%' , fontsize = 20)
    

    
plt.xlabel('Episode', fontsize = 22)







#%%
for e in range_episodes:
    
    for m in results.keys():
        fig, axes = plt.subplots(4,1, sharex=True, figsize=(16,9))
        episode = results[m][results[m].episode == e]
        grid_ev_bought = episode.loc[:,['grid_ev']].sum(axis = 0).sum()
        grid_load_bought = episode.loc[:,['grid_load']].sum(axis = 0).sum()
        p_grid_bought = episode.loc[:,['grid_ev', 'grid_load']].sum(axis = 1)
        total = grid_ev_bought + grid_load_bought
        plt.suptitle(f'{m}, Episode {e}, Power bought: {int(total/1000)} kWh ')
        x = np.arange(len(episode))
        
        axes[0].plot(x, list(episode.pv/1000), label = 'PV')
        axes[0].plot(x, list(episode.pv_ev/1000), label = 'PV_EV')
        axes[0].plot(x, list(episode.pv_load/1000), label = 'PV_Load')
        axes[0].set_title('PV')
        axes[0].legend(ncol = 3, loc = 'upper left')
        
        
        axes[1].plot(x, list(episode.load/1000), label = 'Load')
        axes[1].plot(x, list(episode.grid_load/1000), label = 'Grid_load')
        axes[1].plot(x, list(episode.pv_load/1000), label = 'PV_Load')
        axes[1].set_title('Load')
        axes[1].legend(ncol = 3, loc = 'upper left')
        
        axes[2].plot(list(p_grid_bought/1000) , label = 'Grid')
        axes[2].plot(list(episode.grid_load/1000), label = 'Grid_load')
        axes[2].plot(list(episode.grid_ev/1000), label = 'Grid_EV')
        axes[2].set_title('Grid')
        axes[2].legend(ncol = 3, loc = 'upper left')
        
        axes[3].plot(list(episode.pv_ev/1000), label = 'pv_ev')
        axes[3].plot(list(episode.grid_ev/1000), label = 'grid_ev')
        ax3 = axes[3].twinx()
        ax3.plot(x,list(episode.soc*100),'--', color = 'grey')
        ax3.set_ylabel('EV SoC [%]', color='black')
        ax3.set_ylim([0,110])
        ax3.set_title('SOC')
        axes[3].set_xlabel('Hours')
        axes[3].legend(ncol = 2, loc = 'upper left')
        
        try:
            t_dep = list(episode.avail).index(0)
        except:
            t_dep = x[-1]
            
        for i in range(3):
            axes[i].axvline(t_dep, color = 'black')
            
            
        axes[3].axvline(t_dep, color = 'black')
        time = episode.index
        ticks = np.arange(0, max(x)+1, step=6)
        ticks = [i for i in range(0, max(x) + 1, 4)]
        labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
        plt.xticks(ticks = ticks, labels = labels)
        
        
        plt.show()
#%%

for e in range_episodes:
    fig, axes = plt.subplots(3,1, sharex=True, figsize=(16,9))
    plt.suptitle(f' Episode {e}')
    for i, m in enumerate(results.keys()):
        
        episode = results[m][results[m].episode == e]
        grid_ev_bought = episode.loc[:,['grid_ev']].sum(axis = 0).sum()
        grid_load_bought = episode.loc[:,['grid_load']].sum(axis = 0).sum()
        p_grid_bought = episode.loc[:,['grid_ev', 'grid_load']].sum(axis = 1)
        total = grid_ev_bought + grid_load_bought
        
        x = np.arange(len(episode))
        
        axes[i].plot(list(episode.pv_ev/1000), label = 'pv_ev')
        axes[i].plot(list(episode.grid_ev/1000), label = 'grid_ev')
        ax1 = axes[i].twinx()
        ax1.plot(x,list(episode.soc*100),'--', color = 'grey')
        ax1.set_ylabel('EV SoC [%]', color='black')
        ax1.set_ylim([0,110])
        axes[i].set_title(f'{m}: cost: {int(costs[m][e - episode_start]/1000)}')
        
        axes[i].legend(ncol = 2, loc = 'upper left')
        
        try:
            t_dep = list(episode.avail).index(0)
        except:
            t_dep = x[-1]
            
        
        axes[i].axvline(t_dep, color = 'black')
            
            
        
    time = episode.index
    ticks = np.arange(0, max(x)+1, step=6)
    ticks = [i for i in range(0, max(x) + 1, 4)]
    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
    plt.xticks(ticks = ticks, labels = labels)
    
    
    plt.show()
    
#%%

import seaborn as sns
fig, axes = plt.subplots(len(methods),3, figsize=(20,12))


for i, m in enumerate(methods):
    r = stats[m]
    soc_last = []
    p_bought = []
    self_cons = []
    
    for e in range_episodes:
        soc_last.append(r[e]['absolut']['SOC_last']*100)
        p_bought.append(r[e]['absolut']['P_G_bought']/1000)
        self_cons.append(r[e]['relative']['Self_consumption']*100)

    sns.boxplot(self_cons, ax = axes[i,0], color = 'green')
    sns.boxplot(p_bought, ax = axes[i,1])
    sns.boxplot(soc_last, ax = axes[i,2], color = 'red')
    axes[i,2].set_xlim([50,102])

    

axes[0,1].set_title('Power Bought')
axes[2,1].set_xlabel('kW')
axes[0,0].set_title('PV self consumption')
axes[2,0].set_xlabel('%')
axes[0,2].set_title('SOC at departure')
axes[2,2].set_xlabel('%')
plt.grid()
plt.show()
    
#%%


