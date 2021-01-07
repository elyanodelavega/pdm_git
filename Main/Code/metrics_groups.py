# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:03:49 2020

@author: Yann
"""
## IMPORTS

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import pandas as pd
import seaborn as sns
import numpy as np

from df_prepare import  prices_romande_energie

folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/Full/'

sns.set_style('whitegrid')

img_path = 'C:/Users/Yann/Documents/EPFL/PDM/Images/'

#%% DEFINITION
objectives = ['cost', 'pv', 'peak']
methods_short = ['opti', 'mpc_d', 'mpc_s', 'mpc_s_cvar']
palette = sns.color_palette(n_colors = len(objectives)*len(methods_short) + 4)

color_codes = {}
c = 0
for  o in objectives:
    for m in methods_short:
        
        name = f'v2g_{m}_{o}'
        color_codes[name] = palette[c]
        c += 1



methods = ['Perfect Foresight  \ncost', 
           'MPC deterministic \ncost', 
           'MPC stochastic \nExp: cost \nExp: SOC', 
           'MPC stochastic \nCVaR: cost, \nExp: SOC',
            'Perfect Foresight  \nPV ', 
           'MPC deterministic \nPV', 
           'MPC stochastic \nExp: PV \nExp: SOC', 
           'MPC stochastic \nCVaR: PV, \nExp: SOC',
           'Perfect Foresight  \nAPR', 
           'MPC deterministic \nAPR ', 
           'MPC stochastic \nExp: APR \nExp: SOC', 
           'MPC stochastic \nCVaR: APR, \nExp: SOC']


# import random


# color_codes = {f'opti_pv_{i}': palette[random.randint(0, 10)] for j,i in enumerate(np.arange(0,1.1, 0.1))}

# methods = [f'SOC weight {i}' for j,i in enumerate(np.arange(0,1.1, 0.1))]

names = list(color_codes.keys())
methods_aglo = {methods[i]: names[i]
                  for i in range(len(names))}

csv_code = '.csv'
decisions = {n: pd.read_csv(folder_path+'results_'+n+csv_code, index_col = 0) for n in names}

for n in names:
    df = decisions[n][decisions[n].episode < 36]
    new_index = pd.to_datetime(df.index, dayfirst = True)
    df.index = new_index
    decisions[n] = df

algorithms = {names[i]: methods[i] for i in range(len(names))}

#%%
metrics = ['self_cons', 'soc_dep', 'cost', 'peak_factor']
metrics_title = ['PV self-consumption','SOC at Departure', 'Median Cost', 'APR' ]
metrics_unit = ['%','%', 'CHF','%']
metrics_props = {metrics[i]: {'title': metrics_title[i],'unit': metrics_unit[i]} for i in range(len(metrics))}

labels_radar = {'self_cons': 'PV',
          'cost': 'CP',
          'peak_factor': 'APR',
          'soc_dep': 'SOC '}
#%%
group_code = {'cost': [], 'peak': [], 'pv': [],
          'opti':[], 'mpc_d': [], 'mpc_s': [], 'mpc_s_cvar': []}

group_names = ['Objective: cost','Objective: Peak Shaving', 'Objective: PV',
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
groups_mpc = {}

groups_mpc['Objective: cost'] = ['v2g_'+m+'_cost' for m in methods_short[1:]]
groups_mpc['Objective: PV'] = ['v2g_'+m+'_pv' for m in methods_short[1:]]
groups_mpc['Objective: Peak Shaving'] = ['v2g_'+m+'_peak' for m in methods_short[1:]]

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
    
    df_sum = d.groupby('episode').sum()
    
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

    df['cost'] = df_med['cost']
    
    # df['cost_tot'] = df_sum['cost']
    
    df['peak_factor'] = 100*df_mean.grid_bought / df_max.grid_bought
    
    return df

def loss_ratio(stats_loss, quantile_low = 0.1, quantile_high = 0.9):
    
    df = stats_loss.copy()
    df_med = df.mean()
    
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



#%% box plot metrics


def ax_boxplot_metrics(ax, metric, g, algos, ylim = None, leg = True):
    # fig, axes = plt.subplots(len(metrics),1, sharex = True, figsize=(25,16))
    
    
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
    else:
        ax.set(xticklabels = [algos_specs[a][s] for a in algos])
    ax.grid()
    if ylim is not None:
        ax.set_ylim(ylim)
        
    return ax

#%% AX RADAR

def ax_radar(ax, g, algos, x_legend = 0.5, y_legend = -0.5, start = np.pi/4, leg = True, lw = 2, fs = 15):
    

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

        if m == 'cost':
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


        ax.plot(angles, vals_med, 'o-',  label = algos_specs[a][s], color = color_codes[a], linewidth = lw)
        ax.errorbar(angles, vals_med, yerr = errors_low,  uplims=True,color = color_codes[a])
        ax.errorbar(angles, vals_med, yerr = errors_high,  lolims=True,color = color_codes[a])
        
    
        ax.set_thetagrids(angles[:-1] * 180/np.pi, group_df_low.columns, fontsize = 16)
        
    
    ax.grid(True)
    if leg == True:
        ax.legend(loc = 'lower center',bbox_to_anchor=(x_legend,y_legend), ncol = 1, fontsize = fs)

    ax.set_xlabel('%', fontsize = fs)
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
    
        if m == 'cost':
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


color_indices = {'Grid': palette[12],
                  'PV': palette[13],
                  'EV': palette[14],
                  'Load': palette[15]}
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
    ax.set_ylim([0,100])
    ax.set_title(f'{name.upper()}')
    ax.grid(True)

    return ax


#%%
metrics_ylim =  {'self_cons': [0,105],
          'cost': None,
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
group_plot = groups_mpc
plot_reduce = True
if plot_reduce:
    for g in list(group_plot.keys()):
        if 'Objective' in g:
            s = 'Method'
        else:
            s = 'Objective'
        
        algos = group_plot[g]
        fig = plt.figure(figsize=(16, 9), dpi = 800)
        plt.suptitle(g, fontsize = 18)
        ax1 = fig.add_subplot(1,2,1, polar = True)
        ax_radar(ax1, g, algos,  leg = False, lw = 3, fs = 18)
        
        
        ax3= fig.add_subplot(4,2,2)
        ax4= fig.add_subplot(4,2,4)
        ax5= fig.add_subplot(4,2,6)
        ax6= fig.add_subplot(4,2,8)
        
        ax_box = [ax3, ax4, ax5, ax6]
        
        for i in range(len(ax_box)):
            m = metrics[i]
            ax = ax_box[i]

            ax_boxplot_metrics(ax, m, g,algos, ylim = metrics_ylim[m], leg = False)
        
        ncol = 3
        if len(algos) > 3:
            ncol = int(len(algos)/2)
        patches1 = [mpatches.Patch(color=color_codes[n], label=algos_specs[n][s]) for n in algos]
        ax1.legend(handles=patches1, loc='upper center',ncol=ncol,bbox_to_anchor = (0.5,1.15), fontsize = 13) 
        
        fig.savefig(img_path+g.replace(': ','_')+'.png', dpi = 800)
        
        fig.show()

#%%
full_plot = True
if full_plot:
    for g in list(groups.keys()):
        algos = groups[g]
        fig = plt.figure(figsize=(28, 15), dpi = 500)
        plt.suptitle(g, fontsize = 18)
        ax1 = fig.add_subplot(1,4,1, polar = True)
        ax_radar(ax1, g,algos,  leg = False)
        
        ax3= fig.add_subplot(4,4,2)
        ax4= fig.add_subplot(4,4,6)
        ax5= fig.add_subplot(4,4,10)
        ax6= fig.add_subplot(4,4,14)
        
        ax_box = [ax3, ax4, ax5, ax6]
        
        for i in range(len(ax_box)):
            m = metrics[i]
            ax = ax_box[i]
            if i == len(ax_box)-1:
                leg = True
            ax_boxplot_metrics(ax, m,  g,algos, ylim = metrics_ylim[m], leg = False)
            
        patches1 = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in algos]
        ax1.legend(handles=patches1, loc='upper center',ncol=int(len(algos)/2),bbox_to_anchor = (0.5,1.5), fontsize = 16) 
        
        
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

import pickle
time_algo = {}
for i, o in enumerate(objectives):
    for j, m in enumerate(methods_short):
        name = f'{m}_{o}'
        file_name = f'time_{name}'
        algo = f'v2g_{name}'
        file_inter = open(folder_path+file_name+'.pickle', 'rb')
        time_algo[algorithms[algo]] = pickle.load(file_inter)[:35]
        print(name,len(time_algo[algorithms[algo]]))
        file_inter.close()
        
#%%
df_time = pd.DataFrame.from_dict(time_algo)
fig, ax = plt.subplots(1,1,figsize = (18,11), dpi = 600)
sns.boxplot(data = df_time, ax = ax, palette = [color_codes[methods_aglo[n]] for n in methods])
ax.set_ylabel('Sec.', fontsize = 15)
ax.grid(True)
ax.set_title('Time per episode', fontsize = 17)

#%%
# variables = ['pv_ev','pv_load','grid_ev','grid_load','ev_grid','y_buy','y_sell','y_ch','y_dis','avail','cost','self_cons','peak_factor']
subm =  ['self_cons', 'cost', 'peak_factor']
fig, axes = plt.subplots(1,len(subm),figsize = (18,8), dpi = 600, sharey = True)
for k in range(len(subm)):
    g = list(groups.keys())[k]
    if 'Method' in g:
        break
    algos = groups[g]
    new_df = stats[algos[0]][subm]
    for i in range(1,len(algos)):
        a = algos[i]
        s = stats[a][subm]
        new_df.append(s)
    c = new_df.corr()
    c.rename(index={m: labels_radar[m] for m in subm}, inplace = True)
    c.rename(columns={m: labels_radar[m] for m in subm}, inplace = True)
    
    axes[k].set_title(g)
    sns.heatmap(c,ax = axes[k], annot=True, annot_kws={"size":12} , linewidths=.5)
#%%
avail = np.unique(stats[names[0]]['avail'])

bins = list(np.arange(11,30,4))
bins.append(31)
x = [np.round(0.1*i,2) for i in range(11)]



ticks = np.arange(0,11,2.5)
labels = [str(int(t)*10)+'%' for t in ticks]
labels[0] = labels[0] + '\n' + 'Arr.'
labels[-1] = labels[-1] + '\n' + 'Dep.'
#plt.suptitle(f'SOC at arrival: {s_arr}%',fontsize = 18)
for g in groups:
    algos = groups[g]
    fig, axes = plt.subplots(len(algos)+2,len(bins)-1, sharex = True, figsize=(16,11))
    for i in range(len(bins)-1):
        a_low = int(bins[i])
    
        a_high = int(bins[i+1])
    
        for j, n in enumerate(algos):
            
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
        
            sns.boxplot(data = df, ax = axes[0,i], orient = 'v', color = color_indices['PV'])
            
            df = pd.DataFrame(data = {int(x[q]*100): load_array[:,q] for q in range(len(x))})
        
            sns.boxplot(data = df, ax = axes[1,i], orient = 'v', color = color_indices['Load'])
            
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
    fig.legend(handles=patches, loc='lower center',ncol=int(len(names)/2))
            