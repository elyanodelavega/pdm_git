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
# color_codes = {'v2g_opti':palette[0],
#                 'v2g_mpc_d':palette[1],
#                 'v2g_mpc_s':palette[2],
#                 'v2g_mpc_s_cvar_cost':palette[4],
#                 'v2g_mpc_s_cvar_soc':palette[5],
#                 'v2g_opti_no_pen':palette[3],
#                 'v2g_mpc_d_no_pen':palette[6],
#                 'v2g_mpc_s_no_pen':palette[7],
#                 'v2g_mpc_s_cvar_cost_no_pen':palette[8],
#                 'v2g_mpc_s_cvar_soc_no_pen':palette[9],
#                 'v2g_opti_pv':palette[3],
#                 'v2g_mpc_d_pv':palette[6],
#                 'v2g_mpc_s_pv':palette[7],
#                 'v2g_mpc_s_cvar_soc_pv':palette[9]}



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

    df['Loss'] = df['cost']
    
    df['p_ev'] = p_ev
    
    df['pv_load_perc'] = pv_load_perc
    
    df['pv_ev_perc'] = pv_ev_perc
    
    df['pv_grid_perc'] = pv_grid_perc
    
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
    
#%%

# methods = ['Fully deterministic ',  'MPC deterministic', 
#            'MPC stochastic', 
#            'MPC stochastic \nExp: SOC, \nCVaR 75%: Cost',
#            'MPC stochastic \nExp: Cost, \nCVaR 75%: SOC',
#             'Fully deterministic \nSOC No penalty',  
#             'MPC deterministic \nSOC No penalty', 
#             'MPC stochastic \nSOC No penalty', 
#             'MPC stochastic \nExp: SOC, \nCVaR 75%: Cost \nSOC No penalty',
#             'MPC stochastic \nExp: Cost, \nCVaR 75%: SOC \nSOC No penalty',
#             'Fully deterministic \n PV',
#             'MPC deterministic \n PV', 
#             'MPC stochastic \n PV', 
#             'MPC stochastic \nExp: PV, \nCVaR 75%: SOC ']

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


stats = {n: quick_stats(decisions[n],prices_romande_energie) for n in names}

metrics = ['self_cons', 'soc_dep', 'Loss', 'peak_factor']
           # 'pv_load_perc','pv_ev_perc','pv_grid_perc']

stats_df = {m: pd.DataFrame(data = {n: list(stats[n].loc[:,m] )
                             for n in names}) for m in metrics}

benchmark = 'v2g_opti_cost'
n_episodes = len(stats[benchmark])

range_episodes = range(int(stats[benchmark].index[0]), int(stats[benchmark].index[-1] + 1))

metrics_title = ['PV self-consumption','SOC at Departure', 'Median Loss', 'Peak factor' ]
metrics_label = ['%','%', 'CHF','%']
metrics_props = {metrics[i]: {'title': metrics_title[i],'label': metrics_label[i]} for i in range(len(metrics))}

#%%     
group_code = {'cost': [], 'peak': [], 'pv': [],
          'opti':[], 'mpc_d': [], 'mpc_s': [], 'mpc_s_cvar': []}

group_names = ['Objective: Cost','Objective: Peak Shaving', 'Objective: PV',
               'Method: Perfect Foresight','Method: MPC deterministic',
               'Method: MPC stochastic, Expected', 'Method: MPC stochastic, CVaR']
groups = {}
for n in names:
    
    for i,g in enumerate(group_code.keys()):
        
        if g in n:
            
            group_code[g].append(n)
            
        groups[group_names[i]] = group_code[g]
#%% box plot variation

fig, axes = plt.subplots(len(metrics),1, figsize=(16,9), sharex = True)
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
    axes[j].set_title(metrics_props[i]['title'])

        

handles, labels= axes[0].get_legend_handles_labels()
fig.legend(handles,labels, loc='lower center',ncol=len(names))

for i in range(len(axes)):
    axes[i].axhline(0, color = 'black')
    axes[i].set_ylabel('%' )
    axes[i].grid()


fig.show()

#%% Box plot self, soc, cons

fig, axes = plt.subplots(len(metrics),1, sharex = True, figsize=(25,16))

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

    axes[i].set_title(metrics_props[m]['title'])
    axes[i].set_ylabel(metrics_props[m]['label'])

    axes[i].grid()
    if i != 2:
        axes[i].set_ylim([0,105])
    

fig.show()


#%% RADAR

labels = {'self_cons': 'PV \nself \nconsumption ',
          'Loss': 'Cost \nPerformance',
          'peak_factor': 'Peak \nAverage Ratio ',
          'soc_dep': 'SOC \nat departure '}



gr_1 = list(groups.keys())[:3]
gr_2 = list(groups.keys())[3:]


for gr in [gr_1,gr_2]:
    fig, ax =plt.subplots(1,len(gr),figsize = (30,14),subplot_kw=dict(polar=True))
    for i, g in enumerate(gr):
        
        algos = groups[g]
        group_df_low = pd.DataFrame(index = algos)
        group_df_high = pd.DataFrame(index = algos)
        group_df_med = pd.DataFrame(index = algos)
        errors = {}
        
        
        
        angles=np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles=np.concatenate((angles,[angles[0]]))
        for m in metrics:
    
            if m == 'Loss':
                low, med, high = loss_ratio(stats_df[m])
                group_df_low[labels[m]] = low[algos]
                group_df_high[labels[m]] = high[algos]
                group_df_med[labels[m]] = med[algos]
            else:
                group_df_low[labels[m]] = stats_df[m].quantile(0.1)[algos]
                group_df_high[labels[m]] = stats_df[m].quantile(0.9)[algos]
                group_df_med[labels[m]] = stats_df[m].median()[algos]
            

        for a in algos:
            vals_low =group_df_low.loc[a,:]
            errors_low = group_df_med.loc[a,:] - vals_low
            
            errors_low=np.concatenate((errors_low,[errors_low[0]]))
            vals_high =group_df_high.loc[a,:]
            errors_high =  vals_high - group_df_med.loc[a,:]
            
            errors_high=np.concatenate((errors_high,[errors_high[0]]))
            vals_med =group_df_med.loc[a,:]
            
            vals_med=np.concatenate((vals_med,[vals_med[0]]))
    

            ax[i].plot(angles, vals_med, 'o-',  label = algorithms[a], color = color_codes[a])
            ax[i].errorbar(angles, vals_med, yerr = errors_low,  uplims=True,color = color_codes[a])
            ax[i].errorbar(angles, vals_med, yerr = errors_high,  lolims=True,color = color_codes[a])
            
            ax[i].set_thetagrids(angles[:-1] * 180/np.pi, group_df_low.columns)
            
        
        ax[i].grid(True)
        ax[i].legend(loc = 'lower center',bbox_to_anchor=(0.5,-0.5), ncol = 2, fontsize = 14)
        ax[i].set_title(f'{g}')
        ax[i].set_xlabel('%', fontsize = 18)
        ax[i].set_ylim([0,105])
        
        
        fig.show()


    
#%% RADAR

labels = {'self_cons': 'PV \nself \nconsumption ',
          'Loss': 'Cost \nPerformance',
          'peak_factor': 'Peak \nAverage Ratio ',
          'soc_dep': 'SOC \nat departure '}



angles=np.linspace(0, 2*np.pi, len(names), endpoint=False)
angles=np.concatenate((angles,[angles[0]]))
fig, ax = plt.subplots(1,1,figsize = (16,9),subplot_kw=dict(polar=True))
for m in metrics:

    if m == 'Loss':
        _, med, _ = loss_ratio(stats_df[m])
        

    else:
        
        med = stats_df[m].median()
    

    
    vals_med = med
    
    vals_med=np.concatenate((vals_med,[vals_med[0]]))


    ax.plot(angles, vals_med, 'o-',  label = labels[m])
    
    ax.set_thetagrids(angles[:-1] * 180/np.pi, methods)
    

ax.grid(True)
ax.legend(loc = 'lower center',bbox_to_anchor=(0.5,-0.2), ncol = len(metrics))
ax.set_xlabel('%', fontsize = 18)
ax.set_ylim([0,105])


fig.show()
    
#%% RADAR MPC

labels = {'self_cons': 'PV \nself \nconsumption ',
          'Loss': 'Cost \nPerformance',
          'peak_factor': 'Peak \nAverage Ratio ',
          'soc_dep': 'SOC \nat departure '}


algos = [n for n in names if 'opti' not in n]
angles=np.linspace(0, 2*np.pi, len(algos), endpoint=False)
angles=np.concatenate((angles,[angles[0]]))
fig, ax = plt.subplots(1,1,figsize = (16,9),subplot_kw=dict(polar=True))

algo_names = [algorithms[n] for n in algos]
for m in metrics:

    if m == 'Loss':
        _, med, _ = loss_ratio(stats_df[m])
        

    else:
        
        med = stats_df[m].median()
    

    
    vals_med = med[algos]
    
    vals_med=np.concatenate((vals_med,[vals_med[0]]))


    ax.plot(angles, vals_med, 'o-',  label = labels[m])
    
    ax.set_thetagrids(angles[:-1] * 180/np.pi, algo_names)
    

ax.grid(True)
ax.legend(loc = 'lower center',bbox_to_anchor=(0.5,-0.2), ncol = len(metrics))
ax.set_xlabel('%', fontsize = 18)
ax.set_ylim([0,105])
plt.title('Overall scores,\nMPC')

fig.show()  
    
    
#%% RADAR MPC

labels_metrics = {'self_cons': 'PV \nself \nconsumption ',
          'Loss': 'Cost \nPerformance',
          'peak_factor': 'Peak \nAverage Ratio ',
          'soc_dep': 'SOC \nat departure '}




for g in groups.keys():
    algos = groups[g]
    angles=np.linspace(0, 2*np.pi, len(algos), endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))
    fig, ax = plt.subplots(1,1,figsize = (16,9),subplot_kw=dict(polar=True))
    
    algo_names = [algorithms[n] for n in algos]
    for m in metrics:
    
        if m == 'Loss':
            _, med, _ = loss_ratio(stats_df[m])
            
    
        else:
            
            med = stats_df[m].median()
        
    
        
        vals_med = med[algos]
        
        vals_med=np.concatenate((vals_med,[vals_med[0]]))
    
    
        ax.plot(angles, vals_med, 'o-',  label = labels[m])
        
        ax.set_thetagrids(angles[:-1] * 180/np.pi, algo_names)
        
    
    ax.grid(True)
    ax.legend(loc = 'lower center',bbox_to_anchor=(0.5,-0.2), ncol = len(metrics))
    ax.set_xlabel('%', fontsize = 18)
    ax.set_ylim([0,105])
    plt.title(f'Overall scores,\n{g}')
    
    fig.show()    
  
    
#%% repartition plot



def plot_repartition(ax, decisions, names, variable, indices, labels, color_indices, methods, name = None):
    
    new_df = pd.DataFrame(index = indices, columns=list(algorithms.keys()))

    x = np.arange(len(names))
    if name == None:
        name = variable[0]
    
    for  n in names:
        
        s_df = decisions[n]
        
        
        tot = s_df[variable].sum().sum()
        
        for ind in indices:
            
            new_df.loc[ind,n] = 100*s_df[ind].sum()/tot
    
    for k,ind in enumerate(indices):
        if k == 0: 
            ax.bar(x, new_df.loc[ind,:],  color = color_indices[labels[k]])
        
        elif k== 1:
            ax.bar(x, new_df.loc[ind,:], bottom=new_df.loc[indices[0],:], color = color_indices[labels[k]])
        
        else:
            ax.bar(x, new_df.loc[ind,:], bottom= new_df.loc[:indices[k-1],:].sum().values, color = color_indices[labels[k]])
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('%')
    ax.set_title(f'{name.upper()}')

    return ax
    
    
#%%
labels_tot = ['PV', 'Grid', 'EV', 'Load']
indices = ['pv_load', 'grid_load', 'ev_load']

labels = ['PV', 'Grid', 'EV']

colors = sns.color_palette()

color_indices = {'Grid': colors[0],
                  'PV': colors[1],
                  'EV': colors[2],
                  'Load': colors[3]}

variable = ['load']

fig, axes = plt.subplots(5,1, sharex = True, figsize=(18,9))

plot_repartition(axes[0], decisions, names,variable, indices, labels, color_indices, methods)

indices = ['pv_load', 'pv_ev', 'pv_grid']

labels = ['Load', 'EV', 'Grid']   

variable = ['pv']

plot_repartition(axes[1], decisions, names,variable, indices, labels,color_indices, methods)

indices = ['grid_load', 'grid_ev']

labels = ['Load', 'EV']   

variable = ['grid_load','grid_ev']

plot_repartition(axes[2], decisions, names,variable, indices, labels,
                 color_indices, methods, name = 'Grid')


new_dec = {}
for n in names:
    dec = decisions[n].copy()
    dec['residual_pv'] = dec['pv']- dec['pv_load']
    new_dec[n] = dec.copy()

indices = ['pv_ev', 'pv_grid']

labels = ['EV', 'Grid']   

variable = ['residual_pv']

plot_repartition(axes[3], new_dec, names,variable, indices, labels,
                 color_indices, methods, name = 'Residual pv (PV - PV to Load)')

new_dec = {}
for n in names:
    dec = decisions[n].copy()
    dec['p_ev'] = dec['grid_ev'] +  dec['pv_ev']
    new_dec[n] = dec.copy()

indices = ['pv_ev', 'grid_ev']

labels = ['PV', 'Grid']   

variable = ['p_ev']

plot_repartition(axes[4], new_dec, names,variable, indices, labels,
                 color_indices, methods, name = 'EV')

patches = [mpatches.Patch(color=color_indices[labels_tot[i]], label= labels_tot[i]) for i in range(len(labels_tot))]

fig.legend(handles=patches, loc='lower center',bbox_to_anchor= (0.5,0),ncol=int(len(labels_tot)), fontsize = 16)

fig.show()
#%%
metrics_comp = ['pv', 'load', 'PV self consumption [%]', 'Cost [Cents]']
colors_parameters = ['green','blue']

fig, axes = plt.subplots(len(metrics_comp),1, sharex = True, figsize=(20,12))

algos = names.copy()
for i, n in enumerate(algos):
        
        
    data = decisions[n].copy()
    
    data['hour'] = [t.hour for t in data.index]
    data = prices_romande_energie(data)
    data['PV self consumption [%]'] = 100*(data['pv_ev']+data['pv_load'])/(data['pv'])
    data['Cash out [Cents]'] = 100*(data['grid_load'] + data['grid_ev'])*data['buy']

    data['Cash in [Cents]'] = 100*(data['pv_grid'] + data['ev_grid'])*data['buy']
    
    data['Cost [Cents]'] = data['Cash out [Cents]'] - data['Cash in [Cents]']
    data['soc'] = data['soc']*100
    
    d = data.groupby('hour').mean()
    
    for a in range(2,len(axes)):
        axes[a].plot(d.index, d[metrics_comp[a]], label = algorithms[n])

        # axes[3].plot(d.index,)
        #sns.boxplot(x="hour", y="pv", data=decision, ax = axes[0], color = 'green')
    for a in range(len(metrics_comp[:2])):
        axes[a].bar(d.index, d[metrics_comp[a]]/1000,color = colors_parameters[a])
        axes[a].set_ylabel('kW')
            
    for a in range(len(axes)):
        axes[a].grid(True)
        axes[a].set_title(metrics_comp[a].upper())
        axes[a].set_xlim([0,23])
axes[-1].set_xticks([i for i in range(24)])
axes[-1].set_xticklabels([str(i)+':00' for i in range(24)])
plt.legend(loc = 'lower center', ncol = int(len(names)/2), bbox_to_anchor = (0.5,-1.05))

    
#%% RADAR MPC
def add_metrics(decisions):
    
    data = decisions.copy()
        
    data['hour'] = [t.hour for t in data.index]
    data = prices_romande_energie(data)
    data['self_cons'] = 100*(data['pv_ev']+data['pv_load'])/(data['pv'])
    data['cost_buy'] = (data['grid_load'] + data['grid_ev'])*data['buy']

    data['cost_sell'] = (data['pv_grid'] + data['ev_grid'])*data['buy']
    
    data['Loss'] = data['cost_buy'] - data['cost_sell']

    soc_dep = [data.soc[i] *100 for i in range(1, len(data)) if data.avail[i] < data.avail[i-1]]
    
    data['grid_bought'] = 100*(data['grid_load'] + data['grid_ev'])
    df_mean = data.groupby('episode').mean()
    
    df_max = data.groupby('episode').max()
    
    peak_factor = 100*df_mean.grid_bought / df_max.grid_bought
    
    return data, soc_dep, peak_factor

labels_2 = {'self_cons': 'PV self consumption [%]',
          'Loss': 'Cost [CHF]',
          'peak_factor': 'Peak Average Ratio [%]',
          'soc_dep': 'SOC at departure [%]'}


def scatter_plot_comp(variable, metrics, decisions, algos,labels, special = None,
                      ratio = 1, suptitle = None, unit = None):
    new_df = pd.DataFrame(index = list(algorithms.keys()), columns=metrics)


    for  n in algos:
        
        s_df = decisions[n]
        
        data, soc_dep, peak_factor = add_metrics(s_df)
        tot = s_df[variable].sum().sum()/ratio
        if special is not None:
            tot = special[n]
         
        new_df.loc[n,'self_cons'] = data['self_cons'].mean()
        new_df.loc[n,'Loss'] = data['Loss'].sum()
        new_df.loc[n, 'peak_factor'] = peak_factor.mean()
        new_df.loc[n, 'soc_dep'] = np.mean(soc_dep)
        new_df.loc[n, variable] = tot
    fig, axes = plt.subplots(2,2, figsize = (16,9), sharex = True)
    for i in range(len(axes)*2):
        row = int(i/2)
        if i == 1 or i == 3:
            col = 1
        else:
            col = 0

        axes[row,col].scatter(new_df[variable], new_df[metrics[i]], label = list(algorithms.keys()), 
                              color = list(color_codes.values()), s = 200)
        axes[row,col].set_ylabel(labels[metrics[i]])
        axes[row,col].grid()
        if row == 1:
            axes[row,col].set_xlabel(unit)
    plt.suptitle(suptitle, fontsize = 16)    
    patches = [mpatches.Patch(color=color_codes[n], label=algorithms[n]) for n in algos]
    fig.legend(handles=patches, loc='lower center',ncol=int(len(names)/2), bbox_to_anchor = (0.5,-0.1)) 

#%% EV to load
scatter_plot_comp('ev_load', metrics, decisions, names, labels_2, ratio = 10e3, suptitle = 'EV to Load', unit = 'kWh')

scatter_plot_comp('pv_ev', metrics, decisions, names, labels_2, ratio = 10e3, suptitle = 'PV to EV', unit = 'kWh')

scatter_plot_comp('ev_grid', metrics, decisions, names, labels_2, ratio = 10e3, suptitle = 'EV to Grid', unit = 'kWh')

scatter_plot_comp('pv_grid', metrics, decisions, names, labels_2, ratio = 10e3, suptitle = 'PV to Grid', unit = 'kWh')


#%%

scatter_plot_comp('y_dis', metrics, decisions, names, labels_2, ratio = 1, suptitle = 'Discharges', unit = 'N')

#%%
special = {}
for o in decisions:
    df = decisions[o].copy()
    special[o] = 100*df[df.avail > 0].soc.median()

scatter_plot_comp('soc', metrics, decisions, names, labels_2, special = special, ratio = 1, suptitle = 'Median SOC', unit = '%')

#%%
special = {}
for o in decisions:
    df = decisions[o].copy()
    special[o] = 100*df[df.avail > 0].soc.mean()

scatter_plot_comp('soc', metrics, decisions, names, labels_2, special = special, ratio = 1, suptitle = 'Mean SOC', unit = '%')

#%%
special = {}
for o in decisions:
    df = decisions[o].copy()
    y_dis = list(df[df.avail > 0].y_dis)
    count = 0
    for ind in range(len(y_dis)-1):
        if y_dis[ind+1] > y_dis[ind]:
           count += 1
    special[o] = count
scatter_plot_comp('y_dis', metrics, decisions, names, labels_2, special = special, ratio = 1, suptitle = 'Charge switch', unit = 'N')


#%%
labels_metrics_flat = {'self_cons': 'PV self-consumption ',
 'Loss': 'Losses',
 'peak_factor': 'Peak Average Ratio ',
 'soc_dep': 'SOC at departure '}

metrics = ['self_cons', 'Loss', 'peak_factor']
for g in groups.keys():
    fig, axes = plt.subplots(len(metrics),1, figsize=(25,18))
    plt.suptitle(f'{g}', fontsize = 22)
    algos = groups[g]
    for i,m in enumerate(metrics):
        df = stats_df[m]

        for n in algos:
            
            # sns.kdeplot(data = data, x = df[n], cumulative=True, linewidth = 2,
            #                 color = color_codes[n], label = algorithms[n], ax = axes[i], fill = True)
            axes[i].hist(df[n], color = color_codes[n], bins = 60, cumulative = True, density = True,
                         label = algorithms[n], alpha = 0.4)

    for m in range(len(metrics)):
        axes[m].grid(True)
        axes[m].set_title(labels_metrics_flat[metrics[m]], fontsize = 15)
        axes[m].set_xlabel(None)
        axes[m].set_yticks(np.arange(0,1.2,0.2))
        axes[m].set_yticklabels([int(i*100) for i in np.arange(0,1.2,0.2)])
        axes[m].set_ylabel('%')
    axes[-1].legend(fontsize = 15, loc = 'upper left')
    


