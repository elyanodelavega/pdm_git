# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:33:36 2020
​
@author: dorokhov
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

sns.set_style('darkgrid',{'legend.frameon':True})
palette = sns.color_palette()
plt.rcParams.update({'font.size': 14})

def confidence_interval(df, q):
    arr = np.array(df.values).astype('float64')
    med = np.median(arr, axis = 1)
    low = np.quantile(arr, 1-q, axis = 1)
    high = np.quantile(arr, q, axis = 1)

    return low, med, high

def plot_MPC(results_forecast,decisions, figname = None):
    
    t_decision = results_forecast.index[0]
    
    results_forecast = results_forecast.copy()

    results_forecast['pv'] = results_forecast['pv']/1000 # to kW
    results_forecast['pv_forecast'] = results_forecast['pv_forecast']/1000
    results_forecast['load'] = results_forecast['load']/1000 # to kW
    results_forecast['pv_ev'] = results_forecast['pv_ev']/1000 # to kW
    results_forecast['grid_ev'] = results_forecast['grid_ev']/1000 # to kW
    results_forecast['soc'] = results_forecast['soc']*100 # to %
    
    results_forecast.drop(t_decision, inplace = True)
    n = max(len(decisions), 12)
        
    decisions = decisions.tail(n).copy()

    decisions['pv'] = decisions['pv']/1000 # to kW
    decisions['pv_forecast'] = decisions['pv_forecast']/1000
    decisions['load'] = decisions['load']/1000 # to kW
    decisions['pv_ev'] = decisions['pv_ev']/1000 # to kW
    decisions['grid_ev'] = decisions['grid_ev']/1000 # to kW
    decisions['soc'] = decisions['soc']*100 # to %
    
    dataset = decisions.append(results_forecast, ignore_index = False)
    
    time = dataset.index

    
    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,7))
    plt.title(f'Decision at {t_decision}')
    x = range(0,len(dataset))
    
    plt.xlabel('Hours')
    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,max(dataset['pv'].max(),dataset['load'].max())+3])
    # ticks = [time[i].hour for i in range(0, len(dataset), 6)]
    # plt.xticks(ticks)
    ticks = np.arange(0, max(x)+1, step=6)
    ticks = [i for i in range(0, max(x) + 1, 6)]
    
    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
    
    plt.xticks(ticks = ticks, labels = labels)
    
    
    ax2 = axes[1].twinx() # ax for plotting EV SOC evolution
    ax2.set_ylabel('EV state of charge [%]', color='black')
    
#    ax1.grid(False)
    ax2.grid(False)
    

    # top subplot

    
    # top subplot
    axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.5)
    axes[0].plot(x, dataset['load'], color=palette[3], label='Building load')
    
    axes[0].fill_between(x, dataset['pv_forecast'], color=palette[0], alpha=0.5)
    axes[0].plot(x, dataset['pv_forecast'], color=palette[0], label='PV')
    
    
    axes[0].axvline(x=n-1, color = 'black')
    
    
    axes[0].axvspan(x[n-1], len(dataset) , alpha=0.2, color='grey', hatch = '//', label = 'forecast')
    
    # bottom subplot
    ax2.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
    ax2.set_ylim([0,110])
    
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[1].set_ylim([0,6])
    
    axes[1].axvline(x=n-1, color = 'black')
    
    
    axes[1].axvspan(x[n-1], len(dataset), alpha=0.2, color='grey' , hatch = '//')
    
    #axes[0].axvspan(x[0], x[n-1], alpha=0.5, color='grey')
    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    axes[0].legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(1, 0.9), facecolor='white', ncol = 3)
    axes[1].legend(handles_2, labels_2, loc='upper right', facecolor='white')
    
    if figname:
        plt.savefig(figname)
        
def plot_MPC_uncertainty(decisions, SOC, predictions, load, figname = None):
    
    t_decision = decisions.index[-1]
    
    decisions = decisions.copy()
    predictions = predictions.copy()
    load = load[:24].copy()
    SOC = SOC.copy()

    load = load/1000 # to kW
    predictions = predictions/1000

    n = min(len(decisions), 12)
    m = 24 - n
        
    decisions = decisions.tail(n).copy()

    decisions['pv'] = decisions['pv']/1000 # to kW
    decisions['pv_ev'] = decisions['pv_ev']/1000 # to kW
    decisions['grid_ev'] = decisions['grid_ev']/1000 # to kW
   
    time = list(decisions.index)
    
    time.extend(list(predictions.index)[:m])

    
    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,7))
    plt.title(f'Decision at {t_decision}')
    x = range(24)
    
    
    plt.xlabel('Hours')
    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,max(decisions['pv'].max(),load.max())+3])
    # ticks = [time[i].hour for i in range(0, len(dataset), 6)]
    # plt.xticks(ticks)
    ticks = np.arange(0, max(x)+1, step=6)
    ticks = [i for i in range(0, max(x) + 1, 6)]
    
    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
    
    plt.xticks(ticks = ticks, labels = labels)
    
    
    ax2 = axes[1].twinx() # ax for plotting EV SOC evolution
    ax2.set_ylabel('EV state of charge [%]', color='black')
    
#    ax1.grid(False)
    ax2.grid(False)

    
    # top subplot
    axes[0].fill_between(x, load, color=palette[3], alpha=0.5, label='Building load')
    axes[0].plot(x, load, color=palette[3])
    
    
    axes[0].fill_between(x[:n], decisions['pv'], color=palette[0], alpha=0.5)
    axes[0].plot(x[:n], decisions['pv'], color=palette[0], label='Actual PV')
    
    low_PV, med_PV, high_PV = confidence_interval(predictions.head(m), 0.8)
    axes[0].fill_between(x[n-1:n+m-1], low_PV[:m], high_PV[:m], color=palette[0], alpha=0.5, hatch = '//', label='Expected PV')
    #axes[0].plot(x[n-1:n+m-1], med_PV[:m], color=palette[0])
    

    axes[0].axvline(x=n-1, color = 'black')
    
    
    # axes[0].axvspan(x[n-1], len(dataset) , alpha=0.2, color='grey', hatch = '//', label = 'forecast')
    
    # bottom subplot
    low_SOC, med_SOC, high_SOC = confidence_interval(SOC.head(m), 0.95)
    
    soc_values = list(decisions['soc'])
    soc_values.extend(med_SOC)
    
    soc_values = [s*100 for s in soc_values]
    high_SOC = [s*100 for s in high_SOC]
    low_SOC = [s*100 for s in low_SOC]
    
    ax2.plot(x, soc_values, color=palette[7], marker='.', label='EV state of charge')
    ax2.fill_between(x[n:n+m], low_SOC[:m], high_SOC[:m], color=palette[7], alpha = 0.5)
    ax2.set_ylim([0,110])
    
    axes[1].bar(x[:n], decisions['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x[:n], decisions['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[1].set_ylim([0,6])
    
    
    
    
    axes[1].axvline(x=n-1, color = 'black')
    
    
    # axes[1].axvspan(x[n-1], len(dataset), alpha=0.2, color='grey' , hatch = '//')
    
    #axes[0].axvspan(x[0], x[n-1], alpha=0.5, color='grey')
    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    axes[0].legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(1, 0.9), facecolor='white', ncol = 3)
    axes[1].legend(handles_2, labels_2, loc='upper right', facecolor='white')
    
    if figname:
        plt.savefig(figname)
    

def plot_MPC_results(df, figname = None):
    
    dataset = df.copy()
    

    dataset['pv'] = dataset['pv']/1000 # to kW
    dataset['pv_forecast'] = dataset['pv_forecast']/1000
    dataset['load'] = dataset['load']/1000 # to kW
    dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
    dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
    dataset['soc'] = dataset['soc']*100 # to %
    time = dataset.index
    
    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,7))
    plt.title(time[0])
    x = range(0,len(dataset))
    
    plt.xlabel('Hours')
    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,max(dataset['pv'].max(),dataset['load'].max())+3])
    # ticks = [time[i].hour for i in range(0, len(dataset), 6)]
    # plt.xticks(ticks)
    ticks = np.arange(0, max(x)+1, step=6)
    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
    
    plt.xticks(ticks = ticks, labels = labels)
    
    
    ax2 = axes[1].twinx() # ax for plotting EV SOC evolution
    ax2.set_ylabel('EV state of charge [%]', color='black')
    
#    ax1.grid(False)
    ax2.grid(False)
    

    # top subplot
    axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.3)
    axes[0].plot(x, dataset['load'], color=palette[3], label='Building load')
    
    axes[0].fill_between(x, dataset['pv_forecast'], color=palette[0], alpha=0.3)
    axes[0].plot(x, dataset['pv_forecast'], color=palette[0], label='PV forecast')

    
    # bottom subplot
    ax2.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
    ax2.set_ylim([0,110])
    
    #if max(max(dataset['grid_ev']),max(dataset['pv_ev'])) > 10e-5:
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[1].set_ylim([0,6])
    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    axes[0].legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(1, 0.9), facecolor='white')
    axes[1].legend(handles_2, labels_2, loc='upper right', facecolor='white')
    if figname:
        plt.savefig(figname)
    
def plot_results(df):
    
    dataset = df.copy()
    
    if 'episode' not in dataset.columns:
        episodes = []
        count = 0
        for row in range(0,len(dataset)-1):
            episodes.append(count)
            if dataset['avail'][row+1] > dataset['avail'][row]:
                count += 1
        episodes.append(episodes[-1])
        dataset['episode'] = episodes
    if 'avail' in dataset.columns:
        dataset = dataset[dataset['avail']>0]
        dataset = dataset.drop('avail',axis=1)
    
    # transform data
    dataset['pv'] = dataset['pv']/1000 # to kW
    dataset['load'] = dataset['load']/1000 # to kW
    dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
    dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
    dataset['soc'] = dataset['soc']*100 # to %
    
    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,7))
    plt.title(df.index[0])
    x = range(0,len(dataset))
    
    plt.xlabel('Hours')
    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,max(dataset['pv'].max(),dataset['load'].max())+3])
    plt.xticks(np.arange(min(x), max(x)+1, step=12))
    
#    ax1 = axes[0].twinx() # ax for plotting EV availability
#    ax1.set_yticks([0,1])
#    ax1.set_ylabel('EV availability', color='black')
    
    ax2 = axes[1].twinx() # ax for plotting EV SOC evolution
    ax2.set_ylabel('EV state of charge [%]', color='black')
    
#    ax1.grid(False)
    ax2.grid(False)
    
    # episodes annotation
    ypos = math.floor(axes[0].get_ylim()[1])-0.5
    xpos = np.insert(dataset.groupby('episode').count()['pv'].values,0,0)
    for row in range(0,len(xpos)-1):
        axes[0].annotate(s='', xy=(xpos[:row+2].sum()-1,ypos), xytext=(xpos[:row+1].sum(),ypos), arrowprops=dict(arrowstyle='<->',color='black'))
        axes[0].annotate(s='Episode '+str(row+1),xy=(((xpos[:row+2].sum()-1+xpos[:row+1].sum())/2), ypos+0.2), fontsize=12.0, ha='center')
    
    # top subplot
    axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.3)
    axes[0].plot(x, dataset['load'], color=palette[3], label='Building load')
    
    axes[0].fill_between(x, dataset['pv'], color=palette[0], alpha=0.3)
    axes[0].plot(x, dataset['pv'], color=palette[0], label='PV production forecast')
    
#    sns.scatterplot(x, avail, color=palette[7], edgecolor=palette[7],
#                    ax=ax1, label='EV availability', legend=False)
    
    # bottom subplot
    ax2.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    
    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    axes[0].legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(1, 0.9), facecolor='white')
    axes[1].legend(handles_2, labels_2, loc='upper left', facecolor='white')
def plot_boxplot_marina(df, capacity, name):
    
    dataset = df.copy()
    
    if 'episode' not in dataset.columns:
        episodes = []
        count = 0
        for row in range(0,len(dataset)-1):
            episodes.append(count)
            if dataset['avail'][row+1] > dataset['avail'][row]:
                count += 1
        episodes.append(episodes[-1])
        dataset['episode'] = episodes
    if 'avail' in dataset.columns:
        dataset = dataset[dataset['avail']>0]
        dataset = dataset.drop('avail',axis=1)
    
    # calculate pv self-consumption per episode
    grouped_data = dataset.groupby(dataset['episode']).sum()
    soc_start = dataset.groupby(dataset['episode']).min()['soc'].values
    soc_end = dataset.groupby(dataset['episode']).max()['soc'].values
    charge = (1 - soc_start)*capacity
    pv_self_cons = []
    pv_opt_self_cons = []
    for row in range(0,len(grouped_data)):
        pv_self_cons.append((grouped_data['pv_load'][row]+grouped_data['pv_ev'][row])/
                            grouped_data['pv'][row])
        pv_opt_self_cons.append((grouped_data['pv_load'][row]+grouped_data['pv_ev'][row])/
                                (grouped_data['pv_load'][row]+min(grouped_data['pv'][row]-grouped_data['pv_load'][row],charge[0])))
        
    pv_self_cons = [x if x <= 1.0 else 1.0 for x in pv_self_cons]
    pv_opt_self_cons = [x if x <= 1.0 else 1.0 for x in pv_opt_self_cons]
   
    # plot
    plt.figure(figsize=(10,7))
    plt.title(name)
    sns.boxplot(x=['PV self-consumption','Optimal PV self-consumption','SOC at departure'],
                y=[pv_self_cons,pv_opt_self_cons,soc_end])
    
def plot_training_results(pv_self_cons, soc_depart, name):
    
    df = pd.DataFrame(data={'pv':pv_self_cons,
                            'soc':soc_depart})
    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,7))
    plt.xlabel('Number of episodes')
    plt.title(name)
    axes[0].set_ylabel('Optimal PV self-consumption')
    axes[1].set_ylabel('SOC at departure')
    
    # rolling values
    roll_mean = df.rolling(10).mean()
    
    # upper plot
    axes[0].plot(df.index, roll_mean['pv'], color=palette[3])
    axes[0].plot(df.index, df['pv'], color=palette[0], alpha=0.5)
    
    # bottom plot    
    axes[1].plot(df.index, roll_mean['soc'], color=palette[3])
    axes[1].plot(df.index, df['soc'], color=palette[0], alpha=0.5)


def plot_dropout_results(data_full,pred_variable, predictions, 
                         predictions_dropout, n_hour_future,
                         plot_histogram = False, plot_kde = False, cumulative = True,
                         plot_cumulative = False, plot_boxplot = False,
                         boxplot_dropouts = []):
    ts = list(predictions.keys())
    dropouts = list(predictions.keys())
    for t_forecast in ts:
        t = t_forecast
        t_end = t + pd.Timedelta(hours = n_hour_future)
       
        for d in dropouts:
            dr = predictions_dropout[d][t]
            nd = predictions[t][pred_variable]
            medians = np.median(dr.T, axis = 1)
            dr_scaled = np.zeros(dr.shape)
            nd_scaled = np.zeros(nd.shape)
            test = list(data_full.loc[t: t_end][pred_variable])
            for i in range(dr.shape[1]):
                dr_scaled[:,i] = dr[:,i] - medians[i]
                nd_scaled[i] = nd[i] - test[i]

    
    dropout_values = {d:[] for d in dropouts}
    for d in dropouts:
        nd_total = []
        dr_total = []
        for t_forecast in ts:
            t = t_forecast
            t_end = t + pd.Timedelta(hours = n_hour_future)
            dr = predictions_dropout[d][t]
            nd = predictions[t][pred_variable]
            medians = np.median(dr.T, axis = 1)
            dr_scaled = np.zeros(dr.shape)
            nd_scaled = np.zeros(nd.shape)
            test = list(data_full.loc[t: t_end][pred_variable])
            for i in range(dr.shape[1]):
                dr_scaled[:,i] = dr[:,i] - test[i]
                nd_scaled[i] = nd[i] - test[i]
            nd_total.append(nd_scaled)
            dr_total.append(dr_scaled)
        nd_flat = np.array(nd_total).reshape(-1,1)
        dr_flat = np.array(dr_total).reshape(-1,1)
        dropout_values[d] = dr.reshape(-1,1)
        if plot_histogram:
            sns.distplot(nd_flat, label = '(predicted - real)', kde = True)
            sns.distplot(dr_flat.reshape(-1,1), label = f'dropout: {d} - median')
            plt.title(f'Dropout: {d}')
            plt.legend()
            plt.show()
    

    if plot_kde:
        for d in dropouts:
            values = np.concatenate(dropout_values[d])
            sns.kdeplot(values, label = d, cumulative = cumulative, common_norm=False, common_grid=True)
        
        t_start = ts[0]
        t_end = ts[-1]
        actual_values = list(data_full.loc[t_start:t_end][pred_variable])
            
        sns.kdeplot(actual_values,linestyle = '--', cumulative = cumulative, common_norm=False, common_grid=True, label = 'actual')
        
        plt.title('Dropout Comparison')
        plt.grid()    
        plt.legend()
        plt.show()
    
    if plot_cumulative:
        actual_values = list(data_full[pred_variable])
        #actual_values = list(data[idx_start:idx_end+n_hour_future][pred_variable])
        N1 = len(actual_values)
        X1 = np.sort(actual_values)
        F1 = np.array(range(N1))/float(N1)
        
        
        
        for d in dropouts:
            values = np.concatenate(dropout_values[d])
            N2 = len(values)
            X2 = np.sort(values)
            F2 = np.array(range(N2))/float(N2)
            plt.plot(X2, F2, label = d)
            
        plt.plot(X1, F1,'--', color = 'black', label = 'actual')
        plt.title(f'Distribution of {pred_variable} values')
        plt.xlabel('Watts')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.show()
        
    if plot_boxplot:
        for d in boxplot_dropouts:
            for t in ts:
                t_end = t +  pd.Timedelat(hours = n_hour_future)
                actual = list(data_full.loc[t:t_end, pred_variable])
                predict_no_drop = predictions[t][pred_variable]
                pred = predictions_dropout[d][t]
                sns.boxplot(data = pred)
                plt.plot(actual, color = 'black', label = 'Actual')
                plt.plot(predict_no_drop, linestyle = '--', color = 'black', label = 'Prediction no dropout')
                plt.legend()
                plt.title(f'{t.day}.{t.month}, prediction at {t.hour}:00, dropout: {d}')
                plt.show()