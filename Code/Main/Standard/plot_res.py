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
scale_PV = 10000/2835

def confidence_interval(df, q, scale = 1):
    arr = np.array(df.values).astype('float64')
    med = np.median(arr, axis = 1)
    low = np.quantile(arr, 1-q, axis = 1)
    high = np.quantile(arr, q, axis = 1)
    
    
    med = [m*scale for m in med]
    low = [l*scale for l in low]
    high = [h*scale for h in high]
    return low, med, high


        
        
def plot_MPC(decisions, results, SOC_distribution = None, predictions_distribution = None, figname = None, img_path = None):
    
    t_decision = decisions.index[-1]
    
    dataset = decisions.append(results[1:])
    
    n = min(len(decisions), 12)
    m = 24 - n
    
    t_start = t_decision - pd.Timedelta(hours = n)
    t_end = t_decision + pd.Timedelta(hours = m-1)
    dataset = dataset[t_start:t_end]
    dataset.fillna(0, inplace = True)
    
    positions = []
    if 'episode' not in dataset.columns:
        episodes = []
        count = 1
        row = 0
                    
        while row < len(dataset):
            if dataset['avail'][row] == 1:
                pos = row
                while dataset['avail'][row] == 1:
                    episodes.append(count)
                    row += 1
                    if row == len(dataset):
                        break
                positions.append((pos,row-1))
            else:
                count += 1
                while dataset['avail'][row] == 0:
                    episodes.append(0)
                    row += 1
                    if row == len(dataset):
                        break
        dataset['episode'] = episodes
    
    dataset['pv'] = dataset['pv']/1000 # to kW
    dataset['pv_forecast'] = dataset['pv_forecast']/1000 # to kW
    dataset['load'] = dataset['load']/1000 # to kW
    dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
    dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
    # dataset['pv_ev_real'] = dataset['pv_ev_real']/1000 # to kW
    # dataset['grid_ev_real'] = dataset['grid_ev_real']/1000 # to kW
    dataset['soc'] = dataset['soc']*100
    print(dataset.shape)

   
    time = list(dataset.index)

    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,7))
    plt.title(f'Decision at {t_decision}')
    x = range(len(dataset))

    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,11])

    ticks = np.arange(0, max(x)+1, step=6)
    ticks = [i for i in range(0, max(x) + 1, 6)]
    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
    plt.xticks(ticks = ticks, labels = labels)

    ax2 = axes[1].twinx() # ax for plotting EV SOC evolution
    ax2.set_ylabel('EV SoC [%]', color='black')
    ax2.grid(False)
     # episodes annotation
    #num_episodes = max(dataset['episode'])
    ypos = math.floor(axes[0].get_ylim()[1])-0.5
    for e in range(len(positions)):
        xpos = positions[e]
        axes[0].annotate(s='', xy=(xpos[0],ypos), xytext=(xpos[1],ypos), arrowprops=dict(arrowstyle='<->',color='black'))
        axes[0].annotate(s='Episode '+str(e+1),xy=((xpos[0] + xpos[1])/2, ypos+0.2), fontsize=12.0, ha='center')

    
    # top subplot
    axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.3, label='Building load')
    axes[0].plot(x, dataset['load'], color=palette[3])
    
    
    axes[0].fill_between(x[:n], dataset['pv'][:n], color=palette[0], alpha=0.3)
    axes[0].plot(x[:n], dataset['pv'][:n],marker='o', color=palette[0], label='Actual PV')
    
    if predictions_distribution is not None:
        predictions = predictions_distribution/1000
        low_PV, med_PV, high_PV = confidence_interval(predictions.head(m), 0.8, scale = scale_PV)
        axes[0].fill_between(x[n-1:n+m-1], low_PV[:m], high_PV[:m], color=palette[2], alpha=0.3, hatch = '//', label='Expected PV')
        axes[0].plot(x[n-1:n+m-1], med_PV[:m],'--', color=palette[2])
    else:
        axes[0].fill_between(x[n-1:n+m-1], dataset.loc[t_decision:t_end,'pv_forecast'], color=palette[2], alpha=0.5)
        axes[0].plot(x[n-1:n+m-1], dataset.loc[t_decision:t_end,'pv_forecast'], color=palette[2], label='PV forecast')

    axes[0].axvline(x=n-1, color = 'black')

    # bottom subplot
    
    if SOC_distribution is not None:
        SOC = SOC_distribution.head(m)
        low_SOC, med_SOC, high_SOC = confidence_interval(SOC, 0.95, scale = 100)
        ax2.plot(x[n-1:n+m-1], med_SOC, color=palette[7], marker='.')
        ax2.fill_between(x[n-1:n+m-1], low_SOC, high_SOC, color=palette[7], alpha = 0.3)
        ax2.set_ylim([0,110])
    else:
        ax2.plot(x[n-1:n+m-1], dataset['soc'][n-1:n+m-1], color=palette[7], marker='.', label='EV state of charge')
    
    ax2.plot(x[:n], dataset['soc'][:n], color=palette[7], marker='.', label='EV state of charge')
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[1].set_ylim([0,6])

    axes[1].axvline(x=n-1, color = 'black')

    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    axes[0].legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(1, 0.9), facecolor='white', ncol = 3)
    axes[1].legend(handles_2, labels_2, loc='upper right', facecolor='white')
    
    if figname:
        fig.savefig(img_path + figname)
        plt.close(fig)
        
def plot_results_day_ahead(df,predictions = None, figname = None, img_path = None):
    
    dataset = df.copy()
    t_decision = dataset.index[0]
    
    positions = []
    if 'episode' not in dataset.columns:
        episodes = []
        count = 1
        row = 0
                    
        while row < len(dataset):
            if dataset['avail'][row] == 1:
                pos = row
                while dataset['avail'][row] == 1:
                    episodes.append(count)
                    row += 1
                    if row == len(dataset):
                        break
                positions.append((pos,row-1))
            else:
                count += 1
                while dataset['avail'][row] == 0:
                    episodes.append(0)
                    row += 1
                    if row == len(dataset):
                        break
        dataset['episode'] = episodes

    # transform data
    dataset['pv_real'] = dataset['pv_real']/1000 # to kW
    dataset['pv_forecast'] = dataset['pv_forecast']/1000 # to kW
    dataset['load'] = dataset['load']/1000 # to kW
    dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
    dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
    dataset['pv_ev_real'] = dataset['pv_ev_real']/1000 # to kW
    dataset['grid_ev_real'] = dataset['grid_ev_real']/1000 # to kW
    dataset['soc'] = dataset['soc']*100 # to %
    time = dataset.index
    
    # plot
    fig, axes = plt.subplots(3,1, sharex=True, figsize=(10,7))
    axes[0].set_title(time[0])
    x = range(0,len(dataset))

    
    #plt.xlabel('Hours')
    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[2].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,max(dataset['pv_real'].max(),dataset['load'].max())+3])
    #plt.xticks(np.arange(min(x), max(x)+1, step=4))
    
    
    ax2 = axes[1].twinx() # ax for plotting EV SOC evolution
    ax2.set_ylabel('EV SOC [%]', color='black')
    
#    ax1.grid(False)
    ax2.grid(False)
    
    # episodes annotation
    num_episodes = max(dataset['episode'])
    ypos = math.floor(axes[0].get_ylim()[1])-0.5
    for e in range(len(positions)):
        xpos = positions[e]
        axes[0].annotate(s='', xy=(xpos[0],ypos), xytext=(xpos[1],ypos), arrowprops=dict(arrowstyle='<->',color='black'))
        axes[0].annotate(s='Episode '+str(e+1),xy=((xpos[0] + xpos[1])/2, ypos+0.2), fontsize=12.0, ha='center')
    
    # top subplot
    axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.3)
    axes[0].plot(x, dataset['load'], color=palette[3], label='Building load')
    
    axes[0].fill_between(x, dataset['pv_real'], color=palette[0], alpha=0.3)
    axes[0].plot(x, dataset['pv_real'], color=palette[0], label='PV real')
    
    if predictions is not None:
        predictions_array = predictions/1000
        low_PV, med_PV, high_PV = confidence_interval(predictions_array, 0.8, scale= scale_PV)
        axes[0].fill_between(x[1:], low_PV, high_PV, color=palette[2], alpha=0.5, hatch = '//', label='PV_forecast')
        axes[0].plot(x[1:], med_PV,'--', color=palette[2])
    else:
        axes[0].fill_between(x, dataset['pv_forecast'], color=palette[2], alpha=0.5)
        axes[0].plot(x, dataset['pv_forecast'], color=palette[2], label='PV forecast')
        
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(1, 0.9), facecolor='white', fontsize="small")

    ticks = np.arange(0, max(x)+1, step=6)
    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
    
    plt.xticks(ticks = ticks, labels = labels)
    
    # middle subplot
    ax2.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
    ax2.set_ylim([0,110])
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[1].set_title('Expected')
    # plot legend
    
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    
    axes[1].legend(handles_2, labels_2, loc='upper right', facecolor='white', fontsize="small")
    
    ax3 = axes[2].twinx() # ax for plotting EV SOC evolution
    ax3.set_ylabel('EV SOC [%]', color='black')
    
#    ax1.grid(False)
    ax3.grid(False)
    ax3.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
    ax3.set_ylim([0,110])
    
    axes[2].bar(x, dataset['pv_ev_real'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[2].bar(x, dataset['grid_ev_real'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[2].set_title('Actual')
    # plot legend
    
    handles_3, labels_3 = [(a + b) for a, b in zip(axes[2].get_legend_handles_labels(),
                                            ax3.get_legend_handles_labels())]
    
    axes[2].legend(handles_3, labels_3, loc='upper right', facecolor='white', fontsize="small")
    
    if figname:
        fig.savefig(img_path + figname )
        plt.close(fig)


    
def plot_results_deterministic(df, figname = None, img_path = None):
    
    dataset = df.copy()
    
    positions = []
    if 'episode' not in dataset.columns:
        episodes = []
        count = 1
        row = 0
                    
        while row < len(dataset):
            if dataset['avail'][row] == 1:
                pos = row
                while dataset['avail'][row] == 1:
                    episodes.append(count)
                    row += 1
                    if row == len(dataset):
                        break
                positions.append((pos,row-1))
            else:
                count += 1
                while dataset['avail'][row] == 0:
                    episodes.append(0)
                    row += 1
                    if row == len(dataset):
                        break
        dataset['episode'] = episodes
    # if 'avail' in dataset.columns:
    #     dataset = dataset[dataset['avail']>0]
    #     dataset = dataset.drop('avail',axis=1)
    
    # transform data
    dataset['pv_real'] = dataset['pv_real']/1000 # to kW
    dataset['load'] = dataset['load']/1000 # to kW
    dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
    dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
    dataset['soc'] = dataset['soc']*100 # to %
    time = dataset.index
    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,7))
    plt.title(time[0])
    x = range(0,len(dataset))
    
    #plt.xlabel('Hours')
    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,max(dataset['pv_real'].max(),dataset['load'].max())+3])
    #plt.xticks(np.arange(min(x), max(x)+1, step=4))
    ticks = np.arange(0, max(x)+1, step=6)
    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]
    
    plt.xticks(ticks = ticks, labels = labels)
    
    ax2 = axes[1].twinx() # ax for plotting EV SOC evolution
    ax2.set_ylabel('EV state of charge [%]', color='black')
    
#    ax1.grid(False)
    ax2.grid(False)
    
    # episodes annotation
    ypos = math.floor(axes[0].get_ylim()[1])-0.5
    for e in range(len(positions)):
        xpos = positions[e]
        axes[0].annotate(s='', xy=(xpos[0],ypos), xytext=(xpos[1],ypos), arrowprops=dict(arrowstyle='<->',color='black'))
        axes[0].annotate(s='Episode '+str(e+1),xy=((xpos[0] + xpos[1])/2, ypos+0.2), fontsize=12.0, ha='center')
    
    # top subplot
    axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.3)
    axes[0].plot(x, dataset['load'], color=palette[3], label='Building load')
    
    axes[0].fill_between(x, dataset['pv_real'], color=palette[0], alpha=0.3)
    axes[0].plot(x, dataset['pv_real'], color=palette[0], label='PV production')
    

    
    # bottom subplot
    ax2.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    
    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    axes[0].legend(handles_1, labels_1, loc='upper right', bbox_to_anchor=(1, 0.9), facecolor='white')
    axes[1].legend(handles_2, labels_2, loc='upper right', facecolor='white')
    
    if figname:
        fig.savefig(img_path + figname )
        plt.close(fig)

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