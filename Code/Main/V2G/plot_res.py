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

def confidence_interval(realization, df, q, scale = 1):
    arr = np.array(df.values).astype('float64')
    med = np.median(arr, axis = 1)
    low = np.quantile(arr, 1-q, axis = 1)
    high = np.quantile(arr, q, axis = 1)
    
    med_list = [realization]
    low_list = [realization]
    high_list = [realization]
    
    med_list.extend([m*scale for m in med])
    low_list.extend([l*scale for l in low])
    high_list.extend([h*scale for h in high])

    return low_list, med_list, high_list


def plot_MPC_det(decisions, t_decision, results, pv_pred, load_pred, figname = None, img_path = None):

    decision = decisions[:t_decision]

    
    episode = decision['episode'][0]
    
    n = len(decision)

    
    results['pv'] = pv_pred*scale_PV
    results['load'] = load_pred
    dataset = decision.append(results[1:])
    
    episode_length = len(dataset)

    t_start = dataset.index[0]
    t_end = dataset.index[-1]

    
    dataset['pv'] = dataset['pv']/1000 # to kW
    dataset['load'] = dataset['load']/1000 # to kW
    dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
    dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
    dataset['ev_grid'] = -dataset['ev_grid']/1000 # to kW
    dataset['ev_load'] = -dataset['ev_load']/1000 # to kW

    dataset['soc'] = dataset['soc']*100


   
    time = list(dataset.index)

    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(16,9))
    plt.title(f'Decision at {t_decision}')
    x = range(len(dataset))

    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,16])

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

    ta = 0
    if 0 in list(dataset.avail):

        td = list(dataset.avail).index(0)
        arrowstyle='<->'
    else:
        td = len(dataset)-1
        arrowstyle='<-'

    axes[0].annotate(text='', xy=(ta,ypos), xytext=(td,ypos), arrowprops=dict(arrowstyle=arrowstyle,color='black'))
    axes[0].annotate(text='Episode '+str(int(episode)),xy=(((ta+td)/2), ypos+0.2), fontsize=12.0, ha='center')

    # top subplot
    
    axes[0].fill_between(x[:n], dataset['pv'][:n], color=palette[2], alpha=0.3, label='Actual PV')
    axes[0].plot(x[:n], dataset['pv'][:n], color=palette[2])
    
    axes[0].fill_between(x[:n], dataset['load'][:n], color=palette[3], alpha=0.3, label='Actual load')
    axes[0].plot(x[:n], dataset['load'][:n], color=palette[3])
    
    
    
    #axes[0].fill_between(x[n-1:], dataset['pv'][n-1:], color=palette[2], alpha=0.5)
    axes[0].plot(x[n-1:], dataset['pv'][n-1:],marker='o', color=palette[2], label='PV forecast')
    
    #axes[0].fill_between(x[n-1:], dataset['load'][n-1:], color=palette[3], alpha=0.5)
    axes[0].plot(x[n-1:], dataset['load'][n-1:],marker='o', color=palette[3], label='load forecast')

    axes[0].axvline(x=n-1, color = 'black')

    # bottom subplot

    ax2.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
    
    #ax2.plot(x[:n], dataset['soc'][:n], color=palette[7], marker='.', label='EV state of charge')
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[1].bar(x, dataset['ev_grid'], color=palette[6], edgecolor=palette[6], label='EV supplied to Grid')
    axes[1].bar(x, dataset['ev_load'], color=palette[5], edgecolor=palette[5], label='EV supplied to Load')
    ax2.set_ylim([-100,110])
    ticks = np.arange(0,110,25)
    ax2.set_yticks(ticks)

    ax2.set_yticklabels([str(t) for t in ticks])
    axes[1].set_ylim([-4,4])

    axes[1].axvline(x=n-1, color = 'black')

    # plot legend
    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    fig.legend(handles_1, labels_1, loc='upper center',facecolor='white', ncol = 4)
    fig.legend(handles_2, labels_2, loc='lower center', bbox_to_anchor=(0.5,-0.05),facecolor='white', ncol = 3)
    
    if figname:
        fig.savefig(img_path + figname )
        plt.close(fig)
        
def plot_MPC_sto(decisions, t_decision, results, pv_pred, load_pred, soc_pred, figname = None, img_path = None):

    decision = decisions[:t_decision]

    episode = decision['episode'][0]
    
    idx_t = len(decision)-1

    results['pv'] = pv_pred*scale_PV
    results['load'] = load_pred
    dataset = decision.append(results[1:])
    
    episode_length = len(dataset)

    t_start = dataset.index[0]
    t_end = dataset.index[-1]

    
    dataset['pv'] = dataset['pv']/1000 # to kW
    dataset['load'] = dataset['load']/1000 # to kW
    dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
    dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
    dataset['ev_grid'] = -dataset['ev_grid']/1000 # to kW
    dataset['ev_load'] = -dataset['ev_load']/1000 # to kW

    dataset['soc'] = dataset['soc']*100

    time = list(dataset.index)

    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(16,9))
    plt.title(f'Decision at {t_decision}')
    x = range(len(dataset))

    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,16])

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
    ta = 0
    if 0 in list(dataset.avail):

        td = list(dataset.avail).index(0)
        arrowstyle='<->'
    else:
        td = len(dataset)-1
        arrowstyle='<-'

    axes[0].annotate(text='', xy=(ta,ypos), xytext=(td,ypos), arrowprops=dict(arrowstyle=arrowstyle,color='black'))
    axes[0].annotate(text='Episode '+str(int(episode)),xy=(((ta+td)/2), ypos+0.2), fontsize=12.0, ha='center')

    # top subplot
    
    axes[0].fill_between(x[:idx_t+1], dataset['pv'][:idx_t+1], color=palette[2], alpha=0.3, label='Actual PV')
    axes[0].plot(x[:idx_t+1], dataset['pv'][:idx_t+1], color=palette[2])
    
    axes[0].fill_between(x[:idx_t+1], dataset['load'][:idx_t+1], color=palette[3], alpha=0.3, label='Actual load')
    axes[0].plot(x[:idx_t+1], dataset['load'][:idx_t+1], color=palette[3])
    
    rest = len(dataset) - (idx_t+1)

    
    predictions = pv_pred[:rest]/1000
    
    low_PV, med_PV, high_PV = confidence_interval(dataset.loc[t_decision,'pv'], predictions, 0.8, scale = scale_PV)
    x_stop = idx_t + len(low_PV)
    axes[0].fill_between(x[idx_t:x_stop], low_PV, high_PV, color=palette[2], alpha=0.3, hatch = '//')
    axes[0].plot(x[idx_t:x_stop], med_PV, marker='o', color=palette[2], label='Forecast PV')
    
    axes[0].plot(x[idx_t:], dataset['load'][idx_t:],marker='o', color=palette[3], label='Forecast load')

    axes[0].axvline(x=idx_t, color = 'black')

    # bottom subplot

    ax2.plot(x[:idx_t+1], dataset['soc'][:idx_t+1], color=palette[7], marker='.', label='EV state of charge')
    SOC = soc_pred[1:rest]
    low_SOC, med_SOC, high_SOC = confidence_interval(dataset.loc[t_decision,'soc'],SOC, 0.95, scale = 100)
    x_stop = idx_t + len(low_SOC)
    ax2.plot(x[idx_t:x_stop], med_SOC, color=palette[7], marker='.')
    ax2.fill_between(x[idx_t:x_stop], low_SOC, high_SOC, color=palette[7], alpha = 0.3)
    ax2.set_ylim([-100,110])
    ticks = np.arange(0,110,25)
    ax2.set_yticks(ticks)

    ax2.set_yticklabels([str(t) for t in ticks])
    #ax2.plot(x[n:], dataset['soc'][:n], color=palette[7], marker='.', label='EV state of charge')
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[1].bar(x, dataset['ev_grid'], color=palette[6], edgecolor=palette[6], label='EV supplied to Grid')
    axes[1].bar(x, dataset['ev_load'], color=palette[5], edgecolor=palette[5], label='EV supplied to Load')
    axes[1].set_ylim([-3.7,5])
    # ax2.set_ylim([15,105])
    axes[1].axvline(x=idx_t, color = 'black')

    # plot legend
    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    fig.legend(handles_1, labels_1, loc='upper center',facecolor='white', ncol = 4)
    fig.legend(handles_2, labels_2, loc='lower center',  bbox_to_anchor=(0.5,-0.05),facecolor='white', ncol = 3)
    
    if figname:
        fig.savefig(img_path + figname )
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


    
def plot_results_deterministic(decisions, episodes,method, figname = None, img_path = None):
    
    dataset = decisions[decisions.episode.isin(episodes)]
    
    # transform data
    dataset['pv_real'] = dataset['pv']/1000 # to kW
    dataset['load'] = dataset['load']/1000 # to kW
    dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
    dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
    dataset['soc'] = dataset['soc']*100 # to %
    dataset['ev_grid'] = -dataset['ev_grid']/1000 # to kW
    dataset['ev_load'] = -dataset['ev_load']/1000 # to kW

    time = dataset.index
    # plot
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(16,9))

    x = range(0,len(dataset))

    axes[1].set_title(method)
    #plt.xlabel('Hours')
    axes[0].set_ylabel('Power [kW]', color='black')
    axes[1].set_ylabel('Power [kW]', color='black')
    axes[0].set_ylim([0,max(dataset['pv_real'].max(),dataset['load'].max())+3])
    #plt.xticks(np.arange(min(x), max(x)+1, step=4))
    ticks = np.arange(0, max(x)+1, step=int(len(dataset)/6))

    labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]

    plt.xticks(ticks = ticks, labels = labels)
    
    ax2 = axes[1].twinx() # ax for plotting EV SOC evolution
    ax2.set_ylabel('EV state of charge [%]', color='black')
    
#    ax1.grid(False)
    ax2.grid(False)
    # episodes annotation
    ypos = math.floor(axes[0].get_ylim()[1])-0.5
    td = [i for i in range(1, len(dataset)) if dataset.avail[i-1] > dataset.avail[i]]
    ta = [0]
    ta.extend([i for i in range(1, len(dataset)) if dataset.avail[i-1] < dataset.avail[i]])
    
    #episodes = np.unique(dataset.episode)
    for i in range(len(td)):
        x1 = ta[i]
        x2 = td[i]
    
        axes[0].annotate(s='', xy=(x1,ypos), xytext=(x2,ypos), arrowprops=dict(arrowstyle='<->',color='black'))
        axes[0].annotate(s='Episode '+str(int(episodes[i])),xy=(((x1+x2)/2), ypos+0.2), fontsize=12.0, ha='center')
    

    # top subplot
    axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.3, label='Building load')
    axes[0].plot(x, dataset['load'], color=palette[3])
    
    axes[0].fill_between(x, dataset['pv_real'], color=palette[2], alpha=0.3, label='PV production')
    axes[0].plot(x, dataset['pv_real'], color=palette[2])
    

    
    # bottom subplot
    ax2.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
    axes[1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
    axes[1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
    axes[1].bar(x, dataset['ev_grid'], color=palette[6], edgecolor=palette[6], label='EV supplied to Grid')
    axes[1].bar(x, dataset['ev_load'], color=palette[5], edgecolor=palette[5], label='EV supplied to Load')
    axes[1].set_ylim([-3.7,5])
    ax2.set_ylim([-100,110])
    ticks = np.arange(0,110,25)
    ax2.set_yticks(ticks)

    ax2.set_yticklabels([str(t) for t in ticks])

    # plot legend
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                            ax2.get_legend_handles_labels())]
    fig.legend(handles_1, labels_1, loc='upper center',facecolor='white', ncol = 4)
    fig.legend(handles_2, labels_2, loc='lower center',  bbox_to_anchor=(0.5,-0.05),facecolor='white', ncol = 3)
    
    if figname:
        fig.savefig(img_path + figname )
        plt.close(fig)
        
def plot_results_comparison(decisions, episodes, figname = None, img_path = None):
    
    fig, axes = plt.subplots(len(decisions)+1,1, sharex=True, figsize=(20,13))
    for k,m in enumerate(decisions.keys()):
        df = decisions[m]
        dataset = df[df.episode.isin(episodes)]
        
        # transform data
        dataset['pv_real'] = dataset['pv']/1000 # to kW
        dataset['load'] = dataset['load']/1000 # to kW
        dataset['pv_ev'] = dataset['pv_ev']/1000 # to kW
        dataset['grid_ev'] = dataset['grid_ev']/1000 # to kW
        dataset['ev_grid'] = -dataset['ev_grid']/1000 # to kW
        dataset['ev_load'] = -dataset['ev_load']/1000 # to kW
        dataset['soc'] = dataset['soc']*100 # to %
        time = dataset.index
        # plot
        
    
        x = range(0,len(dataset))
    
        
        if k == 0:
            #plt.xlabel('Hours')
            axes[0].set_ylabel('Power [kW]', color='black')
            
            axes[0].set_ylim([0,max(dataset['pv_real'].max(),dataset['load'].max())+3])
            #plt.xticks(np.arange(min(x), max(x)+1, step=4))
            ticks = np.arange(0, max(x)+1, step=int(len(dataset)/6))

            labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]

            plt.xticks(ticks = ticks, labels = labels)
            
            
            # episodes annotation
            ypos = math.floor(axes[0].get_ylim()[1])-0.5
            td = [i for i in range(1, len(dataset)) if dataset.avail[i-1] > dataset.avail[i]]
            ta = [0]
            ta.extend([i for i in range(1, len(dataset)) if dataset.avail[i-1] < dataset.avail[i]])
            
            #episodes = np.unique(dataset.episode)
            for i in range(len(td)):
                x1 = ta[i]
                x2 = td[i]
            
                axes[0].annotate(s='', xy=(x1,ypos), xytext=(x2,ypos), arrowprops=dict(arrowstyle='<->',color='black'))
                axes[0].annotate(s='Episode '+str(int(episodes[i])),xy=(((x1+x2)/2), ypos+0.2), fontsize=12.0, ha='center')
            
            
            # top subplot
            axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.3, label='Building load')
            axes[0].plot(x, dataset['load'], color=palette[3])
            
            axes[0].fill_between(x, dataset['pv_real'], color=palette[2], alpha=0.3, label='PV production')
            axes[0].plot(x, dataset['pv_real'], color=palette[2])
        
        ax2 = axes[k+1].twinx() # ax for plotting EV SOC evolution
        ax2.set_ylabel('SOC [%]', color='black')
        
    #    ax1.grid(False)
        ax2.grid(False)
        axes[k+1].set_ylabel('Power [kW]', color='black')
        axes[k+1].set_title(m)
        # bottom subplot
        ax2.plot(x, dataset['soc'], color=palette[7], marker='.', label='EV state of charge')
        axes[k+1].bar(x, dataset['pv_ev'], color=palette[8], edgecolor=palette[8], label='PV supplied to EV')
        axes[k+1].bar(x, dataset['grid_ev'], color=palette[4], edgecolor=palette[4], label='Grid supplied to EV')
        ax2.set_ylim([-100,110])
        ticks = np.arange(0,110,50)
        ax2.set_yticks(ticks)
    
        ax2.set_yticklabels([str(t) for t in ticks])
        
        axes[k+1].bar(x, dataset['ev_grid'], color=palette[6], edgecolor=palette[6], label='EV supplied to Grid')
        axes[k+1].bar(x, dataset['ev_load'], color=palette[5], edgecolor=palette[5], label='EV supplied to Load')
        axes[k+1].set_ylim([-3.7,5])
        if k == 0:
        # plot legend
            handles_1, labels_1 = axes[0].get_legend_handles_labels()
            handles_2, labels_2 = [(a + b) for a, b in zip(axes[1].get_legend_handles_labels(),
                                                    ax2.get_legend_handles_labels())]
            axes[0].legend(handles_1, labels_1, loc='upper center',facecolor='white',bbox_to_anchor=(0.5,1.8), ncol = 2)
            fig.legend(handles_2, labels_2, loc='lower center', facecolor='white', ncol = 3)
    
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