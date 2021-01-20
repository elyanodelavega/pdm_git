
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

# paramters
sns.set_style('whitegrid',{'legend.frameon':True})
palette = sns.color_palette()
plt.rcParams.update({'font.size': 14})
scale_PV = 10000/2835

        
def plot_results_comparison(title, decisions, episodes):
    ''' Plot EV management 
        Input:
            title: str, title for the plot
            decisions: dict of df, results of the methods or objectives
            episodes: [[int, int], [int, int int]], episodes to appear on the graph
            '''
    # number of row in the plots (+1 for load and PV)
    n_row = len(decisions)+1
    
    # figure
    fig, axes = plt.subplots(n_row,1, sharex=True, figsize=(20,13), dpi = 500)
    
    # title
    plt.suptitle(title, fontsize = 18)
    
    
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
       
        x = range(0,len(dataset))
    
        # top subplot
        if k == 0:
            
            axes[0].set_ylabel('Power [kW]', color='black')
            
            axes[0].set_ylim([0,max(dataset['pv_real'].max(),dataset['load'].max())+3])
            
            ticks = np.arange(0, max(x)+1, step=int(len(dataset)/6))

            labels = [str(time[ticks[i]].hour) + ':00' for i in range(len(ticks))]

            plt.xticks(ticks = ticks, labels = labels)
            
            
            # episodes annotation (when EV is plugged only)
            ypos = math.floor(axes[0].get_ylim()[1])-0.5
            td = [i for i in range(1, len(dataset)) if dataset.avail[i-1] > dataset.avail[i]]
            ta = [0]
            ta.extend([i for i in range(1, len(dataset)) if dataset.avail[i-1] < dataset.avail[i]])
            

            for i in range(len(td)):
                x1 = ta[i]
                x2 = td[i]
            
                axes[0].annotate(s='', xy=(x1,ypos), xytext=(x2,ypos), arrowprops=dict(arrowstyle='<->',color='black'))
                axes[0].annotate(text='EV plugged, Episode '+str(int(episodes[i])),xy=(((x1+x2)/2), ypos+0.2), fontsize=12.0, ha='center')
            
            
            # plot load and PV
            axes[0].fill_between(x, dataset['load'], color=palette[3], alpha=0.3, label='Building load')
            axes[0].plot(x, dataset['load'], color=palette[3])
            
            axes[0].fill_between(x, dataset['pv_real'], color=palette[2], alpha=0.3, label='PV production')
            axes[0].plot(x, dataset['pv_real'], color=palette[2])
        
        # ax for plotting EV SOC evolution
        ax2 = axes[k+1].twinx() 
        ax2.set_ylabel('SOC [%]', color='black')

        ax2.grid(False)
        axes[k+1].set_ylabel('Power [kW]', color='black')
        axes[k+1].set_title(m)
        
        # EV strategy of the algorithm
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
            axes[0].legend(handles_1, labels_1, loc='upper right',facecolor='white', ncol = 2)
            fig.legend(handles_2, labels_2, loc='lower center', facecolor='white', ncol = 3)

        
def plot_dropout_results(data_full,pred_variable, predictions, 
                         predictions_dropout, n_hour_future,
                         plot_cumulative = False, plot_boxplot = False,
                         boxplot_dropouts = []):
    
    ''' Plot function for th dropout evaluation
        Input: 
            data_full: df, actual values
            pred_variable: 'PV'
            predictions: dict, predictions deterministic at specific forecast time
            predictions_dropout: dict, predicitons stochastic  at specific forecast time of all dropouts
            n_hour_future: int, number of hours in the future
            plot_cumulative: bool
            plot_boxplot: bool
            boxplot_dropouts: list, dropouts results to be plotted'''
            
    # times of forecast
    ts = list(predictions.keys())
    
    # intialization
    dropouts = list(predictions_dropout.keys())
    sns.set_style("whitegrid")
    
    # last t_end
    last = list(predictions.keys())[-1] + pd.Timedelta(hours = n_hour_future)
    
    # reduce data set
    data = data_full[:last]
    
    # scale data to evaluate the dropouts performance
    
    dropout_values = {d:[] for d in dropouts}
    for d in dropouts:
        nd_total = []
        dr_total = []
        for t_forecast in ts:
            t = t_forecast
            t_end = t + pd.Timedelta(hours = n_hour_future)
            
            # stochastic predicition with dropout
            dr = predictions_dropout[d][t]
            
            # deterministic predicition
            nd = predictions[t]
            
            # scale initialization
            dr_scaled = np.zeros(dr.shape)
            nd_scaled = np.zeros(nd.shape)
            
            test = list(data_full.loc[t: t_end][pred_variable])
            
            # substract the difference with actual data
            for i in range(dr.shape[1]):
                dr_scaled[:,i] = dr[:,i] - test[i]
                nd_scaled[i] = nd[i] - test[i]
            nd_total.append(nd_scaled)
            dr_total.append(dr_scaled)
        
        # flatten to evaluate CDF
        dropout_values[d] = dr.reshape(-1,1)
    
    if plot_cumulative:
        
        # evaluate empirical CDF
        actual_values = list(data[pred_variable])
        N1 = len(actual_values)

        X1 = np.sort(actual_values)
        F1 = np.array(range(N1))/float(N1)
        
        
        plt.figure(figsize = (9,5), dpi = 600)
        for d in dropouts:
            values = np.concatenate(dropout_values[d])
            N2 = len(values)
            X2 = np.sort(values)
            F2 = np.array(range(N2))/float(N2)
            plt.plot(X2, F2, label = f'dropout = {d}')
            
        plt.plot(X1, F1,'--', color = 'black', label = 'actual')
        plt.title(f'Distribution of {pred_variable} values')
        plt.xlabel('Watts')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # plot MSE difference
        plt.figure(figsize = (9,5), dpi = 600)
        mse = []
        
        # MSE based on quantiles
        X1_quant = np.quantile(X1/float(N1), np.arange(0,1,0.05))
        for d in dropouts:
            values = np.concatenate(dropout_values[d])

            X2 = np.sort(values)
            X2_quant = np.quantile(X2/float(N2), np.arange(0,1,0.05))
            

            mse.append(np.mean(np.square(X1_quant - X2_quant)))


        plt.plot(dropouts, mse, marker = 'o')
        plt.title('MSE evaluation of dropouts from cdf')
        plt.ylabel('MSE')
        plt.xlabel('Dropout')
        plt.xticks(dropouts)

        plt.grid(True)
        plt.show()
        
    # plot distribution prediction with boxplot over 23hours time window
    if plot_boxplot:
        for d in boxplot_dropouts:

            for t in ts:
                plt.figure(figsize = (9,6), dpi = 600)
                t_end = t +  pd.Timedelta(hours = n_hour_future)
                actual = list(data_full.loc[t:t_end, pred_variable])
                predict_no_drop = predictions[t]
                
                pred = predictions_dropout[d][t]
                pred_df = pd.DataFrame(pred)
                pred_df.rename(columns = {i: i+1 for i in range(23)}, inplace = True)

                sns.boxplot(data = pred_df)
                
                plt.plot(actual, color = 'black', label = 'Actual')
                plt.plot(predict_no_drop, linestyle = '--', color = 'black', label = 'Prediction no dropout')
                plt.legend()
                plt.xlim(left = -1)
                ticks = np.arange(-1,24)
                labels = [t+1 for t in ticks]
                plt.xticks(ticks,labels)
                plt.ylabel('W')
                plt.xlabel('Hours ahead forecast')
                plt.title(f'{t.day}.{t.month}.{t.year}, prediction at {t.hour}:00, dropout: {d}')
                plt.show()