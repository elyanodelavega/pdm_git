# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:03:49 2020

@author: Yann
"""
''' Class to plot metrics'''
## IMPORTS

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import pandas as pd
import seaborn as sns
import numpy as np

from df_prepare import  prices_romande_energie



sns.set_style('whitegrid')

dpi = 400

class Metrics_groups:
    def __init__(self, res_folder_path):
        ''' res_folder_path: str, folder path to results'''
        
        self.res_folder_path = res_folder_path
        self.objectives = ['cost', 'pv', 'peak']
        self.methods_short = ['opti', 'mpc_d', 'mpc_s', 'mpc_s_cvar']
        
        # color for each algorithm
        self.palette = sns.color_palette(n_colors = len(self.objectives)*len(self.methods_short) + 4)
        
        self.color_algorithms = {}
        c = 0
        for  o in self.objectives:
            for m in self.methods_short:
                
                name = f'v2g_{m}_{o}'
                self.color_algorithms[name] = self.palette[c]
                c += 1
        
        # color for each model
        self.palette_models = sns.color_palette(n_colors = 10)
        all_models = self.objectives.copy()
        all_models.extend(self.methods_short)
        
        self.color_models = {all_models[i]: self.palette_models[i] for i in range(len(all_models))}
        
        # methods name complete
        self.methods = ['Perfect Foresight  \ncost', 
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
        
        
        # csv names
        self.names = list(self.color_algorithms.keys())
        
        # plotting names
        self.metrics = ['PVSC', 'soc_dep', 'cost', 'APR']
        self.metrics_title = ['PVSC','SOC at departure', 'Median cost', 'APR' ]
        self.metrics_unit = ['%','%', 'CHF','%']
        self.metrics_props = {self.metrics[i]: {'title': self.metrics_title[i],'unit': self.metrics_unit[i]} for i in range(len(self.metrics))}
        
        self.labels_radar = {'PVSC': 'PVSC',
                  'cost': 'CP',
                  'APR': 'APR',
                  'soc_dep': 'SOC '}
        
        # limits for boxplot
        self.metrics_ylim =  {'PVSC': [0,105],
                          'cost': None,
                          'APR': None,
                          'soc_dep': [0,105]}

    def import_decisions(self):
        ''' import the results'''
        
        # import
        csv_code = '.csv'
        self.decisions = {n: pd.read_csv(self.res_folder_path+'results_'+n+csv_code, index_col = 0) for n in self.names}
        
        # assert uneven dataset
        self.l_min = min([self.decisions[n].shape[0] for n in self.names ])
        
        # assess index into datetime
        for n in self.names:
            df = self.decisions[n]
            df = df[:self.l_min]
            new_index = pd.to_datetime(df.index, dayfirst = True)
            df.index = new_index
            self.decisions[n] = df
        
        # map csv name and full method name
        self.algorithms = {self.names[i]: self.methods[i] for i in range(len(self.names))}


        # split into group
        group_code = {'cost': [], 'pv': [], 'peak': [],
                  'opti':[], 'mpc_d': [], 'mpc_s': [], 'mpc_s_cvar': []}
        
        group_names = ['Objective: cost', 'Objective: PVSC','Objective: APR',
                       'Perfect Foresight','MPC deterministic',
                       'MPC stochastic, Expected', 'MPC stochastic, CVaR']
        
        # map models csv code and groups names
        self.models_codes = list(self.color_models.keys())
        self.color_groups = {group_names[i]: self.color_models[self.models_codes[i]] for i in range(len(group_names))}
        self.groups = {}
        for n in self.names:
            
            for i,g in enumerate(group_code.keys()):
                
                if g in n:
                    
                    group_code[g].append(n)
                    
                self.groups[group_names[i]] = group_code[g]
        
        # manipulation to remove mpc_s_cvar from mpc_s group
        algos_mpc_s =  list(self.groups['MPC stochastic, Expected'])
        for a in algos_mpc_s:    
            if 'cvar' in a:
                algos_mpc_s.remove(a)
        
        self.groups['MPC stochastic, Expected'] = algos_mpc_s
        
        # give each algo its objective and its method
        self.algos_specs = {n: {'Objective': None, 'Method': None} for n in self.names}
        
        for g_name in self.groups.keys():
        
            algos = self.groups[g_name]
            
            if 'Objective' in g_name:
                
                for a in algos:
                    self.algos_specs[a]['Objective'] = g_name
                    
            else:
        
                for a in algos:
                    self.algos_specs[a]['Method'] = g_name

        # groups without perfect foresight
        self.groups_mpc = {}
        
        self.groups_mpc['Objective: cost'] = ['v2g_'+m+'_cost' for m in self.methods_short[1:]]
        self.groups_mpc['Objective: PVSC'] = ['v2g_'+m+'_pv' for m in self.methods_short[1:]]
        self.groups_mpc['Objective: APR'] = ['v2g_'+m+'_peak' for m in self.methods_short[1:]]
        
        
        # add metrics to "stats", where each dict entry is a weight and its corresponding metrics
        self.stats = {n: self.compute_metrics(self.decisions[n]) for n in self.names}
        
        

        # dict with each entry is a metric, and the values for each weight
        self.stats_df = {m: pd.DataFrame(data = {n: list(self.stats[n].loc[:,m] )
                             for n in self.names}) for m in self.metrics}
        
        
        # import execution time from each algo
        import pickle
        self.time_algo = {}
        for i, o in enumerate(self.objectives):
            for j, m in enumerate(self.methods_short):
                name = f'{m}_{o}'
                file_name = f'time_v2g_{name}'
                algo = f'v2g_{name}'
                file_inter = open(self.res_folder_path+file_name+'.pickle', 'rb')
                self.time_algo[algo] = pickle.load(file_inter)
        
                file_inter.close()
                
        # assess all same length
        
        self.t_min = min([len(self.time_algo[algo]) for algo in self.time_algo])
        
        self.time_algo = {algo: self.time_algo[algo][:self.t_min] for algo in self.time_algo}
        
    def get_obj(self, algo):
        
        # get objective from algo (csv_code)
        if 'cost' in algo:
            return 'cost'
        elif 'pv' in algo:
            return 'PVSC'
        else:
            return 'APR'

    def compute_metrics(self, decisions):
        ''' Function to return metrics per episode of the results (decision)
            Input: 
                decisions: df, results of an algorithm
            Output:
                df: metrics per episode'''
    
        # copy results
        data = decisions.copy()
        
        # add hourly prices
        data = prices_romande_energie(data)
    
        # compute cost
        data['cost_buy'] = (data['grid_load'] + data['grid_ev'])*data['buy']
        
        data['cost_sell'] = (data['pv_grid'] + data['ev_grid'])*data['buy']
        
        data['cost'] = data['cost_buy'] - data['cost_sell']
        
        # compute power drawn from grid
        data['grid_bought'] = (data['grid_load'] + data['grid_ev'])/1000
        
        d = data.copy()
        
        # groupby
        df = d.groupby('episode').sum()
        
        df_mean = data.groupby('episode').mean()
        
        df_max = data.groupby('episode').max()
        
        df_med = d.groupby('episode').median()
        
        # soc at departure for each episode
    
        df.drop(['soc'], axis = 1, inplace = True)
        
        soc_dep = [data.soc[i] *100 for i in range(1, len(data)) if data.avail[i] < data.avail[i-1]]
        
        # pv self consumption
            
        PVSC = 100*(df['pv_load'] + df['pv_ev'])/df['pv']
        
        # APR
        
        APR = 100*df_mean.grid_bought / df_max.grid_bought
    
        df['soc_dep'] = soc_dep
        
        df['PVSC'] = PVSC
    
        df['cost'] = df_med['cost']
        
        df['APR'] = APR
        
        return df


    def compute_cost_perforance(self, stats_cost, quantile_low = 0.1, quantile_high = 0.9):
        ''' compute the cost performance
            Input:
                stats_cost: df, stats_df['cost']
            Output:
                quantile low, median, and quantile high of stats_cost, with respect to best cost (see. report)'''
        
        df = stats_cost.copy()
        
        # median
        df_med = df.median()
        
        # find best median cost algo
        best_algo = df_med[df_med == df_med.min()].index.values[0]
        
        # compute cost performance of each algo
        new_df = pd.DataFrame(columns = df.columns)
        for col in new_df.columns:
            values = 100*df[best_algo]/df[col]
            new_df[col] = values
            
        low = new_df.quantile(quantile_low)
        med = new_df.median()
        high = new_df.quantile(quantile_high)
        
        return low,med,high
    
    def adjust(self, s):
        # put str s on multiple line for plot (if name is too long)
        if ' ' in s:
            s = s.replace(' ', '\n')
        return s    


    # box plot metrics
    def ax_boxplot_metrics(self, ax, metric, g, algos, ylim = None, legend = True):
        # return ax for boxplot of a particular metric, for the algos in group g        
        
        # if the group is an objectve, we compare the methods and vice-versa
        if 'Objective' in g:
            s = 'Method'
        else:
            s = 'Objective'
    
        m = metric
        
        # df of the metric m
        s_df = self.stats_df[m]
        
        # new df with the specific algos
        new_df = {}
        for n in algos:
    
            values = list(s_df[n].values)
            new_df[self.algos_specs[n][s]] = values
        
        df = pd.DataFrame(new_df)
        
        # plot
        sns.boxplot(data = df, ax = ax, palette = [self.color_groups[self.algos_specs[a][s]] for a in algos])

        ax.set_title(self.metrics_props[m]['title'])
        ax.set_ylabel(self.metrics_props[m]['unit'])
        if legend == False:
            ax.set(xticklabels=[])
        else:
            ax.set(xticklabels = [self.algos_specs[a][s] for a in algos])
        ax.grid()
        if ylim is not None:
            ax.set_ylim(ylim)
            
        return ax

    # RADAR
    
    def ax_radar(self, ax, g, algos, x_legend = 0.5, y_legend = -0.5, start = np.pi/4, legend = True):
        
        # radar plot for specific algos in group g
        # start: first angle
        
        if 'Objective' in g:
            s = 'Method'
        else:
            s = 'Objective'
            
        group_df_low = pd.DataFrame(index = algos)
        group_df_high = pd.DataFrame(index = algos)
        group_df_med = pd.DataFrame(index = algos)
    
        # define angles for radar plot
        angles=np.linspace(start, 2*np.pi + start, len(self.metrics), endpoint=False)
        angles=np.concatenate((angles,[angles[0]]))
        
        # compute values for each metrics, with quantiles
        for m in self.metrics:
    
            if m == 'cost':
                low, med, high = self.compute_cost_perforance(self.stats_df[m])
                group_df_low[self.labels_radar[m]] = low[algos]
                group_df_high[self.labels_radar[m]] = high[algos]
                group_df_med[self.labels_radar[m]] = med[algos]
            else:
                group_df_low[self.labels_radar[m]] = self.stats_df[m].quantile(0.1)[algos]
                group_df_high[self.labels_radar[m]] = self.stats_df[m].quantile(0.9)[algos]
                group_df_med[self.labels_radar[m]] = self.stats_df[m].median()[algos]
            
            # plot
        for a in algos:
            vals_low =group_df_low.loc[a,:]
            errors_low = group_df_med.loc[a,:] - vals_low
            
            errors_low=np.concatenate((errors_low,[errors_low[0]]))
            vals_high =group_df_high.loc[a,:]
            errors_high =  vals_high - group_df_med.loc[a,:]
            
            errors_high=np.concatenate((errors_high,[errors_high[0]]))
            vals_med =group_df_med.loc[a,:]
            
            vals_med=np.concatenate((vals_med,[vals_med[0]]))
    
            c = self.color_groups[self.algos_specs[a][s]]
    
            ax.plot(angles, vals_med, 'o-',  label = self.algos_specs[a][s], color = c, linewidth = 2)
            ax.errorbar(angles, vals_med, yerr = errors_low,  uplims=True,color = c)
            ax.errorbar(angles, vals_med, yerr = errors_high,  lolims=True,color = c)
            
            # angles legend
            ax.set_thetagrids(angles[:-1] * 180/np.pi, group_df_low.columns, fontsize = 16)
            
        
        ax.grid(True)
        if legend == True:
            ax.legend(loc = 'lower center',bbox_to_anchor=(x_legend,y_legend), ncol = 1, fontsize = 15)
    
        ax.set_xlabel('%', fontsize = 15)
        ax.set_ylim([0,105])
        
        return ax

    def plot_groups(self):
        
        # full plot (radar + boxplot) for each group
        for g in list(self.groups.keys()):
            if 'Objective' in g:
                s = 'Method'
            else:
                s = 'Objective'
            
            algos = self.groups[g]
            
            
            fig = plt.figure(figsize=(13, 7), dpi = dpi)
            plt.suptitle(g, fontsize = 18)
            
            # radar plot
            ax1 = fig.add_subplot(1,2,1, polar = True)
            self.ax_radar(ax1, g, algos,  legend = False)
            
            # boxplot metrics
            ax3= fig.add_subplot(4,2,2)
            ax4= fig.add_subplot(4,2,4)
            ax5= fig.add_subplot(4,2,6)
            ax6= fig.add_subplot(4,2,8)
            
            ax_box = [ax3, ax4, ax5, ax6]
            
            for i in range(len(ax_box)):
                m = self.metrics[i]
                ax = ax_box[i]
    
                self.ax_boxplot_metrics(ax, m, g, algos, ylim = self.metrics_ylim[m], legend = False)
            
            ncol = 3
            if len(algos) > 3:
                ncol = int(len(algos)/2)
            patches1 = [mpatches.Patch(color=self.color_groups[self.algos_specs[n][s]], label=self.algos_specs[n][s]) for n in algos]
            ax1.legend(handles=patches1, loc='upper center',ncol=ncol,bbox_to_anchor = (0.5,1.15), fontsize = 12) 
            
            
            fig.show()


    def ax_time(self, ax, g, legend = False, log_scale = False):
        # return ax to plot time of execution in group g
        
        df_time = pd.DataFrame.from_dict(self.time_algo)
        algos = self.groups[g].copy()
        
        df = df_time[algos].copy()
        
        if 'Objective' in g:
            s = 'Method'
        else:
            s = 'Objective'
        
        # rename columns for plot
        df.rename(columns = {a: self.algos_specs[a][s] for a in algos}, inplace = True)
        
        # plot
        sns.boxplot(data = df, ax = ax, palette = [self.color_groups[self.algos_specs[a][s]] for a in algos])

        if legend == False:
            ax.set(xticklabels=[])
        else:
            ax.set(xticklabels = [self.algos_specs[a][s] for a in algos])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title(g, fontsize = 18)
        return ax

    def plot_time(self):
        # call ax_time to plot
        
        fig, axes = plt.subplots(3,1,figsize = (16,9), dpi = dpi)
        legend = False
        for i,g in enumerate(self.groups):
            
            if 'Objective' in g:
                if i == 2:
                    legend = True
                self.ax_time(axes[i], g, legend = legend)
        fig.show()

    def heatmap_obj(self):
        # heatmap of each objective
        
        # remove soc_dep
        subm =  [ 'cost','PVSC', 'APR']
        # plot
        fig, ax = plt.subplots(1,1,figsize = (16,9), dpi = dpi, sharey = True)
        
        new_df = pd.DataFrame()
        for k in range(len(subm)):
            # group name
            g = list(self.groups.keys())[k]
            
            # only objective
            if 'Method' in g:
                break
            algos = self.groups[g]
            obj = self.get_obj(algos[0])
            new_df[g] = np.concatenate(self.stats_df[obj][algos].values)
            
        # correlation matrix
        c = new_df.corr()
        plt.yticks( va="center", fontsize = 15)
        plt.xticks( va="center", fontsize = 16)
        sns.heatmap(c, annot=True, annot_kws={"size":20} , linewidths=.5)
    
    def plot_SOC(self):

        # differentiation between episode length into bins
        bins = list(np.arange(11,30,7))
        bins.append(31)
        
        # deciles
        x = [np.round(0.1*i,2) for i in range(11)]
        
        # plot parameters
        ticks = np.arange(0,11,2.5)
        labels = [str(int(t)*10)+'%' for t in ticks]
        labels[0] = labels[0] + '\n' + 'Arr.'
        labels[-1] = labels[-1] + '\n' + 'Dep.'

        for g in self.groups:
            
            algos = self.groups[g]
            if 'Objective' in g:
                s = 'Method'
        
                fig, axes = plt.subplots(len(algos)+2,len(bins)-1, sharex = True, figsize=(16,9), dpi = dpi)
                plt.suptitle(g, fontsize = 16)
                
                for i in range(len(bins)-1):
                    
                    #bin lower bound
                    lower_bound = int(bins[i])
                
                    # bin upper bound
                    upper_bound = int(bins[i+1])
                
                    for j, n in enumerate(algos):
                        
                        decision = self.decisions[n]
                        
                        # keep only the time charging between the bounds
                        lower = self.stats[n][self.stats[n].avail < upper_bound]
                        upper = lower[lower.avail >= lower_bound]
                        
                        # list of episodes between these bounds
                        episodes = upper.index
                        
                        # initialize
                        soc_array = np.zeros((len(episodes),len(x)))
                        pv_array = np.zeros((len(episodes),len(x)))
                        load_array = np.zeros((len(episodes),len(x)))
                        
                        # iterate over each episode
                        for k, e in enumerate(episodes):
                            
                            # get the episode result
                            ep = decision[decision.episode == int(e)]
                            
                            # time of departure of the episode
                            td = list(ep.avail).index(0)
                            
                            # all soc, pv gen. and load in this window
                            soc = ep.soc[:td+1]*100
                            pv = ep.pv[:td+1]/1000
                            load = ep.load[:td+1]/1000
                            
                            # how long is the EV charging
                            time_charging = ep.avail.sum()
                            
                            # get the deciles of the time charging
                            t = np.dot(x,time_charging)
                            t = [math.floor(i) for i in t]
                            
                            # the corresponding variables deciles
                            deciles_soc = soc[t]
                            deciles_pv = pv[t]
                            deciles_load = load[t]
                            
                            soc_array[k,:] = deciles_soc
                            pv_array[k,:] = deciles_pv
                            load_array[k,:] = deciles_load
                            
                        # concatenate the results in df and plot PV, load and SOC 
                
                        df = pd.DataFrame(data = {int(x[q]*100): soc_array[:,q] for q in range(len(x))})
                    
                        sns.boxplot(data = df, ax = axes[j+2,i], orient = 'v', color = self.color_groups[self.algos_specs[n][s]])
                        
                        df = pd.DataFrame(data = {int(x[q]*100): pv_array[:,q] for q in range(len(x))})
                    
                        sns.boxplot(data = df, ax = axes[0,i], orient = 'v', color = self.palette[11])
                        
                        df = pd.DataFrame(data = {int(x[q]*100): load_array[:,q] for q in range(len(x))})
                    
                        sns.boxplot(data = df, ax = axes[1,i], orient = 'v', color = self.palette[12])
                        
                        axes[j+2,i].grid()
                        
                        axes[j+2,i].set_xticks(ticks)
                        
                        if i > 0:
                            axes[j+2,i].set_yticklabels([])
                        
                        axes[j+2,0].set_ylabel('SOC [%]')
                        
                        axes[j+2,i].set_ylim([15,105])
                        
                        
                    # plot specifications
                    power = ['PV','Load']
                    
                    for k in range(2):
                        axes[k,i].grid()
                        axes[k,i].set_ylim([0,20])
                        if i > 0:
                            axes[k,i].set_yticklabels([])
                        axes[k,0].set_yticks(np.arange(0,21,10))
                        axes[k,0].set_ylabel(f'{power[k]} [kW]')
                    
                    
                    axes[4,i].set_xticklabels(labels)
                    axes[0,i].set_title(f'Charging time {lower_bound} - {upper_bound-1}h', fontsize = 14)  
                    
                patches = [mpatches.Patch(color=self.color_groups[self.algos_specs[n][s]], label=self.algos_specs[n][s]) for n in algos]
                fig.legend(handles=patches, loc='lower center',ncol=int(len(algos)),bbox_to_anchor=(0.5,-0.02), fontsize = 15)
