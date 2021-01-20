

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:03:49 2020

@author: Yann
"""
## IMPORTS

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np

from df_prepare import  prices_romande_energie

folder_path = 'C:/Users/Yann/Documents/EPFL/PDM/V2G/Results/opti/'

dpi = 400


#%% DEFINITION
class V2X_VARIATION:
    def __init__(self, res_folder_path, method = 'opti'):
        
        # folder path where results v1g and v2g are (df)
        
        self.res_folder_path = res_folder_path
        
        # method used
        self.method = method
        
        # color initialization
        palette = sns.color_palette()
        
        self.v1g_color = palette[9]
        self.v2g_color = palette[6]
        self.v2x_color = {'v1g': palette[9],
                         'v2g':palette[6]}
        
        colors = sns.color_palette()
        
        self.color_indices = {'Grid': colors[0],
                          'PV': colors[1],
                          'EV': colors[2],
                          'Load': colors[3]}
        
        # labels and names
        self.metrics = ['PVSC', 'soc_dep', 'Cost', 'APR']
        self.metrics_title = ['PVSC','SOC at departure', 'Median Cost', 'APR']
        self.metrics_unit = ['%','%', 'CHF','%', '%']
        self.metrics_props = {self.metrics[i]: {'title': self.metrics_title[i],'unit': self.metrics_unit[i]} for i in range(len(self.metrics))}
        
        self.labels_radar = {'PVSC': 'PVSC',
                  'Cost': 'CP',
                  'APR': 'APR',
                  'soc_dep': 'SOC '
                  }
        
        
    def import_decisions(self):
        
        # methods with respective objective
        self.methods = ['V1X Cost ', 
                   'V1X APR ', 
                   'V1X PV',
                   'V2X Cost', 
                   'V2X APR', 
                   'V2X PV'
                   ]
        # objectives codes
        objectives = ['cost', 'pv', 'peak']
        
        n_v1g = ['v1g_'+self.method+'_'+o for o in objectives]
        n_v2g = ['v2g_'+self.method+'_'+o for o in objectives]
        
        # names codes
        self.names = n_v1g + n_v2g
        
        # import
        csv_code = '.csv'
        self.decisions = {n: pd.read_csv(self.res_folder_path+'results_'+n+csv_code, index_col = 0) for n in self.names}
        
        # asses index into datetime
        for n in self.names:
            df = self.decisions[n].copy()
            new_index = pd.to_datetime(df.index, dayfirst = True)
            df.index = new_index
            self.decisions[n] = df
        
        # map between name code and real name
        self.algorithms = {self.names[i]: self.methods[i] for i in range(len(self.names))}
        
        # groups separation (objective and methods)
        group_code = {'cost': [], 'peak': [], 'pv': [], 'v1g': [], 'v2g': []}
    
        group_names = ['Objective: Cost','Objective: APR', 'Objective: PVSC',
                       'V1X','V2X']
        self.groups = {}
        for n in self.names:
            
            for i,g in enumerate(group_code.keys()):
                
                if g in n:
                    
                    group_code[g].append(n)
                    
                self.groups[group_names[i]] = group_code[g]
        
        
        
        self.algos_specs = {n: {'Objective': None, 'Method': None} for n in self.names}
        
        for g_name in self.groups.keys():
        
            algos = self.groups[g_name]
            
            if 'Objective' in g_name:
                
                for a in algos:
                    self.algos_specs[a]['Objective'] = g_name
                    
            else:
        
                for a in algos:
                    self.algos_specs[a]['Method'] = g_name
        
        
        # map between objective and respective metric
        metrics_to_algos = {}
        metrics_to_algos['Cost'] = self.groups['Objective: Cost']
        metrics_to_algos['PVSC'] = self.groups['Objective: PVSC']
        metrics_to_algos['APR'] = self.groups['Objective: APR']
        metrics_to_algos['share_PV_ext'] = self.groups['Objective: PVSC']
        
        self.metrics_to_algos = metrics_to_algos
        
        # compute metrics for each algorithm
        self.stats = {n: self.compute_metrics(self.decisions[n]) for n in self.names}
                
        # dict for each metric
        self.stats_df = {m: pd.DataFrame(data = {n: list(self.stats[n].loc[:,m] )
                                 for n in self.names}) for m in self.metrics}
    
        

    def get_obj(self, algo):
        # get objective from algo name code
        if 'cost' in algo:
            return 'Cost'
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
    
    def loss_ratio(self, stats_loss, quantile_low = 0.1, quantile_high = 0.9):
        
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
    
    def adjust(self, s):
        # put str s on multiple line for plot (if name is too long)
        if ' ' in s:
            s = s.replace(' ', '\n')
        return s   
    


    def ax_boxplot_metrics(self, ax, metric, g, ylim = None, leg = True):
        # return ax with boxplot of the metric for the group g
        
        # algos in the group
        algos = self.groups[g]
        
        # if group is objective, we plot the methods and vice-versa
        if 'Objective' in g:
            s = 'Method'
        else:
            s = 'Objective'
            
        m = metric
        # get metrc df for all algos
        s_df = self.stats_df[m]
        
        # concatenate df with only specified algos
        new_df = {}
        for n in algos:
    
            values = list(s_df[n].values)
            new_df[self.algos_specs[n][s]] = values
        
        df = pd.DataFrame(new_df)
        
        # plot
        sns.boxplot(data = df, ax = ax, palette = [self.v1g_color, self.v2g_color])
    
    
        ax.set_title(self.metrics_props[m]['title'])
        ax.set_ylabel(self.metrics_props[m]['unit'])
        if leg == False:
            ax.set(xticklabels=[])
    
        ax.grid()
        if ylim is not None:
            ax.set_ylim(ylim)
            
        return ax
    
    
    #box plot metrics
    def return_variation(self,metric,v1g, v2g):
        
        # return  variation of v2g algo from v1g algo of particular metric
        s_df = self.stats_df[metric]
        
        df = pd.DataFrame()
        df['V2X variation [%]'] = 100*(s_df[v2g]/s_df[v1g])-100
    
        return df
    
    def ax_boxplot_metrics_variation(self, ax, metric, v1g, v2g, v2g_c = True):
        # return boxplot of the variation above
    
        df = self.return_variation(metric, v1g, v2g)
        
        # color
        if v2g_c == True:
            c = self.v2g_color
        else:
            c = self.v2x_color[v2g]
            
        # plot
        sns.boxplot(data = df, ax = ax, color = c)
    
        ax.set_title(self.metrics_props[metric]['title'], fontsize = 15)
        ax.set_ylabel('%')
    
    
        ax.grid()
    
            
        return ax
    
    
    # AX RADAR
    
    def ax_radar(self, ax, g, x_legend = 0.5, y_legend = -0.5, start = np.pi/4, leg = True):
        
        # radar plot for specific algos in group g
        # start: first angle
        algos = self.groups[g]
        if 'Objective' in g:
            s = 'Method'
        else:
            s = 'Objective'
        
        # intialization
        group_df_low = pd.DataFrame(index = algos)
        group_df_high = pd.DataFrame(index = algos)
        group_df_med = pd.DataFrame(index = algos)
    
         # define angles for radar plot
        angles=np.linspace(start, 2*np.pi + start, len(self.metrics), endpoint=False)
        angles=np.concatenate((angles,[angles[0]]))
        
         # compute values for each metrics, with quantiles
        for m in self.metrics:
    
            if m == 'Cost':
                low, med, high = self.loss_ratio(self.stats_df[m])
                group_df_low[self.labels_radar[m]] = low[algos]
                group_df_high[self.labels_radar[m]] = high[algos]
                group_df_med[self.labels_radar[m]] = med[algos]
            else:
                group_df_low[self.labels_radar[m]] = self.stats_df[m].quantile(0.1)[algos]
                group_df_high[self.labels_radar[m]] = self.stats_df[m].quantile(0.9)[algos]
                group_df_med[self.labels_radar[m]] = self.stats_df[m].median()[algos]
            
        # plot
        for a in algos:
            if self.algos_specs[a]['Method'] == 'V1X':
                c = self.v1g_color
            else:
                c = self.v2g_color
                
            # errors plot
            vals_low =group_df_low.loc[a,:]
            errors_low = group_df_med.loc[a,:] - vals_low
            
            errors_low=np.concatenate((errors_low,[errors_low[0]]))
            
            
            vals_high =group_df_high.loc[a,:]
            errors_high =  vals_high - group_df_med.loc[a,:]
            
            errors_high=np.concatenate((errors_high,[errors_high[0]]))
            
            
            vals_med =group_df_med.loc[a,:]
            vals_med=np.concatenate((vals_med,[vals_med[0]]))
    
            
            ax.plot(angles, vals_med, 'o-',  label = self.algos_specs[a][s], color = c, linewidth = 2)
            ax.errorbar(angles, vals_med, yerr = errors_low,  uplims=True,color = c)
            ax.errorbar(angles, vals_med, yerr = errors_high,  lolims=True,color = c)
            
            # angles legend
            ax.set_thetagrids(angles[:-1] * 180/np.pi, group_df_low.columns, fontsize = 16)
            
        
        ax.grid(True)
        if leg == True:
            ax.legend(loc = 'lower center',bbox_to_anchor=(x_legend,y_legend), ncol = 1, fontsize = 14)
    
        ax.set_xlabel('%', fontsize = 18)
        ax.set_ylim([0,105])
        ax.set_title(g)
        
        return ax
    
    #bar plot per obj
    def ax_bar_v2x_obj(self, g, v1g,v2g, ax, leg = False):
        # return ax of objectve and the mean metric variation
        
        new_df = pd.DataFrame()
        for m in self.metrics:
            if m == 'soc_dep':
                continue
            elif m == 'Cost':
                # turn cost into revenue for positive impact
                new_df[m] = -self.return_variation(m, v1g, v2g).mean()
            else:
                new_df[m] = self.return_variation(m, v1g, v2g).mean()
        
        new_df.rename(columns = {c: self.labels_radar[c] for c in new_df.columns}, inplace = True)
        new_df.rename(columns = {'CP':'Revenue'}, inplace = True)
        ax.set_ylabel('%')
        
        # plot
        sns.barplot(data = new_df, ax = ax)
        if leg == False:
            ax.set(xticklabels=[])
        else:
            ax.set(xticklabels = new_df.columns)
        ax.grid(True)
        ax.set_title(g)
        return ax

    
    def plot_all_obj(self):
        
        # plot call for boxplot metrics variation
        fig, axes = plt.subplots(1, len(self.metrics)-1, sharey = True, figsize=(16, 9), dpi = dpi)
        

        for i, g in enumerate(self.groups.keys()):
            if 'Objective' in g:
                
                algos = self.groups[g]
                obj = self.get_obj(algos[0])
                
                for a in algos:
                    # identify which is v1g which is v2g
                    if self.algos_specs[a]['Method'] == 'V1X':
                        v1g = a
                    else:
                        v2g = a
        
                ax = axes[i]
                self.ax_boxplot_metrics_variation(ax,obj,v1g,v2g)
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.grid(True)
            else:
                continue
        
        fig.show()

    def plot_variation_per_objective(self):
        # plot call for ax_bar_v2x_obj
        fig, axes = plt.subplots(3,1, figsize=(9, 6), dpi = dpi)
        plt.suptitle('V2X metric variation per objective', fontsize = 14)
        leg = False
        for i, g in enumerate(self.groups.keys()):
            if 'Objective' in g:
                
                algos = self.groups[g]
         
                for a in algos:
                    if self.algos_specs[a]['Method'] == 'V1X':
                        v1g = a
                    else:
                        v2g = a
        
                ax = axes[i]
                if i == 2:
                    leg = True
                
                self.ax_bar_v2x_obj(g,v1g, v2g, ax, leg)
                ax.axhline(0, color = 'black')
            else:
                continue

        fig.show()
    
    
    def plot_complete_graph(self):
        # plot full results per group
        
        # colors
        colors_v = [self.v1g_color, self.v2g_color]
        
        for g in self.groups.keys():
            # we want only to compare the objectives
            if 'Objective' in g:
                
                # same as above
                algos = self.groups[g]
                obj = self.get_obj(algos[0])
                for a in algos:
                    if self.algos_specs[a]['Method'] == 'V1X':
                        v1g = a
                    else:
                        v2g = a
                
                fig = plt.figure(figsize=(16, 9), dpi = dpi)
                plt.suptitle(g, fontsize = 18)
                
                # radar plot
                ax1 = fig.add_subplot(1,3,1, polar = True)
                self.ax_radar(ax1, g, leg = False)
                
                # boxplot
                ax2 = fig.add_subplot(1,3,2)
                self.ax_boxplot_metrics_variation(ax2,obj,v1g,v2g, v2g_c = True)
                ax2.tick_params(axis='both', which='major', labelsize=16)
                ax2.grid(True)
                ax3= fig.add_subplot(4,3,3)
                ax4= fig.add_subplot(4,3,6)
                ax5= fig.add_subplot(4,3,9)
                ax6= fig.add_subplot(4,3,12)
                
                ax_box = [ax3, ax4, ax5, ax6]
                leg = False
                for i in range(len(ax_box)):
                    m = self.metrics[i]
                    ax = ax_box[i]
                    if i == len(ax_box)-1:
                        leg = True
                    self.ax_boxplot_metrics(ax, m, g, ylim = None, leg = leg)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    ax.grid(True)
                    
                # add legend
                patches1 = [mpatches.Patch(color=colors_v[k], label=self.algorithms[n]) for k,n in enumerate(algos)]
                ax1.legend(handles=patches1, loc='lower center',ncol=int(len(algos)), bbox_to_anchor = (0.5,-0.4)) 
                
                
                fig.show()
            else:
                continue
