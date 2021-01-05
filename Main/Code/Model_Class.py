# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:29:01 2020

@author: Yann
"""


import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from inputimeout import inputimeout, TimeoutOccurred


class EV:
    def __init__(self, BC_EV = 16e3, eta_EV_ch = 1, 
                        P_EV_chmax = 3.7e3,P_EV_dismax = 3.7e3, SOC_min = 0.2, SOC_max = 1,SOC_min_departure = 1):
        ''' EV class with parameters'''
        self.BC_EV = BC_EV
        self.eta_EV_ch = eta_EV_ch
        self.eta_EV_dis = 1/self.eta_EV_ch
        self.P_EV_chmax = P_EV_chmax
        self.P_EV_dismax = P_EV_chmax
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.SOC_min_departure = SOC_min_departure
        
        
class House:
    def __init__(self, installed_PV = 10000, pv_nominal = 2835):
        ''' House class with parameters
        Input:
            Installed PV: int, PV capacity [kWp] of the installation
            pv_nominal: int, Nominal capacity of the PV forecast'''
        self.installed_PV = installed_PV
        self.pv_nominal = pv_nominal
        self.scale_PV = float(installed_PV/pv_nominal)
        
class Model:
    
    def __init__(self, data_EV, t_start, t_res, House, EV, spot_prices, V2X = 0):
        ''' Model
        Input:
            Data_EV: Actual Data
            t_start: pd.Timestamp, episode start
            t_res: float, dataset resolution
            House: class
            EV: class
            spot_prices: df, price of electricity
            V2X: bool, V2X on or off
            '''
        
        # Initialize class parameters
        self.set_house_parameter(House)

        self.set_EV_parameters(EV)
        
        self.data_EV = data_EV
        
        # column name of output decision
        self.col_names = ['pv', 'load', 'pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load','ev_load','ev_grid','y_buy',
           'y_sell','y_ch','y_dis','soc',
       'avail', 'episode']
        
        # decision variables
        self.variables = ['pv_ev', 'pv_load','pv_grid', 'grid_ev', 'grid_load','ev_load','ev_grid','y_buy',
           'y_sell','y_ch','y_dis']
        
        self.t_start = t_start
        
        self.EV = EV
        
        # intialize decisions, with soc at arrival
        previous_decision = {self.col_names[i]: [0] for i in range(len(self.col_names))}
        previous_decision['soc'] = self.data_EV.loc[t_start, 'soc']
        
        self.decisions = pd.DataFrame(data = previous_decision, index = [t_start])
        
        self.spot_prices = spot_prices
        
        self.V2X = V2X

        
    def set_house_parameter(self, House):
        self.scale_PV = House.scale_PV
    
    def set_EV_parameters(self, EV):
        self.BC_EV = EV.BC_EV
        self.eta_EV_ch = EV.eta_EV_ch
        self.eta_EV_dis = EV.eta_EV_dis
        self.P_EV_chmax = EV.P_EV_chmax
        self.P_EV_dismax = EV.P_EV_dismax
        self.SOC_min = EV.SOC_min
        self.SOC_max = EV.SOC_max
        self.SOC_min_departure = EV.SOC_min_departure
        
        
    def set_parameters(self):
        
        self.EV_availability = self.data.EV_availability
        
        self.preprocess_time()
        
        # Forecasted parameters (first and second stage)
        self.P_PV = {t: list(self.predictions_PV.loc[t]) for t in self.time_horizon}

        self.P_PV[self.t_decision] = self.realization_PV
        
        self.P_load = {t: list(self.predictions_load.loc[t]) for t in self.time_horizon}
        
        self.P_load[self.t_decision] = self.realization_load
        
        # spot prices
        self.buy_spot_price = self.spot_prices.loc[self.time_vector, 'buy']
        
        self.sell_spot_price = self.spot_prices.loc[self.time_vector, 'sell']
        
    def preprocess_data(self, t_decision, t_end, predictions_PV, predictions_load):
        ''' Data preprocessing 
        Input: 
            t_decision, t_end: pd.Timestamp
            predictions_PV, predictions_load: dataframe'''
        
        self.data = self.data_EV[t_decision:t_end]
        
        self.t_res = ((self.data.index[1] - self.data.index[0]).seconds)/3600
        
        self.episode = self.data_EV.loc[t_decision, 'episode']
        
        # optimization over min(episode_length, n_hour_future + 1)
        
        self.delta = t_end - t_decision
        
        self.duration = int((self.delta.days * 24  + self.delta.seconds/3600)/self.t_res)
        
        self.realization_PV = self.data_EV.loc[t_decision, 'PV']
        
        self.realization_load = self.data_EV.loc[t_decision, 'load']
        
        # rescale data to actual installation
        if self.forecasting:
        
            self.predictions_PV = predictions_PV[:self.duration]*self.scale_PV
            
            
        else:
            self.predictions_PV = predictions_PV[:self.duration]
            
        self.predictions_load = predictions_load[:self.duration]
        
        # size of scenarios
        
        self.n_samples = predictions_PV.shape[1]

        self.range_samples = range(self.n_samples)
        
        self.t_decision = t_decision
        
        
        
        
    def preprocess_time(self):
        
        # [t_decision: t_decision + 24]
        self.time_vector = list(self.data.index)
        
        self.t_start = self.time_vector[0]
        self.t_end = self.time_vector[-1]
        
        # second stage time vector
        self.time_horizon = self.time_vector[1:]
        self.t_forecast = self.time_horizon[0]
        
        # time vector for soc
        self.time_soc = self.time_horizon[1:]
        
        # if t_departure is not yet in time horizon
        try:
            self.t_departure = self.data[self.data.EV_availability == 0].index[0]
        except:
           self.t_departure = self.data.index[-1]
    
    def add_first_stage_variables(self):
        # first stage variables
        
        self.P_PV_2EV_1 = self.m.addVar(name = 'pv_ev_1')
        self.P_PV_2L_1 = self.m.addVar(name = 'pv_load_1')
        self.P_PV_2G_1 = self.m.addVar(name = 'pv_grid_1')

        self.current_SOC = self.decisions.loc[self.t_decision, 'soc']
        self.SOC_2 = self.m.addVar(lb = self.SOC_min, ub = self.SOC_max, name = 'soc_2')
        self.P_EV_ch_1 = self.m.addVar(ub = self.P_EV_chmax)
        self.P_EV_dis_1 = self.m.addVar(ub = self.P_EV_dismax)
        self.P_EV2L_1 = self.m.addVar(name = 'ev_load_1')
        self.P_EV2G_1 = self.m.addVar(name = 'ev_grid_1')
        
        
        self.P_grid_bought_1 = self.m.addVar(name = 'P_bought_1')
        self.P_grid_sold_1 = self.m.addVar()
        self.y_g_buy_1 = self.m.addVar(vtype=GRB.BINARY, name = 'y_buy_1')
        self.y_g_sell_1 = self.m.addVar(vtype=GRB.BINARY, name = 'y_sell_1')
        
        self.P_grid_2EV_1 = self.m.addVar(name = 'grid_ev_1')
        self.P_grid_2L_1 = self.m.addVar(name = 'grid_load_1')
        
        self.y_ev_ch_1 = self.m.addVar(vtype=GRB.BINARY, name = 'y_ch_1')
        self.y_ev_dis_1 = self.m.addVar(vtype=GRB.BINARY, name = 'y_dis_1')
        
    def add_first_stage_constraints(self):
        
        # 1st stage constraints
        t = self.t_decision
        
        self.Grid_balance_1 = self.m.addConstr(self.P_PV[t] + self.y_g_buy_1 * self.P_grid_bought_1
                                               + self.y_ev_dis_1*self.P_EV_dis_1 - (self.y_g_sell_1) * self.P_grid_sold_1 
                                                - self.y_ev_ch_1*self.P_EV_ch_1 
                                                - self.P_load[t] == 0, name = 'Grid_balance_1')

        
        self.PV_balance_1 = self.m.addConstr(self.P_PV[t] == self.P_PV_2EV_1 + self.P_PV_2L_1 + self.P_PV_2G_1, name = 'PV_balance_1')
        
        self.Load_balance_1 = self.m.addConstr((self.P_load[t] == self.P_PV_2L_1+ self.P_grid_2L_1 + self.P_EV2L_1),
                                  name = 'Load_balance_1')
        
        self.Grid_bought_balance_1 = self.m.addConstr(self.P_grid_bought_1 == self.P_grid_2EV_1 
                                            + self.P_grid_2L_1, name = 'Grid_bought_balance_1')
        
        self.Grid_sold_balance_1 = self.m.addConstr(self.P_grid_sold_1 == self.P_PV_2G_1 + self.P_EV2G_1, name = 'Grid_sold_balance_1')
        
        self.ch_EV_balance_1 = self.m.addConstr(self.P_EV_ch_1 == self.P_grid_2EV_1
                                    + self.P_PV_2EV_1, name = 'ch_EV_balance_1')
        
        self.dis_EV_balance_1 = self.m.addConstr(self.P_EV_dis_1 == self.P_EV2L_1
                                    + self.P_EV2G_1, name = 'dis_EV_balance_1')
        
        self.P_chmax_1 = self.m.addConstr(self.P_EV_ch_1 <= self.y_ev_ch_1*self.EV_availability[t] * self.P_EV_chmax, name = 'P_EV_chmax_1')
        self.P_dismax_1 = self.m.addConstr(self.P_EV_dis_1 <= self.y_ev_dis_1*self.EV_availability[t] * self.P_EV_dismax * self.V2X, name = 'P_EV_dismax_1')

        # update soc for t_forecast, depending on first stage decisions
        self.SOC_update_1 = self.m.addConstr(self.SOC_2 == 
                                  self.current_SOC + 
                                  (self.eta_EV_ch*self.P_EV_ch_1/self.BC_EV - self.eta_EV_dis*self.P_EV_dis_1/self.BC_EV) , name = 'SOC_update_1')
        
        self.grid_bin_1 = self.m.addConstr(self.y_g_buy_1 + self.y_g_sell_1 <= 1, name = 'y_grid_1')
        self.EV_bin_1 = self.m.addConstr(self.y_ev_ch_1 + self.y_ev_dis_1 <= 1,name = 'y_ev_1')
        self.EV_grid_1 = self.m.addConstr((1-self.y_g_sell_1)*self.P_EV2G_1 + (1-self.y_g_buy_1)*self.P_grid_2EV_1 == 0, 'y_ev_grid_1')

    def add_second_stage_variables(self):
        
        # second stage variables on time horizon
        self.P_PV_2EV = self.m.addVars(self.time_horizon, self.n_samples, name = 'pv_ev')
        self.P_PV_2L = self.m.addVars(self.time_horizon, self.n_samples, name = 'pv_load')
        self.P_PV_2G = self.m.addVars(self.time_horizon, self.n_samples, name = 'pv_grid')        
        
        self.SOC = self.m.addVars(self.time_horizon, self.n_samples,lb = self.SOC_min, ub = self.SOC_max, name = 'soc')
        
        self.P_EV_ch = self.m.addVars(self.time_horizon, self.n_samples,lb = 0, ub = self.P_EV_chmax)
        self.P_EV_dis = self.m.addVars(self.time_horizon, self.n_samples,lb = 0, ub = self.P_EV_dismax)
        self.P_EV2L = self.m.addVars(self.time_horizon, self.n_samples, name = 'ev_load')
        self.P_EV2G = self.m.addVars(self.time_horizon, self.n_samples, name = 'ev_grid')
        
        self.P_grid_bought = self.m.addVars(self.time_horizon, self.n_samples, name = 'P_bought')
        self.P_grid_sold = self.m.addVars(self.time_horizon, self.n_samples)
        self.y_g_buy = self.m.addVars(self.time_horizon, self.n_samples, vtype=GRB.BINARY, name = 'y_buy')
        self.y_g_sell = self.m.addVars(self.time_horizon, self.n_samples, vtype=GRB.BINARY, name = 'y_sell')
        
        self.P_grid_2EV = self.m.addVars(self.time_horizon, self.n_samples, name = 'grid_ev')
        self.P_grid_2L = self.m.addVars(self.time_horizon, self.n_samples, name = 'grid_load')
        
        self.y_ev_ch = self.m.addVars(self.time_horizon, self.n_samples, vtype=GRB.BINARY, name = 'y_ch')
        self.y_ev_dis = self.m.addVars(self.time_horizon, self.n_samples, vtype=GRB.BINARY, name = 'y_dis')
        
        
    
    def add_second_stage_constraints(self):
        
        
        self.Grid_balance = self.m.addConstrs((self.P_PV[t][i] + self.y_g_buy[t,i] * self.P_grid_bought[t,i] + self.y_ev_dis[t,i]*self.P_EV_dis[t,i]
                                               - self.y_g_sell[t,i] * self.P_grid_sold[t,i] 
                                                -  self.y_ev_ch[t,i]*self.P_EV_ch[t,i] 
                                                - self.P_load[t][0] == 0 
                                                for t in self.time_horizon for i in self.range_samples), name = 'Grid_balance')
        
        self.PV_balance = self.m.addConstrs((self.P_PV[t][i] == self.P_PV_2EV[t,i] + self.P_PV_2L[t,i] + self.P_PV_2G[t,i] 
                                  for t in self.time_horizon for i in self.range_samples), name = 'PV_balance')
        
        
        self.Load_balance = self.m.addConstrs((self.P_load[t][0] == self.P_PV_2L[t,i] + self.P_grid_2L[t,i] + self.P_EV2L[t,i]
                                  for t in self.time_horizon for i in self.range_samples), name = 'Load_balance')
        
        self.Grid_bought_balance = self.m.addConstrs((self.P_grid_bought[t,i] == self.P_grid_2EV[t,i] 
                                            + self.P_grid_2L[t,i]
                                            for t in self.time_horizon for i in self.range_samples), name = 'Grid_bought_balance')
        
        self.Grid_sold_balance = self.m.addConstrs((self.P_grid_sold[t,i] == self.P_PV_2G[t,i] + self.P_EV2G[t,i]
                                            for t in self.time_horizon for i in self.range_samples), name = 'Grid_sold_balance')
        
        self.ch_EV_balance = self.m.addConstrs((self.P_EV_ch[t,i] == self.P_grid_2EV[t,i]
                                  + self.P_PV_2EV[t,i]
                                  for t in self.time_horizon for i in self.range_samples), name = 'ch_EV_balance')
        
        self.dis_EV_balance = self.m.addConstrs((self.P_EV_dis[t,i] == self.P_EV2L[t,i]
                                  + self.P_EV2G[t,i]
                                  for t in self.time_horizon for i in self.range_samples), name = 'dis_EV_balance')
       
        self.P_chmax = self.m.addConstrs((self.P_EV_ch[t,i]) <= self.EV_availability[t] * self.y_ev_ch[t,i]*self.P_EV_chmax for t in self.time_horizon for i in self.range_samples)
        
        self.P_dismax = self.m.addConstrs((self.P_EV_dis[t,i]) <= self.EV_availability[t] * self.y_ev_dis[t,i]*self.P_EV_dismax * self.V2X for t in self.time_horizon for i in self.range_samples)
        
        self.SOC_continuity = self.m.addConstrs((self.SOC[self.time_horizon[0],j] == self.SOC_2 
                                                 for j in self.range_samples), name = 'SOC_continuity')
        
        self.SOC_update = self.m.addConstrs((self.SOC[self.time_horizon[t+1],i] == 
                                  self.SOC[self.time_horizon[t],i] + 
                                  (self.P_EV_ch[self.time_horizon[t],i]/self.BC_EV) * self.eta_EV_ch
                                  - (self.P_EV_dis[self.time_horizon[t],i]/self.BC_EV) * self.eta_EV_dis
                                  for t in range(len(self.time_horizon)-1) for i in self.range_samples 
                                  ), name = 'SOC_update')
        
        self.grid = self.m.addConstrs((self.y_g_buy[t,i] + self.y_g_sell[t,i] <= 1 for t in self.time_horizon for i in self.range_samples), name = 'y_grid')
        self.EV_bin = self.m.addConstrs((self.y_ev_ch[t,i] + self.y_ev_dis[t,i] <= 1 for t in self.time_horizon for i in self.range_samples), name = 'y_ev')
        self.EV_grid = self.m.addConstrs(((1-self.y_g_sell[t,i])*self.P_EV2G[t,i] + (1-self.y_g_buy[t,i])*self.P_grid_2EV[t,i] == 0 for t in self.time_horizon for i in self.range_samples), name = 'y_ev_grid' )

        
        
    def optimize(self, t_decision, t_end, predictions_PV, predictions_load, forecasting = True, objective_1 = 'cost', objective_2 = 'soc',
                 method = 'deterministic', parameters = None, lambda_soc = 0.5, soc_penalty = 1.5):
        
        if predictions_PV.shape[1] == 1:
            self.stochastic = False
            self.m = gp.Model()
        else:
            self.stochastic = True
            self.m = gp.Model()
        
        self.m.Params.OutputFlag = 0
        self.m.Params.TimeLimit=60
        self.forecasting = forecasting
        self.preprocess_data(t_decision,t_end, predictions_PV, predictions_load)
        self.set_parameters()
        # first_stage
        self.add_first_stage_variables()
        self.add_first_stage_constraints()
        
        self.add_second_stage_variables()
        self.add_second_stage_constraints()
        
        
        PV_consumed_decision = self.P_PV_2EV_1 + self.P_PV_2L_1
        PV_decision = self.realization_PV
        
        if PV_decision > 0:
            self_cons_d = PV_consumed_decision/PV_decision
        else:
            self_cons_d = 0
            
        
        
        SOC_difference =  gp.quicksum([self.SOC_min_departure - self.SOC[self.t_end,i] for i in self.range_samples])
        
        objective = {}
        
        if 'pv'  in [objective_1, objective_2] :
            # PV_consumed_forecast = [np.sum([(self.P_PV_2EV[t,i] + self.P_PV_2L[t,i]) for t in self.time_horizon]) for i in self.range_samples]
            # PV_total_forecast = [self.predictions_PV.sum()[i] for i in self.range_samples]
            
            # self_cons = np.sum([PV_consumed_forecast[i]/PV_total_forecast[i] for i in self.range_samples])
            
            PV_grid_1 = self.P_PV_2G_1
            PV_grid = gp.quicksum(self.P_PV_2G)
            
            penalty = self.BC_EV
            self.m.Params.MIPGap = 1e-2
            
            decision = PV_grid_1
            future = PV_grid
            
            
        elif 'peak' in [objective_1, objective_2]:
            
            self.z_apr = self.m.addVars(self.n_samples, name = 'z_apr')
            self.P_max_var = self.m.addVars(self.n_samples, name = 'P_max_var')

            self.P_max = self.m.addConstrs(self.P_max_var[i] == gp.max_(self.P_grid_bought[t,i] for t in self.time_horizon) for i in self.range_samples)
            self.peak = self.m.addConstrs((gp.quicksum(self.P_grid_bought[t,i] for t in self.time_horizon))/len(self.time_horizon) + self.z_apr[i] == self.P_max_var[i] for i in self.range_samples)
            
            penalty = self.BC_EV
            self.m.Params.MIPGap = 1e-2
            
            decision = 0
            future = gp.quicksum(self.z_apr)
        elif 'cost' in [objective_1, objective_2]:
            cost_decision = self.P_grid_bought_1* self.buy_spot_price[self.t_decision] - self.P_grid_sold_1* self.sell_spot_price[self.t_decision]
            cost_forecast = gp.quicksum([(self.P_grid_bought[t,i] * self.buy_spot_price[t] - self.P_grid_sold[t,i] * self.sell_spot_price[t]) for t in self.time_horizon for i in self.range_samples])  
        
            penalty = soc_penalty * self.buy_spot_price[self.t_departure]*self.BC_EV
            
            decision = cost_decision
            future = cost_forecast
            
        Power_difference = penalty * SOC_difference
        objective['soc'] = Power_difference/self.n_samples
        
        lambda_main = 1 - lambda_soc
        
        
        if  method == 'deterministic' or method == 'day_ahead' or method == 'expected value':
            objective[objective_1] = decision + future/self.n_samples
            # objective['cost'] = cost_decision + cost_forecast/self.n_samples 
            # objective['soc'] = Power_difference/self.n_samples
            # objective['pv'] = PV_grid_1 + PV_grid/self.n_samples
            # #objective['pv'] = (1 - self_cons_d) + (1 - self_cons/self.n_samples)
            
            # if 'peak' in [objective_1, objective_2]: 
            #     objective['peak'] = gp.quicksum(self.z_apr) / self.n_samples
            

        elif method == 'Markowitz':
            alpha_obj1 = parameters['alpha_obj1']
            objective[objective_1] = decision + future/((1-alpha_obj1)*self.n_samples) 
            # objective['cost'] = (cost_decision +  cost_forecast/((1-alpha_obj1)*self.n_samples)) 
            # objective['soc'] =  Power_difference/((1-alpha_soc)*self.n_samples)
            # objective['pv'] =  PV_grid_1 + PV_grid/((1-alpha_obj1)*self.n_samples)
            # #objective['pv'] = (1 - self_cons_d) + (1 - self_cons/ ((1-alpha_obj1)*self.n_samples))
            # if 'peak' in [objective_1, objective_2]: 
            #     objective['peak'] = gp.quicksum(self.z_apr) / ((1-alpha_obj1)*self.n_samples)
            
        self.m.setObjective(lambda_main * objective[objective_1] + lambda_soc* objective['soc'])

            
        self.m.optimize()
                

        self.update_decisions()

        
    def update_decisions(self):
        
        
        SOC = self.current_SOC
        
        dataset = {}
        dataset['pv'] = self.realization_PV
        dataset['load'] = self.realization_load
        for var in self.variables:
            dataset[var] = self.m.getVarByName(f'{var}_1').x
            
        dataset['soc'] = SOC
        dataset['avail'] = self.EV_availability[self.t_decision]
        dataset['episode'] = self.episode

        self.decision = pd.Series(data = dataset, name = self.t_decision)
        
        next_decision = {self.col_names[i]: 0 for i in range(len(self.col_names))}
        next_decision['soc'] = self.SOC_2.x
        next_decision = pd.Series(next_decision, name = self.t_forecast)
        
        self.decisions.loc[self.t_decision] = self.decision
        self.decisions.loc[self.t_forecast] = next_decision
        
    
    def predictions_SOC(self):
        
        df_SOC = pd.DataFrame(index = self.time_vector, columns=self.range_samples)
        
        for t,i in self.SOC:
            df_SOC.loc[t,i] = self.SOC[t,i].x
            
        return df_SOC
    
    def results_deterministic(self):
        

        dataset = pd.DataFrame(index = self.time_vector)
        dataset['pv'] = self.data_EV.loc[self.time_vector, 'PV']
        dataset['load'] = self.data_EV.loc[self.time_vector, 'load']

        for var in self.variables:
            var_1 = [self.m.getVarByName(f'{var}_1').x]
            var_2 = [self.m.getVarByName(f'{var}[{t},{i}]').x for t in self.time_horizon for i in self.range_samples]
            var_1.extend(var_2)
            dataset[var] = var_1
        
        SOC = [self.current_SOC]
        SOC.extend([self.SOC[t,0].x for t in self.time_horizon])
        


        dataset['soc'] = SOC
        dataset['avail'] = list(self.EV_availability)
        dataset['episode'] = self.episode

            
        return dataset
    
    def results_stochastic(self):
        
        dataset = pd.DataFrame(index = self.time_vector)
        dataset['pv'] = self.data_EV.loc[self.time_vector, 'PV']
        dataset['load'] = self.data_EV.loc[self.time_vector, 'load']

        for var in self.variables:
            var_1 = [self.m.getVarByName(f'{var}_1').x]
            var_2 = [np.sum([self.m.getVarByName(f'{var}[{t},{i}]').x for i in self.range_samples])/self.n_samples for t in self.time_horizon]
            var_1.extend(var_2)
            dataset[var] = var_1
        
        SOC = [self.current_SOC]
        SOC.extend([np.sum([self.SOC[t,i].x for i in self.range_samples])/self.n_samples for t in self.time_horizon])

        dataset['soc'] = SOC
        dataset['avail'] = list(self.EV_availability)
        dataset['episode'] = self.episode

        return dataset

    
    def day_ahead_update(self):
        
        actual_PV = self.data_EV.loc[self.time_vector, 'PV']
        
        if self.stochastic:
            decisions = self.results_stochastic()
        else:
            decisions = self.results_deterministic()
           

        data_real = {'pv_ev':[], 'pv_load':[], 'grid_ev':[],'grid_load':[]}
        
        for t in decisions.index:
            
               pv_ev = decisions.loc[t,'pv_ev']
               grid_ev = decisions.loc[t,'grid_ev']
               pv_load = decisions.loc[t,'pv_load']
               pv_grid = decisions.loc[t,'pv_grid']
               avail = decisions.loc[t,'avail']
               
               p_ev = pv_ev + grid_ev
               load = decisions.loc[t,'load']
               delta = decisions.loc[t,'delta_pv']
               pv = decisions.loc[t,'pv_real']
               
               if delta < 0:
                   delta = -delta
                   # less pv than expected
                   # reduce selling
                   #data_real['pv_grid'].append(max(0, pv_grid-delta))
                   delta = max(0,delta - pv_grid)
                
                   # reduce pv to load
                   data_real['pv_load'].append(max(0, pv_load - delta))
                   data_real['grid_load'].append(load - max(0, pv_load - delta))
                   delta = max(0, delta - pv_load)
                    
                   # switch EV charging
                   if avail > 0:
                       if pv >= p_ev:
                           data_real['pv_ev'].append(p_ev)
                           data_real['grid_ev'].append(0)
                       else:
                           data_real['pv_ev'].append(0)
                           data_real['grid_ev'].append(p_ev)
                   else:
                       data_real['pv_ev'].append(0)
                       data_real['grid_ev'].append(0)
  
               else:
                   # check if we can switch car to ev
                   if avail > 0:
                       if pv >= p_ev:
                           pv_ev = p_ev
                           data_real['pv_ev'].append(pv_ev)
                           data_real['grid_ev'].append(0)
                           
                       else:
                           data_real['pv_ev'].append(0)
                           data_real['grid_ev'].append(p_ev)
                           pv_ev = 0
                   else:
                       data_real['pv_ev'].append(0)
                       data_real['grid_ev'].append(0)
                       pv_ev = 0
                   
                   delta = pv - pv_ev
                   
                   data_real['pv_load'].append(min(load, delta))
                   data_real['grid_load'].append(load - min(load, delta))

                   #data_real['pv_grid'].append(rest)

        decisions['pv_ev_real'] = data_real['pv_ev']
        decisions['grid_ev_real'] = data_real['grid_ev']
        decisions['pv_load_real'] = data_real['pv_load']
        decisions['grid_load_real'] = data_real['grid_load']
        
        actual_cost = np.sum(data_real['grid_ev']) + np.sum(data_real['grid_load'])
        
        return actual_cost, decisions
    


