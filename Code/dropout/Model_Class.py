# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:29:01 2020

@author: Yann
"""


import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np


class EV:
    def __init__(self, BC_EV = 40e3, eta_EV_ch = 0.9, 
                        P_EV_chmax = 5e3, SOC_min = 0.2, SOC_max = 1,SOC_min_departure = 0.8, V2G = False ):
        
        self.BC_EV = BC_EV
        self.eta_EV_ch = eta_EV_ch
        self.P_EV_chmax = P_EV_chmax
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.SOC_min_departure = SOC_min_departure
        
        
class House:
    def __init__(self, installed_PV = 10000, pv_capacity = 2835, **kwargs):
        self.installed_PV = installed_PV
        self.pv_capacity = pv_capacity
        self.scale_PV = float(installed_PV/pv_capacity)
        
class Model:
    
    def __init__(self, name, data_EV, t_start, t_res, House, EV):

        self.name = name
        
        self.set_house_parameter(House)

        self.set_EV_parameters(EV)
        
        self.data_EV = data_EV
        
        self.col_names = ['pv', 'pv_forecast', 'load', 'pv_ev', 'pv_load', 'grid_ev', 'grid_load',
       'soc', 'avail']
        
        self.t_start = t_start
        
        t_previous = t_start - pd.Timedelta(hours = t_res)
        
        previous_decision = {self.col_names[i]: [0] for i in range(len(self.col_names))}
        previous_decision['soc'] = self.data_EV.loc[t_previous, 'EV_SOC']
        
        self.decisions = pd.DataFrame(data = previous_decision, index = [t_previous])
        
        self.constraints_violation = {}

        
    def set_house_parameter(self, House):
        self.scale_PV = House.scale_PV
    
    def set_EV_parameters(self, EV):
        self.BC_EV = EV.BC_EV
        self.eta_EV_ch = EV.eta_EV_ch
        self.P_EV_chmax = EV.P_EV_chmax
        self.SOC_min = EV.SOC_min
        self.SOC_max = EV.SOC_max
        self.SOC_min_departure = EV.SOC_min_departure
        
        
    def set_parameters(self):
        
        self.EV_Availability = self.data.EV_availability
        
        self.prepare_time()
        
        self.P_PV = {t: list(self.predictions.loc[t]) for t in self.time_horizon}

            
        self.P_PV[self.t_decision] = self.realization
        
        self.load = self.data.load
        self.P_load = {self.time_vector[i]:self.load[i] for i in range(len(self.time_vector))}
        
        SOC_ini_fixed = {}
        SOC_ini_fixed[self.t_previous] = self.decisions.loc[self.t_previous, 'soc']
        
        for t in self.T_away:
            SOC_ini_fixed[t] = self.data_EV.loc[t, 'EV_SOC']
        
        # for t in self.T_departure:
            # SOC_ini_fixed[t] = self.SOC_max 
        
        self.SOC_ini_fixed = SOC_ini_fixed
        
        
        
    def prepare_data(self, t_decision, predictions, pred_variable):
        
        t_end = predictions.index[-1]
        
        self.realization = self.data_EV.loc[t_decision, pred_variable]
        
        if self.forecasting:
        
            self.predictions = predictions*self.scale_PV
            
        else:
            self.predictions = predictions

        self.n_samples = predictions.shape[1]

        self.range_samples = range(self.n_samples)
        
        self.t_decision = t_decision
        
        self.data = self.data_EV[t_decision:t_end]
        
        self.data.drop(pred_variable, axis = 1, inplace = True)           
        
    def prepare_time(self):
        
        self.t_previous = self.decisions.index[-1]
        print(self.t_previous)
        
        self.time_vector = list(self.data.index)
        
        self.t_start = self.time_vector[0]
        self.t_end = self.time_vector[-1]
        self.t_res = int((self.time_vector[1] - self.time_vector[0]).seconds)/3600
        
        self.time_horizon = self.time_vector[1:]
        
        self.T_arrival = list(self.data.index[i] for i in range(self.data.shape[0]-1)
                      if (self.EV_Availability[i+1] == 1 and self.EV_Availability[i] ==0))
       
        
        self.T_departure = list(self.data.index[i] for i in range(self.data.shape[0]-1)
                     if (self.EV_Availability[i+1] == 0 and self.EV_Availability[i] ==1))
        
        self.T_away = list(self.data.index[i] for i in range(self.data.shape[0]-1)
                     if self.EV_Availability[i] == 0)
        
        Time_parked = list(self.data.index[i] for i in range(self.data.shape[0]-1)
                     if self.EV_Availability[i] == 1)
        
        Time_parked.extend(self.T_arrival)
        
        self.time_parked = Time_parked
       
    
    def add_first_stage_variables(self):
        
        self.P_PV_2EV_1 = self.m.addVar()
        self.P_PV_2L_1 = self.m.addVar()
        self.P_PV_2G_1 = self.m.addVar()

        
        self.SOC_1 = self.m.addVar(lb = self.SOC_min, ub = self.SOC_max)
        self.P_EV_1 = self.m.addVar(ub = self.P_EV_chmax)
        
        self.P_grid_bought_1 = self.m.addVar()
        self.P_grid_sold_1 = self.m.addVar()
        self.y_grid_1 = self.m.addVar(vtype=GRB.BINARY)
        
        self.P_grid_2EV_1 = self.m.addVar()
        self.P_grid_2L_1 = self.m.addVar()
        
        self.y_grid_2EV_1 = self.m.addVar(vtype=GRB.BINARY)
        
    def add_first_stage_constraints(self):
        
        t = self.t_decision
        
        self.Grid_balance_1 = self.m.addConstr(self.P_PV[t] + self.y_grid_1 * self.P_grid_bought_1 - 
                                    (1-self.y_grid_1) * self.P_grid_sold_1 
                                    - self.EV_Availability[t]*self.P_EV_1 
                                    - self.P_load[t] == 0)
        
        self.PV_balance_1 = self.m.addConstr(self.P_PV[t] == self.P_PV_2EV_1 + self.P_PV_2L_1 + self.P_PV_2G_1)
        
        self.Grid_bought_balance_1 = self.m.addConstr(self.P_grid_bought_1 == (self.y_grid_2EV_1)*self.P_grid_2EV_1 
                                            + self.P_grid_2L_1)
        
        self.EV_balance_1 = self.m.addConstr(self.P_EV_1 == (self.y_grid_2EV_1)*self.P_grid_2EV_1
                                  + (1-self.y_grid_2EV_1)*self.P_PV_2EV_1)
        
        if t in self.SOC_ini_fixed.keys():
            print('fixed')
            self.SOC_fixed_1 = self.m.addConstr(self.SOC_1 == self.SOC_ini_fixed[t])
        
        else:
            print('not fixed')
            self.SOC_update_1 = self.m.addConstr(self.SOC_1 == 
                                      self.SOC_ini_fixed[self.t_previous] + self.EV_Availability[t] * 
                                      (self.P_EV_1/self.BC_EV) * self.eta_EV_ch)
            
            if t in self.T_departure:
                self.SOC_dep_1 = self.m.addConstr(self.SOC_1 >= self.SOC_min_departure)
        
        
    def add_second_stage_variables(self):
        
        self.P_PV_2EV = self.m.addVars(self.time_horizon, self.n_samples)
        self.P_PV_2L = self.m.addVars(self.time_horizon, self.n_samples)
        self.P_PV_2G = self.m.addVars(self.time_horizon, self.n_samples)        
        
        self.SOC = self.m.addVars(self.time_vector, self.n_samples,lb = self.SOC_min, ub = self.SOC_max)
        self.P_EV = self.m.addVars(self.time_horizon, self.n_samples,lb = 0, ub = self.P_EV_chmax)
        
        self.P_grid_bought = self.m.addVars(self.time_horizon, self.n_samples, lb = 0)
        self.P_grid_sold = self.m.addVars(self.time_horizon, self.n_samples)
        self.y_grid = self.m.addVars(self.time_horizon, self.n_samples, vtype=GRB.BINARY)
        
        self.P_grid_2EV = self.m.addVars(self.time_horizon, self.n_samples)
        self.P_grid_2L = self.m.addVars(self.time_horizon, self.n_samples)
        
        self.y_grid_2EV = self.m.addVars(self.time_horizon, self.n_samples, vtype=GRB.BINARY)
    
    def add_second_stage_constraints(self):
        
        
        self.Grid_balance = self.m.addConstrs(self.P_PV[t][i] + self.y_grid[t,i] * self.P_grid_bought[t,i] - 
                                    (1-self.y_grid[t,i]) * self.P_grid_sold[t,i] 
                                    - self.EV_Availability[t]*self.P_EV[t,i] 
                                    - self.P_load[t] == 0 
                                    for t in self.time_horizon for i in self.range_samples)
        
        self.PV_balance = self.m.addConstrs(self.P_PV[t][i] == self.P_PV_2EV[t,i] + self.P_PV_2L[t,i] + self.P_PV_2G[t,i] 
                                  for t in self.time_horizon for i in self.range_samples)
        
        self.Grid_bought_balance = self.m.addConstrs(self.P_grid_bought[t,i] == (self.y_grid_2EV[t,i])*self.P_grid_2EV[t,i] 
                                            + self.P_grid_2L[t,i]
                                            for t in self.time_horizon for i in self.range_samples)
        
        self.EV_balance = self.m.addConstrs(self.P_EV[t,i] == (self.y_grid_2EV[t,i])*self.P_grid_2EV[t,i]
                                  + (1-self.y_grid_2EV[t,i])*self.P_PV_2EV[t,i]
                                  for t in self.time_horizon for i in self.range_samples)
        
        
        self.SOC_update = self.m.addConstrs(self.SOC[self.time_vector[t],i] == 
                                  self.SOC[self.time_vector[t-1],i] +
                                  (self.P_EV[self.time_vector[t],i]/self.BC_EV) * self.eta_EV_ch
                                  for t in range(1,len(self.time_vector)) for i in self.range_samples 
                                  if self.time_vector[t] not in self.T_away)
        
        self.SOC_dep = self.m.addConstrs(self.SOC[t,i] >= self.SOC_min_departure for t in self.T_departure for i in self.range_samples)
        
        self.SOC_fixed = self.m.addConstrs(self.SOC[t,i] == self.SOC_ini_fixed[t] for t in self.SOC_ini_fixed.keys() 
                                           for i in self.range_samples if t > self.t_previous)
        
        self.SOC_continuity = self.m.addConstrs(self.SOC[self.time_vector[0],j] == self.SOC_1 for j in self.range_samples)
        
    def optimize(self, t_decision, pred_variable, predictions, forecasting = True, method = 'deterministic', parameters = None, OutputFlag = 0):
        
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', OutputFlag)
            env.start()
            with gp.Model(env=env) as self.m:
        
                if predictions.shape[1] == 1:
                    self.stochastic = False
                    self.m = gp.Model(self.name+'_deterministic')
                else:
                    self.stochastic = True
                    self.m = gp.Model(self.name+'_stochastic')
                
                self.forecasting = forecasting
                self.prepare_data(t_decision, predictions, pred_variable)
                self.set_parameters()
                # first_stage
                self.add_first_stage_variables()
                self.add_first_stage_constraints()
                
                self.add_second_stage_variables()
                self.add_second_stage_constraints()
                
                if  method == 'deterministic' or method == 'day_ahead' or method == 'expected value':
                    self.E_bought = (gp.quicksum(self.P_grid_bought)/self.n_samples)
                    
                elif method == 'CVaR':
                    alpha = parameters['alpha']
                    self.E_bought = (gp.quicksum(self.P_grid_bought)/((1-alpha) *self.n_samples))
                
                elif method == 'Markowitz':
                    alpha = parameters['alpha']
                    beta = parameters['beta']
                    self.E_bought = (gp.quicksum(self.P_grid_bought)*beta/((1-alpha( *self.n_samples))))
                
                self.m.setObjective(self.P_grid_bought_1 + self.E_bought)
                
                self.m.write('model_1.lp')
                self.m.optimize()
                
                if self.m.status == GRB.INFEASIBLE:
                    print('Model infasible, relaxing SOC constraint')
                    self.m.feasRelaxS(2, False, False,True)
                    self.m.optimize()
                    self.constraints_violation[t_decision] = self.SOC_1
                    
                self.update_decisions()

                self.cost = self.m.ObjVal

        
    def update_decisions(self):
        
        PV_2EV = self.P_PV_2EV_1.x
        PV_2L = self.P_PV_2L_1.x
        PV_2G = self.P_PV_2G_1.x 
        
        Grid_2EV = self.P_grid_2EV_1.x
        Grid_2L = self.P_grid_2L_1.x 
        Grid_bought = self.P_grid_bought_1.x 
        
        SOC_f = self.SOC_1.x 
        
        dataset = pd.DataFrame(index = self.time_vector)
        dataset['pv'] = self.realization
        dataset['pv_forecast'] = self.realization
        dataset['load'] = list(self.load)
        dataset['pv_ev'] = PV_2EV
        dataset['pv_load'] = PV_2L
        dataset['grid_ev'] = Grid_2EV
        dataset['grid_load'] = Grid_2L
        dataset['soc'] = SOC_f
        dataset['avail'] = list(self.EV_Availability)

        decision = pd.Series(dataset.loc[self.t_decision], name = self.t_decision)
        
        self.decisions = self.decisions.append(decision, ignore_index=False)
    
    def model_decisions(self):
        
        return self.decisions[1:]
    
    def predictions_SOC(self):
        
        df_SOC = pd.DataFrame(index = self.time_vector, columns=self.range_samples)
        
        for t,i in self.SOC:
            df_SOC.loc[t,i] = self.SOC[t,i].x
            
        return df_SOC
    
    def results_deterministic(self):
        
        PV = [self.realization]
        if self.forecasting:
            PV.extend(np.concatenate(list(self.predictions.values)))
        else:
            PV.extend((list(self.predictions.values)))
        
        Load = list(self.load)
        EV_availability = list(self.EV_Availability)
        
        PV_2EV = [self.P_PV_2EV_1.x]
        PV_2EV.extend([self.P_PV_2EV[t,0].x for t in self.time_horizon])
        PV_2L = [self.P_PV_2L_1.x]
        PV_2L.extend([self.P_PV_2L[t,0].x for t in self.time_horizon])
        PV_2G = [self.P_PV_2G_1.x]
        PV_2G.extend([self.P_PV_2G[t,0].x for t in self.time_horizon])
        
        Grid_2EV = [self.P_grid_2EV_1.x]
        Grid_2EV.extend([self.P_grid_2EV[t,0].x for t in self.time_horizon])
        Grid_2L = [self.P_grid_2L_1.x]
        Grid_2L.extend([self.P_grid_2L[t,0].x for t in self.time_horizon])

        Grid_bought = [self.P_grid_bought_1.x]
        Grid_bought.extend([self.P_grid_bought[t,0].x for t in self.time_horizon])
        
        SOC_f = [self.SOC[t,0].x for t in self.time_vector]
        
        dataset = pd.DataFrame(index = self.time_vector)
        dataset['pv_real'] = self.data_EV.loc[self.time_vector, 'PV']
        dataset['pv_forecast'] = PV
        dataset['delta_pv'] =  dataset['pv_real'] -dataset['pv_forecast']
        dataset['pv_grid'] = PV_2G

        dataset['load'] = Load
        dataset['pv_load'] = PV_2L
        dataset['grid_load'] = Grid_2L

        dataset['grid_ev'] = Grid_2EV
        dataset['pv_ev'] = PV_2EV
        dataset['soc'] = SOC_f
        dataset['avail'] = EV_availability
            
        return dataset
    
    def results_stochastic(self):
        
        Load = list(self.load)
        EV_availability = list(self.EV_Availability)
        
        PV = [self.realization]
        PV.extend([np.mean(self.predictions.loc[t,:]) for t in self.time_horizon])
        
        PV_2EV = [self.P_PV_2EV_1.x]
        PV_2EV.extend([np.sum([self.P_PV_2EV[t,i].x for i in self.range_samples])/self.n_samples for t in self.time_horizon])
        
        PV_2L = [self.P_PV_2L_1.x]
        PV_2L.extend([np.sum([self.P_PV_2L[t,i].x for i in self.range_samples])/self.n_samples for t in self.time_horizon])
        
        PV_2G = [self.P_PV_2G_1.x]
        PV_2G.extend([np.sum([self.P_PV_2G[t,i].x for i in self.range_samples])/self.n_samples for t in self.time_horizon])
        
        Grid_2EV = [self.P_grid_2EV_1.x]
        Grid_2EV.extend([np.sum([self.P_grid_2EV[t,i].x for i in self.range_samples])/self.n_samples for t in self.time_horizon])
        
        Grid_2L = [self.P_grid_2L_1.x]
        Grid_2L.extend([np.sum([self.P_grid_2L[t,i].x for i in self.range_samples])/self.n_samples for t in self.time_horizon])
        
        SOC = [self.SOC_1.x]
        SOC.extend([np.sum([self.SOC[t,i].x for i in self.range_samples])/self.n_samples for t in self.time_horizon])
 
        dataset = pd.DataFrame(index = self.time_vector)
        
        dataset = pd.DataFrame(index = self.time_vector)
        dataset['pv_real'] = self.data_EV.loc[self.time_vector, 'PV']
        dataset['pv_forecast'] = PV
        dataset['delta_pv'] = dataset['pv_real'] -dataset['pv_forecast']
        dataset['pv_grid'] = PV_2G

        dataset['load'] = Load
        dataset['pv_load'] = PV_2L
        dataset['grid_load'] = Grid_2L

        dataset['grid_ev'] = Grid_2EV
        dataset['pv_ev'] = PV_2EV
        dataset['soc'] = SOC
        dataset['avail'] = EV_availability


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
    
    