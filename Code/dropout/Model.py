# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:29:01 2020

@author: Yann
"""


import gurobipy as gp
from gurobipy import GRB
import plot_res as pr

class EV_model:
    def __init__(self, BC_EV = 40e3, eta_EV_ch = 0.9, 
                        P_EV_chmax = 5e3, SOC_min = 0.2, SOC_max = 1, V2G = False ):
        
        self.BC_EV = BC_EV
        self.eta_EV_ch = eta_EV_ch
        self.P_EV_chmax = P_EV_chmax
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        
        
class House:
    def __init__(self, installed_PV = 10000, pv_capacity = 2835, **kwargs):
        self.installed_PV = installed_PV
        self.pv_capacity = pv_capacity
        self.scale_PV = float(installed_PV/pv_capacity)
        
class Model:
    
    def __init__(self, name, data_full, t_start, t_res, House, EV):

        self.name = name
        
        self.set_house_parameter(House)

        self.set_EV_parameters(EV)
        
        self.data_full = data_full
        
        self.col_names = ['pv', 'pv_forecast', 'load', 'pv_ev', 'pv_load', 'grid_ev', 'grid_load',
       'soc', 'avail']
        
        self.t_start = t_start
        
        t_previous = t_start - pd.Timedelta(hours = t_res)
        
        previous_decision = {self.col_names[i]: [0] for i in range(len(self.col_names))}
        previous_decision['soc'] = self.data_full.loc[t_previous, 'EV_SOC']
        
        self.decisions = pd.DataFrame(data = previous_decision, index = [t_previous])
        
        
    def set_house_parameter(self, House):
        self.scale_PV = House.scale_PV
    
    def set_EV_parameters(self, EV):
        self.BC_EV = EV.BC_EV
        self.eta_EV_ch = EV.eta_EV_ch
        self.P_EV_chmax = EV.P_EV_chmax
        self.SOC_min = EV.SOC_min
        self.SOC_max = EV.SOC_max
        
        
        
    def set_parameters(self):
        
        self.EV_Availability = self.data.EV_availability
        
        self.load = self.data.load
        self.PV_forecast = self.data.PV_forecast
        
        self.prepare_time()
        self.SOC_ini_fixed = {t: self.data_full.loc[t, 'EV_SOC'] for t in self.T_arrival}
        #self.SOC_ini_fixed = {t: self.data_full.loc[t, 'EV_SOC'] for t in self.T_away}
        self.SOC_ini_fixed[self.t_previous] = self.decisions.loc[self.t_previous, 'soc']
        
                
        
    def prepare_time(self):
        
        self.t_previous = self.decisions.index[-1]
        
        self.time_vector = list(self.data.index)
        self.t_start = self.time_vector[0]
        self.t_end = self.time_vector[-1]
        self.t_res = int((self.time_vector[1] - self.time_vector[0]).seconds)/3600
        
        time_vector_extended = self.time_vector.copy()
        time_vector_extended.append(self.time_vector[0]-pd.Timedelta(hours = self.t_res))
        time_vector_extended.sort()
        self.time_vector_extended = time_vector_extended
        
        self.T_arrival = list(self.data.index[i] for i in range(self.data.shape[0]-1)
                      if (self.EV_Availability[i+1] == 1 and self.EV_Availability[i] ==0))
       
        
        self.T_departure = list(self.data.index[i] for i in range(self.data.shape[0]-1)
                     if (self.EV_Availability[i+1] == 0 and self.EV_Availability[i] ==1))
        
        self.T_away = list(self.data.index[i] for i in range(self.data.shape[0]-1)
                     if self.EV_Availability[i] == 0)
    
        
    def add_variables(self):
        
        self.P_PV = {self.time_vector[i]:self.PV_forecast[i] for i in range(len(self.time_vector))}
        self.P_PV_2EV = self.m.addVars(self.time_vector, name = 'P_PV_2EV')
        self.P_PV_2L = self.m.addVars(self.time_vector, name = 'P_PV_2L')
        self.P_PV_2G = self.m.addVars(self.time_vector, name = 'P_PV_2G')
        
        
        self.P_load = {self.time_vector[i]:self.load[i] for i in range(len(self.time_vector))}
        
        
        self.SOC = self.m.addVars(self.time_vector_extended,lb = self.SOC_min, ub = self.SOC_max, name = 'SOC')
        self.P_EV = self.m.addVars(self.time_vector,lb = 0, ub = self.P_EV_chmax, name = 'P_EV')
        
        self.P_grid_bought = self.m.addVars(self.time_vector, lb = 0, name = 'P_grid_bought')
        self.P_grid_sold = self.m.addVars(self.time_vector, name = 'P_grid_sold')
        self.y_grid = self.m.addVars(self.time_vector, vtype=GRB.BINARY, name = 'y_grid')
        
        self.P_grid_2EV = self.m.addVars(self.time_vector, name = 'P_grid_2EV')
        self.P_grid_2L = self.m.addVars(self.time_vector, name = 'P_grid_2L')
        
        self.y_grid_2EV = self.m.addVars(self.time_vector, vtype=GRB.BINARY, name = 'y_grid_2EV')
    
    def add_constraints(self):
        self.Grid_balance = self.m.addConstrs(self.P_PV[t] + self.y_grid[t] * self.P_grid_bought[t] - 
                                    (1-self.y_grid[t]) * self.P_grid_sold[t] 
                                    - self.EV_Availability[t]*self.P_EV[t] 
                                    - self.P_load[t] == 0 
                                    for t in self.time_vector)
        
        self.PV_balance = self.m.addConstrs(self.P_PV[t] == self.P_PV_2EV[t] + self.P_PV_2L[t] + self.P_PV_2G[t] 
                                  for t in self.time_vector)
        
        self.Grid_bought_balance = self.m.addConstrs(self.P_grid_bought[t] == (self.y_grid_2EV[t])*self.P_grid_2EV[t] 
                                            + self.P_grid_2L[t]
                                            for t in self.time_vector)
        self.EV_balance = self.m.addConstrs(self.P_EV[t] == (self.y_grid_2EV[t])*self.P_grid_2EV[t]
                                  + (1-self.y_grid_2EV[t])*self.P_PV_2EV[t]
                                  for t in self.time_vector)
        
        
        
        self.SOC_update = self.m.addConstrs(self.SOC[self.time_vector_extended[i]] == 
                                  self.SOC[self.time_vector_extended[i-1]] +
                                  self.EV_Availability[self.time_vector_extended[i]] * (self.P_EV[self.time_vector_extended[i]]/self.BC_EV) * self.eta_EV_ch
                                  for i in range(1,len(self.time_vector_extended)) 
                                  if self.time_vector_extended[i] not in self.T_arrival)
        
        self.SOC_dep = self.m.addConstrs(self.SOC[t] >= 1 for t in self.T_departure)
        
        self.SOC_fixed = self.m.addConstrs(self.SOC[t] == self.SOC_ini_fixed[t] for t in self.SOC_ini_fixed.keys())
        
    
    def optimize(self, t_decision, df_forecasted):
        
        self.t_decision = t_decision
        self.data = df_forecasted
        
        self.m = gp.Model(self.name)

        self.set_parameters()
        self.add_variables()
        self.add_constraints()
        
        self.E_bought = gp.quicksum(self.P_grid_bought[t] for t in self.time_vector)
        
        # if goal is PV self consumption
        self.E_sold = gp.quicksum(self.P_grid_sold[t] for t in self.time_vector)
        
        self.m.setObjective(self.E_bought)
        
        self.m.write('model_1.lp')
        self.m.optimize()
        
        if self.m.status == GRB.INFEASIBLE:
            self.m.feasRelaxS(0, False, False,True)
            self.m.optimize()
    
    def results(self, plot_results = True):
        
        PV = list(self.data['PV'])
        PV_forecast = list(self.PV_forecast)
        Load = list(self.load)
        EV_availability = list(self.EV_Availability)
        
        PV_2EV = [self.P_PV_2EV[t].x for t in self.time_vector]
        PV_2L = [self.P_PV_2L[t].x for t in self.time_vector]
        PV_2G = [self.P_PV_2G[t].x for t in self.time_vector]
        
        Grid_2EV = [self.P_grid_2EV[t].x for t in self.time_vector]
        Grid_2L = [self.P_grid_2L[t].x for t in self.time_vector]
        Grid_bought = [self.P_grid_bought[t].x for t in self.time_vector]
        
        SOC_f = [self.SOC[t].x for t in self.time_vector]
        
        dataset = pd.DataFrame(index = self.time_vector)
        dataset['pv'] = PV
        dataset['pv_forecast'] = PV_forecast
        dataset['load'] = Load
        dataset['pv_ev'] = PV_2EV
        dataset['pv_load'] = PV_2L
        dataset['grid_ev'] = Grid_2EV
        dataset['grid_load'] = Grid_2L
        dataset['soc'] = SOC_f
        dataset['avail'] = EV_availability

        decision = pd.Series(dataset.loc[self.t_decision], name = self.t_decision)
        
        self.decisions = self.decisions.append(decision, ignore_index=False)
            
        return dataset
        

