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
    def __init__(self, BC_EV = 16e3, eta_EV_ch = 0.95, 
                        P_EV_chmax = 3.7e3, SOC_min = 0.2, SOC_max = 1,SOC_min_departure = 1, V2G = False ):
        
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
        
        self.col_names = ['pv', 'load', 'pv_ev', 'pv_load', 'pv_grid', 'grid_ev', 'grid_load',
       'soc', 'avail', 'episode']
        
        self.t_start = t_start
        
        self.EV = EV
        previous_decision = {self.col_names[i]: [0] for i in range(len(self.col_names))}
        previous_decision['soc'] = self.data_EV.loc[t_start, 'soc']
        
        self.decisions = pd.DataFrame(data = previous_decision, index = [t_start])
        
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
        
        self.EV_availability = self.data.EV_availability
        
        self.prepare_time()
        
        self.P_PV = {t: list(self.predictions_PV.loc[t]) for t in self.time_horizon}
        

        self.P_PV[self.t_decision] = self.realization_PV
        
        self.P_load = {t: list(self.predictions_load.loc[t]) for t in self.time_horizon}
        
        self.P_load[self.t_decision] = self.realization_load
                
        
        
    def prepare_data(self, t_decision, t_end, predictions_PV, predictions_load):
        
        self.data = self.data_EV[t_decision:t_end]
        
        self.episode = self.data_EV.loc[t_decision, 'episode']
        
        self.delta = t_end - t_decision
        
        self.duration = int(self.delta.days * 24 + self.delta.seconds/3600) 
        
        self.realization_PV = self.data_EV.loc[t_decision, 'PV']
        
        self.realization_load = self.data_EV.loc[t_decision, 'load']
        
        if self.forecasting:
        
            self.predictions_PV = predictions_PV[:self.duration]*self.scale_PV
            
            
        else:
            self.predictions_PV = predictions_PV[:self.duration]
            
        self.predictions_load = predictions_load[:self.duration]
        
        self.n_samples = predictions_PV.shape[1]

        self.range_samples = range(self.n_samples)
        
        self.t_decision = t_decision
        
        
    def prepare_time(self):
        
        
        self.time_vector = list(self.data.index)
        
        self.t_start = self.time_vector[0]
        self.t_end = self.time_vector[-1]
        self.t_res = int((self.time_vector[1] - self.time_vector[0]).seconds)/3600
        
        self.time_horizon = self.time_vector[1:]
        self.t_forecast = self.time_horizon[0]
        
        self.time_soc = self.time_horizon[1:]
        
        try:
            self.t_departure = self.data[self.data.EV_availability == 0].index[0]
        except:
           self.t_departure = self.data.index[-1]
    
    def add_first_stage_variables(self):
        
        self.P_PV_2EV_1 = self.m.addVar()
        self.P_PV_2L_1 = self.m.addVar()
        self.P_PV_2G_1 = self.m.addVar()

        self.current_SOC = self.decisions.loc[self.t_decision, 'soc']
        self.SOC_2 = self.m.addVar(lb = self.SOC_min, ub = self.SOC_max)
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
                                    - self.EV_availability[t] * self.P_EV_1 
                                    - self.P_load[t] == 0, name = 'Grid_balance_1')
        
        self.PV_balance_1 = self.m.addConstr(self.P_PV[t] == self.P_PV_2EV_1 + self.P_PV_2L_1 + self.P_PV_2G_1, name = 'PV_balance_1')
        
        self.Load_balance_1 = self.m.addConstr((self.P_load[t] == self.P_PV_2L_1+ self.P_grid_2L_1),
                                  name = 'Load_balance_1')
        
        self.Grid_bought_balance_1 = self.m.addConstr(self.P_grid_bought_1 == self.P_grid_2EV_1 
                                            + self.P_grid_2L_1, name = 'Grid_bought_balance_1')
        
        self.Grid_sold_balance_1 = self.m.addConstr(self.P_grid_sold_1 == self.P_PV_2G_1, name = 'Grid_sold_balance_1')
        
        self.EV_balance_1 = self.m.addConstr(self.EV_availability[t] * self.P_EV_1 == (self.y_grid_2EV_1)*self.P_grid_2EV_1
                                  + (1-self.y_grid_2EV_1)*self.P_PV_2EV_1, name = 'EV_balance_1')
        

        # update soc for t_forecast, depending on first stage decisions
        self.SOC_update_1 = self.m.addConstr(self.SOC_2 == 
                                  self.current_SOC + self.EV_availability[t] * 
                                  (self.P_EV_1/self.BC_EV) * self.eta_EV_ch , name = 'SOC_update_1')
        
        # if self.forecasting == False:
        #     self.SOC_dep = self.m.addConstrs((self.SOC[self.t_departure,i] >= self.SOC_min_departure for i in self.range_samples if self.t_forecast == self.t_departure), name = 'SOC_departure_1')
        
        
    def add_second_stage_variables(self):
        
        self.P_PV_2EV = self.m.addVars(self.time_horizon, self.n_samples)
        self.P_PV_2L = self.m.addVars(self.time_horizon, self.n_samples)
        self.P_PV_2G = self.m.addVars(self.time_horizon, self.n_samples)        
        
        self.SOC = self.m.addVars(self.time_horizon, self.n_samples,lb = self.SOC_min, ub = self.SOC_max)
        self.P_EV = self.m.addVars(self.time_horizon, self.n_samples,lb = 0, ub = self.P_EV_chmax)
        
        self.P_grid_bought = self.m.addVars(self.time_horizon, self.n_samples, lb = 0)
        self.P_grid_sold = self.m.addVars(self.time_horizon, self.n_samples)
        self.y_grid = self.m.addVars(self.time_horizon, self.n_samples, vtype=GRB.BINARY)
        
        self.P_grid_2EV = self.m.addVars(self.time_horizon, self.n_samples)
        self.P_grid_2L = self.m.addVars(self.time_horizon, self.n_samples)
        
        self.y_grid_2EV = self.m.addVars(self.time_horizon, self.n_samples, vtype=GRB.BINARY)
    
    def add_second_stage_constraints(self):
        
        
        self.Grid_balance = self.m.addConstrs((self.P_PV[t][i] + self.y_grid[t,i] * self.P_grid_bought[t,i] - 
                                    (1-self.y_grid[t,i]) * self.P_grid_sold[t,i] 
                                    - self.EV_availability[t] * self.P_EV[t,i] 
                                    - self.P_load[t][0] == 0 
                                    for t in self.time_horizon for i in self.range_samples), name = 'Grid_balance')
        
        self.PV_balance = self.m.addConstrs((self.P_PV[t][i] == self.P_PV_2EV[t,i] + self.P_PV_2L[t,i] + self.P_PV_2G[t,i] 
                                  for t in self.time_horizon for i in self.range_samples), name = 'PV_balance')
        
        
        self.Load_balance = self.m.addConstrs((self.P_load[t][0] == self.P_PV_2L[t,i] + self.P_grid_2L[t,i] 
                                  for t in self.time_horizon for i in self.range_samples), name = 'Load_balance')
        
        self.Grid_bought_balance = self.m.addConstrs((self.P_grid_bought[t,i] == self.P_grid_2EV[t,i] 
                                            + self.P_grid_2L[t,i]
                                            for t in self.time_horizon for i in self.range_samples), name = 'Grid_bought_balance')
        self.Grid_sold_balance = self.m.addConstrs((self.P_grid_sold[t,i] == self.P_PV_2G[t,i]
                                            for t in self.time_horizon for i in self.range_samples), name = 'Grid_bought_balance')
        
        self.EV_balance = self.m.addConstrs((self.EV_availability[t] * self.P_EV[t,i] == (self.y_grid_2EV[t,i])*self.P_grid_2EV[t,i]
                                  + (1-self.y_grid_2EV[t,i])*self.P_PV_2EV[t,i]
                                  for t in self.time_horizon for i in self.range_samples), name = 'EV_balance')
        
        self.SOC_continuity = self.m.addConstrs((self.SOC[self.time_horizon[0],j] == self.SOC_2 for j in self.range_samples), name = 'SOC_continuity')
        
        
        
        self.SOC_update = self.m.addConstrs((self.SOC[self.time_horizon[t+1],i] == 
                                  self.SOC[self.time_horizon[t],i] + self.EV_availability[self.time_horizon[t]]
                                  *(self.P_EV[self.time_horizon[t],i]/self.BC_EV) * self.eta_EV_ch
                                  for t in range(len(self.time_horizon)-1) for i in self.range_samples 
                                  ), name = 'SOC_update')
        
        # if self.forecasting == False:
        #     self.SOC_dep = self.m.addConstrs((self.SOC[self.t_departure,i] >= self.SOC_min_departure for i in self.range_samples if self.t_departure > self.t_decision), name = 'SOC_departure')
        
        
        
        
    def optimize(self, t_decision, t_end, predictions_PV, predictions_load, forecasting = True, method = 'deterministic', parameters = None, OutputFlag = 0):
        
        if predictions_PV.shape[1] == 1:
            self.stochastic = False
            self.m = gp.Model(self.name+'_deterministic')
        else:
            self.stochastic = True
            self.m = gp.Model(self.name+'_stochastic')
        
        self.forecasting = forecasting
        self.prepare_data(t_decision,t_end, predictions_PV, predictions_load)
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
        
        
        # if self.t_decision < self.t_departure and self.forecasting:
            
        self.SOC_difference =  gp.quicksum([self.SOC_min_departure - self.SOC[self.t_end,i] for i in self.range_samples])/self.n_samples
        self.Power_difference = self.SOC_difference*self.BC_EV
        penalty = 1.1
        
        self.m.setObjective(self.P_grid_bought_1 + self.E_bought + penalty*self.Power_difference)
        
        # else:
        #     self.m.setObjective(self.P_grid_bought_1 + self.E_bought)
        
        self.m.write('model_1.lp')
        self.m.optimize()
        
        # if self.m.status == GRB.INFEASIBLE:
        #     count = 0
        #     #while self.m.status == GRB.INFEASIBLE or count <5:
        #     count +=1
        #     print(count)
        #     print(f'Decision: {self.t_decision}')
        #     print(f'Departure: {self.t_departure}')
        #     print(f'SOC: {self.current_SOC}')
            
            
        #     for i in self.range_samples:
        #         self.m.remove(self.m.getConstrByName(f'SOC_departure[{i}]'))
            

        #     self.m.setObjective(gp.quicksum(self.SOC)/self.n_samples, GRB.MAXIMIZE)
        #     self.m.update()
        #     self.m.optimize()
        #     #print(self.SOC[self.t_forecast,0].x)

        #     self.constraints_violation[t_decision] = self.t_decision
                
                
        self.SOC_min_departure = self.EV.SOC_min_departure 
        self.update_decisions()

        self.cost = self.m.ObjVal
        
        self.extra_cost = max(0, self.BC_EV * np.sum([self.SOC_min_departure - self.SOC[self.t_end,i].x for i in self.range_samples])/self.n_samples)
        
        
    def update_decisions(self):
        
        PV_2EV = self.P_PV_2EV_1.x
        PV_2L = self.P_PV_2L_1.x
        PV_2G = self.P_PV_2G_1.x 
        
        Grid_2EV = self.P_grid_2EV_1.x
        Grid_2L = self.P_grid_2L_1.x 
        Grid_bought = self.P_grid_bought_1.x
        
        SOC = self.current_SOC
        
        dataset = pd.DataFrame(index = self.time_vector)
        dataset['pv'] = self.realization_PV
        dataset['load'] = self.realization_load
        dataset['pv_ev'] = PV_2EV
        dataset['pv_load'] = PV_2L
        dataset['pv_grid'] = PV_2G
        dataset['grid_ev'] = Grid_2EV
        dataset['grid_load'] = Grid_2L
        dataset['soc'] = SOC
        dataset['avail'] = list(self.EV_availability)
        dataset['episode'] = self.episode

        decision = pd.Series(dataset.loc[self.t_decision], name = self.t_decision)
        
        next_decision = {self.col_names[i]: [0] for i in range(len(self.col_names))}
        next_decision['soc'] = self.SOC_2.x
        next_decision = pd.Series(next_decision, name = self.t_forecast)
        
        self.decisions.loc[self.t_decision] = decision
        self.decisions.loc[self.t_forecast] = next_decision
    
    
    def predictions_SOC(self):
        
        df_SOC = pd.DataFrame(index = self.time_vector, columns=self.range_samples)
        
        for t,i in self.SOC:
            df_SOC.loc[t,i] = self.SOC[t,i].x
            
        return df_SOC
    
    def results_deterministic(self):
        
        # PV = [self.realization_PV]
        # if self.forecasting:
        #     PV.extend(np.concatenate(list(self.predictions_PV.values)))
        # else:
        #     PV.extend((list(self.predictions_PV.values)))

        
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
        
        SOC = [self.current_SOC]
        SOC.extend([self.SOC[t,0].x for t in self.time_horizon])
        
        dataset = pd.DataFrame(index = self.time_vector)
        dataset['pv'] = self.data_EV.loc[self.time_vector, 'PV']
        dataset['load'] = self.data_EV.loc[self.time_vector, 'load']
        dataset['pv_ev'] = PV_2EV
        dataset['pv_load'] = PV_2L
        dataset['pv_grid'] = PV_2G
        dataset['grid_ev'] = Grid_2EV
        dataset['grid_load'] = Grid_2L
        dataset['soc'] = SOC
        dataset['avail'] = list(self.EV_availability)
        dataset['episode'] = self.episode

            
        return dataset
    
    def results_stochastic(self):
        
        Load = list(self.load)
        EV_availability = list(self.EV_availability)
        
        PV = [self.realization_PV]
        PV.extend([np.mean(self.predictions_PV.loc[t,:]) for t in self.time_horizon])
        
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
    
def quick_stats(decisions):
    data = decisions.copy()
    
    time = list(data.index)
    avail  = list(data.avail)
    departure = time[avail.index(0)]
    
    # Time absolut
    P_G_bought_t = data.loc[:,['grid_load','grid_ev']].sum(axis = 1)
    P_G_sold_t = data['pv_grid']
    P_EV_t = data.loc[:,['pv_ev','grid_ev']].sum(axis = 1)
    PV_t = data[data.pv > 0]['pv']
    PV_consumed_t = data.loc[:,['pv_ev','pv_load']].sum(axis = 1)
    SOC_t = data['soc']
    P_load_t = data['load']
    
    P_consumed_t = P_EV_t + P_load_t
    
    data_time_absolut = {'P_G_bought': P_G_bought_t, ' P_G_sold': P_G_sold_t, 'P_EV': P_EV_t,
                          'PV_consumed': PV_consumed_t, 'P_consumed': P_consumed_t}
    
    df_time_absolut = pd.DataFrame(data = data_time_absolut)
    
    
    # Time Relative
    EV_PV_t = P_EV_t/PV_t
    Load_PV_t = data['pv_load']/PV_t
    
    EV_G_bought_t = data['grid_ev']/P_G_bought_t
    
    Load_G_bought_t = P_load_t/P_G_bought_t
    
    Sold_PV_t = P_G_sold_t/PV_t
    
    Self_consumption_t = PV_consumed_t/P_consumed_t
    
    data_time_relative = {'EV_PV': EV_PV_t, 'Load_PV': Load_PV_t, 'EV_G_bought': EV_G_bought_t,
                          'Load_G_bought': Load_G_bought_t, 'Sold_PV': Sold_PV_t,
                          'Self_consumption': Self_consumption_t }
    
    df_time_relative = pd.DataFrame(data = data_time_relative)
    
    # Absolut
    P_G_bought = P_G_bought_t.sum()
    P_G_sold = P_G_sold_t.sum()
    P_EV = P_EV_t.sum()
    PV = PV_t.sum()
    PV_consumed = PV_consumed_t.sum()
    SOC_last = data.loc[departure,'soc']
    P_load = P_load_t.sum()
    
    P_consumed = P_consumed_t.sum()
    
    absolut = {'P_G_bought': P_G_bought, 'P_G_sold': P_G_sold,'P_EV': P_EV, 'PV': PV, 
               'PV_consumed': PV_consumed, 'SOC_last': SOC_last, 'Load': P_load, 
               'P_consumed': P_consumed}
    
    
    
    
    # Relative
    EV_PV = P_EV/PV
    Load_PV = data['pv_load'].sum()/PV
    
    EV_G_bought = data['grid_ev'].sum()/P_G_bought
    
    Load_G_bought = P_load/P_G_bought
    
    Sold_PV = P_G_sold/PV
    
    Self_consumption = PV_consumed/P_consumed
    
    
    relative = {'EV_PV': EV_PV, 'Load_PV': Load_PV, 'EV_G_bought': EV_G_bought,
                          'Load_G_bought': Load_G_bought, 'Sold_PV': Sold_PV,
                          'Self_consumption': Self_consumption }
    
    
    stats = {'time_relative':df_time_relative,
             'time_absolut':df_time_absolut,
             'relative': relative,
             'absolut': absolut}
    
    
    return stats