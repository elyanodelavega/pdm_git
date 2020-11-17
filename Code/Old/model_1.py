# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:40:17 2020

@author: Yann
"""
from pathlib import Path
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import gurobipy as gp
from gurobipy import GRB
import df_prepare as dfp
import plot_res as pr

pickle_file_name = 'df_PV_load_EV_big'

Data_EV_availability, Data_PV, Data_Load, time_vector, time_vector_extended,T_arrival, T_departure = dfp.df_from_pickle(pickle_file_name, duration_day = 7)
m = gp.Model('LP EV')

# parameters


BC_EV = 40e3
eta_EV_ch = 0.9
P_EV_chmax = 5e3
SOC_ini_fixed = {t: 0.2 + np.random.random()/3 for t in T_arrival}
SOC_ini_fixed[time_vector_extended[0]] = 0.2 + np.random.random()/3

Availability = Data_EV_availability

P_PV = {time_vector[i]:Data_PV[i] for i in range(len(time_vector))}
P_PV_2EV = m.addVars(time_vector, name = 'P_PV_2EV')
P_PV_2L = m.addVars(time_vector, name = 'P_PV_2L')
P_PV_2G = m.addVars(time_vector, name = 'P_PV_2G')


P_load = {time_vector[i]:Data_Load[i] for i in range(len(time_vector))}


SOC = m.addVars(time_vector_extended,lb = 0.2, ub = 1, name = 'SOC')
P_EV = m.addVars(time_vector,lb = 0, ub = P_EV_chmax, name = 'P_EV')

P_grid_bought = m.addVars(time_vector, lb = 0, name = 'P_grid_bought')
P_grid_sold = m.addVars(time_vector, name = 'P_grid_sold')
y_grid = m.addVars(time_vector, vtype=GRB.BINARY, name = 'y_grid')

P_grid_2EV = m.addVars(time_vector, name = 'P_grid_2EV')
P_grid_2L = m.addVars(time_vector, name = 'P_grid_2L')

y_grid_2EV = m.addVars(time_vector, vtype=GRB.BINARY, name = 'y_grid_2EV')


Grid_balance = m.addConstrs(P_PV[t] + y_grid[t] * P_grid_bought[t] - 
                            (1-y_grid[t]) * P_grid_sold[t] 
                            - Availability[t]*P_EV[t] 
                            - P_load[t] == 0 
                            for t in time_vector )

PV_balance = m.addConstrs(P_PV[t] == P_PV_2EV[t] + P_PV_2L[t] + P_PV_2G[t] 
                          for t in time_vector)

Grid_bought_balance = m.addConstrs(P_grid_bought[t] == (y_grid_2EV[t])*P_grid_2EV[t] 
                                   + P_grid_2L[t]
                                   for t in time_vector)
EV_balance = m.addConstrs(P_EV[t] == (y_grid_2EV[t])*P_grid_2EV[t]
                          + (1-y_grid_2EV[t])*P_PV_2EV[t]
                          for t in time_vector)



SOC_update = m.addConstrs(SOC[time_vector_extended[i]] == 
                          SOC[time_vector_extended[i-1]] +
                          Availability[time_vector_extended[i]] * (P_EV[time_vector_extended[i]]/BC_EV) * eta_EV_ch
                          for i in range(1,len(time_vector_extended)) 
                          if time_vector_extended[i] not in T_arrival)

SOC_dep = m.addConstrs(SOC[t] >= 1 for t in T_departure)
SOC_arr = m.addConstrs(SOC[t] == SOC_ini_fixed[t] for t in SOC_ini_fixed.keys())


E_bought = gp.quicksum(P_grid_bought[t] for t in time_vector)

# if goal is PV self consumption
E_sold = gp.quicksum(P_grid_sold[t] for t in time_vector)

m.setObjective(E_bought)


m.write('model_1.lp')
m.optimize()

if m.status == GRB.INFEASIBLE:
    m.feasRelaxS(0, False, False,True)
    m.optimize()


PV = list(Data_PV)
Load = list(Data_Load)
EV_availability = list(Availability)

PV_2EV = [P_PV_2EV[t].x for t in time_vector]
PV_2L = [P_PV_2L[t].x for t in time_vector]
PV_2G = [P_PV_2G[t].x for t in time_vector]

Grid_2EV = [P_grid_2EV[t].x for t in time_vector]
Grid_2L = [P_grid_2L[t].x for t in time_vector]
Grid_bought = [P_grid_bought[t].x for t in time_vector]

SOC_f = [SOC[t].x for t in time_vector]

dataset = pd.DataFrame(index = time_vector)
dataset['pv'] = PV
dataset['load'] = Load
dataset['pv_ev'] = PV_2EV
dataset['pv_load'] = PV_2L
dataset['grid_ev'] = Grid_2EV
dataset['grid_load'] = Grid_2L
dataset['soc'] = SOC_f
dataset['avail'] = Availability


dataset.to_pickle("./model_1.pkl")


pr.plot_results(dataset)

