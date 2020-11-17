# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:19:35 2020

@author: Yann
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout 
from sklearn.metrics import mean_squared_error
from matplotlib.dates import drange
import df_prepare as dfp

#%% functions

class Forecast:
    
    def __init__(self, pred_variable, data, n_hour_future):
        
        self.name = pred_variable
        self.data = data
        self.n_hour_future = n_hour_future

        self.pred_variable = pred_variable
        
        _, self.pred_pos = self.return_indices(pred_var = self.pred_variable)
        
        self.scale_data()
        
        train_days = 90
        test_days = 7
        n_hour_past = n_hour_future
        
        self.period_past = int(self.f*n_hour_past)
        self.period_future = int(self.f*n_hour_future) 
        
        
        train_res = int(train_days*24*self.f)
        test_res = int(test_days*24*self.f)
        
        self.train_data = self.dataset_scaled[:train_res]
        self.test_data = self.dataset_scaled[train_res+1: train_res+1+test_res]
        
        self.features_set = self.reshape_features(self.train_data, self.period_past, self.period_future)
        
        labels = []  
        for i in range(self.period_past, len(self.train_data) - self.period_future):  
            labels.append(self.train_data[i:i+self.period_future, self.pred_pos])
            
        self.labels = np.array(labels)
        
        self.test_features = self.reshape_features(self.test_data, self.period_past, self.period_future)

        self.build_LSTM()
        
    def scale_data(self):
        
        dataset = self.data.copy()
        
        dataset = dataset._get_numeric_data()
        
        self.t_res = int((dataset.index[1] - dataset.index[0]).seconds)/3600
        self.f = 1/self.t_res
        
        self.dataset = dataset.fillna(0)
        
        # scale data
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        # standard scaler
        self.dataset_scaled = self.scaler.fit_transform(dataset)

    
    def return_indices(self,time = None, pred_var = None):
        
        times = list(self.data.index)
        self.col_names = list(self.data.columns)
        
        if time is not None:
            idx_t = times.index(time)
        else:
            idx_t = None
        
        if pred_var is not None:
            idx_p = self.col_names.index(pred_var)
        else:
            idx_p = None
            
        return idx_t, idx_p
    
    
    def reshape_features(self,data_scaled, period_past, period_future):
        num_features = data_scaled.shape[1]
        
        features = []
        for col in np.arange(0, num_features):
            for i in range(period_past, len(data_scaled)-period_future):
                features.append(data_scaled[i-period_past:i, col])
        
        
        features = np.array(features)
        samples_size = int(len(features)/num_features)

        
        features_reshaped = np.zeros((samples_size,period_past, num_features))

        for i in range(num_features):
            n_in = samples_size*i
            n_out = samples_size*(i+1)
            
            # if n_out < features
            features_reshaped[:,:,i] = features[n_in:n_out, :]
        
        return features_reshaped

        
        
    def plot_results_forecast(self, predictions_scaled, future_scaled, time):

        future_unscaled = self.scaler.inverse_transform(future_scaled)
        predictions_scaled = np.concatenate(predictions_scaled)
        timestep = 4
        step = int(self.period_future/timestep)

        labels = [str((time + pd.Timedelta(hours = 4 * i)).hour) + ':00' for i in range(step)]
        
        fig, ax = plt.subplots()
        ax.plot( predictions_scaled, label = 'Predicted')
        ax.plot(future_unscaled[:,self.pred_pos],label = 'Expected')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          ncol=2)
        
        ax.set_xticks(range(0,self.period_future,timestep))
        ax.set_xticklabels(labels)
        
        #ax.set_ylim([0,max(max(predictions_scaled,future_unscaled[:,self.pred_pos]))])
        
        plt.title(f'{self.period_future}h {self.pred_variable} forecast, {time}')
        plt.show()
        
    def plot_results_test(self, predictions_scaled, comparison_scaled):

        comparison_unscaled = self.scaler.inverse_transform(comparison_scaled)
        
        
        plt.plot(predictions_scaled[:,0], label = 'Predicted')
        plt.plot(comparison_unscaled[self.period_past:-self.period_future,self.pred_pos],label = 'Expected')
        plt.legend()
        plt.title(f'Model test results for {self.pred_variable}')
        plt.show()
        
    def build_LSTM(self, batch_size = 72, epochs = 50, dropout= 0, plot_results = False):
        # create LSTM neural network

        model = Sequential()
        model.add(LSTM(units=self.period_past, return_sequences=True, input_shape=(self.features_set.shape[1], self.features_set.shape[2])))  
        model.add(Dropout(dropout)) 
        model.add(LSTM(units=self.period_past))  
        model.add(Dropout(dropout)) 
        model.add(Dense(self.period_future,activation='sigmoid')) 
        
        # compile the model
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        
        model.fit(self.features_set, self.labels, epochs = epochs, batch_size = batch_size)

        
        # make predictions using trained model
        if plot_results:
            predictions_scaled = model.predict(self.test_features)
            predictions_unscaled = self.unscale_predictions(predictions_scaled)
            
            self.plot_results_test(predictions_unscaled, self.test_data)
           
        self.model = model
    
    def unscale_predictions(self, predictions_scaled):
        
        rows = predictions_scaled.shape[0]
        cols = len(self.col_names)
        
        dummy_array = np.zeros((rows,cols))
        
        if len(predictions_scaled.shape) == 1:
            horizon = 1
            dummy_array[:,self.pred_pos] = predictions_scaled
        
            predictions_unscaled = self.scaler.inverse_transform(dummy_array)[:,self.pred_pos]
        else:
            horizon = predictions_scaled.shape[1]
            
            predictions_unscaled = np.zeros((rows, horizon))
            
            for h in range(horizon):
            
                dummy_array[:,self.pred_pos] = predictions_scaled[:,h]
            
                predictions_unscaled[:,h] = self.scaler.inverse_transform(dummy_array)[:,self.pred_pos]

        return predictions_unscaled
    
    
    def predict(self, time, plot_results = False):
        
        t_pos, pred_pos = self.return_indices(time = time, pred_var = self.pred_variable)
        
        dataset_forecast = self.dataset_scaled[t_pos -self.period_past -1:t_pos, :]
        
        data_to_pred = self.reshape_features(dataset_forecast, self.period_past, 0)

        predictions_scaled = self.model.predict(data_to_pred)

        predictions_unscaled = self.unscale_predictions(predictions_scaled)
        
        if plot_results:
        
            expected_scaled = self.dataset_scaled[t_pos:t_pos + self.period_future, :]
            
            self.plot_results_forecast(predictions_unscaled, expected_scaled, time)
        
        predictions = {self.pred_variable: np.concatenate(predictions_unscaled)}
        
        
        return predictions
    


def df_pred(data, time, n_hour_future, forecast, t_res):
        
        t_start = time - pd.Timedelta(hours = t_res)
        t_end = time + pd.Timedelta(hours = n_hour_future- t_res)
        
        df = data.loc[t_start:t_end].copy()
        
        df.drop(['hours_trigo','days_trigo'], axis = 1, inplace = True)
        
        pred_variables = list(forecast.keys())
        for key in pred_variables:
            
            col_name = key+'_forecast'
            df.loc[time:t_end,col_name] = forecast[key]
            df.loc[t_start,col_name] = df.loc[t_start,key]
        
            
        return df 


#%%

import gurobipy as gp
from gurobipy import GRB
import plot_res as pr

class EV_model:
    def __init__(self, BC_EV = 40e3, eta_EV_ch = 0.9, 
                        P_EV_chmax = 5e3, SOC_min = 0.2, SOC_max = 1 ):
        
        self.BC_EV = BC_EV
        self.eta_EV_ch = eta_EV_ch
        self.P_EV_chmax = P_EV_chmax
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        
        

class Model:
    
    def __init__(self, name, data_full, t_start, t_res,  EV):

        self.name = name

        self.set_EV_parameters(EV)
        
        self.data_full = data_full
        
        self.col_names = ['pv', 'pv_forecast', 'load', 'pv_ev', 'pv_load', 'grid_ev', 'grid_load',
       'soc', 'avail']
        
        self.t_start = t_start
        
        t_previous = t_start - pd.Timedelta(hours = t_res)
        
        previous_decision = {self.col_names[i]: [0] for i in range(len(self.col_names))}
        previous_decision['soc'] = self.data_full.loc[t_previous, 'EV_SOC']
        
        self.decisions = pd.DataFrame(data = previous_decision, index = [t_previous])
        
        
        
    
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
        


#%%
img_path = 'C:/Users/Yann/Documents/EPFL/PDM_git/Images'
pickle_file_name = 'df_PV_load_EV_big'

data = dfp.df_from_pickle(pickle_file_name)

#%%
n_hour_future = 23
PV_model_forecast = Forecast(pred_variable = 'PV', data = data,  n_hour_future = n_hour_future)


#%%
import os
from to_video import to_video
path = 'C:/Users/Yann/Documents/EPFL/PDM_git/Images/'

model_number = 1
name = 'model'
#%%

t_res = PV_model_forecast.t_res
save = True

MPC_days = 10
idx_start = 560
idx_end = int(idx_start + MPC_days * 24)
t_MPC_start = data.index[idx_start]

EV = EV_model()
model = Model(name = name + str(model_number), data_full = data, t_start = t_MPC_start, t_res = t_res,  EV = EV)

img_path = path+name+str(model_number)+'/'

if save:
    os.mkdir(img_path)

figname = None
for i in range(idx_start, idx_end):
    t_decision = data.index[i]
    t_forecast = data.index[i+1]
    
    
    PV_predictions = PV_model_forecast.predict(t_forecast, plot_results= False)
    df = df_pred(data = data, time = t_forecast, n_hour_future = n_hour_future, 
                 forecast = PV_predictions, t_res = t_res)
    
    model.optimize(t_decision = t_decision, df_forecasted = df)    
    
    results = model.results()
    
    decisions = model.decisions
    if save:
        figname = img_path+name + str(i)
    
    pr.plot_MPC(results,decisions, figname = figname)

if save:
    to_video(img_path)

model_number += 1