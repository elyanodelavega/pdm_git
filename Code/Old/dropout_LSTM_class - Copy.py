# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 08:51:52 2020

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
from keras.models import Model, Sequential
from keras import backend as K
from sklearn.metrics import mean_squared_error
from matplotlib.dates import drange
import df_prepare as dfp
import tensorflow as tf

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
        
    def plot_results_test(self, predictions_scaled, comparison_unscaled):

        comparison_unscaled = self.scaler.inverse_transform(comparison_scaled)
        
        
        plt.plot(predictions_scaled[:,0], label = 'Predicted')
        plt.plot(comparison_unscaled[self.period_past:-self.period_future,self.pred_pos],label = 'Expected')
        plt.legend()
        plt.title(f'Model test results for {self.pred_variable}')
        plt.show()
        
    def LSTM(self, batch_size = 72, epochs = 50, dropout= 0, plot_results = False):
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
        
        predictions_scaled = model.predict(self.test_features)
        predictions_unscaled = self.unscale_predictions(predictions_scaled)
        
        # make predictions using trained model
        if plot_results:
            
            self.plot_results_test(predictions_unscaled, self.test_data)
            
        print(predictions_unscaled.shape)

        
        return predictions_unscaled
    
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
    
#%%
img_path = 'C:/Users/Yann/Documents/EPFL/PDM_git/Images'
pickle_file_name = 'df_PV_load_EV_big'

data = dfp.df_from_pickle(pickle_file_name)

#%%
import time

start_time = time.time()
n_hour_future = 24
PV_model_forecast = Forecast(pred_variable = 'PV', data = data,  n_hour_future = n_hour_future)
t = PV_model_forecast.scaler.inverse_transform(PV_model_forecast.test_data)

position = np.arange(0, 7*24, 24)
test = t[position[1]:position[-1],0]
position2 = position[:-2]

dropouts = [0.25, 0.3, 0.4, 0.5]

num_iter = 15

predict_no_drop = PV_model_forecast.LSTM(epochs = 50)[position2, :]

predict_drop = {}

for d in dropouts:

    predict_drop[d] = np.zeros((num_iter, test.shape[0]))

    for i in range(num_iter):
        print(f'dropout: {d}, iteration {i}')

        p2 = PV_model_forecast.LSTM(epochs = 50, dropout = d)[position2, :]
        p2 = p2.reshape(1,-1)
        predict_drop[d][i,:] = p2

#%%
predict_no_drop = np.zeros((num_iter, test.shape[0]))
for i in range(num_iter):
    print(f'iterartion {i}')
    p1 = PV_model_forecast.LSTM(epochs = 50)[position2, :]
    p1 = p1.reshape(1,-1)
    predict_no_drop[i,:] = p2

    
end_time = time.time()
print(f'time: {end_time - start_time}')    
#%%
import seaborn as sns
for d in dropouts:
    dr = predict_drop[d]
    nd = predict_no_drop
    medians = np.median(dr.T, axis = 1)
    dr_scaled = np.zeros(dr.shape)
    nd_scaled = np.zeros(nd.shape)
    
    for i in range(dr.shape[0]):
        dr_scaled[:,i] = dr[:,i] - medians[i]
        nd_scaled[:,i] = nd[:,i] - test[i]
    
    sns.distplot(nd_scaled.reshape(-1,1), label = '(predicted - real)', kde = True)
    sns.distplot(dr_scaled.reshape(-1,1), label = f'dropout: {d} - median')
    
    plt.legend()
    plt.show()


#%%
for i in range(48):
    for d in dropouts:
        sns.distplot(predict_drop[d][:, i], label = f'dropout: {d}')
    day = int(i/24)
    plt.axvline(test[i],linestyle = '--', label = 'real value', color = 'k')
    plt.title(f'day {int(i/24)}, {i - day * 24}h ahead forecast')
    plt.legend()
    plt.show()
    
#%%
values = np.arange(0.3, 0.9, step = 0.1 )
quantiles_real = np.quantile(test, values)

# for i in range(quantiles
# quantiles_n_values = [t if ]




























