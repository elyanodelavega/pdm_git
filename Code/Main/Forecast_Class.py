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

import tensorflow as tf
from tensorflow import keras  
from keras.models import Sequential  
from tensorflow.keras.layers import Dense, LSTM, Dropout 

from sklearn.metrics import mean_squared_error

import warnings

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error




K = keras.backend

#%% function
def time_to_trigo(time_list):
    
    hours_list = [t.hour for t in time_list]
    hours_trigo = [math.sin(h/24 * math.pi) for h in hours_list]
    
    days_list = [t.dayofyear for t in time_list]
    days_trigo = [math.sin(d/365 * math.pi) for d in days_list]
    
    return hours_trigo, days_trigo

#%% LSTM

class Forecast_LSTM:
    
    def __init__(self, pred_variable, data, n_hour_future, ratio = 1):
        
        self.name = pred_variable
        self.data = data
        self.n_hour_future = n_hour_future

        self.pred_variable = pred_variable
        
        _, self.pred_pos = self.return_indices(pred_var = self.pred_variable)
        
        self.scale_data()
        
        number_days_full = math.floor(self.data.shape[0]/(24 * self.f))
        
        train_days = math.floor(0.7 * number_days_full)
        test_days = math.floor(0.07 * number_days_full)
        n_hour_past = n_hour_future
        
        self.period_past = int(self.f*n_hour_past)*ratio
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
        
    def scale_data(self):
        
        dataset = self.data.copy()
        
        dataset = dataset._get_numeric_data()
        
        dataset['trigo_hours'], dataset['trigo_days'] = time_to_trigo(dataset.index)
        
        self.t_res = int((dataset.index[1] - dataset.index[0]).seconds)/3600
        self.f = 1/self.t_res

        self.dataset = dataset.fillna(0, inplace = True)
        
        if dataset.isnull().any().any():
            print('NaN presents')
        
        # scale data
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        # standard scaler
        self.dataset_scaled = self.scaler.fit_transform(dataset)
        
    def unscale_predictions(self, predictions_scaled):
        
        rows = predictions_scaled.shape[0]
        cols = self.dataset_scaled.shape[1]
        
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
    
        
    def prepare_data_to_pred(self, time):
        
        t_pos, pred_pos = self.return_indices(time = time, pred_var = self.pred_variable)
        
        dataset_forecast = self.dataset_scaled[t_pos -self.period_past -1:t_pos, :]
        
        data_to_pred = self.reshape_features(dataset_forecast, self.period_past, 0)
        
        return t_pos, data_to_pred

        
    
    def build_LSTM(self, batch_size = 72, epochs = 50, dropout = 0, plot_results = False):
        # create LSTM neural network
        
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.features_set.shape[1], self.features_set.shape[2])))  
        model.add(Dropout(dropout)) 
        model.add(LSTM(units=self.period_past))  
        model.add(Dropout(dropout)) 
        model.add(Dense(self.period_future,activation='sigmoid')) 
        
        # compile the model
        model.compile(optimizer = 'adam', loss = 'mse')
       
        model.fit(self.features_set, self.labels, epochs = epochs, batch_size = batch_size)
        
        self.LSTM_model = model
        
        self.predict_test(model, plot_results)
        
        
    def predict_test(self, model, plot_results = False):
        predictions_scaled = model.predict(self.test_features)
        predictions_unscaled = self.unscale_predictions(predictions_scaled)
        
        # make predictions using trained model
        if plot_results:
            
            self.plot_results_test(predictions_unscaled, self.test_data)
            
        self.predictions_unscaled = predictions_unscaled
        self.predictions_scaled = predictions_scaled
        
        comparison_unscaled = self.scaler.inverse_transform(self.test_data)

        self.final_comparison = comparison_unscaled[self.period_past:-self.period_future,self.pred_pos]
        
    
    
    def predict(self, model, time, dataframe = False, plot_results = False):

        t_pos, data_to_pred = self.prepare_data_to_pred(time)

        predictions_scaled = model.predict(data_to_pred)
    
        predictions_unscaled = self.unscale_predictions(predictions_scaled)
        
        if plot_results:
        
            expected_scaled = self.dataset_scaled[t_pos:t_pos + self.period_future, :]
            
            self.plot_results_forecast(predictions_unscaled, expected_scaled, time)
        
        predictions = np.concatenate(predictions_unscaled)
        
        
        if dataframe:
            return self.predictions_to_df(time, predictions)
        
        else:
            
            return predictions
        

    def build_dropout_model(self, model, dropout):

        # Load the config of the original model
        conf = model.get_config()
        # Add the specified dropout to all layers
        for layer in conf['layers']:
            # Dropout layers
            if layer["class_name"]=="Dropout":
                layer["config"]["rate"] = dropout
            # Recurrent layers with dropout
            elif "dropout" in layer["config"].keys():
                layer["config"]["dropout"] = dropout
        # Create a new model with specified dropout
        if type(model)==Sequential:
            # Sequential
            model_dropout = Sequential.from_config(conf)
        model_dropout.set_weights(model.get_weights())
        
        return model_dropout
    
    def predictions_to_df(self, time, results_flat):
        
        data = results_flat.T
        
        if len(results_flat.shape) == 1:
            
            indices = [time + pd.Timedelta(hours = self.t_res * i) for i in range(len(results_flat))]
            
            col_names = [self.pred_variable + '_forecast']
        else:
            indices = [time + pd.Timedelta(hours = self.t_res * i) for i in range(data.shape[0])]
            
            col_names = [self.pred_variable + '_forecast_' + str(i) for i in range(data.shape[1])]
        
        
        df = pd.DataFrame(data = data, index = indices, columns = col_names)
        
        return df

    def predict_distribution(self, model, time, dropout, num_iter = 20, 
                             dataframe = False, plot_results = False):
        
        model_dropout = self.build_dropout_model(model, dropout)
        
         # Create a function to predict with the dropout on
        predict_with_dropout = K.function(
                    [model_dropout.layers[0].input],
                    [model_dropout.layers[-1].output])
        
        K.set_learning_phase(1)
        results = []
        
        t_pos, data_to_pred = self.prepare_data_to_pred(time)
        expected_scaled = self.dataset_scaled[t_pos:t_pos + self.period_future, :]
        
        for _ in range(num_iter):
            pred_dropout_scaled = predict_with_dropout([data_to_pred])
            results.append(pred_dropout_scaled)
        
        ''' at this stage we have result list in shape of 20 num_iter, test_features.shape[0]
        (num_samples), period_future, now we need to reshape it'''
        predictions_array = []
        

        for i in np.arange(0,num_iter):
            results_unscaled = self.unscale_predictions(np.array(results[i][0]))
            if plot_results:
                title = f'dropout: {dropout}, iteration: {i}'
                self.plot_results_test(results_unscaled, expected_scaled, title = title)
            predictions_array.append(results_unscaled)
            
        result_final = np.array(predictions_array)# shape (num_iter,num_samples,period_future)
        
        results_flat = result_final.reshape(num_iter, self.n_hour_future)
        
        if dataframe:
            return self.predictions_to_df(time, results_flat)
        
        else:
            return results_flat

    
    def uncertainty_evaluation(self, model, test_days, idx_start, dropouts, n_iterations):

        idx_end = int(idx_start + test_days * 24)
        dropouts = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
        predictions = {}
        predictions_dropout = {d: {} for d in dropouts}
        n_iter = 20

        for i in range(idx_start,idx_end,self.n_hour_future):
            t_forecast = self.data.index[i]
            print(t_forecast)
            predictions[t_forecast] = self.predict(model = model, time = t_forecast)
            
            for d in dropouts:
                predictions_dropout[d][t_forecast] = self.predict_distribution(model = model, 
                                                                               time = t_forecast, dropout = d).reshape(n_iter, self.n_hour_future)
        return predictions, predictions_dropout
    
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
        
    
    def plot_results_test(self, predictions_scaled, comparison_unscaled, title = None):

        comparison_unscaled = self.scaler.inverse_transform(comparison_unscaled)
        
        plt.figure(figsize=(10,7))
        plt.plot(predictions_scaled[:,0], label = 'Predicted')
        plt.plot(comparison_unscaled[self.period_past:-self.period_future,self.pred_pos],label = 'Expected')
        plt.annotate('Model Settings: ' + 'Epochs: 50, ' + 'Batchsize: 72, ' + 'Optimizer: Adam, '+ f'MSE: 0.0059', xy=(0.06, .015),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='bottom', fontsize=13)
        plt.legend()
        plt.xlabel('Hours')
        plt.ylabel('Watts')
        
        if title is None:
            title = f'Model test results for {self.pred_variable}'
            
        plt.title(title)
        plt.show()
    
#%% Functions
class Forecast_ARIMA:
    
    def __init__(self, data, pred_variable):
        
        self.data = data.astype('float32')
        
        self.pred_variable = pred_variable
        
        self.dataset = data.loc[:,pred_variable]
        
        
    def evaluate_arima_model(self, arima_order):
    
        X = self.dataset
        train_size = int(len(X) * 0.66)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        
        predictions = list()
        for t in range(len(test)):
            arima = ARIMA(history, order=arima_order)
            model = arima.fit(disp=0)
            yhat = model.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        error = mean_squared_error(test, predictions)
        return error
 

    def evaluate_models(self, p_values, d_values, q_values):
        dataset = self.dataset.astype('float32')
        best_score, best_conf = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = self.evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score, best_conf = mse, order
                        print(f'ARIMA{order} MSE=(int{mse})')
                    except:
                        continue
    
        print(f'Best: ARIMA{order} MSE=(int{mse})')
        
        self.best_conf = best_conf
    
    
    def predict(self, t_forecast, t_end, dataframe = True, best_conf = (2, 1, 1), 
                        plot_results = False, conf_interval = False, iterative = False, n_past = 300):
        
        t_decision = t_forecast - pd.Timedelta(hours = 1)
        previous = list(self.data.loc[:t_decision, self.pred_variable])[-n_past:]
        forecast_window = list(self.data.loc[t_forecast:t_end, self.pred_variable])
        time_window = self.data.loc[t_forecast:t_end].index
    
        history = [p for p in previous]
        
        if iterative:
            predictions = []
            #predictions_low = []
            #predictions_high = []
            
            
            for t in range(len(forecast_window)):
                arima = ARIMA(history, order=best_conf)
                model = arima.fit(disp=0)
                yhat = model.forecast()[0][0]
                #yhat_low = model.forecast()[2][0][0]
                #yhat_high = model.forecast()[2][0][1]
                predictions.append(yhat)
                #predictions_low.append(yhat_low)
                #predictions_high.append(yhat_high)
                
                history.append(forecast_window[t])
                #print(f'{time_window[t]} Expected: {int(forecast_window[t])}, Predicted: {int(yhat)}')
        else:
            arima = ARIMA(history, order = best_conf)
            try:
                model = arima.fit(disp = 0)
                predictions = model.forecast(len(time_window))[0]
                data = {f'{self.pred_variable}': predictions}
                if (data>0).all():
                    predictions = self.predict_with_average(t_forecast, t_end)
                    data = {f'{self.pred_variable}': predictions}
                
            except:
                predictions = self.predict_with_average(t_forecast, t_end)
                data = {f'{self.pred_variable}': predictions}
            
        if plot_results:
            plt.figure(figsize = (10,7))
            plt.plot(forecast_window, label = 'expected')
            plt.plot(predictions, label = 'predicted')
            # plt.fill_between(x, predictions_low, predictions_high, color = 'grey', alpha = 0.2)
            plt.legend()
            plt.title(f'{len(forecast_window)}h {self.pred_variable} forecast')
            plt.show()
            
        
        
        # if conf_interval:
        #     data[f'{self.pred_variable}_low'] = predictions_low
        #     data[f'{self.pred_variable}_high'] = predictions_high
        
        df = pd.DataFrame(data = data, index = time_window)
        
        
        
        return df
        
        
    def predict_with_average(self, t_forecast, t_end):
        
        t_decision = t_forecast - pd.Timedelta(hours = 1)
        hours = [t.hour for t in self.data.index]
        self.data['hour'] = hours
        hours_list = np.unique(hours)
        previous = self.data.loc[:t_decision]
        average = {h: np.mean([previous[previous['hour'] == h][self.pred_variable]]) for h in hours_list}
        forecast_time = self.data.loc[t_forecast:t_end].index
        
        forecast = [average[t.hour] for t in forecast_time]
        
        #forecast = pd.DataFrame(data = forecast_variable, index = forecast_time, columns=[self.pred_variable])
        
        
        return forecast
    

    
    
    
    
        
