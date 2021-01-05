# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 08:51:52 2020

@author: Yann
"""
# LIBRARY IMPORT
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
def time_to_trigo(Data_index):
    ''' Encode time as trigonometric function
        Input: dataframe timestamp index
        Output: Days and hours encoded as trigo function'''
    hours_list = [t.hour for t in Data_index]
    hours_trigo = [math.sin(h/24 * math.pi) for h in hours_list]
    
    days_list = [t.dayofyear for t in Data_index]
    days_trigo = [math.sin(d/365 * math.pi) for d in days_list]
    
    return hours_trigo, days_trigo

def MSE(x_real, x_pred):
    return np.mean(np.square(x_real - x_pred))

#%% LSTM

class Forecast_LSTM:
    
    def __init__(self, pred_variable, data, n_hour_future, ratio = 1,
                 test_days = 21):
        ''' pred_variable: str, column name of the variable to predict
            data: pandas Dataframe, PV simulation 
            n_hour_future: int, number of hour to predict in the future
            ratio: int, n_hour_future*ratio = n_hour_past'''
        
        #Initialization
        self.name = pred_variable
        self.data = data
        self.n_hour_future = n_hour_future
        self.test_days = test_days
        self.pred_variable = pred_variable
        
        # position in columns of the pred_variable
        _, self.pred_pos = self.return_indices(pred_var = self.pred_variable)
        
        #MinMax scaler of data
        self.scale_data()
        
        # Number of days in total in the data
        number_days_full = math.floor(self.data.shape[0]/(24 * self.f))
        
        # Train/test split
        train_days = math.floor(0.94 * number_days_full)
        test_days = math.floor(0.06 * number_days_full)
        n_hour_past = n_hour_future
        
        self.period_past = int(self.f*n_hour_past)*ratio
        self.period_future = int(self.f*n_hour_future) 
        
        train_res = int((number_days_full-test_days)*24*self.f)
        test_res = int(test_days*24*self.f)
        
        
        self.train_data = self.dataset_scaled[:train_res]
        self.test_data = self.dataset_scaled[train_res+1: train_res+1+test_res]
        self.features_set = self.reshape_features(self.train_data, self.period_past, self.period_future)
        
        # y
        labels = []  
        for i in range(self.period_past, len(self.train_data) - self.period_future):  
            labels.append(self.train_data[i:i+self.period_future, self.pred_pos])
            
        self.labels = np.array(labels)
        
        # X, reshape in (samples_size,period_past, num_features)
        self.test_features = self.reshape_features(self.test_data, self.period_past, self.period_future)
    
    def return_indices(self,time = None, pred_var = None):
        
        ''' Return indices of the time and pred_var when df is transformed in array
            Input:
                time: timestamp
                pred_var: str
            Output:
                idx_t, idx_p: int
            '''
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
        '''Scales data for LSTM with MinMaxScaler '''
        
        dataset = self.data.copy()
        
        dataset = dataset._get_numeric_data()
        
        dataset['trigo_hours'], dataset['trigo_days'] = time_to_trigo(dataset.index)
        
        # Period (hours)
        self.t_res = int((dataset.index[1] - dataset.index[0]).seconds)/3600
        
        # Frequency
        self.f = 1/self.t_res
        
        # NaNs removal (if data is plugged in with df_prepare.py, shouldn't be any)
        self.dataset = dataset.fillna(0, inplace = True)
        
        if dataset.isnull().any().any():
            print('NaN presents')
        
        # scale data
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        # standard scaler
        self.dataset_scaled = self.scaler.fit_transform(dataset)
        
    def unscale_predictions(self, predictions_scaled):
        
        ''' From scale predictions, return unscaled from the MinMax scaler used previously
        Input:
            predictions_scaled: array
        Output:
             predictions_unscaled: array
        '''
        rows = predictions_scaled.shape[0]
        cols = self.dataset_scaled.shape[1]
        
        dummy_array = np.zeros((rows,cols))
        
        # differentiation if preidctions are single values or distribution
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
        ''' Shape features in (samples_size,period_past, num_features) for LSTM input
        Input:
            data_scaled: array, scaled data
            period_past: int, number of past timestep used to predict
            period_future, int, number of timestep to predict
        Output:
            features_reshaped: array of size (samples_size,period_past, num_features)'''
        
        #number of features
        num_features = data_scaled.shape[1]
        
        # append of a rolling basis contaning each time the previous period_past timestep
        features = []
        for col in np.arange(0, num_features):
            for i in range(period_past, len(data_scaled)-period_future):
                features.append(data_scaled[i-period_past:i, col])
        # list to array
        features = np.array(features)
        samples_size = int(len(features)/num_features)
        
        # intialization
        features_reshaped = np.zeros((samples_size,period_past, num_features))
        
        for i in range(num_features):
            n_in = samples_size*i
            n_out = samples_size*(i+1)

            features_reshaped[:,:,i] = features[n_in:n_out, :]
        
        return features_reshaped
    
        
    def prepare_data_to_pred(self, time):
        ''' funtion to prepare data to predict from a specific time
            Input:
                time: timestamp
            Output:
                t_pos: int, position of time in sclaed array
                data_to_pred: array, array to feed in predict LSTM'''
        
        t_pos, pred_pos = self.return_indices(time = time, pred_var = self.pred_variable)
        
        dataset_forecast = self.dataset_scaled[t_pos -self.period_past -1:t_pos, :]
        
        data_to_pred = self.reshape_features(dataset_forecast, self.period_past, 0)
        
        return t_pos, data_to_pred

        
    
    def build_LSTM(self, neurons_2 = 100, neurons_3 = 50, batch_size = 72, epochs = 50, dropout = 0, plot_results = False):
        '''Build LSTM architecture
            Input:
                batch_size: int
                epochs: int
                dropout: int
                plot_results: bool, output of the prediction in graph
            Output:
                None
                '''
        self.neurons_2 = neurons_2
        self.neurons_2 = neurons_3
        self.epochs = epochs
        self.batch_size = batch_size
        # LSTM architecture
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(self.features_set.shape[1], self.features_set.shape[2])))  
        model.add(Dropout(dropout)) 
        model.add(LSTM(units=50))  
        model.add(Dropout(dropout)) 
        model.add(Dense(self.period_future,activation='sigmoid')) 
        
        # Compile the model
        model.compile(optimizer = 'adam', loss = 'mse')
       
        # Fit
        model.fit(self.features_set, self.labels, epochs = epochs, batch_size = batch_size)
        
        # Save Model
        self.LSTM_model = model
        
        # Test
        self.predict_test(model, plot_results)
        
        
    def predict_test(self, model, plot_results = False):
        ''' Test after fitting the model
            Input: 
                model: LSTM model
                plot_results: bool, output of the prediction in graph
            Output:
                None '''
        #predict
        self.predictions_scaled = model.predict(self.test_features)
        
        #unscale predictions
        self.predictions_unscaled = self.unscale_predictions(self.predictions_scaled)
        

        # actual values
        comparison_unscaled = self.scaler.inverse_transform(self.test_data)

        self.final_comparison = comparison_unscaled[self.period_past:-self.period_future,self.pred_pos]
        
        
        
        # plot results
        if plot_results:
            
            self.plot_results_test(self.predictions_unscaled, self.test_data)
            
        
        
        
    
    def predict(self, model, time, dataframe = False, plot_results = False):
        ''' Predict at specific time
            Input:
                model: LSTM model
                time: timestamp
                datframe: bool, if results in dataframe wanted
                plot_results: bool, if output comparison graph wanted
            Output:
                predictions: array or dataframe'''
        
        # Prepare data
        t_pos, data_to_pred = self.prepare_data_to_pred(time)

        # Predict
        predictions_scaled = model.predict(data_to_pred)
    
        # Unscale
        predictions_unscaled = self.unscale_predictions(predictions_scaled)
        
        # plot results
        if plot_results:
        
            expected_scaled = self.dataset_scaled[t_pos:t_pos + self.period_future, :]
            
            self.plot_results_forecast(predictions_unscaled, expected_scaled, time)
        
        predictions = np.concatenate(predictions_unscaled)
        
        if dataframe:
            return self.predictions_to_df(time, predictions)
        
        else:
            
            return predictions
        

    def build_dropout_model(self, model, dropout):
        ''' build a dropout model for predict distribution
            Input:
                model: LSTM model
                dropout: int
            Output:
                model_dropout: LSTM model with dropouts added'''
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
        ''' array of predictions to dataframe
            Input: 
                time: timestamp
                results_flat: flatten array of predictions
            Output:
                df: dataframe with predictions'''
        
        # Transform
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
        ''' predict distribution form initial LSTM model
            Input:
                model: LSTM model
                time: timesamp
                dropout: int, dropout wanted
                num_iter: int, number of wanted predictions
                dataframe: bool, results to dataframe
                plot_results: bool
            Output:
                predictions'''
                
        model_dropout = self.build_dropout_model(model, dropout)
        
        # Create a function to predict with the dropout on the model
        predict_with_dropout = K.function(
                    [model_dropout.layers[0].input],
                    [model_dropout.layers[-1].output])
        
        # Artificially set learning phase to 1
        K.set_learning_phase(1)
        results = []
        
        # Prepare data
        t_pos, data_to_pred = self.prepare_data_to_pred(time)
        expected_scaled = self.dataset_scaled[t_pos:t_pos + self.period_future, :]
        
        # Predict samples
        for _ in range(num_iter):
            pred_dropout_scaled = predict_with_dropout([data_to_pred])
            results.append(pred_dropout_scaled)
        
        
        predictions_array = []
        
        # Unscale predictions
        for i in np.arange(0,num_iter):
            results_unscaled = self.unscale_predictions(np.array(results[i][0]))
            if plot_results:
                title = f'dropout: {dropout}, iteration: {i}'
                self.plot_results_test(results_unscaled, expected_scaled, title = title)
            predictions_array.append(results_unscaled)
            
        result_final = np.array(predictions_array)# shape (num_iter,num_samples,period_future)
        
        # From 3D to 2D
        results_flat = result_final.reshape(num_iter, self.n_hour_future)
        
        if dataframe:
            return self.predictions_to_df(time, results_flat)
        
        else:
            return results_flat

    
    def uncertainty_evaluation(self, model, test_days, idx_start, dropouts, n_iterations):
        ''' Evaluate different dropouts results day to day
            Input:
                model: LSTM model
                test_days: int, test size
                idx_start: int, starting index (time)
                droputs: list, list of dropouts to test
                n_iterations: number of samples
            Output:
                predictions: array, simple prediction
                predictions_dropout: array, distribution prediction'''
        # Prepare data
        idx_end = int(idx_start + test_days * 24)
        
        predictions = {}
        predictions_dropout = {d: {} for d in dropouts}
        n_iter = 20

        for i in range(idx_start,idx_end,self.n_hour_future):
            # Predict simple values
            t_forecast = self.data.index[i]

            predictions[t_forecast] = self.predict(model = model, time = t_forecast)
            
            # Predict distriution with specified dropout
            for d in dropouts:
                predictions_dropout[d][t_forecast] = self.predict_distribution(model = model, 
                                                                               time = t_forecast, dropout = d).reshape(n_iter, self.n_hour_future)
        return predictions, predictions_dropout
    
    def plot_results_forecast(self, predictions_scaled, future_scaled, time):
        ''' Plot function to visualize predictions
            Input:
                predictions_scaled: array
                future_scaled: array, actual future values to be compared with
                time: timestamp'''
        
        # Unscale future
        future_unscaled = self.scaler.inverse_transform(future_scaled)
        
        # concatenate predictions
        predictions_scaled = np.concatenate(predictions_scaled)
        
        # Plot options
        timestep = 4
        step = int(self.period_future/timestep)

        labels = [str((time + pd.Timedelta(hours = 4 * i)).hour) + ':00' for i in range(step)]
        
        # Figure
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
        
       
        plt.title(f'{self.period_future}h {self.pred_variable} forecast, {time}')
        plt.show()
        
    
    def plot_results_test(self, predictions_scaled, comparison_unscaled, title = None):
        ''' Plot function for test results
            Input:
                predictions_scaled: array
                comparison_unscaled: array
            Output: Figure'''
        comparison_unscaled = self.scaler.inverse_transform(comparison_unscaled)
        plt.figure(figsize=(12,7), dpi = 500)
        
        self.MSEs = {}
        l = len(self.predictions_unscaled)
        for t in range(self.n_hour_future):
            self.MSEs[t] = mean_squared_error(self.final_comparison[t:], self.predictions_unscaled[:l-t,t])
            
        plt.plot(range(1,self.n_hour_future+1),self.MSEs.values(), marker = 'o')
        
        plt.annotate(f'Model Settings: \nEpochs: {self.epochs}, batchsize: {self.batch_size}, Optimizer: Adam\nAverage MSE: {int(np.mean(list(self.MSEs.values())))} ', 
                     xy=(0.06, .005),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='bottom', fontsize=13)
        plt.xlabel('Hours ahead forecasting')
        plt.ylabel('MSE')
        plt.xticks(np.arange(24))
        
        title = f'MSE results for {self.pred_variable}'
        
        plt.ylim(bottom = 0)
        plt.title(title)
        plt.grid(True)
        plt.show()

        
        plt.figure(figsize=(12,7), dpi = 500)
        plt.plot(predictions_scaled[:,0], label = 'Predicted')
        plt.plot(comparison_unscaled[self.period_past:-self.period_future,self.pred_pos],label = 'Expected')
        plt.annotate(f'Model Settings: \nEpochs: {self.epochs}, batchsize: {self.batch_size}, Optimizer: Adam\nAverage MSE: {int(np.mean(list(self.MSEs.values())))} ',
                     xy=(0.06, .005),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='bottom', fontsize=13)
        plt.legend()
        ticks = np.arange(0,len(comparison_unscaled), 24)
        labels = [i for i in range(len(ticks))]
        plt.xticks(ticks, labels)
        plt.xlabel('Days')
        plt.xlim(right = len(predictions_scaled))
        plt.ylabel('Watts')
        
        
        title = f'Model Validation results for {self.pred_variable}'
        plt.ylim(bottom = 0)
        plt.title(title)
        plt.grid(True)
        plt.show()
        
        
    
#%% Functions
class Forecast_ARIMA:
    
    def __init__(self, data, pred_variable):
        
        ''' data: dataframe with pred_variable present
            pred_variable: str, name of vairable to predict'''
        
        self.data = data.astype('float32')
        
        self.pred_variable = pred_variable
        
        self.dataset = data.loc[:,pred_variable]
        
        
    def evaluate_arima_model(self, dataset, arima_order):
        ''' Fit an arima model with a particular order (p,d,q)
            and evaluate the MSE
            input:
                dataset: dataframe
                arima_order: (int,int,int)
            output:
                error: float'''
        
        # Train/Test split
        X = dataset
        train_size = int(len(X) * 0.66)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        
        # predict each time step on a rolling manner
        predictions = list()
        for t in range(len(test)):
            # Fit
            arima = ARIMA(history, order=arima_order)
            model = arima.fit(disp=0)
            # Predict
            yhat = model.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # Calculate out of sample error
        error = mean_squared_error(test, predictions)
        return error
 

    def evaluate_models(self, p_values, d_values, q_values):
        ''' Find the best (p,d,q) order
            input:
                p,d,q values: [int],[int],[int]
            output:
                results: dict'''
        dataset = self.dataset.astype('float32')
        best_score, best_conf = float("inf"), None
        results =  {}
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = self.evaluate_arima_model(dataset, order)
                        results[order] = mse
                        if mse < best_score:
                            best_score, best_conf = mse, order
                        print(f'ARIMA{order} MSE={np.round((mse),2)}')
                    except:
                        continue
    
        print(f'Best: ARIMA{best_conf} MSE={np.round((best_score),2)}')
        
        self.best_conf = best_conf
    
        return results
    
    def predict(self, t_forecast, t_end, dataframe = True, best_conf = (2, 1, 1), 
                        plot_results = False, conf_interval = False, iterative = False, n_past = 300):
        '''Predict values with ARIMA model at given time t_forecast until t_end
            Input:
                t_forecast, t_end: pd.Timestamp()
                dataframe: results in pd.Dataframe, bool
                best_conf: (int,int,int), (p, d, q)
                plot_results: bool
                conf_interval: bool, results with confidence interval
                iterative: bool, forecast on a rolling manner or by batch, bool
                n_past: int, dataset size
            Output:
                prediction: df or list
                '''
        # Initialization
        t_decision = t_forecast - pd.Timedelta(hours = 1)
        previous = list(self.data.loc[:t_decision, self.pred_variable])[-n_past:]
        forecast_window = list(self.data.loc[t_forecast:t_end, self.pred_variable])
        time_window = self.data.loc[t_forecast:t_end].index
    
        history = [p for p in previous]

        
        # Rolling manner
        if iterative:
            predictions = []
            #predictions_low = []
            #predictions_high = []
            
            
            for t in range(len(forecast_window)):
                # fit
                arima = ARIMA(history, order=best_conf)
                model = arima.fit(disp=0)
                # Forecast
                yhat = model.forecast()[0][0]
                #yhat_low = model.forecast()[2][0][0]
                #yhat_high = model.forecast()[2][0][1]
                predictions.append(yhat)
                #predictions_low.append(yhat_low)
                #predictions_high.append(yhat_high)
                
                history.append(forecast_window[t])
                #print(f'{time_window[t]} Expected: {int(forecast_window[t])}, Predicted: {int(yhat)}')
            data = predictions
        else:
            
            arima = ARIMA(history, order = best_conf)
            # Precaution if the model doesn't converge
            try:
                # Fit
                model = arima.fit(disp = 0)
                predictions = model.forecast(len(time_window))[0]
                data = {f'{self.pred_variable}': predictions}
                # if negative values predicted
                print((data>0).all())
                if (data>0).all():
                    # predict with average
                    
                    predictions = self.predict_with_stats(t_forecast, t_end)
                    data = {f'{self.pred_variable}': predictions}
                    
            # predict with average if model did not converge    
            except:
                
                predictions = self.predict_with_stats(t_forecast, t_end)
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
        
        if dataframe == True:
            return pd.DataFrame(data = data, index = time_window)
        
        else:
            return data

        
        
    def predict_with_stats(self, t_forecast, t_end, method = 'mean'):
        ''' predict with average with previous data, based on hour of the day
        Input:
            t_forecast, t_end: pd.Timestamp()
        Output:
            forecast: list
        '''
        t_decision = t_forecast - pd.Timedelta(hours = 1)
        hours = [t.hour for t in self.data.index]
        self.data['hour'] = hours
        
        hours_list = np.unique(hours)
        previous = self.data.loc[:t_decision]
        if method == 'mean':
            stats = {h: np.mean([previous[previous['hour'] == h][self.pred_variable]]) for h in hours_list}
        
        elif method == 'median':
            stats = {h: np.median([previous[previous['hour'] == h][self.pred_variable]]) for h in hours_list}
            
        forecast_time = self.data.loc[t_forecast:t_end].index
        
        forecast = [stats[t.hour] for t in forecast_time]

        
        return forecast
    

    
    
    
    
        
