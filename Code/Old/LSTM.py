#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:53:02 2020

@author: Yann
"""
import numpy as np
import time
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import df_prepare as dfp
 
# convert series to supervised learning

def year_label(y):
    if y == 1:
        label = str(y) + ' year'
    else:
        label = str(y) + ' years'
    return label

def series_to_supervised(data, pred_var, n_past=1, n_horizon=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
	# input sequence (t-n, ... t-1)
    for i in range(n_past, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_horizon):
        cols.append(df[pred_var].shift(-i))
        if i == 0:
            names += [('var%d(t)' % (1)) ]
        else:
            names += [('var%d(t+%d)' % (1, i))]
	#put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
 	# # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	rmse_final = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return rmse_final, scores

def plot_rmse_time(rmse_list):
    for train_year, rmse, tt in rmse_list:
        print(f'train years: {train_year}, rmse: {rmse}, time: {tt} ')
        plt.scatter(rmse, tt, label = year_label(train_year))
    plt.xlabel('RMSE')
    plt.ylabel('time (s)')
    plt.legend()
    plt.show()
    

def save_best_model(model_results):
    
    rmse_list = [m[2] for m in model_results]
    mu_rmse, std_rmse = np.mean(rmse_list), np.std(rmse_list)
    rmse_st = [(r-mu_rmse)/std_rmse for r in rmse_list]
    
    tt_list = [m[3] for m in model_results]
    mu_tt, std_tt = np.mean(tt_list), np.std(tt_list)
    tt_st = [(r-mu_tt)/std_tt for r in tt_list]
    
    ratio = [rmse_st[i]/ tt_st[i] for i in range(len(rmse_list))]
    
    argmin = np.argmin(ratio)
    
    
    best = model_results[argmin]
    best_model = best[0]

    best_train_years = best[5]
    best_test_y = best[6]
    best_predictions = best[7]
    
    best_model.save('best_lstm.h5')
    
    return best_train_years, best_model, best_test_y, best_predictions

def reshape_values_3D(values, timestep):
    n_samples = values.shape[0]
    n_features = values.shape[1]
    
    n_samples2 = n_samples - timestep
    reshaped = np.zeros((n_samples2, timestep, n_features))
    for f in range(n_features):
        for i in range(n_samples2):
            reshaped[i,:,f] = values[i:i+timestep,f]
    
    return reshaped
            
        

def LSTM_run(data, pred_var, n_past, n_horizon, dropout = 0):
    start_time = time.time()
    y = data.index[0].year
    last_year = data.index[-1].year
    train_years = last_year - y
    values = data.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    data_scaled = pd.DataFrame(data=scaled, index = data.index, columns = data.columns)
    # frame as supervised learning
    reframed = series_to_supervised(data_scaled, pred_var, n_past, n_horizon)
    
    values = reframed.values
    
    n_train_hours = (train_years) *365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    
    # split into input and outputs
    train_X, train_y = train[:, :-n_horizon], train[:, -n_horizon:]
    test_X, test_y = test[:, :-n_horizon], test[:, -n_horizon:]
    # reshape input to be 3D [samples, timesteps, features]
    
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0],  1, test_X.shape[1]))
    train_y = train_y.reshape((train_y.shape[0],  1, train_y.shape[1]))
    test_y = test_y.reshape((test_y.shape[0],  1, test_y.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)  
    
    
    # design network
    model = keras.Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1],train_X.shape[2]), dropout = dropout , return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(n_horizon))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=1, shuffle=False)
    # plot history
    if train_years == 1:
        label = 'train: ' + str(y)
    else:
        label = 'train: ' + str(y)+ ' to ' + str(last_year-1)
        
    plt.plot(history.history['loss'], label=label)
    plt.plot(history.history['val_loss'], label='test: ' + str(last_year))
    plt.legend()
    plt.show()
   
    # make predictions
    
    predictions = model.predict(test_X)
    rmse, scores = evaluate_forecasts(test_X[:, :, 0], predictions)

    tt =  time.time()-start_time
    rmse_list = (train_years, rmse, tt)
    test_y_flat = test_y.reshape(test_y.shape[0], test_y.shape[2])
    model_results = (model, history, rmse, 
                           tt, float(tt/rmse), train_years, test_y_flat, predictions)
    label_2 = year_label(train_years)
    print(f'Test score for {label_2}: {rmse}')
        
    return model_results, rmse_list

#%%.
# load dataset
data_raw = dfp.df_from_time_series('Timeseries_PV.csv', beg_year = 2015)
first_year = data_raw.index[0].year
last_year = data_raw.index[-1].year
pred_var, n_past, n_horizon = 'G_i', 24, 12
rmse_list = []
model_results = []

    
for y in range(first_year, last_year):
    data = dfp.df_from_time_series('Timeseries_PV.csv', beg_year = y )
    model, rmse =  LSTM_run(data, pred_var, n_past, n_horizon)
    rmse_list.append(rmse)
    model_results.append(model)

best_train_years, best_model, best_test_y, best_predictions = save_best_model(model_results)

plot_rmse_time(rmse_list)

#%%
hours = 240
for i in range(best_predictions.shape[1]):
    plt.plot(best_test_y[:,i][-hours:], label = 'Expected')
    plt.plot(best_predictions[:,i][-hours:], label = 'Predicted')
    plt.title('Predictions at t + '+ str(i))
    plt.legend()
    plt.show()

#%%
final_year = data_raw.index [-1].year
beg_year = final_year - best_train_years


sample_size = 10

dropout = 0.5

model_results = []

start_time = time.time()

for i in range(sample_size):
    
    data = dfp.df_from_time_series('Timeseries_PV.csv', beg_year = beg_year , to_trigo = False)
    model, rmse = LSTM_run(data, pred_var, n_past, n_horizon, dropout = dropout)
    rmse_list.append(rmse)
    model_results.append(model)

total_time = time.time() - start_time

print(f'total_time:{round(total_time, 0)}')


#%%
predictions = [m[7] for m in model_results]
n_days = 5
n_hours = n_days*24

for i in range(sample_size):
    rows = predictions[i].shape[0]
    columns = len(predictions)
    pred_do = np.zeros((rows, columns))
    for j in range(len(predictions)):
        pred_do[:,j] = predictions[j][:rows,i]
    for k in range(columns):
        plt.plot(pred_do[-n_hours:,k])
    plt.title('t + ' +str(i))
    plt.show()
        
        
        
        
    

# import pandas as pd
# pred_df = pd.DataFrame(index = range(len(predictions[0])), columns = range(len(predictions)))
# for i in range(len(predictions)):
#     pconc = np.concatenate(predictions[i])
#     pred_df.iloc[:,i] = pconc
    
#%%