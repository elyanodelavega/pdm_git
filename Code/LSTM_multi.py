# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:36:47 2020

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
import df_prepare as dfp

#%% functions

# def unscale_predictions(scaler, predictions, test_data, 
#             period_back, pred_var, pred_pos, col_names):
#     df_size = len(test_data)- len(predictions)
#     dummy = test_data[df_size:,:].copy()

#     dummy2 = pd.DataFrame(dummy, columns = col_names)

#     dummy2[pred_var] = predictions

#     dummy3 = np.array(dummy2)

#     new_predictions = scaler.inverse_transform(dummy3)[:,pred_pos]
    
#     return new_predictions

def unscale_predictions(scaler, predictions, col_names, pred_pos):
    
    rows = len(predictions)
    cols = len(col_names)
    
    dummy_array = np.zeros((rows,cols))
    
    dummy_array[:,pred_pos] = predictions

    new_predictions = scaler.inverse_transform(dummy_array)[:,pred_pos]
    
    return new_predictions

def reshape_features(data_scaled, period_past, period_future):
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
    
#%% Data preparation

# path_to_data_folder = 'C:/Users/Yann/Documents/EPFL/PDM_git/Code/'
# name = 'Montana_Timeseries'
# file_format = '.csv'
# data = dfp.df_from_time_series2(path_to_data_folder+name+file_format, beg_year = 2018)
# print('Data loaded')

pickle_file_name = 'df_PV_load_EV_big'

data_period = dfp.df_from_pickle(pickle_file_name)
#%%
## PARAMETERS
train_days = 90
test_days = 7
n_hour_past = 12
n_hour_horizon = 8
pred_var = 'PV'

dropout = 0

my_epochs = 50
my_batch = 8


dataset = data_period.copy()
dataset = dataset._get_numeric_data()
t_res = int((dataset.index[1] - dataset.index[0]).seconds)/3600
f = 1/t_res
dataset = dataset.fillna(0)

#dataset['trigo_hours'], dataset['trigo_days'] = time_to_trigo(dataset.index)


# separate into train and test
'''took short sets just to speed up the code to see if it works'''
y = dataset.index[0].year
last_year = dataset.index[-1].year
train_years = last_year - y



#%% LSTM
col_names = list(dataset.columns)
print(col_names)

number_of_features = len(col_names)
pred_pos = col_names.index(pred_var)

# define the period in past you want to use for prediction
period_past = int(f*n_hour_past)
# define the period in future you want to forecast for
period_future = int(f*n_hour_horizon) 

# scale data
scaler = MinMaxScaler(feature_range = (0, 1))
# standard scaler
dataset_scaled = scaler.fit_transform(dataset)

train_res = int(train_days*24*f)


test_res = int(test_days*24*f)

train_data = dataset_scaled[:train_res]
test_data = dataset_scaled[train_res+1: train_res+1+test_res]

features_set = reshape_features(train_data, period_past, period_future)
        
''' we do the same to create our Y but only using the column we want
to predict, in this case it's GHI'''

labels = []  
for i in range(period_past, len(train_data)-period_future):  
    labels.append(train_data[i:i+period_future, pred_pos])
    
labels = np.array(labels)

test_features = reshape_features(test_data, period_past, period_future)



#%%

# create LSTM neural network

model = Sequential()
model.add(LSTM(units=period_past, return_sequences=True, input_shape=(features_set.shape[1], features_set.shape[2])))  
model.add(Dropout(dropout)) 
model.add(LSTM(units=period_past))  
model.add(Dropout(dropout)) 
model.add(Dense(period_future,activation='sigmoid')) 

# compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')



    #%% 

# fit the model: epochs and batch_size to be chosen


model.fit(features_set, labels, epochs = my_epochs, batch_size = my_batch)

# make predictions using trained model

predictions = model.predict(test_features) 
#%%


new_pred = unscale_predictions(scaler, predictions[:,0], col_names, pred_pos)

test_unscaled = scaler.inverse_transform(test_data)

plt.plot(new_pred, label = 'predicted')
plt.plot(test_unscaled[period_past:,pred_pos],label = 'expected')
plt.legend()