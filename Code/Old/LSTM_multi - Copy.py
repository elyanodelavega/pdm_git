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

def unscale_predictions(scaler, predictions, test_data, 
            period_back, pred_var, pred_pos, col_names):
    df_size = len(test_data)- len(predictions)
    dummy = test_data[df_size:,:].copy()

    dummy2 = pd.DataFrame(dummy, columns = col_names)

    dummy2[pred_var] = predictions

    dummy3 = np.array(dummy2)

    new_predictions = scaler.inverse_transform(dummy3)[:,pred_pos]
    
    return new_predictions

def reshape_features(data_scaled, period_past, period_future):
    num_features = data_scaled.shape[1]
    
    features = []
    for col in np.arange(0, num_features):
        for i in range(period_past, len(data_scaled)-period_future):
            features.append(data_scaled[i-period_past:i, col])
    
    
    features = np.array(features)
    samples_size = int(len(features)/num_features)
    print(samples_size)
    features_reshaped = np.zeros((samples_size,period_past, num_features))
    for i in range(num_features):
        n_in = samples_size*i
        n_out = samples_size*(i+1)
        
        # if n_out < features
        features_reshaped[:,:,i] = features[n_in:n_out, :]
    
    return features_reshaped
    
#%% Data preparation

path_to_data_folder = 'C:/Users/Yann/Documents/EPFL/PDM_git/Code/'
name = 'Montana_Timeseries'
file_format = '.csv'
data = dfp.df_from_time_series2(path_to_data_folder+name+file_format, beg_year = 2018)
print('Data loaded')

#%%
dataset = data.copy()
t_res = int(data['ts'][0].split(':')[1]) # mins
dataset.drop(['dt','ts'],axis = 1, inplace = True)
dataset = dataset.fillna(0)

#dataset['trigo_hours'], dataset['trigo_days'] = time_to_trigo(dataset.index)


# separate into train and test
'''took short sets just to speed up the code to see if it works'''
f = int(60/t_res)
y = dataset.index[0].year
last_year = dataset.index[-1].year
train_years = last_year - y



#%% LSTM
col_names = list(dataset.columns)
print(col_names)

number_of_features = len(col_names)
pred_var = 'DCPmp'

pred_pos = col_names.index(pred_var)




# define the period in past you want to use for prediction
n_hour_past = 12
period_past = int(f*n_hour_past)
# define the period in future you want to forecast for
n_hour_horizon = 5
period_future = int(f*n_hour_horizon) 



# scale data
scaler = MinMaxScaler(feature_range = (0, 1))
# standard scaler
dataset_scaled = scaler.fit_transform(dataset)

train_days = 180
train_res = int(train_days*24*f)

test_days = 7
test_res = int(test_days*24*f)

train_data = dataset_scaled[:train_res]
test_data = dataset_scaled[train_res+1: train_res+1+test_res]

# prepare features and label sets
''' we take input dataset and shift it using period as rolling window,
this is how we create our X'''

# features_set = []  
# for col in np.arange(0,number_of_features):
#         for i in range(period_past, len(train_data)-period_future):  
#             features_set.append(train_data[i-period_past:i, col])

      
# ''' the size of features set is (samples,period_past,number_of_features)'''        
# features_set = np.array(features_set)
# features_set = np.reshape(features_set, (-1, period_past, number_of_features))
features_set = reshape_features(train_data, period_past, period_future)
        
''' we do the same to create our Y but only using the column we want
to predict, in this case it's GHI'''

labels = []  
for i in range(period_past, len(train_data)-period_future):  
    labels.append(train_data[i:i+period_future, pred_pos])
    
labels = np.array(labels)

#%%
#labels = np.reshape(labels,((train_res,period_past,1)))

# create LSTM neural network
dropout = 0
model = Sequential()
model.add(LSTM(units=period_past, return_sequences=True, input_shape=(features_set.shape[1], features_set.shape[2])))  
model.add(Dropout(dropout)) 
model.add(LSTM(units=period_past))  
model.add(Dropout(dropout)) 
model.add(Dense(period_future,activation='sigmoid')) 

# compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# prepare test set
''' we want to predict GHI values on the test data, so to predict the first
value of the test dataset we need to use the last 'period' values of the 
train set, further we will use the data from test set in a rolling manner'''

#%%
test_inputs_scaled = dataset_scaled[len(dataset) - len(test_data) - period_past:]

''' test inputs should be scaled using the transformation values fitted 
on the train set, otherwise the prediction will be poor'''

#test_inputs_scaled = scaler.transform(test_inputs) 

''' we need to create our X in the test dataset'''
test_features = []

# for col in np.arange(0, test_inputs_scaled.shape[1]):
#     for i in range(period_past, len(test_inputs_scaled)):
#         test_features.append(test_inputs_scaled[i-period_past:i, col])


# test_features = np.array(test_features)
# test_features2 = np.zeros((1008,72,28))
# for i in range(28):
#     n_in = 1008*i
#     n_out = 1008*(i+1)
#     test_features2[:,:,i] = test_features[n_in:n_out, :]
    
# # plt.plot(test_features[:,0])
# # plt.title('test features with only DCPmp, sclaed, before reshaping')
# # plt.show()
# print(test_features.shape)
# test_features_reshaped = np.reshape(test_features, (-1, test_features.shape[1], test_inputs_scaled.shape[1]))

# plt.plot(test_features2[:,0,pred_pos])

test_features = reshape_features(test_data, period_past, period_future)


    #%% 

# fit the model: epochs and batch_size to be chosen

my_epochs = 5
my_batch = 72

model.fit(features_set, labels, epochs = my_epochs, batch_size = my_batch)

# make predictions using trained model
'''our prediction is of shape (test_samples,period_future)'''
predictions = model.predict(test_features) 
#%%
# # inverse transform the data
# ''' be careful here because we applied scaler to all features, while
# predicting only GHI at the output, need a little trick to inverse transform'''
# col_names = dataset.columns.values
# dummy = pd.DataFrame(np.zeros((len(predictions), len(col_names))), columns=col_names)
# dummy['DCPmp'] = predictions
# #%%
# dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=col_names)

# final_prediction = dummy['DCPmp'].values

# # check RMSE value
# performance = mean_squared_error(test_data['DCPmp'].values, final_prediction, squared=False)

new_pred = unscale_predictions(scaler, predictions, dataset_scaled, 
            period_past, pred_var, pred_pos, col_names)

test_unscaled = scaler.inverse_transform(test_data)

plt.plot(new_pred, label = 'predicted')
plt.plot(test_unscaled[period_past:,pred_pos],label = 'expected')
plt.legend()