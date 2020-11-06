# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:54:26 2020

@author: Yann
"""

# Setting up packages for data manipulation and machine learning
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, Activation
import df_prepare as dfp
#%% functions

def unscale_predictions(scaler, predictions, test_data, 
            period_back, pred_var, pred_pos, col_names):
    dummy = test_data[period_back:,:].copy()

    dummy2 = pd.DataFrame(dummy, columns = col_names)

    dummy2[pred_var] = predictions

    dummy3 = np.array(dummy2)

    new_predictions = mmscaler.inverse_transform(dummy3)[:,pred_pos]
    
    return new_predictions
# Creating the sample sinus curve dataset
#%% Data preparation

path_to_data_folder = 'C:/Users/Yann/Documents/EPFL/PDM_git/Code/'
name = 'Montana_Timeseries'
file_format = '.csv'
data = dfp.df_from_time_series2(path_to_data_folder+name+file_format, beg_year = 2019)
print('Data loaded')

#%%
#df_raw = pd.DataFrame({"valid": data['DCPmp']}, columns=["valid"])
dataset = data.copy()
t_res = int(data['ts'][0].split(':')[1]) # mins
dataset.drop(['dt','ts'],axis = 1, inplace = True)
dataset = dataset.fillna(0)
col_names = list(dataset.columns)
pred_var = 'DCPmp'
pred_pos = col_names.index(pred_var)
num_features = len(col_names)

df = dataset[:5000]
df['indices'] = range(len(df))
df.set_index('indices', inplace=True, drop=True)

# Settings
epochs = 5; batch_size = 72; lstm_neuron_number = 110

# Get the number of rows to train the model on 80% of the data
npdataset = df.values
training_data_length = math.ceil(len(npdataset) * 0.8)

# Transform features by scaling each feature to a range between 0 and 1
mmscaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = mmscaler.fit_transform(npdataset)

# Create a scaled training data set
train_data = scaled_data[0:training_data_length, :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []
trainingdatasize = len(train_data)
for i in range(lstm_neuron_number, trainingdatasize):
    x_train.append(
        train_data[i - lstm_neuron_number : i, :]
    )  # contains lstm_neuron_number values 0-lstm_neuron_number
    y_train.append(train_data[i, pred_pos])  # contains all other values

# Convert the x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)


# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],num_features))
print("x_tain.shape: " + str(x_train.shape) + " -- y_tain.shape: " + str(y_train.shape))

# Configure and compile the neural network model
model1 = Sequential()
model1.add(
    LSTM(lstm_neuron_number, return_sequences=False, input_shape=(x_train.shape[1],num_features ))
)
model1.add(Dense(1))
model1.compile(optimizer="adam", loss="mean_squared_error")

# Create the test data set
test_data = scaled_data[training_data_length - lstm_neuron_number :, :]

# Create the data sets x_test and y_test
x_test = []
y_test = npdataset[training_data_length:, pred_pos]
for i in range(lstm_neuron_number, len(test_data)):
    x_test.append(test_data[i - lstm_neuron_number : i, :])


#%%
# Train the model
history = model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Reshape the data, so that we get an array with multiple test datasets
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_features))

# Get the predicted values
predictions = model1.predict(x_test)

dummy = test_data[lstm_neuron_number:,:].copy()

dummy2 = pd.DataFrame(dummy, columns = col_names)

dummy2[pred_var] = predictions

dummy3 = np.array(dummy2)

predictions = mmscaler.inverse_transform(dummy3)[:,pred_pos]


# Get the root mean squarred error (RMSE) and the meadian error (ME)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
me = np.median(y_test - predictions)
print("me: " + str(round(me, 4)) + ", rmse: " + str(round(rmse, 4)))

train = df[pred_var][:training_data_length]
valid = pd.DataFrame(df[pred_var][training_data_length:])
valid["Predictions"] = predictions
fig, ax1 = plt.subplots(figsize=(32, 5), sharex=True)
yt = train
yv = valid[[pred_var, "Predictions"]]
ax1.tick_params(axis="x", rotation=0, labelsize=10, length=0)
plt.title("Predictions vs Ground Truth", fontsize=18)
plt.plot(yv["Predictions"], color="#F9A048")
plt.plot(yv[pred_var], color="#A951DC")
plt.legend(["Ground Truth", "Train"], loc="upper left")
plt.grid()
plt.show()

#%%
# Settings and Model Labels
rolling_forecast_range = 30
titletext = "Forecast Chart Model A"
ms = [
    ["epochs", epochs],
    ["batch_size", batch_size],
    ["lstm_neuron_number", lstm_neuron_number],
    ["rolling_forecast_range", rolling_forecast_range],
    ["layers", "LSTM, DENSE(1)"],
]
settings_text = ""
lms = len(ms)
for i in range(0, lms):
    settings_text += ms[i][0] + ": " + str(ms[i][1])
    if i < lms - 1:
        settings_text = settings_text + ",  "

# Making a Multi-Step Prediction
new_df = df.copy()
#new_df['DCPmp forecast'] = np.zeros((len(new_df),1))

for i in range(0, rolling_forecast_range):
    print(i)
    last_values = new_df[-lstm_neuron_number:].values
    last_values_scaled = mmscaler.transform(last_values)
    X_input = []
    X_input.append(last_values_scaled)
    X_input = np.array(X_input) 
    X_test = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], num_features))
    pred_value = model1.predict(X_input)
    pred_value_unscaled = unscale_predictions(mmscaler, pred_value, last_values, lstm_neuron_number, pred_var, pred_pos, col_names)
    #pred_value_unscaled = mmscaler.inverse_transform(pred_value)
    pred_value_f = np.round(pred_value_unscaled, 4)
    next_index = new_df.iloc[[-1]].index.values + 1
    new_df = new_df.append(pd.DataFrame({"valid": pred_value_f}, index=next_index))
    new_df_length = new_df.size
forecast = new_df[new_df_length - rolling_forecast_range : new_df_length].rename(
    columns={"valid": "Forecast"}
)

#Visualize the results
validxs = valid.copy()
dflen = new_df.size - 1
validxs.insert(2, "Forecast", forecast, True)
dfs = pd.concat([validxs, forecast], sort=False)
#dflen = int(len(dfs)-1)
dfs.at[dflen, "Forecast"] = dfs.at[dflen, "Predictions"]

# Zoom in to a closer timeframe
dfs = dfs[dfs.index > 200]
yt = dfs[["valid"]]
yv = dfs[["Predictions"]]
yz = dfs[["Forecast"]]
xz = dfs[["Forecast"]].index

# Visualize the data
fig, ax1 = plt.subplots(figsize=(16, 5), sharex=True)
ax1.tick_params(axis="x", rotation=0, labelsize=10, length=0)
ax1.xaxis.set_major_locator(plt.MaxNLocator(30))
plt.title('Forecast Basic Model', fontsize=18)
plt.plot(yt, color="#039dfc", linewidth=1.5)
plt.plot(yv, color="#F9A048", linewidth=1.5)
plt.scatter(xz, yz, color="#F332E6", linewidth=1.0)
plt.plot(yz, color="#F332E6", linewidth=0.5)
plt.legend(["Ground Truth", "TestPredictions", "Forecast"], loc="upper left")
ax1.annotate('ModelSettings: ' + settings_text, xy=(0.06, .015),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='bottom', fontsize=10)
plt.grid()
plt.show()