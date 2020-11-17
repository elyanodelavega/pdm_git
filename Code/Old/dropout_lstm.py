# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:33:25 2020

@author: Yann
"""
from tensorflow import keras
import numpy as np
from keras.models import load_model
from keras.models import Model, Sequential
from keras import backend as K
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import df_prepare.py as dfp



def create_dropout_predict_function(model, dropout):
    """
    Create a keras function to predict with dropout
    model : keras model
    dropout : fraction dropout to apply to all layers
    
    Returns
    predict_with_dropout : keras function for predicting with dropout
    """
    
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
    else:
        # Functional
        model_dropout = Model.from_config(conf)
    model_dropout.set_weights(model.get_weights()) 
    
    # Create a function to predict with the dropout on
    #predict_with_dropout = K.function(model_dropout.inputs+[K.learning_phase()], model_dropout.outputs)
    
    return predict_with_dropout

dropout = 0.5
num_iter = 20

data_df = dfp.df_from_time_series('Timeseries_PV.csv', beg_year = 2015)
data_np = data_df.to_numpy()
num_samples = data_np.shape[0]

model = load_model('best_lstm.h5')

predict_with_dropout = create_dropout_predict_function(model, dropout)

predict = np.zeros((num_samples, num_iter))

#for i in range(num_iter):
    
    