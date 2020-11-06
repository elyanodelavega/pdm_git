# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:11:50 2020

@author: Yann
"""
import pickle 
import pandas as pd

class MPC:
    def __init__(self, duration):
        self.duration = duration
        
    def df_from_pickle(self, pickle_file_name, 
                 start = 'default', start_hour = '00:00'):
        
        path_to_data_folder = 'C:/Users/Yann/Documents/EPFL/PDM_git/Code/'
        file = open(path_to_data_folder, pickle_file_name, 'rb')
         
        Data = pickle.load(file)
        self.resolution = 1
        
        if start == 'default':
            start_time = str(Data.index[0])
        else:
            start_time = start
        if start_hour == 'default':
            start_hour = '00:00'
        
        start_time = pd.Timestamp(start_time + ' '+ start_hour)
    
        end_time = start_time + pd.Timedelta(hours = 24 * self.duration)
        
        Data_period = Data[list(Data.index).index(start_time):list(Data.index).index(end_time)]
        
        self.Data_EV_availability = Data_period.EV_availability
        self.Data_PV = Data_period.PV
        self.Data_Load = Data_period.load
        
        time_vector = list(Data_period.index)
        time_vector_extended = time_vector.copy()
        time_vector_extended.append(time_vector[0]-pd.Timedelta(hours = 1))
        time_vector_extended.sort()
        
        self.T_arrival = list(Data_period.index[i+1] for i in range(Data_period.shape[0]-1)
                      if (Data_period.EV_availability[i+1] == 1 and Data_period.EV_availability[i] ==0))
        self.T_arrival.append(time_vector[0] - pd.Timedelta(hours = 1))
        self.T_arrival.sort()
        
        
        
        
        self.T_departure = list(Data_period.index[i+1] for i in range(Data_period.shape[0]-1)
                     if (Data_period.EV_availability[i+1] == 0 and Data_period.EV_availability[i] ==1))
        
        self.Data_period = Data_period
        
        def split_train_test(data, train_days, test_days):
            self.Data_train = self.Data_period[train_days * self.resolution:]
            self.Data_test = self.Data_period[test_days * self.resolution:]
            
        