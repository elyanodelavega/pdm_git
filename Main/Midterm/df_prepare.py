# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 08:38:50 2020

@author: Yann
"""

from pathlib import Path
import os, pickle
import numpy as np
import pandas as pd

import math

def data_EV_csv(file_name, to_trigo = False, 
                 add_EV_SOC = True, soc_min = 0.2):
    
    Data = pd.read_csv(file_name, index_col = 0)
    Data.index.rename('time')
    Data.index = pd.to_datetime(Data.index, dayfirst= True)
    
    
    time_vector = list(Data.index)

        
    if add_EV_SOC:
        Data['EV_SOC'] = np.zeros(Data.shape[0])
        T_arrival = list(time_vector[i] for i in range(Data.shape[0]-1)
                  if (Data.EV_availability[i+1] == 1 and Data.EV_availability[i] ==0))
        T_arrival.append(time_vector[0])
        #T_arrival.append(time_vector[-1])
        T_arrival.sort()
        
        T_departure = list(time_vector[i+1] for i in range(Data.shape[0]-1)
                     if (Data.EV_availability[i+1] == 0 and Data.EV_availability[i] ==1))
        for i in range(len(T_arrival)-1):
            if i == 0:
                Data.loc[T_arrival[i]:T_arrival[i+1],'EV_SOC'] = 0.2 + np.random.random()/3
            else:
                t1 = T_arrival[i]
                tbefore = time_vector[time_vector.index(t1)-1]
                Data.loc[t1:T_arrival[i+1],'EV_SOC'] = 0.2 + np.random.random()/3
        # for i in range(len(T_arrival)-1):
        #     if i == 0:
        #         Data.loc[T_arrival[i]:T_arrival[i+1],'EV_SOC'] = 0.2 + np.random.random()/3
        #     else:
                
        #         td = T_departure[i-1]
        #         ta = T_arrival[i+1]
                
        #         Data.loc[td:ta,'EV_SOC'] = 0.2 + np.random.random()/3
    
    return Data

def df_from_pickle_complete(pickle_file_name, duration_day = 1, 
                 start = 'default', start_hour = '00:00'):
    current_directory = Path(__file__).parent #Get current directory
    file = open(os.path.join(current_directory, pickle_file_name), 'rb')
     
    Data = pickle.load(file) 
    
    if start == 'default':
        start_time = str(Data.index[0])
    else:
        start_time = start
    if start_hour == 'default':
        start_hour = '00:00'
    
    start_time = pd.Timestamp(start_time + ' '+ start_hour)
    
    
    Data_period = Data[list(Data.index).index(start_time):]
    
    Data_EV_availability = Data_period.EV_availability
    Data_PV = Data_period.PV
    Data_Load = Data_period.load
    
    time_vector = list(Data_period.index)
    time_vector_extended = time_vector.copy()
    time_vector_extended.append(time_vector[0]-pd.Timedelta(hours = 1))
    time_vector_extended.sort()
    
    T_arrival = list(Data_period.index[i+1] for i in range(Data_period.shape[0]-1)
                  if (Data_period.EV_availability[i+1] == 1 and Data_period.EV_availability[i] ==0))
    T_arrival.append(time_vector[0] - pd.Timedelta(hours = 1))
    T_arrival.sort()
    
    
    
    
    T_departure = list(Data_period.index[i+1] for i in range(Data_period.shape[0]-1)
                 if (Data_period.EV_availability[i+1] == 0 and Data_period.EV_availability[i] ==1))
    
    return Data_EV_availability, Data_PV, Data_Load, time_vector, time_vector_extended,T_arrival, T_departure

def time_to_trigo(time_list):
    
    hours_list = [t.hour for t in time_list]
    hours_trigo = [math.sin(h/24 * math.pi) for h in hours_list]
    
    days_list = [t.dayofyear for t in time_list]
    days_trigo = [math.sin(d/365 * math.pi) for d in days_list]
    
    return hours_trigo, days_trigo
    

def data_PV_csv(df_file_name, hourly_data = True, beg_year = 'beginning', fin_year = 'end' , to_trigo = True):
    df = pd.read_csv(df_file_name)
    t_res = float(df['ts'][0].split(':')[1])
    df['dt'] = pd.to_datetime(df['dt'], dayfirst= True)
    df.set_index('dt', inplace=True, drop=True)
    df = df.fillna(0)
    df = df._get_numeric_data()
    
    if beg_year == 'beginning':
        beginning = str(df.index[0].year)
    else:
        beginning = str(beg_year)
        
    if fin_year == 'end':
        end = str(df.index[-1].year)
    else:
        end = str(fin_year)
    
    data = df[beginning:end]
    
    if hourly_data:
        start = data.index[0]
        len_hours = int(data.shape[0]/(60/t_res))
        hours_indices = [start + pd.Timedelta(hours = i) for i in range(len_hours)]
        data = data.loc[data.index.intersection(hours_indices)]
    
    if to_trigo:
        data['trigo_hours'], data['trigo_days'] = time_to_trigo(data.index)
    
    return data