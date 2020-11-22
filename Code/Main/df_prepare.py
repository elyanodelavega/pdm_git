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

def data_EV_csv(file_name):
    
    Data = pd.read_csv(file_name, index_col = 0)
    Data.index.rename('time')
    Data.index = pd.to_datetime(Data.index, dayfirst= True)
    Data.rename(columns = {'EV_soc_arrival':'soc'}, inplace = True)
    Data.drop(['EV_time_to_departure'],axis = 1,  inplace = True)
    
    time_vector = list(Data.index)

    
    return Data


def time_to_trigo(time_list):
    
    hours_list = [t.hour for t in time_list]
    hours_trigo = [math.sin(h/24 * math.pi) for h in hours_list]
    
    days_list = [t.dayofyear for t in time_list]
    days_trigo = [math.sin(d/365 * math.pi) for d in days_list]
    
    return hours_trigo, days_trigo
    

def data_PV_csv(df_file_name, hourly_data = True, to_trigo = True):
    df = pd.read_csv(df_file_name)
    t_res = float(df['ts'][0].split(':')[1])
    df['dt'] = pd.to_datetime(df['dt'], dayfirst= True)
    df.set_index('dt', inplace=True, drop=True)
    df = df.fillna(0)
    df = df._get_numeric_data()
    
    data = df.copy()
    
    
    if hourly_data:
        start = data.index[0]
        len_hours = int(data.shape[0]/(60/t_res))
        hours_indices = [start + pd.Timedelta(hours = i) for i in range(len_hours)]
        data = data.loc[data.index.intersection(hours_indices)]
    
    # if to_trigo:
    #     data['trigo_hours'], data['trigo_days'] = time_to_trigo(data.index)
    
    return data