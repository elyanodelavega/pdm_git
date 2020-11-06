# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:08:10 2020

Controller model

@author: Yann
"""


class EV:
    def __init__(self, availability, BC_EV,SOC_EV_UB,SOC_EV_LB, P_EV_chmax, eta_EV_ch ):
        
        self.availability = availability
        self.BC = BC_EV
        self.SOC_UB = SOC_EV_UB
        self.SOC_LB = SOC_EV_LB
        self.P_chmax = P_EV_chmax
        self.eta_ch = eta_EV_ch
        self.SOC_min = SOC_EV_LB*BC_EV
        self.SOC_max = SOC_EV_UB*BC_EV
        
        self.P_ch = None
        self.SOC = None
        
    def EV_charge(P_ch, time):
        self.P_ch = P_ch
        self.SOC = self.SOC + 

class Event:

    def __init__(self, time):
        self.time = time

class EV_Arrival(Event):
    def __init__(self, time, SOC_EV_ini, time_to_departure):
        super().__init__(time)
        self.SOC_ini = SOC_EV_ini
        self.time_to_departure = time_to_departure
        
class EV_Departure(Event):
    def __init__(self, time, SOC_EV_final):
        super().__init__(time)
        self.SOC_ini = SOC_EV_ini
        self.time_to_departure = time_to_departure
        
	
class PV:
    def __init__(self, PV_prod):
        self.production = PV_prod


class grid:
    def __init__(self):
        f.P_b = None

