#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:11:02 2022

@author: jannik
"""

import netCDF4 as nc
import numpy as np


# Function to compute virtual temperature
def computeTv(temperature, moisture):
    t = temperature
    q = moisture
    return (np.multiply(t, (1 + q / 0.622) / (1 + q)))


class DataLoader:
    
    def __init__(self,dataset,timestep):
        
        self.__ds = dataset
        self.__tstep = timestep
        
        self.__t = (self.__ds['TSRFC'][self.__tstep, :, :]).filled()
        self.__q = (self.__ds['QVSRFC'][self.__tstep, :, :]).filled()
        self.__u = (self.__ds['USFC'][self.__tstep, :, :]).filled()
        self.__v = (self.__ds['VSFC'][self.__tstep, :, :]).filled()
        self.__w = (self.__ds['WSRFC'][self.__tstep, :, :]).filled()
        self.__rint = (self.__ds['Prec'][self.__tstep, :, :]).filled() / 24 # unit from mm/day to mm/h
        self.__tb = (self.__ds['TB'][self.__tstep, :, :]).filled()
        self.__tv = computeTv(self.__t, self.__q)
    
    def __del__(self):
        """
        Deletes the DataLoader object
        """

    def getTimestep(self):
        
        return self.__tstep
        
    def getT(self):
        
        return self.__t
    
    def getQ(self):
        
        return self.__q
    
    def getU(self):
        
        return self.__u

    def getV(self):
        
        return self.__v

    def getW(self):
        
        return self.__w

    def getRint(self):
        
        return self.__rint 

    def getTb(self):
        
        return self.__tb 
    
    def getTv(self):
        
        return self.__tv  



    
if __name__ == "__main__":

    tstep = 400
    
    # Import the datasets with the chosen resolution     
    path = "/home/jannik/PhD/Programming/gust_front/Romain_data/cp-detection/diurnal2K_200m/diurnal2K_200m_240x240km2.nc"
    ds = nc.Dataset(path,mode="r")
    
    # Create DataLoader object    
    dataloader1 = DataLoader(ds,tstep)
    
    tv = dataloader1.getTv()
    
    del dataloader1