#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:52:51 2019

@author: polaris
"""

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import robust


class Regime_shift(pd.Series):
    
    def as_detect(self, dt = 1, lwl = 5, hwl = 1/3):
        """
        Function as_detect based on Chris Boulton & Tim Lenton approach for detecting regime shifts.
        https://f1000research.com/articles/8-746
        The same name as Boulton's function is used:
        https://github.com/caboulton/asdetect
        """
        l = len(self)
        if isinstance(self, (pd.DataFrame,pd.Series)):
            t = self.index
            ts = self.values  #Just if it's a Panda series
        else:
            t = np.arange(0,l,dt)        
        #Validates highest and lowest window lengths
        if hwl > 1/3 or lwl > hwl*l:
            raise TypeError("Invalid value for highest or lowest window length")    
        
        w_sizes = np.arange(lwl, np.floor(l * hwl)+1, dtype = int) #Windows sizes    
        grad = lambda X,Y: sm.OLS(Y, sm.add_constant(X)).fit().params[1] #Obtains the slope of the linear fitted model for a given set of points (X,Y) in R2    
        grad_rank = np.zeros_like(ts) #Ranking for the detected gradients in the Time series
        
        for ws in w_sizes:
            nw = l // ws #Number of windows
            pad = (l % ws) // 2 # Padding to be left at the extremes (beggining and end) of the time series
            #x= np.array([i*dt for i in range(0,ws)])        
            #Obtains the slope for every window of ws size over the series
            gradients = np.array([grad(t[ws*i + pad: ws*(i+1) + pad], ts[ws*i + pad: ws*(i+1) + pad]) for i in range(0,nw)])
            #Finding the absolute distances from the median bigger than 3 MADs        
            outliers = np.where(np.abs(gradients - np.median(gradients)) > 3 * robust.mad(gradients))[0]       
            for o in outliers:            
                grad_rank[o*ws:o*ws+ws] +=  (1/len(w_sizes))*((1,-1)[int(gradients[o]- np.median(gradients) < 0)])
        
        return pd.Series(data = grad_rank, index = t)   
    
    def before_rs(self):
        """
        Returns the part of the timeseries before the regime shift
        """
        rs_time = self.as_detect().idxmax()
        return self.iloc[:rs_time]
    
def Sample_rs():
    """
    sample_rs returns a sample time series with a regime shift
    """
    mu = (2/9)*math.sqrt(3)     #Bifurcation parameter
    t = np.arange(0,999,1)   #Time steps
    m = mu*t/900              #The change in the bifurcation parameter over time (this is the length of t compared to mu which is a single value)    
    a = np.full(len(t)+1,np.nan)    #setting up vector to hold created time series
    a[0] = -1                 #start it in the left well
    for i,e in enumerate(m):    
        a[i+1] = a[i] + (1/2)*((-a[i]**3) + a[i] + e) + 0.1*np.random.normal() #uses forward euler to run the model over times          
    return Regime_shift(pd.Series(a))

    

