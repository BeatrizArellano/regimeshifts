#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:48:00 2019

@author: Polaris

Function as_detect based on Chris Boulton & Tim Lenton approach for detecting regime shifts.
https://f1000research.com/articles/8-746
The same name as Boulton's function is used:
https://github.com/caboulton/asdetect
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import robust

def as_detect(ts, dt = 1, lwl = 5, hwl = 1/3):
    """
    ts: time series
    """
    l = len(ts)
    if isinstance(ts, (pd.DataFrame,pd.Series)):
        t = ts.index
        ts = ts.values  #Just if it's a Panda series
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
      
    
