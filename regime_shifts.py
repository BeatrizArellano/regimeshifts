#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:48:00 2019

@author: Beatriz Arellano-Nava

"""
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import robust


class Regime_shift(pd.Series):
    """
    Regime_shift extends the methods of a Pandas Series 
    to implement a regime shift detection index.
    """
    
    def as_detect(self, dt = 1, lwl = 5, hwl = 1/3):
        """
        Estimates the regime shift detection index according to 
        Chris Boulton & Tim Lenton's approach to detect regime shifts.
        
        Original paper:
        https://f1000research.com/articles/8-746
        Original code in R:
        https://github.com/caboulton/asdetect
        
        Parameters
        ----------
        dt: float
            Time step of the time series. 
            Default: 1.
            
        lwl: int            
            Lowest window length to estimate the gradient values.
            Default:5
            
        hwl: int            
            Highest window length to estimate the gradient values.
            Default: 1/3 of the time series
            
        Returns
        -------
        pandas Series
            A series of the same length as the original series containing
            a value in the interval [-1,1] for each element of the series.
            
            This index indicates the proportion of windows that detected 
            a gradient greater than 3 Median Absoulte Deviations of the 
            distribution of gradients.         
        
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
                grad_rank[o*ws+pad:o*ws+ws+pad] +=  (1/len(w_sizes))*((1,-1)[int(gradients[o]- np.median(gradients) < 0)]) ## Adds +-1/(# of windows) to the ranking of gradients
        
        return pd.Series(data = grad_rank, index = t)  
    
    def before_rs(self):
        """
        Returns the part of the timeseries before the regime shift
        """
        rs_time = self.as_detect().idxmax()
        return self.iloc[:rs_time]
    
def sample_rs(length=999,transition_timing=0.9,std = 0.1):
    """
    sample_rs generates a sample time series with an underlying bifurcation

    Parameters
    ----------
    length : integer, optional
        Length of the time-series. The default is 999.
    transition_timing : float [0,1], optional
        Timing of the regime shift. The default is 0.9.
    std : float, optional
        standard deviation. The default is 0.1.

    Returns
    -------
    Pandas Series
        Time-series of a process approaching a tipping point.

    """
    mu = (2/9)*math.sqrt(3)     #Bifurcation parameter
    t = np.arange(0,length,1)   #Time steps
    m = mu*t / np.floor(length*transition_timing)          #The change in the bifurcation parameter over time (this is the length of t compared to mu which is a single value)    
    a = np.full(len(t)+1,np.nan)    #setting up vector to hold created time series
    a[0] = -1                 #start it in the left well
    for i,e in enumerate(m):    
        a[i+1] = a[i] + (1/2)*((-a[i]**3) + a[i] + e) + std * np.random.normal() #uses forward euler to run the model over times          
    return Regime_shift(pd.Series(a))
      
    
