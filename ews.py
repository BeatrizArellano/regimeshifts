#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:19:31 2019
This script contains the class Ews with the methods to estimate 
the changes in autocorrelation in a timeseries.
This is based on the idea that when a regime shift is approaching 
the autocorrelation is expected to increase.

@author: polaris
"""
import math
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import statsmodels.api as sm


class Ews(pd.Series):
    """
    Ews (Early Warning Signals) extends the methods of a Pandas Series 
    to include useful tools to estimate changes in autocorrelation.
    """
    class Filtered_ts:
        def __init__(self,ts, trend):
            self.trend = trend
            self.res = Ews(ts - trend)
    def gaussian_det(self,bW,**kwargs):
        """
        Detrends a time series using a Gaussian filter
        """        
        trend = gaussian_filter(self.dropna().values, sigma = bW, **kwargs)
        trend = Ews(pd.Series(trend, index = self.dropna().index))                  
        return self.Filtered_ts(self,trend)
    def ar1(self,wL=0.5,detrend=False,**kwargs):
        """
        Estimates the coefficients of the autoregresive model
        """
        if detrend is True:
            self = self.gaussian_det(**kwargs).res
        if wL > len(self.dropna()):
            raise ValueError('Window length cannot be  greater than time series length')
        wL = math.floor(len(self.dropna())*wL) if wL <= 1 else wL
        ar1c = pd.Series(index=self.index)     
        for i in range(0,len(self.dropna())-wL):            
            ar1c.iloc[wL+i] = sm.OLS(self.values[i+1:i+wL+1], sm.add_constant(self.values[i:i+wL])).fit().params[1]              
        return Ews(ar1c)
    def var(self,wL=0.5,detrend=False,**kwargs):
        """
        Estimates the variance for each window along the time series
        """
        if detrend is True:
            self = self.gaussian_det(**kwargs).res
        if wL > len(self):
            raise ValueError('Window length cannot be  greater than time series length')
        wL = math.floor(len(self)*wL) if wL <= 1 else wL
        vari = self.rolling(window=wL).var()
        return Ews(vari)
    def pearsonc(self,wL=0.5,detrend=False,lag=1,**kwargs):
        """
        Estimates the Pearson correlation coefficients between the time series and itself shifted by lag
        """
        if detrend is True:
            self = self.gaussian_det(**kwargs).res
        if wL > len(self):
            raise ValueError('Window length cannot be  greater than time series length')
        wL = math.floor(len(self)*wL) if wL <= 1 else wL
        pcor = self.rolling(window=wL).apply(
                func=lambda x: pd.Series(x).autocorr(lag=lag), raw=True)
        return Ews(pcor)
    
    @property
    def kendall(self):
        """
        Estimates the Kendall correlation coefficient between the time series
        and a perfectly correlated time series of the same length.
        """
        tsCorr = pd.Series(self.dropna().index,self.dropna().index)
        kendall = self.dropna().corr(tsCorr, method="kendall")
        return kendall
    
    def bootstrap(self, method='ar1',n=1000,detrend=False,wL=0.5,lag=1,**kwargs):
        """
        Creates an ensemble of n members in which each member is the same
        length as the original timeseries and its elements are obtained
        sampling from the residuals (after detrending) with replacement.
        Returns an array with the kendall value of the AR(1) or Variance
        changes for each ensemble member.
        """
        if detrend is True:
            self = self.gaussian_det(**kwargs).res
        kendalls = []
        for i in range(0,n):
            sample = Ews(pd.Series(np.random.choice(self.values,len(self))))
            kc = getattr(sample,method)(wL=wL).kendall
            kendalls.append(kc)
        return pd.Series(kendalls)
            
            
            
            
            
            

