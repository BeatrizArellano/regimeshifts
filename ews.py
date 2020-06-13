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
import functools
import inspect
import math
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from statsmodels.tsa.ar_model import AutoReg


class Ews(pd.Series):
    """
    Ews (Early Warning Signals) extends the methods of a Pandas Series 
    to include useful tools to estimate changes in autocorrelation.
    """
    class Filtered_ts:
        def __init__(self,ts, trend):
            self.trend = trend
            self.res = Ews(ts - trend)
    def gaussian_det(self,bW, scale=True,**kwargs):
        """
        Detrends a time-series applying a Gaussian filter.
        
        This method detrends a time series using the `scipy.ndimage.gaussian_filter`
        function.
        
        Parameters
        ----------
        bW: scalar                        
            Bandwidth of the Gaussian smoother kernel. 
            If scale is false, it is the parameter sigma in the original 
            scipy function: Standard deviation for Gaussian kernel.           
        scale: boolean
            If True, scales the standard deviation of the smoothing kernel 
            so that the quartiles of the Gaussian probability distribution 
            are at +-1/4 *(bW).
            The quartiles are +- 0.6745*(sigma), sigma is the standard deviation
            for the Gaussian kernel.
            scaled sigma = 0.25 * (1/0.6745) * bW
        **kwargs:
            The possible parameters for the `scipy.ndimage.gaussian_filter` function.
            
        Returns
        -------
        object
            An object with the properties:
                - trend: Pandas Series containing the filtered time-series.
                - res:   Pandas Series containing the residuals after filtering.
                
         Notes
        -----

        Examples
        --------
        noise = np.random.normal(0,20,1000)
        ts = pd.Series(np.arange(0,100,0.1)*2+ noise)
        ts = Ews(ts)
        trend = ts.gaussian_det(bW=30).trend
        res = ts.gaussian_det(bW=30).res
        """        
        if scale == True:
            sd = 0.25 * (1/0.6745) * bW
            kwargs['truncate'] = 4 * (0.6745) #Bandwidth expressed in number of standard deviations
        else:
            sd = bW
        trend = gaussian_filter(self.dropna().values, sigma = sd, **kwargs)
        trend = Ews(pd.Series(trend, index = self.dropna().index))                  
        return self.Filtered_ts(self,trend)
    def validator(func):
        """
        This function to be used as a decorator performs 3 tasks:            
            - Calls the gaussian_det function according to the value of
              the detrend parameter.
            - Validates the window length: if it's greater than the whole 
              series length it raises an error.
            - Separates the keyword arguments to be used properly in the target
              functions.
        """
        @functools.wraps(func)
        def wrapper(inst,*args, **kwargs):
            """
            The wrapper function receives the instance and the keyword 
            arguments.
            """
            filt_args = set(inspect.signature(inst.gaussian_det).parameters.keys()).union(set(inspect.signature(gaussian_filter).parameters.keys()))                   
            detr_kwargs = {k: kwargs[k] for k in (kwargs.keys() & filt_args)} # Obtains the parameteres to be used for the gaussian filter            
            if 'detrend' in kwargs:            
                if kwargs['detrend'] is True:                                 
                    inst = inst.gaussian_det(**detr_kwargs).res    ## Gets the residuals from the gaussian_det function
            ### Estimates the window length size
            wL = kwargs['wL'] if 'wL' in kwargs else inspect.signature(func).parameters['wL'].default            
            wL = math.floor(len(inst.dropna())*wL) if wL <= 1 else wL
            if wL > len(inst.dropna()):
                raise ValueError('Window length cannot be  greater than the time series length')
            roll_args = set(inspect.signature(inst.rolling).parameters.keys()).union(set(inspect.signature(func).parameters.keys())).difference(detr_kwargs).difference({'detrend','wL'})            
            if 'method' in kwargs:                                     
                roll_args = roll_args.union(set(inspect.signature(getattr(inst,kwargs['method'])).parameters.keys())).difference(detr_kwargs).difference({'detrend','wL'})                
            roll_kwargs = {k: kwargs[k] for k in (kwargs.keys() & roll_args)} #Obtains the parameters to be used when rolling the window
            
            return func(inst, wL=wL, **roll_kwargs) ### Calls the function
        return wrapper
    
    @validator
    def ar1(self,detrend=False,wL=0.5,lag=1,**kwargs):
        """
        Estimates the coefficients of an auautoregresive model of order 1
        for each window rolled over the whole time-series.
        The AR(1) is fitted using the Ordinary Least Squares method embedded 
        in the statsmodels AutoReg function.
    
        Fits an autoregresive model of order 1 over the rolling window.
        
        Returns a pandas series containing the coefficients of the autoregresive
        model.
        """        
        #ar1cb = self.rolling(window=wL,**kwargs).apply(
        #        func=lambda x: sm.OLS(x[lag:], sm.add_constant(x[:-lag])).fit().params[1], raw=True)
        ar1c = self.rolling(window=wL,**kwargs).apply(
                func=lambda x: AutoReg(x, lags=[lag]).fit().params[1], raw=True)
        return Ews(ar1c)
    
    @validator
    def var(self,detrend=False,wL=0.5,**kwargs):
        """
        Estimates the variance for each window along the time series
        """        
        vari = self.rolling(window=wL).var()
        return Ews(vari)
    
    @validator
    def pearsonc(self,detrend=False,wL=0.5,lag=1,**kwargs):
        """
        Estimates the Pearson correlation coefficients between the time series 
        and itself shifted by lag.        
        """
        pcor = self.rolling(window=wL,**kwargs).apply(
                func=lambda x: pd.Series(x).autocorr(lag=lag), raw=True)
        return Ews(pcor)
    
    @property
    def kendall(self):
        """
        Estimates the Kendall Tau correlation coefficient between the 
        indicator time series and time.
        """
        if self.index.dtype == 'datetime64[ns]':
            mannSer = np.arange(1,self.dropna().index.size+1)
        else:
            mannSer = self.dropna().index
        tsCorr = pd.Series(mannSer)
        tsCorr.index = self.dropna().index        
        kendall = self.dropna().corr(tsCorr, method="kendall")
        return kendall
    
    @validator
    def bootstrap(self, method='ar1',n=1000,detrend=False,wL=0.5,**kwargs):
        """
        Creates an ensemble of n members in which each member has the same
        length as the original timeseries and its elements are obtained
        sampling from the residuals (after detrending) with replacement.
        Returns an array with the kendall value of the AR(1) or Variance
        changes for each ensemble member.
        """
        kendalls = []
        for i in range(0,n):
            sample = Ews(pd.Series(np.random.choice(self.values,len(self))))                    
            kc = getattr(sample,method)(wL=wL,**kwargs).kendall
            kendalls.append(kc)
        return pd.Series(kendalls)
            
            
            
            
            

