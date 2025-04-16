#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:19:31 2019

This module provides tools to detect early warning signals of critical transitions
in time-series data by analysing changes in resilience indicators. It extends the
Pandas Series/DataFrame API with methods suitable for any dynamical system where
relatively long time-series are available. Core functionalities include detrending,
indicator computation, trend analysis, robustness assessment, and significance testing.

.. currentmodule:: regimeshifts.ews


Dependencies
------------
* math
* matplotlib
* numpy
* pandas
* scipy
* statsmodels

Classes
-------
Ews
    Extends `pandas.DataFrame` to provide tools for early warning signal detection
    and resilience assessment in time-series data.

Methods
-------

Detrending
----------
gaussian_det
    Applies Gaussian smoothing to remove long-term trends from the time-series.


Resilience indicators
---------------------
ar1
    Computes the lag-1 autocorrelation, a common indicator of critical slowing down.

var
    Computes the variance of the time-series.
    
lambd
    Estimates the linear restoring rate (λ) as proposed by Boers (2021).
    
pearsonc
    Calculates the Pearson correlation between subsequent time steps.

skw
    Estimates changes in the skewness of the time-series.

Trend strength
--------------
kendall
    Computes Kendall’s tau to assess the monotonic trend strength in resilience indicators.

Robustness assessment
---------------------
robustness
    Computes Kendall’s tau across a range of detrending bandwidths and window lengths to assess
    the robustness of trends in resilience indicators.
    
Significance test
-----------------
significance
    Performs significance tests for trends in resilience metrics by comparing the measured trend with
    those observed in an ensemble of surrogate series with the same spectral properties as the original
    series. The surrogate series are obtained by bootstrapping data in the original series.

Plotting
--------
plot
    Provides methods to visualise the significance and robustness of trends. 


Examples
--------
>>> from regimeshifts.ews import Ews
>>> series = pd.Series(np.random.normal(size=1000))
>>> ts = Ews(series)
>>> ar1 = ts.ar1(detrend=True,bW=80,wL=0.4)
>>> ar1_significance = ts.significance(n=1000,detrend=True,wL=0.4,bW=80)
>>> ar1_significance.plot()
>>> robustness = ts.robustness(indicators=['ar1','var'])
>>> robustness.plot()

References
----------
   
- Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H.,
  van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical 
  transitions. Nature, 461(7260), 53–59. https://doi.org/10.1038/nature08227

- Lenton, T. M., Livina, V. N., Dakos, V., van Nes, E. H., & Scheffer, M. (2012). Early 
  warning of climate tipping points from critical slowing down: Comparing methods to improve 
  robustness. Philosophical Transactions of the Royal Society A: Mathematical, Physical and 
  Engineering Sciences, 370(1962), 1185–1204. https://doi.org/10.1098/rsta.2011.0304

- Dakos, V., Carpenter, S. R., van Nes, E. H., & Scheffer, M. (2015). Resilience indicators:
  Prospects and limitations for early warnings of regime shifts. Philosophical Transactions 
  of the Royal Society B: Biological Sciences, 370(1659), 20130263. 
  https://doi.org/10.1098/rstb.2013.0263

- Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S.,
  Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for Detecting Early
  Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data. 
  PLoS ONE, 7(7), e41010. https://doi.org/10.1371/journal.pone.0041010

- Boers, N. (2021). Observation-based early-warning signals for a collapse of the Atlantic 
  Meridional Overturning Circulation. Nature Climate Change, 11(8), 680–688.
  https://doi.org/10.1038/s41558-021-01097-4 

Author
------
Beatriz Arellano-Nava (University of Exeter)
"""

import functools
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import kendalltau
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


class Ews(pd.DataFrame):
    """
    A subclass of pandas.DataFrame to compute early warning signals (EWS) 
    for critical slowing down in time series data.

    This class extends the pandas DataFrame to provide built-in methods for
    detrending and analysing time series using indicators of resilience.
    """
    def __init__(self, data, *args, **kwargs):
        # Initialize as a normal pandas DataFrame
        super().__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        """Ensure DataFrame operations return an instance of Ews."""
        return Ews

#    def __getitem__(self, key):
#        """
#        Overrideing __getitem__ to return Ews instance for column slicing.
#        """
#        result = super().__getitem__(key)        
#        # If a single column is sliced (result is a Series), return it as an Ews instance
#        if isinstance(result, pd.Series):
#            return Ews(result)  # Return the Ews class for single-column slicing        
#        # If multiple columns are sliced (result is a DataFrame), return it as an Ews instance
#        elif isinstance(result, pd.DataFrame):
#            return Ews(result)  # Return the Ews class for multiple-column slicing        
#        return result  # Default behavior for other cases (e.g., row slicing, DataFrame slicing)
    
    class Filtered_ts:
        """
        Container for the result after Gaussian detrending.

        Attributes
        ----------
        trend : pd.DataFrame
            The smoothed version of the original time series (the trend).
        res : Ews
            The residuals after subtracting the trend from the original time series.
            
        """
        def __init__(self,ts, trend):
            self.trend = trend
            self.res = Ews(ts - trend)
            
    def gaussian_det(self,bW, scale=True,**kwargs):
        """
        Applies Gaussian smoothing to detrend a time series.
        
        This method uses a Gaussian filter to smooth each column in the time series,
        extracting the trend component and returning both the trend and residual.

        
        Parameters
        ----------
        bW: int                        
            Bandwidth of the Gaussian smoother kernel. If `scale` is False, this is used 
            directly as the standard deviation (sigma) of the Gaussian filter. 
            If `scale` is True, the bandwidth is scaled such that ±0.25 * bW 
            corresponds to the interquartile range of the Gaussian probability 
            distribution.          
        scale: bool, optional
            Whether to scale the standard deviation of the kernel using the 
            interquartile approximation. Default is True.
        **kwargs:
            Additional keyword arguments passed to `scipy.ndimage.gaussian_filter`.
            
        Returns
        -------
        Filtered_ts
            An object containing the smoothed trend and the residuals as attributes:
                - trend: Pandas Series containing the filtered time-series.
                - res:   Pandas Series containing the residuals after filtering.
             
                
        Notes
        -----
        When `scale` is True, the bandwidth `bW` is scaled to match the interquartile 
        range of a Gaussian distribution. The standard deviation (sigma) of the 
        Gaussian kernel is computed as:
            sigma = 0.25 * (1 / 0.6745) * bW
        This ensures that the quartiles of the Gaussian distribution fall at 
        ±0.25 * bW, since the quartile positions in a Gaussian are at ±0.6745 * sigma.

        The filter is truncated at ±4 standard deviations (sigma), meaning the Gaussian 
        kernel only uses values within that range for smoothing, reducing the influence 
        of distant points.

        References
        ----------        
        - Lenton, T. M., Livina, V. N., Dakos, V., van Nes, E. H., & Scheffer, M. (2012). Early 
          warning of climate tipping points from critical slowing down: Comparing methods to improve 
          robustness. Philosophical Transactions of the Royal Society A: Mathematical, Physical and 
          Engineering Sciences, 370(1962), 1185–1204. https://doi.org/10.1098/rsta.2011.0304            

        Examples
        --------
        >>> noise = np.random.normal(0,20,1000)
        >>> ts = pd.Series(np.arange(0,100,0.1)*2+ noise)
        >>> ts = Ews(np.random.normal(size=1000))
        >>> trend = ts.gaussian_det(bW=30).trend
        >>> residuals = ts.gaussian_det(bW=30).res
        
        """
        if scale == True:
            sd = 0.25 * (1/0.6745) * bW
            kwargs['truncate'] = 4 * (0.6745) #Bandwidth expressed in number of standard deviations
        else:
            sd = bW
        def _apply_gaussian(ts,sd,**kwargs):
            trend = gaussian_filter(ts.dropna().values, sigma = sd, **kwargs)           
            trend_notna = Ews(pd.Series(trend, index=ts.dropna().index))
            df = pd.concat([ts,trend_notna],axis=1)
            trend = df.iloc[:, 1]
            return trend
        trend = self.apply(_apply_gaussian,axis=0,sd=sd,**kwargs)
        return self.Filtered_ts(self,trend)
    
    def validator(func):
        """
        Decorator for preprocessing and organizing keyword arguments 
        before calling early warning signal indicator functions.
        This decorator performs the following tasks:
        1. If the 'detrend' keyword argument is set to True, it applies 
           Gaussian detrending using the `gaussian_det` method.
        2. Separates keyword arguments intended for:
           - the Gaussian filter,
           - rolling window operations, and
           - the target function (the indicator itself).
        This ensures that each function only receives the arguments it requires.
        
        """
        @functools.wraps(func)
        def wrapper(inst,*args, **kwargs):
            """
            Internal wrapper function that receives and processes the instance and the keyword 
            arguments.
            """
            filt_args = set(inspect.signature(inst.gaussian_det).parameters.keys()).union(set(inspect.signature(gaussian_filter).parameters.keys()))                   
            detr_kwargs = {k: kwargs[k] for k in (kwargs.keys() & filt_args)} # Obtains the parameteres to be used for the gaussian filter            
            if 'detrend' in kwargs:            
                if kwargs['detrend'] is True:                                                
                    inst = inst.gaussian_det(**detr_kwargs).res    ## Gets the residuals from the gaussian_det function               
            # Obtains the parameteres for the rolling window, ignoring the detrend parameter as it's been used previously
            roll_args = set(inspect.signature(inst.rolling).parameters.keys()).union(set(inspect.signature(func).parameters.keys())).difference(detr_kwargs).difference({'detrend'})
            if 'indicator' in kwargs:  
                ### Arguments for the rolling window according to the selected indicator. The argument 'detrend' is ignored as it's been used before
                roll_args = roll_args.union(set(inspect.signature(getattr(inst,kwargs['indicator'])).parameters.keys())).difference(detr_kwargs).difference({'detrend'})                
            roll_kwargs = {k: kwargs[k] for k in (kwargs.keys() & roll_args)} #Obtains the keyword arguments to be used when rolling the window
            return func(inst, **roll_kwargs) ### Calls the function
        return wrapper
    
    @staticmethod
    def _window_size(ts,wL):
        """      
        Computes and validates the rolling window length for a time series.

        This method calculates the number of data points to use for a rolling 
        window based on a specified window length. If `wL` is less than or 
        equal to 1, it is treated as a fraction of the time series length. 
        Otherwise, it is interpreted as an absolute number of points. The method 
        also ensures that the resulting window size is valid (i.e., smaller than 
        the length of the time series).
    
        Parameters
        ----------
        ts : pandas.Series
            Input time series.
        wL : float
            Window length. If `wL` ≤ 1, it is treated as a fraction of the total 
            number of valid data points. If `wL` > 1, it is taken as an absolute 
            number of points.
        
        Returns
        -------
        ts : pandas.Series
            Trimmed time series containing only the values between the first and 
            last valid indices.
        wL : int
            Window length expressed as the number of data points.
            
        Raises
        ------
        ValueError
            If the computed window length is greater than the number of valid data 
            points in the time series.
            
        """

        ts =  ts.loc[ts.first_valid_index():ts.last_valid_index()]
        wL = math.floor(len(ts)*wL) if wL <= 1 else int(wL)
        if wL > len(ts):
            raise ValueError('Window length cannot be  greater than the time series length')
        return ts, wL
    
    
    @validator
    def ar1(self,detrend=False,wL=0.5,lag=1,**kwargs):
        """
        Computes the AR(1) coefficient over a rolling window.

        This method estimates the autoregressive coefficient of order 1 (AR(1)) 
        for a time series over a sliding window. The AR(1) model is fitted via 
        Ordinary Least Squares (OLS) using `statsmodels.tsa.AutoReg`.

        If `detrend=True`, the time series is first detrended using a Gaussian 
        filter before estimating the AR(1) coefficient.

        Parameters
        ----------
        detrend : bool, optional (default=False)
            Whether to detrend the time series before analysis using a Gaussian filter.
        wL : float or int, optional (default=0.5)
            Length of the rolling window. If `wL` ≤ 1, it is interpreted as a fraction 
            of the time series length. If `wL` > 1, it is treated as an absolute number 
            of data points.
        lag : int, optional (default=1)
            Lag to use for the autoregressive model.
        **kwargs : dict
            Additional keyword arguments passed to the rolling window function.

        Returns
        -------
        Ews
            A DataFrame containing the AR(1) coefficients computed for each window 
            along the time series.

        Notes
        -----
        This method detects changes in the similarity of the time series from 
        one time step to the next. An increasing similarity over time suggests 
        that the system is becoming slower to recover from perturbations, a 
        key indicator of critical slowing down and a potential early warning 
        of an approaching transition.

        References
        ----------           
        - Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H.,
          van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical 
          transitions. Nature, 461(7260), 53–59. https://doi.org/10.1038/nature08227
        
        - Lenton, T. M., Livina, V. N., Dakos, V., van Nes, E. H., & Scheffer, M. (2012). Early 
          warning of climate tipping points from critical slowing down: Comparing methods to improve 
          robustness. Philosophical Transactions of the Royal Society A: Mathematical, Physical and 
          Engineering Sciences, 370(1962), 1185–1204. https://doi.org/10.1098/rsta.2011.0304
        
        - Dakos, V., Carpenter, S. R., van Nes, E. H., & Scheffer, M. (2015). Resilience indicators:
          Prospects and limitations for early warnings of regime shifts. Philosophical Transactions 
          of the Royal Society B: Biological Sciences, 370(1659), 20130263. 
          https://doi.org/10.1098/rstb.2013.0263
        
        - Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S.,
          Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for Detecting Early
          Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data. 
          PLoS ONE, 7(7), e41010. https://doi.org/10.1371/journal.pone.0041010
        
        Examples
        --------
        >>> ts = Ews(np.random.normal(size=1000))
        >>> ar1_ts = ts.ar1(detrend=True,bW=100,wL=0.4)
        
        """
        def _estimate_ar1c(ts,wL,lag,**kwargs):
            #ar1cb = self.rolling(window=wL,**kwargs).apply(
            #        func=lambda x: sm.OLS(x[lag:], sm.add_constant(x[:-lag])).fit().params[1], raw=True)
            ts, wL = self._window_size(ts,wL)
            ar1c = ts.rolling(window=wL,**kwargs).apply(
                    func=lambda x: AutoReg(x, lags=[lag]).fit().params[1], raw=True)
            return ar1c
        ar1c = self.apply(_estimate_ar1c, axis=0,wL=wL,lag=lag,**kwargs)
        return Ews(ar1c)

    @validator
    def lambd(self,detrend=False,wL=0.5,lindetr=False,**kwargs):
        """
        Estimates the linear restoring rate (lambda) of a time series over a sliding window.

        This method computes the metric described by Boers (2021), which quantifies the
        system's tendency to return to equilibrium after perturbations. Variations in the 
        restoring rate (lambda) reflect changes in the system's stability. Lambda is 
        estimated by fitting a linear model between the rate of change in the signal and 
        its current state, capturing how strongly the system tends to return to its 
        previous state. The restoring rate λ is negative for stable system states, and 
        as the system loses stability, it approaches zero from below.

        Parameters
        ----------
        detrend : bool, optional
            If True, performs Gaussian detrending on the data before estimating lambda.
            Default is False.
        wL : float or int, optional
            Window length, either as a fraction of the series length or an integer.
            Default is 0.5.
        lindetr : bool, optional
            If True, fits and removes a linear trend from each window before
            estimating lambda. Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the Gaussian detrending and 
            pandas rolling functions.

        Returns
        -------
        Ews
            A pandas Series wrapped in an Ews object, containing lambda coefficients 
            (linear restoring rates) for each time-step.

        References
        ----------
        Boers, N. (2021). Observation-based early-warning signals for a collapse of the 
        Atlantic Meridional Overturning Circulation. Nature Climate Change, 11(8), 680–688.

        Notes
        -----
        This function is adapted from code in the Github repository:
        https://github.com/niklasboers/AMOC_EWS/blob/main/EWS_functions.py

        Examples
        --------
        >>> ts = Ews(np.random.normal(size=1000))
        >>> lambda_ts = ts.lambd(detrend=True,bW=100,wL=0.4)
        
        """
        def _estimate_lambda(ts,wL,lindetr,**kwargs):
            ts, wL = self._window_size(ts,wL)
            def _get_lambda_w(xw):
                """
                Calculates the linear restoring rate metric as described in the 2021 publication by Niklas Boers.
                
                This function is adapted from the code in the Github repository:
                https://github.com/niklasboers/AMOC_EWS/blob/main/EWS_functions.py
                """
                if lindetr == True:
                    xw = xw - xw.mean()                                 # Removes the mean
                    p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)  # Fits a linear trend and removes it
                    xw = xw - p0 * np.arange(xw.shape[0]) - p1
                dxw = xw[1:] - xw[:-1]                                 # Estimates delta x
                xw = sm.add_constant(xw)
                model = sm.GLSAR(dxw, xw[:-1], rho=1)
                results = model.iterative_fit(maxiter=10)
                lambda_coeff = results.params[1]
                return lambda_coeff
            
            lambdacoeff = ts.rolling(window=wL,**kwargs).apply(_get_lambda_w, raw=True)
            return lambdacoeff
        lambdacoeff = self.apply(_estimate_lambda, axis=0,wL=wL,lindetr=lindetr,**kwargs)
        return Ews(lambdacoeff)

    
    @validator
    def var(self,detrend=False,wL=0.5,**kwargs):       
        def _estimate_var(ts,wL,**kwargs):
            """
            Estimates the variance of a time series over a sliding window.

            This method computes the rolling variance of the input series,
            which can serve as an early-warning signal for critical transitions.
    
            Parameters
            ----------
            detrend : bool, optional
                If True, applies Gaussian detrending to the time series before
                estimating variance. Default is False.
            wL : float or int, optional
                Window length, either as a fraction of the series length or an integer.
                Default is 0.5.
            **kwargs : dict
                Additional keyword arguments passed to the Gaussian detrending and 
                pandas rolling functions.
    
            Returns
            -------
            Ews
                A pandas Series wrapped in an Ews object, containing the variance 
                estimates for each time-step.

            References
            ----------           
            - Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H.,
              van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical 
              transitions. Nature, 461(7260), 53–59. https://doi.org/10.1038/nature08227
            
            - Dakos, V., Carpenter, S. R., van Nes, E. H., & Scheffer, M. (2015). Resilience indicators:
              Prospects and limitations for early warnings of regime shifts. Philosophical Transactions 
              of the Royal Society B: Biological Sciences, 370(1659), 20130263. 
              https://doi.org/10.1098/rstb.2013.0263
            
            - Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S.,
              Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for Detecting Early
              Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data. 
              PLoS ONE, 7(7), e41010. https://doi.org/10.1371/journal.pone.0041010

            Examples
            --------
            >>> ts = Ews(np.random.normal(size=1000))
            >>> var_ts = ts.var(detrend=True,bW=100,wL=0.4)
            
            """
            ts,wL = self._window_size(ts,wL)
            vari = ts.rolling(window=wL,**kwargs).var()
            return vari
        vari = self.apply(_estimate_var, axis=0,wL=wL,**kwargs)        
        return Ews(vari)

    @validator
    def skw(self,detrend=False,wL=0.5,**kwargs):
       
        def _estimate_skw(ts,wL,**kwargs):
            """
            Estimates the skewness of a time series over a sliding window.

            This method calculates skewness for each time step using a rolling window, 
            providing a measure of asymmetry in the distribution of values.
        
            Parameters
            ----------
            detrend : bool, optional
                If True, applies Gaussian detrending before computing skewness. Default is False.
            wL : float or int, optional
                Window length as a fraction of the series length or as an integer. Default is 0.5.
            **kwargs :
                Additional keyword arguments passed to the rolling function (e.g., `min_periods`).
        
            Returns
            -------
            Ews
                A pandas Series wrapped in an Ews object, containing the skewness 
                estimates for each time-step.

            References
            ----------           
            - Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H.,
              van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical 
              transitions. Nature, 461(7260), 53–59. https://doi.org/10.1038/nature08227
            
            - Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S.,
              Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for Detecting Early
              Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data. 
              PLoS ONE, 7(7), e41010. https://doi.org/10.1371/journal.pone.0041010

            Examples
            --------
            >>> ts = Ews(np.random.normal(size=1000))
            >>> skw_series = ts.skw(detrend=True,bW=100,wL=0.4)

            """
            ts,wL = self._window_size(ts,wL)
            skw = ts.rolling(window=wL,**kwargs).skew()
            return skw
        skw = self.apply(_estimate_skw, axis=0,wL=wL,**kwargs)        
        return Ews(skw)
    
    @validator
    def pearsonc(self,detrend=False,wL=0.5,lag=1,**kwargs):
        """
        Estimates the Pearson autocorrelation coefficient over a sliding window.

        This method computes the Pearson correlation between the time series 
        and a version of itself shifted by a specified lag, using a rolling window. 
        It provides a measure of temporal autocorrelation, often used as an early warning 
        signal for critical transitions.
    
        Parameters
        ----------
        detrend : bool, optional
            If True, applies Gaussian detrending before computing autocorrelation. Default is False.
        wL : float or int, optional
            Window length as a fraction of the series length or as an integer. Default is 0.5.
        lag : int, optional
            Time lag to use for the autocorrelation calculation. Default is 1.
        **kwargs :
            Additional keyword arguments passed to the rolling function (e.g., `min_periods`).
    
        Returns
        -------
        Ews
            A pandas Series wrapped in an Ews object, containing the Pearson 
            autocorrelation coefficients for each time-step.

        References
        ----------           
        - Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H.,
          van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical 
          transitions. Nature, 461(7260), 53–59. https://doi.org/10.1038/nature08227
        
        - Lenton, T. M., Livina, V. N., Dakos, V., van Nes, E. H., & Scheffer, M. (2012). Early 
          warning of climate tipping points from critical slowing down: Comparing methods to improve 
          robustness. Philosophical Transactions of the Royal Society A: Mathematical, Physical and 
          Engineering Sciences, 370(1962), 1185–1204. https://doi.org/10.1098/rsta.2011.0304
        
        - Dakos, V., Carpenter, S. R., van Nes, E. H., & Scheffer, M. (2015). Resilience indicators:
          Prospects and limitations for early warnings of regime shifts. Philosophical Transactions 
          of the Royal Society B: Biological Sciences, 370(1659), 20130263. 
          https://doi.org/10.1098/rstb.2013.0263
        
        - Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S.,
          Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for Detecting Early
          Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data. 
          PLoS ONE, 7(7), e41010. https://doi.org/10.1371/journal.pone.0041010
        
        Examples
        --------
        >>> ts = Ews(np.random.normal(size=1000))
        >>> pearsonc_series = ts.pearsonc(detrend=True,bW=100,wL=0.4)
        
        """
        def _estimate_pearsonc(ts,wL,lag,**kwargs):
            ts,wL = self._window_size(ts,wL)
            pcor = ts.rolling(window=wL,**kwargs).apply(
                    func=lambda x: pd.Series(x).autocorr(lag=lag), raw=True)
            return pcor
        pearsonc = self.apply(_estimate_pearsonc, axis=0,wL=wL,lag=lag,**kwargs)
        return Ews(pearsonc)
    
    @property
    def kendall(self):
        """
        Estimates the strength and direction of the trend using the Kendall Tau correlation coefficient
        between the indicator time-series and time.
        
        The Kendall Tau coefficient measures the ordinal association between two variables, 
        in this case, the time series and time itself, providing an indication of monotonic trends. 
        Kendall Tau ranges from -1 to 1, where a value of 1 indicates a perfect increasing monotonic
        trend (i.e., both variables move in the same direction). A value of -1 indicates a perfect
        decreasing monotonic trend (i.e., the variables move in opposite directions). Values closer
        to 0 suggest a weaker monotonic relationship, where no clear increasing or decreasing trend 
        is observed.
        
        Returns
        -------
        float or pandas.Series
            The Kendall Tau coefficient for each indicator time series as a measure of the trend strength.
            If the time series contains a single column, returns a float; otherwise, returns a Series.

        References
        ---------- 
        - Lenton, T. M., Livina, V. N., Dakos, V., van Nes, E. H., & Scheffer, M. (2012). Early 
          warning of climate tipping points from critical slowing down: Comparing methods to improve 
          robustness. Philosophical Transactions of the Royal Society A: Mathematical, Physical and 
          Engineering Sciences, 370(1962), 1185–1204. https://doi.org/10.1098/rsta.2011.0304
        
        - Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S.,
          Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for Detecting Early
          Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data. 
          PLoS ONE, 7(7), e41010. https://doi.org/10.1371/journal.pone.0041010
            
        Examples
        --------
        >>> ts = Ews(np.random.normal(size=1000))
        >>> ar1_ts = ts.ar1(detrend=True,bW=100,wL=0.4)
        >>> ar1_ts.kendall
        0.87
        
        """
        def _estimate_kendall(ts):
            # if ts.index.dtype == 'datetime64[ns]':
            #     mannSer = np.arange(1,ts.dropna().index.size+1)
            # else:
            #     mannSer = ts.dropna().index
            # tsCorr = pd.Series(mannSer)
            # tsCorr.index = ts.dropna().index

            # kendall = ts.dropna().corr(tsCorr, method="kendall")
            kendall, _ = kendalltau(ts.dropna().values, np.arange(0,len(ts.dropna())))
            return kendall
        kendall = self.apply(_estimate_kendall, axis=0)
        kendall = float(kendall.iloc[0]) if len(self.columns)==1 else kendall
        return kendall
    
    class Significance_test:
        """
        Class for storing the results of a significance test for Kendall Tau correlation 
        coefficients and provide visualization of the statistical significance.

        Attributes
        ----------
        surrog_kendalls : pandas DataFrame
            The Kendall Tau coefficients computed from the surrogate time series.
        kendall_coeff : float or pandas Series
            The observed Kendall Tau coefficient from the original time series.
        pvalue : float or pandas Series
            The p-value indicating the statistical significance of the observed trend.
        test_type : str
            The type of test conducted, either 'positive' or 'negative'.
        indicator : str
            The statistical indicator used for the test (e.g., 'ar1', 'var', etc.).
        
        """
        def __init__(self,kendalls_surrog,kc,pval,test,indicator):
            self.surrog_kendalls = kendalls_surrog
            self.kendall_coeff = kc
            self.pvalue = pval
            self.test_type = test
            self.indicator = indicator
        def plot(self,nbins=30,signif_threshold=0.05):
            """
            Plots the distribution of Kendall Tau coefficients from the surrogate time series
            to visualize the probability that the observed trend in the original time series
            is due to chance.
    
            The plot displays histograms of the Kendall Tau coefficients for each surrogate series,
            with the observed Kendall Tau coefficient shown as a dashed red line. The p-value is annotated
            on the plot to indicate the statistical significance of the observed trend.
    
            Parameters
            ----------
            nbins : int, optional
                The number of bins to use in the histograms. Default is 30.
            signif_threshold : float, optional
                The threshold below which the p-value will be marked with an asterisk (*) to indicate statistical significance.
                Default is 0.05.
    
            Returns
            -------
            None
                This method generates a plot and does not return any values.
            Examples
            --------
            >>> ts = Ews(np.random.normal(size=1000))
            >>> ar1_significance = ts.significance(indicator='ar1',detrend=True,wL=0.4,bW=80)
            >>> ar1_significance.plot()
            
            """
            ncol = len(self.surrog_kendalls.columns)
            kwargsplot = {}
            ## Defines whether to put all plots in one row or in multiple rows
            if ncol < 7:
                nr,nc,figsize = 1, ncol, (2.7*ncol,2.3)
                kwargsplot['sharey'] = True
            else:
                nr,nc,figsize = ncol, 1, (2.7,2.3*ncol)
                kwargsplot['sharex'] = True
            larger_ylim = 0
            fig,axs = plt.subplots(nr,nc,figsize=figsize,**kwargsplot)
            for i,col in enumerate(self.surrog_kendalls.columns):
                ### Histogram
                ax = axs if len(self.surrog_kendalls.columns)==1 else axs[i]
                self.surrog_kendalls[col].hist(bins=nbins, ax=ax,grid=False,edgecolor = "black", color='tab:blue')
                kc = self.kendall_coeff if len(self.surrog_kendalls.columns)==1 else self.kendall_coeff[col]
                ax.axvline(kc,color='r',linestyle='dashed', linewidth=1.5) ## Kendall coefficient measured on the original series
                pval = self.pvalue[col] 
                psig = '*' if pval<signif_threshold else ''  ### Including the p-value as text
                comp = '<' if pval<=0.001 else '='
                pval = 0.001 if pval==0 else pval
                posp = 0.03 if self.test_type=='positive' else 0.4
                ax.text(posp, 0.9, f'p{comp}{pval:.3f}{psig}', transform=ax.transAxes, size=11)
                ax.set_xlim(-1.15,1.15)
                if 'sharey' in kwargsplot:
                    ax.set_xlabel(r'Kendall $\tau$',fontsize=12)
                    if i == 0:
                        ax.set_ylabel('Density',fontsize=12)
                elif 'sharex' in kwargsplot:
                    ax.set_ylabel('Density',fontsize=12)
                    if i == ncol-1:
                        ax.set_xlabel(r'Kendall $\tau$',fontsize=12)              
                larger_ylim = ax.get_ylim()[1] if ax.get_ylim()[1] > larger_ylim else larger_ylim
                ax.set_title(col,fontsize=13,weight='bold')
            if len(self.surrog_kendalls.columns) > 1:
                [axs[i].set_ylim(0,larger_ylim*1.1) for i in range(0,ncol)];
            else:
                axs.set_ylim(0,larger_ylim*1.1)
            
    
    @validator
    def significance(self, indicator='ar1',n=1000,detrend=False,wL=0.5,test='positive',**kwargs):
        """
        Performs a significance test for a given resilience indicator by comparing the 
        measured Kendall Tau coefficient of the original time series with the distribution 
        of Kendall Tau coefficients from an ensemble of surrogate series. The surrogate series 
        are generated by bootstrapping the residuals of the original series after detrending, 
        i.e., by sampling the residuals with replacement.
        
        The p-value is computed as the proportion of surrogate series exhibiting a trend 
        stronger than that of the original series. This approach tests whether the observed trend 
        in the resilience indicator is statistically significant relative to the null hypothesis 
        of no trend in the data.


        Parameters
        ----------
        indicator : str, optional
            The statistical indicator to compute, which can be 'ar1','lambd', 'var', 'pearsonc' or 'skw'. 
            Default is 'ar1'.
        n : int, optional
            The number of surrogate members to generate in the bootstrap ensemble. 
            Default is 1000.
        detrend : bool, optional
            If True, detrends the series before bootstrapping. Default is False.
        wL : float, optional
            The window length as a fraction of the time series. Default is 0.5.
        test : str, optional
            The type of test to perform. Options are 'positive' (test for increasing trends) 
            or 'negative' (test for decreasing trends). Default is 'positive'.
        **kwargs : keyword arguments
            Additional arguments passed to the relevant method for calculating the indicator.
            
        Returns
        -------
        Significance_test
            An instance of the Significance_test class, containing the bootstrapped 
            Kendall Tau coefficients, the observed Kendall coefficient, and the p-value 
            for the significance test.
    
        Notes
        -----
        The significance test compares the Kendall Tau coefficient for the observed series 
        to the distribution of Kendall Tau coefficients obtained from the surrogate series 
        generated by bootstrapping. A p-value is calculated as the proportion of 
        bootstrapped coefficients that are greater than or less than the observed coefficient, 
        depending on the specified test type ('positive' or 'negative').

        References
        ----------
        - Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S.,
          Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for Detecting Early
          Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data. 
          PLoS ONE, 7(7), e41010. https://doi.org/10.1371/journal.pone.0041010
    
        See Also
        --------
        kendalltau : Function used to calculate the Kendall Tau correlation coefficient.

        Examples
        --------
        >>> ts = Ews(np.random.normal(size=1000))
        >>> ar1_significance = ts.significance(indicator='ar1',detrend=True,wL=0.4,bW=80)
        
        """
        if test not in ['positive','negative']:
            print(f'{test} is not an option fot the argument test. Resetting to its default option.')
            test = 'positive'
        ## If detrend is True, the decorator will detrend the dataFrame and return 
        ## the residuals
        def _get_kendalls(ts,indicator,n,wL,**kwargs):
            """
            Estimates the Kendall values measured on the indicator's trend after 
            bootstrapping the time-series n times.
            Parameters
            ----------
            ts : pandas Series
            indicator : string
                statistical indicator: 'ar1', 'var','pearsonc'.
            Returns
            -------
            kendalls : list
               List containing n Kendall coefficients that resulted from 
               bootstrapping a time-series.

            """
            kendalls = []
            ts = ts.loc[ts.first_valid_index():ts.last_valid_index()]  ## Selecting the valid range of the series
            for i in range(0,n):                
                ## Bootstrapping the residuals to obtain a surrogate series
                surrogate_ts = Ews(pd.Series(np.random.choice(ts.values,len(ts))))             
                kc = getattr(surrogate_ts,indicator)(wL=wL,**kwargs).kendall ## Getting the kendall coefficient for each series
                kendalls.append(float(kc))
            return kendalls
        ## Obtaining Kendall coefficients from the surrogate series over each column of the DataFrame
        kendalls_bstr = self.apply(_get_kendalls, axis=0,indicator=indicator,n=n,wL=wL,**kwargs) 
        ### Getting the Kendall values from the original series
        kc = getattr(self,indicator)(wL=wL,**kwargs).kendall
        if test == 'positive':
            tail = kendalls_bstr[kendalls_bstr>=kc] ## Gets the Kendall coefficients that are larger or equal to the observed in the original series
        elif test == 'negative':
            tail = kendalls_bstr[kendalls_bstr<=kc]
        pval = tail.count()/n  ## getting the proportion of coefficients that are larger (less) than the observed one
        return self.Significance_test(kendalls_bstr,kc,pval,test,indicator) ## return an instance of the class Significance_test
    
    class Robustness_dict(dict):
        def plot(self,vmin=-0.2,vmax=1,cmap='Spectral_r', shading='auto',**kwargs):
            """
            Plots the obtained Kendall Tau coefficients for different combinations of 
            window size and detrending bandwidth using colormaps. 
    
            Parameters
            ----------
            vmin : float, optional
                The minimum value for the color scale. Default is -0.2.
            vmax : float, optional
                The maximum value for the color scale. Default is 1.
            cmap : str, optional
                The colormap to use for the heatmaps. Default is 'Spectral_r'.
            shading : str, optional
                The shading method for the heatmap. Default is 'auto'.
            **kwargs : additional keyword arguments, optional
                Any additional arguments to pass to the `pcolormesh` function for customizing the plot.
    
            Returns
            -------
            None
                This method generates a plot.
                
            Examples
            --------
            >>> ts = Ews(np.random.normal(size=1000))
            >>> robustness = ts.robustness(indicators=['ar1','var'])
            >>> robustness.plot()
            
            """
            keys = list(self.keys())
            if isinstance(self[keys[0]], dict):
                indicators = list(self[keys[0]].keys())
                nrows = len(keys)
                title = None
                cols = list(self.keys())
                nested_dict = True
            else:
                indicators = keys
                nrows = 1
                title = self[keys[0]].name
                cols = []
                nested_dict = False
            ind_labels = {'ar1':'AR(1)','var':'Variance','pearsonc': r'Pearson $r$',
                          'lambd':r'Restoring rate ($\lambda$)','skw':'Skewness'}
            fig,axs = plt.subplots(nrows,len(indicators),figsize=(2.7*len(indicators),2*nrows),squeeze=False,sharey='row',gridspec_kw={'hspace': 0.4})
            for nr in range(nrows):
                for i,ind in enumerate(indicators):
                    if nested_dict:
                        colorplot = axs[nr,i].pcolormesh(self[cols[nr]][ind].columns, self[cols[nr]][ind].index,self[cols[nr]][ind],vmin=vmin,vmax=vmax,cmap=cmap,shading=shading,**kwargs)
                    else:
                        colorplot = axs[nr,i].pcolormesh(self[ind].columns, self[ind].index,self[ind],vmin=vmin,vmax=vmax,cmap=cmap, shading=shading,**kwargs)
                    if nr == nrows - 1:
                        axs[nr,i].set_xlabel('Window length',fontsize=12)
                    if nr == 0:
                        axs[nr,i].set_title(ind_labels[ind],fontsize=13,pad=15,weight='bold')
                cbar = plt.colorbar(colorplot, ax=axs[nr,:],pad=0.02)
                cbar.ax.set_ylabel(r'Kendall $\tau$', fontsize=12)
                axs[nr,0].set_ylabel('Bandwidth',fontsize=11)
                axs[nr,0].text(len(indicators)/2, 1.03,cols[nr] if nested_dict else title, transform=axs[nr,0].transAxes, size=13,weight='bold')                

    
    def robustness(self, indicators=['ar1','var'],min_wL=0.2,max_wL=0.7,res_wL=15,min_bW=0.1,max_bW=0.6,res_bW=5,**kwargs):
        """
        Computes the robustness of the measured trends in each resilience indicator by calculating 
        the Kendall Tau coefficient for a combination of different window lengths and detrending 
        bandwidths for each time-series in the dataframe.

        Parameters
        ----------
        indicators : list, optional
            A list of indicators to compute the robustness for. Default is ['ar1', 'var'].
            Each indicator represents a specific method for calculating the robustness of the time-series data.
            
        min_wL : float or int, optional
            The minimum window length to consider for the robustness calculation. Default is 0.2.
            
        max_wL : float or int, optional
            The maximum window length to consider for the robustness calculation. Default is 0.7.
            
        res_wL : int, optional
            The resolution (step size) for window lengths. Default is 15.
            
        min_bW : float or int, optional
            The minimum bandwidth to consider for the robustness calculation. Default is 0.1.
            
        max_bW : float or int, optional
            The maximum bandwidth to consider for the robustness calculation. Default is 0.6.
            
        res_bW : int, optional
            The resolution (step size) for bandwidths. Default is 5.
            
        **kwargs : additional keyword arguments, optional
            Any other arguments to pass to the underlying methods or calculations.
            
        Returns
        -------
            A dictionary containing the robustness analyses for each indicator. 
            The dictionary is structured with columns as keys, where each value is a 
            dictionary containing a DataFrame of Kendall Tau coefficients 
            for each combination of bandwidth and window length.

        References
        ----------           
        - Lenton, T. M., Livina, V. N., Dakos, V., van Nes, E. H., & Scheffer, M. (2012). Early 
          warning of climate tipping points from critical slowing down: Comparing methods to improve 
          robustness. Philosophical Transactions of the Royal Society A: Mathematical, Physical and 
          Engineering Sciences, 370(1962), 1185–1204. https://doi.org/10.1098/rsta.2011.0304
        
        - Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S.,
          Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for Detecting Early
          Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data. 
          PLoS ONE, 7(7), e41010. https://doi.org/10.1371/journal.pone.0041010
            
        Examples
        --------
        >>> ts = Ews(np.random.normal(size=1000))
        >>> robustness = ts.robustness(indicators=['ar1','var'])
        
        """
        def _get_robustness_ts(ts,name,indicators=['ar1','var'],min_wL=0.2,max_wL=0.7,res_wL=15,min_bW=0.1,max_bW=0.6,res_bW=5,**kwargs):
            """
            Helper function that calculates the Kendall Tau correlation coefficient 
            for a single time-series `ts` using different window lengths and bandwidths 
            for each resilience indicator.
            
            Parameters
            ----------
            ts : pandas.Series
                The time-series data to compute the robustness for.
            name : string
                The name of the time-series data (column name).
                
            Returns
            -------
            kendalls : Dictionary
                A `Robustness_dict` object containing the Kendall Tau coefficients for each combination 
                of bandwidth and window length for each indicator.
                
            """
            _,min_wL = self._window_size(ts,min_wL)
            _,max_wL = self._window_size(ts,max_wL)
            _,min_bW = self._window_size(ts,min_bW)
            _,max_bW = self._window_size(ts,max_bW)
            bW_v = np.arange(min_bW,max_bW,res_bW)
            wL_v = np.arange(min_wL,max_wL,res_wL)
            ts = Ews(ts.loc[ts.first_valid_index():ts.last_valid_index()])
            kendalls_arr = {ind:np.zeros((len(bW_v),len(wL_v))) for ind in indicators}
            kendalls = {}
            ### Computing the Kendall coefficient for each combination of parameters
            ### Numpy vectorize is slightly faster than the for loops when computing only one indicator,
            ### Thus, I use for loops here for now, there might be a more efficient way to do it
            for i,b in enumerate(bW_v):     ## Iterating over bandwidths and window lengths
                for j,w in enumerate(wL_v):
                    for ind in indicators:    ## Iterating over indicators
                        kendalls_arr[ind][i][j] = float(getattr(ts,ind)(detrend=True, bW=b, wL=w).kendall)                    
            for ind in indicators:
                kendalls[ind] = pd.DataFrame(kendalls_arr[ind], columns=wL_v,index=bW_v)    ## Converting into dataframes
                kendalls[ind].index.name = 'Bandwidth'
                kendalls[ind].indicator = ind   ## Adds metadata to the dataframes to identify the indicator and column name
                kendalls[ind].name = name
            return self.Robustness_dict(kendalls)
        
        robustness_dict = {} ## Returns a dictionary with the robustness analyses for each one of the indicators
        for col in self.columns:  
            robustness_dict[col] = _get_robustness_ts(self[col],col,indicators=indicators,min_wL=min_wL,max_wL=max_wL,res_wL=res_wL,min_bW=min_bW,max_bW=max_bW,res_bW=res_bW,**kwargs)
        return self.Robustness_dict(robustness_dict)

    
            
            
            
            
            

