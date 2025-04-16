#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:19:31 2019

This module contains the class Ews, which extends a Pandas series adding 
methods to estimate changes in resilience on a time-series. 

.. currentmodule:: regimeshifts.ews

Detrending
==========================================
    gaussian_det

Resilience metrics
==========================================
    ar1
    var
    pearsonc
    skw
    lambd
    
Trend's strength
=========================================
    kendall
    
Significance test
=========================================
    bootstrap


@author: Beatriz Arellano-Nava
"""
import functools
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import kendalltau


class Ews(pd.DataFrame):
    """
    Ews (Early Warning Signals) extends the methods of a Pandas Series 
    to include useful tools to estimate changes in autocorrelation.
    """
    # def __init__(self,column_labels,**kwargs):
    #     if column_labels:
    #         self.columns = column_labels
    
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
        This function to be used as a decorator performs 3 tasks:            
            - Calls the gaussian_det function according to the value of
              the detrend parameter.
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
        Estimates the window length size and validates that is smaller than the 
        time-series.
        Parameters
        ----------
        ts : Pandas Series
        wL : float
            Window length.

        Raises
        ------
        ValueError
            Raises an error when the window is larger than the series.

        Returns
        -------
        wL : float
            window length expressed as number of data points.
        """

        ts =  ts.loc[ts.first_valid_index():ts.last_valid_index()]
        wL = math.floor(len(ts)*wL) if wL <= 1 else int(wL)
        if wL > len(ts):
            raise ValueError('Window length cannot be  greater than the time series length')
        return ts, wL
    
    
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
    def lambd(self,detrend=False,wL=0.5,**kwargs):
        """
        Estimates the linear restoring rate (lambda) for a time series over a sliding window.

        This method implements the metric described by Boers (2021), calculating the system linear restoring rate (lambda) around a given stable equilibrium state.

        Parameters:
            detrend (bool, optional): If True, performs Gaussian detrending on the data before estimation. Default is False.
            wL (float, optional): Window length as a fraction of the series or as an integer. Default is 0.5.
            **kwargs: Additional keyword arguments passed to the pandas rolling function.
        
        Returns:
            pandas.Series: Lambda coefficients (linear restoring rates) for each time-step.
        
        References:
        Boers, N. (2021). Observation-based early-warning signals for a collapse of the Atlantic Meridional Overturning Circulation. 
        Nature Climate Change, 11(8), 680–688.
        """
        def _estimate_lambda(ts,wL,**kwargs):
            ts, wL = self._window_size(ts,wL)
            def _get_lambda_w(xw):
                """
                Calculates the linear restoring rate metric as described in the 2021 publication by Niklas Broers.
                
                This function is adapted from the code in the Github repository:
                https://github.com/niklasboers/AMOC_EWS/blob/main/EWS_functions.py

                References:
                Boers, N. (2021). Observation-based early-warning signals for a collapse of the Atlantic Meridional Overturning Circulation. 
                Nature Climate Change, 11(8), 680–688.
                """
                xw = xw - xw.mean()
                p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)
                xw = xw - p0 * np.arange(xw.shape[0]) - p1
                dxw = xw[1:] - xw[:-1]
                xw = sm.add_constant(xw)
                model = sm.GLSAR(dxw, xw[:-1], rho=1)
                results = model.iterative_fit(maxiter=10)
                lambda_coeff = results.params[1]
                return lambda_coeff
            
            lambdacoeff = ts.rolling(window=wL,**kwargs).apply(_get_lambda_w, raw=True)
            return lambdacoeff
        lambdacoeff = self.apply(_estimate_lambda, axis=0,wL=wL,**kwargs)
        return Ews(lambdacoeff)
    
    @validator
    def var(self,detrend=False,wL=0.5,**kwargs):
       
        def _estimate_var(ts,wL,**kwargs):
            """
            Estimates variance along the sliding window over a pandas Series
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
            Estimates skewness along the sliding window over a pandas Series
            """
            ts,wL = self._window_size(ts,wL)
            skw = ts.rolling(window=wL,**kwargs).skew()
            return skw
        skw = self.apply(_estimate_skw, axis=0,wL=wL,**kwargs)        
        return Ews(skw)
    
    @validator
    def pearsonc(self,detrend=False,wL=0.5,lag=1,**kwargs):
        """
        Estimates the Pearson correlation coefficients between the time series 
        and itself shifted by lag.        
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
        Estimates the Kendall Tau correlation coefficient between the 
        indicator time series and time.
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
        kendall = float(kendall) if len(self.columns)==1 else kendall
        return kendall
    
    class Significance_test:
        """
        
        """
        def __init__(self,kendalls_surrog,kc,pval,test,indicator):
            self.surrog_kendalls = kendalls_surrog
            self.kendall_coeff = kc
            self.pvalue = pval
            self.test_type = test
            self.indicator = indicator
        def plot(self,nbins=30,signif_threshold=0.05):
            """
            Plots the distribution of Kendall coefficients measured on the
            surrogate series to visualise the probability that the measured
            trend on the original series is obtained by chance.

            Parameters
            ----------
            nbins : int, optional
                DESCRIPTION. The default is 30.
            signif_threshold : float, optional
                DESCRIPTION. The default is 0.05.
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
        Creates an ensemble of n members in which each member has the same
        length as the original timeseries and its elements are obtained
        sampling from the residuals (after detrending) with replacement.
        Returns an array with the kendall value of the AR(1) or Variance
        changes for each ensemble member.    
        
        Parameters
        ----------
        indicator : string, optional
            DESCRIPTION. The default is 'ar1'.
        n : int, optional
            DESCRIPTION. The default is 1000.
        detrend : boolean, optional
            DESCRIPTION. The default is False.
        wL : float, optional
            DESCRIPTION. The default is 0.5.
        test : string, optional
            DESCRIPTION. The default is 'positive'.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        An instance of the class Significance_test
            DESCRIPTION.

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
            
            Parameters
            ----------
            vmin : TYPE, optional
                DESCRIPTION. The default is -0.2.
            vmax : TYPE, optional
                DESCRIPTION. The default is 1.
            cmap : TYPE, optional
                DESCRIPTION. The default is 'Spectral_r'.
            shading : TYPE, optional
                DESCRIPTION. The default is 'auto'.
            **kwargs : TYPE
                DESCRIPTION.

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
            ind_labels = {'ar1':'AR(1)','var':'Variance','pearsonc': r'Pearson $r$'}
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

        Parameters
        ----------
        indicators : list, optional
            DESCRIPTION. The default is ['ar1','var'].
        min_wL : float, optional
            DESCRIPTION. The default is 0.2.
        max_wL : float, optional
            DESCRIPTION. The default is 0.7.
        res_wL : float, optional
            DESCRIPTION. The default is 15.
        min_bW : float, optional
            DESCRIPTION. The default is 0.1.
        max_bW : float, optional
            DESCRIPTION. The default is 0.6.
        res_bW : float, optional
            DESCRIPTION. The default is 5.
        **kwargs : 
            DESCRIPTION.

        Returns
        -------
        Dictionary
            DESCRIPTION.
        """
        def _get_robustness_ts(ts,name,indicators=['ar1','var'],min_wL=0.2,max_wL=0.7,res_wL=15,min_bW=0.1,max_bW=0.6,res_bW=5,**kwargs):
            """
            Parameters
            ----------
            ts : TYPE
                DESCRIPTION.
            name : string
                
            Returns
            -------
            kendalls : Dictionary
                DESCRIPTION.
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
        
    
    @property
    def _constructor(self):
        """
        Overriding constructor properties to return an instance of Ews after
        performing an operation on the Pandas Dataframe
        """
        return Ews
    
    # @property
    # def _constructor_sliced(self):
    #     return Ews

            
            
            
            
            

