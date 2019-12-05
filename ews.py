#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:19:31 2019


@author: polaris
"""
import math
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
#from scipy.signal import find_peaks

def ar1(ts, bandWidth = 30, wL = 0.5):
    ts = pd.Series(ts)        
    ts_smooth = pd.Series(gaussian_filter(ts.dropna().values, sigma = bandWidth), index = ts.dropna().index)
    ts_res = ts - ts_smooth
    wLength = math.floor(len(ts)*wL) if wL <= 1 else wL
    var = ts_res.rolling(window=wLength).var()
    ac1 = ts_res.rolling(window=wLength).apply(
        func=lambda x: pd.Series(x).autocorr(lag=1), raw=True)
    return {'tsmooth': ts_smooth,'var': var, 'ar1': ac1} 
    #peaks, _ = find_peaks(np.abs(tp), height = 0.0, prominence=0.05)
    
def ar1_kendall(ts, wL = 0.5, bandWidth = 30):
    acr = ar1(ts, bandWidth, wL)['ar1']
    ts2 = pd.Series(acr.dropna().index,acr.dropna().index)
    kendallAC = acr.dropna().corr(ts2, method="kendall")
    return kendallAC

def kendall_coeff(ts):
    ts2 = pd.Series(ts.dropna().index,ts.dropna().index)
    kendall = ts.dropna().corr(ts2, method="kendall")
    return kendall
