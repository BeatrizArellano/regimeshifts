#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:52:51 2019

@author: polaris
"""

import math
import numpy as np


def sample_rs():
    """
    Returns a sample time series with a regime shift
    """
    mu = (2/9)*math.sqrt(3)     #Bifurcation parameter
    t = np.arange(0,999,1)   #Time steps
    m = mu*t/900              #The change in the bifurcation parameter over time (this is the length of t compared to mu which is a single value)

    a = np.full(len(t)+1,np.nan)    #setting up vector to hold created time series
    a[0] = -1                 #start it in the left well
    for i,e in enumerate(m):    
        a[i+1] = a[i] + (1/2)*((-a[i]**3) + a[i] + e) + 0.1*np.random.normal() #uses forward euler to run the model over times
      
    return a