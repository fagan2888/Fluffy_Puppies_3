#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:02:29 2016

@author: kunmingwu
"""


import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm

import Discount_Functions as disc_func
import Hull_White as hw

import imp
hw = imp.reload(hw)


def calibrate_HW(kappa,sigma):
    n = len(data_vol["Price/Vols"])
    vol_fwd = hw.fwd_vol_HW(kappa, sigma, data_z.loc[3:n+3-1,'Maturity'].values, \
                            data_z.loc[4:n+4-1,'Maturity'].values)
    Z0 = disc_func.poly5_to_Z(z_OLS,data_z.loc[3:n+3-1,'Maturity'].values)
    Z1 = disc_func.poly5_to_Z(z_OLS,data_z.loc[4:n+4-1,'Maturity'].values)
    M = 1.0 + r_cap * delta
    K = 1.0 / M
    d1 = np.log(Z1 / (K * Z0)) / vol_fwd + 0.5 * vol_fwd
    d2 = d1 - vol_fwd
    return M * (K * Z0 * norm.cdf(-d2) - Z1 * norm.cdf(-d1))

def obj_func(x):
    kappa = x[0]
    sigma = x[1]
    return np.sqrt(np.sum(((calibrate_HW(kappa,sigma) - cap_prices))**2))

def calibrate_HW_optimization_res():
    kappa_inputs = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    sigma_inputs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    opt_res_arr = []
    for k in kappa_inputs:
        for s in sigma_inputs:
            opt_res = sp.optimize.minimize(obj_func,(k,s))      
            opt_res_arr.append((opt_res.x,obj_func(opt_res.x)))
            
    opt_res_df = pd.DataFrame(opt_res_arr)
    opt_res_df.columns = ['kappa and sigma', 'error']
    opt_res_df = opt_res_df.sort_values('error')
    return opt_res_df




## LIBOR rates
def libor_rate_10yr_lag_3m(r_arr, num_months):
    libor_arr = [0.0348, 0.0299, 0.0339, 0.0322]
    t_range = np.arange(1, num_months)
    for t in t_range:
        r = r_arr[t]
        r_10yr = r_arr[t+10*12]
        libor_arr.append((r_10yr*(t+10*12) - r*(t)) / (t+10*12 - t))
    return libor_arr

def get_libor_matrix_10yr_lag_3m(r_matrix, num_sims, num_months):
    libor_rate_matrix = []
    for i in range(num_sims):
        libor_rate_matrix.append(libor_rate_10yr_lag_3m(r_matrix[i,:], num_months))
    libor_rate_matrix = np.matrix(libor_rate_matrix)
    return libor_rate_matrix

def libor_rate_1m(r_arr, num_months):
    libor_arr = []
    t_range = np.arange(1, num_months)
    for t in t_range:
        r = r_arr[t]
        r_1m = r_arr[t+1]
        libor_arr.append((r_1m*(t+1) - r*(t)) / (t+1 - t))
    return libor_arr

def get_libor_matrix_1m(r_matrix, num_sims, num_months):
    libor_rate_matrix = []
    for i in range(num_sims):
        libor_rate_matrix.append(libor_rate_1m(r_matrix[i,:], num_months))
    libor_rate_matrix = np.matrix(libor_rate_matrix)
    return libor_rate_matrix


    
    


# Parameters setup
r_cap = 0.0325
short_rate = r0 = 0.01160
delta = 0.25


# Data files setup
data_z = pd.read_csv("Discount_Factors.csv")
data_vol = pd.read_csv("Caplet_Vols.csv")


## Estimating coeff of term structure
data_z["poly"] = np.log(data_z["Price"])
z_OLS = disc_func.OLS(disc_func.power_5(data_z["Maturity"]), data_z["poly"])
print("my estimation of coefficients are:")
print(z_OLS.beta)
(a,b,c,d,e) = z_OLS.beta
coeff = [a,b,c,d,e]


## Using the balck formula, translate to caplet price from vol
cap_prices = disc_func.Black_formula(lambda T: disc_func.poly5_to_Z(z_OLS,T), \
                                     data_vol["Maturity"], r_cap, delta, data_vol["Price/Vols"])



