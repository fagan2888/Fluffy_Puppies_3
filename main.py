#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:01:24 2016

@author: kunmingwu
"""

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from scipy.stats import norm

import Discount_Functions as disc_func
import Hull_White as hw

import imp
hw = imp.reload(hw)

# Data files setup
data_z = pd.read_csv("Discount_Factors.csv")
data_vol = pd.read_csv("Caplet_Vols.csv")

# Parameters setup
r_cap = 0.0325
short_rate = r0 = 0.01160
delta = 0.25

# Q1 Calibrate Hull White

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
    return np.sqrt(np.sum(((calibrate_HW(kappa,sigma) - cap_prices))**2)) * 10000

## Calibrating Hull White
kappa_inputs = [0, 0.2, 0.5, 1.0]
sigma_inputs = [0, 0.05, 0.1, 0.5, 1.0]
for k in kappa_inputs:
    for s in sigma_inputs:
        opt_res = sp.optimize.minimize(obj_func,(k,s))
        print('initial guess:', (k,s))
        print('kappa, sigma = ',opt_res.x)
        print('error = ',obj_func(opt_res.x))

kappa = 3
sigma = -0.19
t_range = np.arange(1.0/24.0, 30, 2.0/24.0)
HW = hw.Hull_White()
theta_arr = HW.Plot_Theta(t_range, kappa, sigma, coeff)
