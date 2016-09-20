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
import matplotlib.pyplot as plt


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


# Data files setup
data_z = pd.read_csv("Discount_Factors.csv")
data_vol = pd.read_csv("Caplet_Vols.csv")


## Estimating coeff of term structure
data_z["poly"] = np.log(data_z["Price"])
z_OLS = disc_func.OLS(disc_func.power_5(data_z["Maturity"]), data_z["poly"])
(a,b,c,d,e) = z_OLS.beta
coeff = [a,b,c,d,e]

# Parameters setup
r_cap = 0.0325
short_rate = r0 = 0.01160
#r0 = -coeff[0]
delta = 0.25


## Using the balck formula, translate to caplet price from vol
cap_prices = disc_func.Black_formula(lambda T: disc_func.poly5_to_Z(z_OLS,T), \
                                     data_vol["Maturity"]-delta, r_cap, delta, data_vol["Price/Vols"])

# Q1 Calibrate Hull White
print('Running optimization to calibrate Hull White model...')
opt_res_df = calibrate_HW_optimization_res()

# Pick the best parameters from above, sigma should take + sign
kappa, sigma = opt_res_df.iloc[0,0]


sigma = np.abs(sigma)

#plt.plot(ir.calibrate_HW(0.56,0.028),label='HW Caplet Price')

'''
plt.plot(calibrate_HW(kappa,sigma),label='HW Caplet Price')
plt.plot(cap_prices,label='Actual Caplet Price')
plt.legend()
plt.show()



## For testing purpose
coeff = [-0.017913777607645898,
 -0.0031169760803777535,
 0.00014448384285259408,
 -2.9207141816756099e-06,
 2.1831624297643029e-08]
kappa = 0.30490045697501322
sigma = 0.020644080934538542

'''

## Similate the LIBOR rate with Hull White
HW = hw.Hull_White()
num_sims = 1000
num_months = 315
T = num_months/12
dt = 1/12
r_cap = 0.0325
r0 = 0.01160
#r0 = -ir.coeff[0]
delta = 0.25
coeff = coeff

## Monte Carlo Simulation
print('Running Monte Carlo simulation to obtain short rates...')
cum_df_matrix, cum_df_anti_matrix, r_matrix, r_anti_matrix = HW.Monte_Carlo(kappa, sigma, r0, T+10, dt, coeff, num_sims)

final_r_matrix = 0.5 * (r_matrix + r_anti_matrix)
final_cum_df_matrix = 0.5 * (cum_df_matrix + cum_df_anti_matrix)

libor_10yr_lag_3m_matrix = get_libor_matrix_10yr_lag_3m(r_matrix, num_sims, num_months-4+1) #we have 4 numbers already
libor_10yr_lag_3m_anti_matrix = get_libor_matrix_10yr_lag_3m(r_anti_matrix, num_sims, num_months-3)
final_libor_10yr_lag_3m_matrix = 0.5 * (libor_10yr_lag_3m_matrix + libor_10yr_lag_3m_anti_matrix)
avg_libor_10yr_lag_3m_arr = np.asarray(final_libor_10yr_lag_3m_matrix.mean(0))[0]

libor_1m_matrix = get_libor_matrix_1m(r_matrix, num_sims, num_months+1)
libor_1m_anti_matrix = get_libor_matrix_1m(r_anti_matrix, num_sims, num_months+1)
final_libor_1m_matrix = 0.5 * (libor_1m_matrix + libor_1m_anti_matrix)
avg_libor_1m_arr = np.asarray(final_libor_1m_matrix.mean(0))[0]

'''
plt.plot(avg_libor_10yr_lag_3m_arr,label=' Avg LIBOR 10yr lag 3m')
plt.plot(avg_libor_1m_arr,label='Avg LIBOR 1m')
plt.legend()
plt.show()
'''

print('Finished Importing interest_rates')