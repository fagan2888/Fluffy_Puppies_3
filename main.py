#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:01:24 2016

@author: kunmingwu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Hull_White as hw
import HW3_MBS_Module as myMBS

import imp
hw = imp.reload(hw)
myMBS = imp.reload(myMBS)


# Q1 Calibrate Hull White
opt_res_df = myMBS.calibrate_HW_optimization_res()

# Pick the best parameters from above, sigma should take + sign
kappa, sigma = opt_res_df.iloc[0,0]
sigma = np.abs(sigma)
plt.plot(myMBS.calibrate_HW(0.56,0.028),label='HW Caplet Price')
#plt.plot(myMBS.calibrate_HW(kappa,sigma),label='HW Caplet Price')
plt.plot(myMBS.cap_prices,label='Actual Caplet Price')
plt.legend()
plt.show()


## Similate the LIBOR rate with Hull White
HW = hw.Hull_White()
num_sims = 1000
num_months = 315
T = num_months/12
dt = 1/12
r_cap = 0.0325
r0 = 0.01160
#r0 = -myMBS.coeff[0]
delta = 0.25
coeff = myMBS.coeff

## Monte Carlo Simulation
cum_df_matrix, cum_df_anti_matrix, r_matrix, r_anti_matrix = HW.Monte_Carlo(kappa, sigma, r0, T+10, dt, coeff, num_sims)

libor_rate_10yr_lag_3m_matrix = myMBS.get_libor_matrix_10yr_lag_3m(r_matrix, num_sims, num_months-4+1) #we have 4 numbers already
libor_rate_10yr_lag_3m_anti_matrix = myMBS.get_libor_matrix_10yr_lag_3m(r_anti_matrix, num_sims, num_months-3)
final_libor_rate_10yr_lag_3m_matrix = 0.5 * (libor_rate_10yr_lag_3m_matrix + libor_rate_10yr_lag_3m_anti_matrix)
avg_libor_rate_10yr_lag_3m_arr = np.asarray(final_libor_rate_10yr_lag_3m_matrix.mean(0))[0]

libor_rate_1m_matrix = myMBS.get_libor_matrix_1m(r_matrix, num_sims, num_months+1)
libor_rate_1m_anti_matrix = myMBS.get_libor_matrix_1m(r_anti_matrix, num_sims, num_months+1)
final_libor_rate_1m_matrix = 0.5 * (libor_rate_1m_matrix + libor_rate_1m_anti_matrix)
avg_libor_rate_1m_arr = np.asarray(final_libor_rate_1m_matrix.mean(0))[0]

plt.plot(avg_libor_rate_10yr_lag_3m_arr,label='LIBOR 10yr lag 3m')
plt.plot(avg_libor_rate_1m_arr,label='LIBOR 1m')
plt.legend()
plt.show()



## Q2
hazard_params = [[0.0097, 1.5832, -100.0000, -0.2145], 
                 [0.0194, 8.0000, 2.1237, np.nan],
                 [0.0045, 1.8651, 38.6501, 0.0144],
                 [0.0200, 8.0000, -0.0267, np.nan]]
hazard_params_df = pd.DataFrame(hazard_params)
hazard_params_df.columns = ['gamma', 'p', 'beta_1', 'beta_2']
hazard_params_df['type'] = ['FRM prepayment','FRM default','ARM prepayment','ARM default']
hazard_params_df

spread_on_libor_1m = 0.055
fixed_wac = 0.07419

arm_coupon_gap = avg_libor_rate_1m_arr + spread_on_libor_1m - avg_libor_rate_10yr_lag_3m_arr
frm_coupon_gap = fixed_wac - avg_libor_rate_10yr_lag_3m_arr
t_range = np.arange(0,315+1)
summer_ind = [myMBS.summer_index_func(t) for t in t_range]

frm_H0 = 52416155/0.856
arm_H0 = 226122657/0.856
home_price_matrix = [myMBS.home_price_GBM(r_arr, frm_H0) for r_arr in r_matrix]
home_prince_anti_matrix = [myMBS.home_price_GBM(r_arr, frm_H0) for r_arr in r_anti_matrix]
final_home_price_matrix = 0.5 * (np.matrix(home_price_matrix) + np.matrix(home_prince_anti_matrix))
avg_home_price_arr = np.asarray(final_home_price_matrix.mean(0))[0][:315]
plt.plot(avg_home_price_arr)