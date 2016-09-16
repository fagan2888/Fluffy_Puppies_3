#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:56:40 2016

@author: kunmingwu
"""



## Outputs from MATLAB
'''
Estimating prepayment rate parameters for ARM...
Output order is [gamma], [p], [beta_coupon_gap], [beta_summer_ind]
Performing Maximum Likelihood with log-logistic Distribution...
 
LnLike      26850.61081
haz_params  0.00969    1.58324 -100.00000   -0.21451
param_se    0.00059  0.02369  7.59822  0.03535
t_stats     16.3169  66.8209 -13.1610  -6.0675
gradient    -0.00000  0.00000  3.38972  0.00000
 
Elapsed time is 14.462471 seconds.
Estimating default rate parameters for ARM...
Output order is [gamma], [p], [beta_coupon_gap], [beta_summer_ind]
Performing Maximum Likelihood with log-logistic Distribution...
 
LnLike      13294.16118
haz_params  0.01935  8.00000  2.12374
param_se    0.00043  0.11581  0.19580
t_stats     45.4522  69.0770  10.8463
gradient    0.00000 -50.90830   0.00000
 
Elapsed time is 15.169382 seconds.
Estimating prepayment rate oarameters for FRM...
Output order is [gamma], [p], [ltv]
Performing Maximum Likelihood with log-logistic Distribution...
 
LnLike      30101.46499
haz_params  0.00452   1.86511  38.65015   0.01444
param_se    0.00028  0.02441  4.68571  0.03314
t_stats     16.1371  76.4084   8.2485   0.4358
gradient    0.82594  0.00187 -0.00004 -0.00003
 
Elapsed time is 32.452440 seconds.
Estimating default rate oarameters for FRM...
Output order is [gamma], [p], [ltv]
Performing Maximum Likelihood with log-logistic Distribution...
 
LnLike      5307.56975
haz_params  0.02003  8.00000 -0.02669
param_se    0.00041  0.20704  0.14031
t_stats     48.7447  38.6393  -0.1902
gradient    0.00007 -17.38296   0.00000
 
Elapsed time is 18.341323 seconds.
'''



## Make CSV data files for modeling prepayment, default rates
data_arm = pd.read_csv('ARM_perf_s-1.csv')
data_frm = pd.read_csv('FIX_perf_s-2.csv')

data_arm.columns = ['id', 'age', 'spread_squared', 'spread', 'ltv', 'remaining_balance',\
                    'summer_indicator', 'default_indicator', 'prepayment_indicator']
data_frm.columns = ['id', 'age', 'spread_squared', 'spread', 'ltv', 'remaining_balance',\
                    'summer_indicator', 'default_indicator', 'prepayment_indicator']

## FRM
fixed_wac = 0.07419
coupon_gap = 0.07419

def frm_coupon_gap_helper(m):
    return fixed_wac - avg_libor_rate_10yr_lag_3m_arr[m]
data_frm['coupon_gap'] = [frm_coupon_gap_helper(m) for m in data_frm['age']]

## ARM
spread_on_index = 0.0550
def arm_coupon_gap_helper(m):
    return spread_on_index + avg_libor_rate_1m_arr[m] - avg_libor_rate_10yr_lag_3m_arr[m]
data_arm['coupon_gap'] = [arm_coupon_gap_helper(m) for m in data_arm['age']]

