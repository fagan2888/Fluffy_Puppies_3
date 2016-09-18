#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 21:30:37 2016

@author: kunmingwu
"""

import numpy as np
import pandas as pd
import interest_rates as ir


def summer_index_func(t):
    # starting from July, therefore plus 6
    t = t + 6
    return 1 if t%12 in [5,6,7,8] else 0

def prepayment_hazard_rate_arr(gamma, p, beta, libor_10yr_lag_3m_arr, libor_1m_arr, mortgage_type):
    #here the time input t should be integer month
    beta = np.asarray(beta).flatten()
    t_range = np.arange(1,num_months+1)
    summer_index_arr = [summer_index_func(t) for t in t_range]
    if mortgage_type == 'F':
        v = np.zeros((num_months,2))
        v[:,0] = fixed_wac - libor_10yr_lag_3m_arr
        v[:,1] = summer_index_arr
        exp_val = np.dot(v, beta)
        up = (gamma * p * (gamma * t_range)**(p-1))
        down = (1 + (gamma * t_range)**p)
        mid = np.exp(exp_val) 
        return np.multiply(np.divide(up,down),mid)
    elif mortgage_type == 'A':
        v = np.zeros((num_months,2))
        v[:,0] = libor_1m_arr + spread_on_libor_1m - libor_10yr_lag_3m_arr
        v[:,1] = summer_index_arr
        exp_val = np.dot(v, beta)
        up = (gamma * p * (gamma * t_range)**(p-1))
        down = (1 + (gamma * t_range)**p)
        mid = np.exp(exp_val) 
        return np.multiply(np.divide(up,down),mid)
    else:
        print('incorrect mortgage type for prepayment_hazard_rate_arr')
        return

def SMM_P_func_arr(gamma, p, beta, libor_10yr_lag_3m_arr, libor_1m_arr, mortgage_type):
    return (1-np.exp(-prepayment_hazard_rate_arr(gamma, p, beta, libor_10yr_lag_3m_arr, libor_1m_arr, mortgage_type)*1))


def get_SMM_P_matrices(gamma, p, beta, libor_10yr_lag_3m_matrix, libor_1m_matrix, mortgage_type, num_sims):
    SMM_matrix = []
    for i in range(num_sims):
        libor_10yr_lag_3m_arr = libor_10yr_lag_3m_matrix[i,:]
        libor_1m_arr = libor_1m_matrix[i,:]
        SMM_matrix.append(SMM_P_func_arr(gamma, p, beta, libor_10yr_lag_3m_arr, libor_1m_arr, mortgage_type))
    return SMM_matrix

def default_hazard_rate(loan_balance, house_price, t, mortgage_type):
    #here the time input t should be integer month
    if mortgage_type == 'F':
        gamma = gamma_D_F
        p = p_D_F
        beta = beta_D_F
    elif mortgage_type == 'A':
        gamma = gamma_D_A
        p = p_D_A
        beta =  beta_D_A
    else:
        print('incorrect mortgage type for default_hazard_rate')
    ltv = loan_balance / house_price
    exp_val = ltv * beta
    up = (gamma * p * (gamma * t)**(p-1))
    down = (1 + (gamma * t)**p)
    mid = np.exp(exp_val) 
    return up / down * mid

def SMM_D_func(loan_balance, house_price, t, mortgage_type):
    return (1-np.exp(-default_hazard_rate(loan_balance, house_price, t, mortgage_type)*1))
    
def house_price_GBM(r_arr, H0):
    iterations = len(r_arr)
    H_arr = [H0]
    H_anti_arr = [H0]
    H = H0
    H_anti = H0

    q = 0.025
    dt = 1/12
    phi = 0.12
    norm_arr = np.random.normal(0,1,size=iterations)    # normal random vars
    norm_anti_arr = [-x for x in norm_arr] 

    for i in range(iterations):
        dH = (r_arr[i]-q) * H * dt + phi * H * norm_arr[i]
        dH_anti = (r_arr[i]-q) * H * dt + phi * H * norm_anti_arr[i]
        H += dH
        H_anti += dH_anti
        H_arr.append(H)
        H_anti_arr.append(H_anti)

    return 0.5 * (np.array(H_arr) + np.array(H_anti_arr))



spread_on_libor_1m = 0.055
fixed_wac = 0.07419
num_months = 315

## Parameteres estimated from MATLAB

hazard_params = [[0.0097, 1.5832, -100.0000, -0.2145], 
                 [0.0194, 8.0000, 2.1237, np.nan],
                 [0.0045, 1.8651, 38.6501, 0.0144],
                 [0.0200, 8.0000, -0.0267, np.nan]]

hazard_params = np.matrix(hazard_params).T
hazard_params_df = pd.DataFrame(hazard_params)
hazard_params_df.columns = ['ARM prepayment','ARM default','FRM prepayment','FRM default']
gamma_D_F = hazard_params_df.loc[0,'FRM default']
p_D_F = hazard_params_df.loc[1,'FRM default']
beta_D_F = hazard_params_df.loc[2,'FRM default']
gamma_D_A = hazard_params_df.loc[0,'ARM default']
p_D_A = hazard_params_df.loc[1,'ARM default']
beta_D_A = hazard_params_df.loc[2,'ARM default']

gamma_P_F = hazard_params_df.loc[0,'FRM prepayment']
p_P_F = hazard_params_df.loc[1,'FRM prepayment']
beta_P_F = hazard_params_df.loc[(2,3),'FRM prepayment'].values
gamma_P_A = hazard_params_df.loc[0,'ARM prepayment']
p_P_A = hazard_params_df.loc[1,'ARM prepayment']
beta_P_A = hazard_params_df.loc[(2,3),'ARM prepayment'].values


         

SMM_P_F = get_SMM_P_matrices(gamma_P_F, p_P_F, beta_P_F, ir.libor_10yr_lag_3m_matrix, ir.libor_1m_matrix, 'F', ir.num_sims)
SMM_P_A = get_SMM_P_matrices(gamma_P_A, p_P_A, beta_P_A, ir.libor_10yr_lag_3m_matrix, ir.libor_1m_matrix, 'A', ir.num_sims)


#SMM_P_F = get_SMM_P_matrices(gamma_P_F, p_P_F, beta_P_F, ir.final_libor_10yr_lag_3m_matrix, ir.final_libor_1m_matrix, 'F', ir.num_sims)
#SMM_P_A = get_SMM_P_matrices(gamma_P_A, p_P_A, beta_P_A, ir.final_libor_10yr_lag_3m_matrix, ir.final_libor_1m_matrix, 'A', ir.num_sims)

frm_H0 = 52416155/0.856
arm_H0 = 226122657/0.856
House_F_m = [house_price_GBM(r_arr, frm_H0) for r_arr in ir.r_matrix]
House_A_m = [house_price_GBM(r_arr, arm_H0) for r_arr in ir.r_matrix]

#House_F_m = [house_price_GBM(r_arr, frm_H0) for r_arr in ir.final_r_matrix]
#House_A_m = [house_price_GBM(r_arr, arm_H0) for r_arr in ir.final_r_matrix]
