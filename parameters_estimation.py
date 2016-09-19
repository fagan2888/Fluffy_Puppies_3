#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:16:27 2016

@author: kunmingwu
"""
import pandas as pd
import interest_rates as ir

## Make CSV data files for modeling prepayment, default rates
data_arm = pd.read_csv('ARM_perf_s-1.csv')
data_frm = pd.read_csv('FIX_perf_s-2.csv')

data_arm.columns = ['id', 'age', 'spread_squared', 'spread', 'ltv', 'remaining_balance',\
                    'summer_indicator', 'default_indicator', 'prepayment_indicator']
data_frm.columns = ['id', 'age', 'spread_squared', 'spread', 'ltv', 'remaining_balance',\
                    'summer_indicator', 'default_indicator', 'prepayment_indicator']

## FRM
fixed_wac = 0.07419

def frm_coupon_gap_helper(m):
    return fixed_wac - ir.avg_libor_10yr_lag_3m_arr[m]
data_frm['coupon_gap'] = [frm_coupon_gap_helper(m) for m in data_frm['age']]

## ARM
spread_on_index = 0.0550
def arm_coupon_gap_helper(m):
    return spread_on_index + ir.avg_libor_1m_arr[m] - ir.avg_libor_10yr_lag_3m_arr[m]
data_arm['coupon_gap'] = [arm_coupon_gap_helper(m) for m in data_arm['age']]

data_arm.to_csv('Modified_ARM_perf_s-1.csv', header=False, index=False)
data_frm.to_csv('Modified_FRM_perf_s-2.csv', header=False, index=False)