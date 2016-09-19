#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:48:36 2016

@author: kunmingwu
"""

import interest_rates as ir
import water_fall as wf
import numpy as np


num_bonds = 13
bond_prices = np.zeros((ir.num_sims, num_bonds))
for i in range(num_bonds):
    cash_flow = wf.CF_total[i,:,:]
    discount_factor = ir.cum_df_matrix
    discount_factor = discount_factor[:,:315]
    discounted_cf = np.multiply(cash_flow, discount_factor)
    bond_prices[:,i] = np.sum(discounted_cf, axis=1)

avg_bond_price = bond_prices.mean(0)