# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:03:42 2016

@author: linshanli
"""
import numpy as np
import hazard_rates as hr
import interest_rates as ir

## Globla variables
J = 315
Path = ir.num_sims
FRM_WAC = 0.07419
ARM_spread = 0.055
FRM_monthly_wac = FRM_WAC/12
B_F_0 = 79036000
B_A_0 = 714395000


tranches = ["A1","A2","A3","M1","M2","M3","M4","M5","M6","M7","M8","M9","M10"]
num_tr = len(tranches)
B_tr0 = np.array([0, 107769000, 24954000, 38481000, 30150000, 18646000,16265000,15075000,
                       13488000, 13092000, 619000,0,0])
Spread_tr = np.array([0.08,0.18,0.28,0.36,0.38,0.39,0.51,0.55,0.62,1.15,1.40,2.25,2.25])/100

def PMT(remaining_principal,monthly_interest,payment_number,i):
    # remaining_principal = M = B
    # monthly_interest = coupon_rate/n, where n -12
    # payment_number = N, total number of payments
    # i = t-1, which is the ith time in the time iteration
    monthly_payment = remaining_principal * \
        (monthly_interest * (1 + monthly_interest) ** (payment_number - i))/ \
        ((1 + monthly_interest) ** (payment_number - i) - 1)   
    return monthly_payment

def SMM_D(B, H, t, mortgage_type): ##Need to be updated as expoential of negative integral of LTV
    return hr.SMM_D_func(B,H,t,mortgage_type)
    
def waterfall_helper(P, B_tr, flag):
    R = P
    if(R > sum(B_tr)):
        B_tr = np.zeros(len(B_tr))
        return B_tr, B_tr
    CF = np.zeros(len(B_tr))        
    if(flag == 'desc'):    
        for k in range(len(B_tr)):
            if R < B_tr[k]:
                B_tr[k] -= R
                CF[k] += R
                R = 0
            else:
                R -= B_tr[k]
                CF[k] +=  B_tr[k]
                B_tr[k] = 0
            if R == 0:
                break
        return B_tr, CF
    elif(flag == 'asc'):
        for k in np.arange(len(B_tr))[::-1]:
            if R < B_tr[k]:
                B_tr[k] -= R
                CF[k] += R
                R = 0

            else:
                R -= B_tr[k]
                CF[k] +=  B_tr[k]
                B_tr[k] = 0                
            if R == 0:
                break
        return B_tr,CF
    else:
        print("waterfall_helper: wrong flag")
        return B_tr, CF

def water_fall(B_F_t, FRM_monthly_wac, B_A_t, Libor_t, House_F_m_t, House_A_m_t, B_tr, B_F_0_ = B_F_0, B_A_0_ = B_A_0):
    #B_t = B_F_t + B_A_t
    #B_0 = B_F_0_ + B_A_0_
    
    ## FRM
    PM_F = PMT(B_F_t, FRM_monthly_wac, J, j)
    I_F = B_F_t * FRM_monthly_wac
    P_F = PM_F - I_F
    PP_F = SMM_P_arr_F[j] * (B_F_t - P_F)
    DP_F = SMM_D(B_F_t, House_F_m_t, j, 'F') * (B_F_t - P_F)
    
    SMM_P_t = SMM_P_arr_F[j]
    SMM_D_t = SMM_D(B_F_t, House_F_m_t, j, 'F')

    ## ARM
    PM_A = PMT(B_A_t, (Libor_t+ARM_spread)/12, J, j)
    I_A = B_A_t * (Libor_t+ARM_spread)/12
    P_A = PM_A - I_A
    PP_A = SMM_P_arr_A[j] * (B_A_t - P_A)
    DP_A = SMM_D(B_A_t, House_A_m_t, j, 'A') * (B_A_t - P_A)
    
    
    ## Aggregation
    I = I_F + I_A
    P = P_F + P_A
    PP = PP_F + PP_A
    DP = DP_F + DP_A
    
    
    I_tr = np.zeros(num_tr)
    for k in range(num_tr):
        I_tr[k] = B_tr[k]*(Libor_t+Spread_tr[k])/12
    ES = I - sum(I_tr) #Excess Spread
    ES_init = ES
    
    if (ES<0):
        print("Warning: In path {} period {},ES<0, ES = {} setting ES = 0".format(i,j,ES))    
        ES = 0
    Neg_int_tr, _ = waterfall_helper(I, I_tr, 'desc')
    if (ES>DP):
        ES -= DP
        DP = 0
    else:
        DP -= ES
        ES = 0
    '''
    OTA = max([0.031*B_0, min([0.031*B_0, 0.062*B_t]), 3967158]) #Overcollateralization Target Amount
    OA = max([0,B_t - sum(B_tr)]) #Overcollateralization Amount
    if (OA>DP):
        OA -= DP
        DP = 0
    else:
        DP -= OA
        OA = 0
    
    #OTA = min([OTA,OA])
    
    EPDA = min([ES,OTA]) #Extra Principal Distribution Amount. 
    '''
    EPDA = ES
    
    B_tr, CF_Principal = waterfall_helper(P, B_tr, flag = 'desc')
    B_tr, CF_Prepay = waterfall_helper(PP, B_tr, flag = 'desc')
    B_tr, CF_Default = waterfall_helper(DP, B_tr, flag = 'asc')
    B_tr, CF_EPDA = waterfall_helper(EPDA, B_tr, flag = 'desc')
    
    ratio = B_F_t/(B_F_t+B_A_t)
    
    B_F_t = max([0,B_F_t - P_F - PP_F - DP_F])
    B_A_t = max([0,B_A_t - P_A - PP_A - DP_A])
    B_F_t = B_F_t + ES_init * ratio
    B_A_t = B_A_t + ES_init * (1-ratio)
    #total_pay = P_F + PP_F + DP_F +P_A + PP_A + DP_A +EPDA
    #total = B_F_t + B_A_t
    CF_total = CF_Principal+CF_Prepay+CF_Default+CF_EPDA + I_tr
    return CF_total, CF_Principal, CF_Prepay+CF_EPDA, CF_Default, I_tr, B_F_t, \
           B_A_t, B_tr, Neg_int_tr, SMM_P_t, SMM_D_t
    


## dummy variable to be change        
Libor_m = np.matrix(ir.final_libor_1m_matrix)
#Libor_m = np.matrix(ir.final_libor_1m_matrix)
House_F_m = np.matrix(hr.House_F_m)
House_A_m = np.matrix(hr.House_A_m)

## storage
CF_total = np.zeros((num_tr,Path, J))
CF_principal = np.zeros((num_tr,Path, J))
CF_prepay = np.zeros((num_tr,Path, J))
CF_default = np.zeros((num_tr,Path, J))
CF_interest = np.zeros((num_tr,Path, J))
CF_neg_int = np.zeros((num_tr,Path, J))

SMM_P_m = np.zeros((Path, J))
SMM_D_m = np.zeros((Path, J))
#i = 0  # ith path
#j = 0  # jth period
for i in range(Path):
    #print('.',end='')
    #initialization for each path
    B_F_t = 52416155
    B_A_t = 226122657
    B_tr = np.array([0, 107769000, 24954000, 38481000, 30150000, 18646000,16265000,15075000,
                       13488000, 13092000, 619000,0,0])
    SMM_P_arr_F = hr.SMM_P_F[i]
    SMM_P_arr_A = hr.SMM_P_A[i]
    for j in range(J):
        Libor_t = Libor_m[i,j]
        House_F_m_t = House_F_m[i,j]
        House_A_m_t = House_A_m[i,j]
        
        CF_total_ij, CF_principal_ij, CF_prepay_ij, CF_default_ij, CF_interest_ij, \
        B_F_t, B_A_t, B_tr , CF_neg_int_ij, SMM_P_ij, SMM_D_ij= \
            water_fall(B_F_t, FRM_monthly_wac, B_A_t, Libor_t, House_F_m_t, House_A_m_t, B_tr)
        
        CF_total[:,i,j]=CF_total_ij
        CF_principal[:,i,j]=CF_principal_ij
        CF_prepay[:,i,j]=CF_prepay_ij
        CF_default[:,i,j]=CF_default_ij
        CF_interest[:,i,j]=CF_interest_ij
        CF_neg_int[:,i,j] = CF_neg_int_ij
        SMM_P_m[i,j] = SMM_P_ij
        SMM_D_m[i,j] = SMM_D_ij
