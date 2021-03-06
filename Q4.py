import pandas as pd
import numpy as np 
import water_fall as wf
import interest_rates as ir

"""
	Principal amounts as of 2015.

	Fixed payments are:
		M2_premium, M5_premium
		M2_coupon * number of period, M5_coupon * number of periods
"""

def Calc_CDS_Legs(M_principal, M_coupon, M_total_buyer_payment,
				M_CF_principal, M_CF_prepay, M_CF_default, M_CF_neg_int, rf_arr):
	"""
		Do for both M2 and M5.
		Payments for seller and buyer made at end of each month.
	"""

	M_total_seller_payment = 0

	arr = []
	arr2 = []
	arr3 = []

	for t in range(num_periods):
		M_missing_principal = 0 	# --> assumption: principal is never guaranteed 
		M_principal_payment = M_CF_principal[t]
		M_missing_interest = M_CF_neg_int[t]
		M_defaulted_principal = M_CF_default[t]
		M_prepayment = M_CF_prepay[t]

		### updated principal = current principal - defaulted principal
		### buyer fixed coupon payment now based on updated principal
		### --> treating M_missing_interest as dollar amount, NOT basis point
		M_principal -= (M_principal_payment + M_prepayment + M_defaulted_principal)

		if M_principal <= 0:
			print ('Principal goes negative at time step t = ', t)
			break

		M_total_buyer_payment += (M_principal * M_coupon * np.exp(-rf_arr[t]))

		arr.append(M_principal * M_coupon * np.exp(-rf_arr[t]))
		arr2.append(M_coupon)
		arr3.append(np.exp(-rf_arr[t]))

		# seller pays out (1 - recovery rate) * defaulted principal
		# sellers pays out 100% of interest and principal shortfalls
		M_total_seller_payment += (M_defaulted_principal * (1 - recovery_rate)) *  np.exp(-rf_arr[t])
		M_total_seller_payment += (M_missing_principal + M_missing_interest) *  np.exp(-rf_arr[t])

	return M_total_seller_payment, M_total_buyer_payment

def Check_Correct_Position(M_buyer_payment, M_seller_payment, tranche):
		print ("For the " + tranche + " tranche, want to be in the position of CDS ", end=' ')
		if M_buyer_payment > M_seller_payment:
			print ("buyer")
		else:
			print ("seller")

### All values given in assignment
M2_principal = 30150000.0
M2_premium_mult = 33.50
M2_premium = (M2_principal / 100.0) * M2_premium_mult
M2_coupon = 17e-4/12.0

M5_principal = 15075000.0
M5_premium_mult = 72.50
M5_premium = (M5_principal / 100.0) * M5_premium_mult
M5_coupon = 44e-4/12.0


"""
	Calculating fixed payments
"""
recovery_rate = 0.40

M2_total_buyer_payment = M2_premium # initial amount as upfront premium
M5_total_buyer_payment = M5_premium 

num_sims = ir.num_sims
num_periods = 315

r_matrix_arr = [ir.r_matrix, ir.r_anti_matrix]


for i in [4, 7]:		# 4 = M2;  7 = M5
	M_seller_array = []
	M_buyer_array = []

	if i == 4:
		M_principal = M2_principal 
		M_coupon = M2_coupon
		M_total_buyer_payment = M2_total_buyer_payment
	elif i == 7:
		M_principal = M5_principal 
		M_coupon = M5_coupon
		M_total_buyer_payment = M5_total_buyer_payment

	M_CF_total = wf.CF_total[i]
	M_CF_principal = wf.CF_principal[i]
	M_CF_prepay = wf.CF_prepay[i]
	M_CF_default = wf.CF_default[i]
	M_CF_interest = wf.CF_interest[i]
	M_CF_neg_int = wf.CF_neg_int[i]

	# for r_matrix and r_anti_matrix
	for j in range(len(r_matrix_arr)):
		r_matrix = r_matrix_arr[j]

		M_buyer_payments = []
		M_seller_payments = []

		for k in range(num_sims):
			### reset total buuyer payment each path to initial premium
			if i == 4:
				M_total_buyer_payment = M2_total_buyer_payment
			elif i == 7:
				M_total_buyer_payment = M5_total_buyer_payment

			M_total_seller_payment, M_total_buyer_payment = \
				Calc_CDS_Legs(M_principal, M_coupon, M_total_buyer_payment, M_CF_principal[k], 
					M_CF_prepay[k], M_CF_default[k], M_CF_neg_int[k], ir.r_matrix[k])

			M_buyer_payments.append(M_total_buyer_payment)
			M_seller_payments.append(M_total_seller_payment)

		M_buyer_array.append(M_buyer_payments)
		M_seller_array.append(M_seller_payments)

	if i == 4:
		M2_avg_buyer_payment = (np.mean(M_buyer_array[0]) + np.mean(M_buyer_array[1])) / 2.0
		M2_avg_seller_payment = (np.mean(M_seller_array[0]) + np.mean(M_seller_array[1])) / 2.0
	elif i == 7:
		M5_avg_buyer_payment = (np.mean(M_buyer_array[0]) + np.mean(M_buyer_array[1])) / 2.0
		M5_avg_seller_payment = (np.mean(M_seller_array[0]) + np.mean(M_seller_array[1])) / 2.0


print (M2_avg_buyer_payment, M2_avg_seller_payment)
print (M5_avg_buyer_payment, M5_avg_seller_payment)

Check_Correct_Position(M2_avg_buyer_payment, M2_avg_seller_payment, "M2")
Check_Correct_Position(M5_avg_buyer_payment, M5_avg_seller_payment, "M5")

"""
	interpret results as which position is more profitable. 

	buyer is long CDS, short credit risk
	seller is short CDS, long credit risk

	if M{2,5}_CDS_value > 0 --> want to be short CDS (i.e. seller position)
	if M{2,5}_CDS_value < 0 --> want to be long CDS (i.e. buyer position)	
"""

"""
	--> Still need to figure out what valuation errors we are 
		making by modeling at pool-level.
"""













