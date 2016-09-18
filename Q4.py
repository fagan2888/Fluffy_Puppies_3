import pandas as pd
import numpy as np 

"""
	Principal amounts as of 2015.

	Fixed payments are:
		M2_premium, M5_premium
		M2_coupon * number of period, M5_coupon * number of periods
"""
### All values given in assignment
M2_principal = 30150.0
M2_premium_mult = 33.50
M2_premium = (M2_principal / 100.0) * M2_premium_mult
M2_coupon = 17e-4

M5_principal = 15075.0
M5_premium_mult = 72.50
M5_premium = (M5_principal / 100.0) * M5_premium_mult
M5_coupon = 44e-4


"""
	Calculating fixed payments
"""

M2_default, M5_default = False, False
M2_differed_interest = 0
M5_differed_interest = 0

num_periods = 315 
recovery_rate = 0.40

M2_total_buyer_payment = M2_premium 
M5_total_buyer_payment = M5_premium 
M2_total_seller_payment = 0
M5_total_seller_payment = 0

M2_defaulted_principal = 0
M5_defaulted_principal = 0

libor_arr = np.ones(num_periods)	# risk-free discounting array

""" Need to discount all payments by LIBOR at that time step."""

def Calc_CDS_Legs(M_principal, M_coupon, M_total_buyer_payment, num_periods):
	"""
		Do for both M2 and M5.
		Payments for seller and buyer made at end of each month.
	"""
	M_principal_payment = M_missing_interest = M_missing_principal = \
	M_total_seller_payment = M_defaulted_principal = M_prepayment = 0

	for t in range(num_periods):
		if M_principal <= 0:
			break

		"""
			Get principal payment for time t for tranche M{2,5}
			Get defaulted principal for time t for tranche M{2,5}
			Get missing interest for time t for tranche M{2,5}
			Get missing principal for time t for tranche M{2,5}
			Get prepayment for time t for tranche M{2,5}
		"""

		# M_principal_payment = ...[t]
		# M_missing_principal = ...[t]
		# M_missing_interest = ...[t]
		# M_defaulted_principal = ...[t]
		# M_prepayment = ...[t]

		### updated principal = current principal - defaulted principal
		### buyer fixed coupon payment now based on updated principal

		### --> treating M_missing_interest as dollar amount, NOT basis point

		M_principal -= (M_principal_payment + M_prepayment + M_defaulted_principal)

		M_total_buyer_payment += ((M_principal) * \
									M_coupon * np.exp(-libor_arr[t]))

		# seller's formula simplified, need to check with TA to line up with formula
		# seller pays out (1 - recovery rate) * defaulted principal
		# sellers pays out 100% of interest and principal shortfalls
		M_total_seller_payment += (M_defaulted_principal * (1 - recovery_rate)) *  np.exp(-libor_arr[t])
		M_total_seller_payment += (M_missing_principal + M_missing_interest) *  np.exp(-libor_arr[t])

		"""
			check if M2 and M5 have defaulted on any of their principal.
			check for any interest guaranteed to M2 and M5 but not given.
			   if there is add to M{2,5}_differed_interest.
		"""

	return M_total_seller_payment, M_total_buyer_payment

M2_total_seller_payment, M2_total_buyer_payment = \
	Calc_CDS_Legs(M2_principal, M2_coupon, M2_total_buyer_payment, num_periods)

M5_total_seller_payment, M5_total_buyer_payment= \
	Calc_CDS_Legs(M5_principal, M5_coupon, M5_total_buyer_payment, num_periods)

### Calculate final value as difference of seller and buyer payments
M2_CDS_value = M2_total_seller_payment - M2_total_buyer_payment
M5_CDS_value = M5_total_seller_payment - M5_total_buyer_payment

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













