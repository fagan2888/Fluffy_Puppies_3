M = csvread('ARM_perf_s-1.csv');
[nrow, ~] = size(M);
ts = zeros(nrow,1);
tf = M(:,2);
coupon_gap = M(:,4)/100;
ltv = M(:,5);
summer_ind = M(:,7);
default_event = M(:,8);
prepayment_event = M(:,9);

fprintf('Estimating prepayment rate parameters for ARM...\n')
fprintf('Output order is [gamma], [p], [beta_coupon_gap], [beta_summer_ind]\n')
prop_haz('log',ts,tf,prepayment_event,[coupon_gap, summer_ind]);

fprintf('Estimating default rate parameters for ARM...\n')
fprintf('Output order is [gamma], [p], [ltv]\n')
prop_haz('log',ts,tf,default_event,ltv);

M = csvread('FIX_perf_s-2.csv');
[nrow, ~] = size(M);
ts = zeros(nrow,1);
tf = M(:,2);
coupon_gap = M(:,4)/100;
ltv = M(:,5);
summer_ind = M(:,7);
default_event = M(:,8);
prepayment_event = M(:,9);

fprintf('Estimating prepayment rate parameters for FRM...\n')
fprintf('Output order is [gamma], [p], [beta_coupon_gap], [beta_summer_ind]\n')
prop_haz('log',ts,tf,prepayment_event,[coupon_gap, summer_ind]);

fprintf('Estimating default rate parameters for FRM...\n')
fprintf('Output order is [gamma], [p], [ltv]\n')
prop_haz('log',ts,tf,default_event,ltv);

