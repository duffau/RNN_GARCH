import pandas as pd
import numpy as np
import scipy.optimize as op
from GARCHnumpy import GARCHnumpy as garch
import time
import json

df = pd.read_csv('./data/sp500.csv',index_col='Date',parse_dates=True).sort_index()
df['return'] = np.log(df['Close']).diff()
df = df.dropna()

print(df.shape)

est_index = df.index[0:4495]
print('From ',min(est_index))
print('To   ',max(est_index))

est_data = pd.DataFrame()
est_data['return']  = df.loc[est_index,'return']
mu       = np.mean(est_data['return'])  
est_data['return_dm']  = est_data['return'] - mu
variance = np.var(est_data['return'])
print('average  = ', mu)
print('variance = ', variance)
print('omega    = ', variance*(1-0.088-0.86))

model = garch()
theta0 = np.array([variance*(1-0.088-0.86),0.088,0.86])
gamma0 = np.log(theta0)
log_like, est_data['sigma2_non-opt'] = model.log_likelihood(gamma=gamma0,y=est_data['return_dm'],fmin=False)
print(log_like)

# Estimate model
res = op.minimize(	model.log_likelihood,
						x0=gamma0,
						args=(est_data['return_dm'],True),
						method='BFGS')
print(res)
print('theta0    =',theta0)
print('theta_opt =',np.exp(res.x))

# Saving results to text-file
est_dct = {'timestamp':str(time.strftime("%a %d %m %Y %H:%M", time.gmtime())),
		   'period':[str(min(est_index)),str(max(est_index))],
		   'n_obs':len(est_data['return']),
		   'mean':mu,
		   'theta_opt':list(np.exp(res.x)),
		   'gamma_opt':list(res.x),
		   'log_like':res.fun}

file_name = 'GARCH_est_' + time.strftime("%a_%d-%m-%Y_%H.%M", time.gmtime())+'.json'
json.dump(est_dct,open(file_name,'w'))

# log_like, est_data['sigma2_opt'] = model.log_likelihood(gamma=res.x,y=est_data['return'],fmin=False)
# print(log_like)

# import matplotlib.pyplot as plt
# plt.subplot(211)
# est_data['return'].plot()

# plt.subplot(212)
# np.sqrt(est_data['return']**2).plot()
# np.sqrt(est_data['sigma2_non-opt']).plot()
# np.sqrt(est_data['sigma2_opt']).plot()
# plt.show()