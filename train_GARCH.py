import pandas as pd
import numpy as np
import scipy.optimize as op
from GARCHnumpy import GARCHnumpy as garch
import time
import json

# Data load and manipulation
# --------------------------
df = pd.read_csv('./data/sp500.csv',index_col='Date',parse_dates=True).sort_index() # Load data from CSV
df['return'] = np.log(df['Close']).diff()*100 # Calculate returns
df = df.dropna() # Remove missings
print('Dimension od loaded CSV file:',df.shape)

# Choosing estimation period
# --------------------------
est_index = df.index[0:4495]
print('From ',min(est_index))
print('To   ',max(est_index))

# Calculating demeaned retuns and variance
# ---------------------------------------- 
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
print('Initial log likelihood =',log_like)

# Minimize loss function
# ----------------------
# Function for printing output during BFGS minimization
def callbackF(Xi):
    global Nfeval
    global log_like
    global w_list
    w_list.append(Xi)  
    loglike_est = model.log_likelihood(Xi,est_data['return_dm_abs']) # Evaluating log like on estimation data (for printing only)
    loglike_val = model.log_likelihood(Xi,val_data['return_dm_abs']) # Evaluating log like on valisation data (for printing only)
    log_like.loc[Nfeval,'EstData'] = loglike_est # Saving estimation data log like value for future plotting
    log_like.loc[Nfeval,'ValData'] = loglike_val # Saving validation data log like value for future plotting
    if (Nfeval % 1) == 0:
    	# Printing number of evaluations and log like values 
    	print('{0:4d} {1: 3.6f} {2: 3.6f}'.format(Nfeval, loglike_est,loglike_val))
    	#print('{0:4d}'.format(Nfeval))
    Nfeval += 1

# Printing header for minimization procedure
print('{0:4s}  {1:9s}  {2:9s}'.format('Iter','EstData','ValData')) 
res = op.minimize(	model.log_likelihood,					# pointer to function to minimize
						x0=gamma0,							# numpy array of intial reparametrized parameters
						args=(est_data['return_dm'],True), 	# arguments for function (y, fmin=True)
						method='BFGS',						# minimization method
						callback=callbackF,					# Function evaluated during each iteration, here used for printing
						options={'maxiter':50})				# Maximum number of BFGS iterations

													
print('Results of BFGS minimizartion\n',res)
print('theta0    =',theta0)
print('theta_opt =',np.exp(res.x))

# Saving results to JSON text-file
# --------------------------------
est_dct = {'timestamp':str(time.strftime("%a %d %m %Y %H:%M", time.gmtime())),
		   'period':[str(min(est_index)),str(max(est_index))],
		   'n_obs':len(est_data['return']),
		   'mean':mu,
		   'theta_opt':list(np.exp(res.x)),
		   'gamma_opt':list(res.x),
		   'log_like':res.fun}

file_name = 'GARCH_est_' + time.strftime("%a_%d-%m-%Y_%H.%M", time.gmtime())+'.json'
json.dump(est_dct,open(file_name,'w'))

# Plotting absolut returns and estimated sigma_t
# ----------------------------------------------- 
# Evaluating model after minimization of loss function and saving in DataFrame for plotting
est_data['sigma2_est'] = model.log_likelihood(y=est_data['return_dm'])

import matplotlib.pyplot as plt
plt.subplot(211)
est_data['return'].plot()

plt.subplot(212)
np.sqrt(est_data['return']**2).plot()
np.sqrt(est_data['sigma2_opt']).plot()
plt.show()