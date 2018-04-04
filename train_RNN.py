import pandas as pd
import numpy as np
from RNNnumpy import RNNnumpy as rnn
import scipy.optimize as op
import json
import time

# Data load and manipulation
# --------------------------
df = pd.read_csv('./data/sp500.csv',index_col='Date',parse_dates=True).sort_index() # Load data from CSV
df['return'] = np.log(df['Close']).diff()*100 # Calculate returns
df = df.dropna() # Remove missings
print('Dimension of loaded CSV file:',df.shape)

# Choosing estimation and validation period
# -----------------------------------------
T = 4495
t = 0
#T = 100
est_index = df.index[t:t+T]       # Date index for estimation
val_index = df.index[t+T:(t+T+500)] # Date index for validation
print('Estimation from ',min(est_index))
print('Estimation to   ',max(est_index))
print('Validation from ',min(val_index))
print('Validation to   ',max(val_index))

# Calculating demeaned retuns and variance
# ---------------------------------------- 
est_data = pd.DataFrame()
val_data = pd.DataFrame()
est_data['return']  = df.loc[est_index,'return']
val_data['return']  = df.loc[val_index,'return']
mu = np.mean(est_data['return'])  
est_data['return_dm']  = est_data['return'] - mu
val_data['return_dm']  = val_data['return'] - mu

est_data['return_dm2'] = est_data['return_dm']**2
val_data['return_dm2'] = val_data['return_dm']**2

variance = np.var(est_data['return_dm'])
print('average  = ', mu)
print('variance = ', variance)

# Initialazing with an instance of the model object
# ------------------------------------------------- 
hidden_dim = 12
model = rnn(1,hidden_dim,1,variance=variance,model_type='Jordan')
print('w dimensions =',model.w_dim)
#state, est_data['sigma2'] = model.foward_prop(est_data['return'])

# Minimize loss function
# ----------------------
# Initializing with random weights uniformly on the interval -1/sqrt(k) to -1/sqrt(k)
# where k = previous layers dimensions.
W_H = np.random.uniform(-np.sqrt(1./3),np.sqrt(1./3),(hidden_dim,3))
W_O = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(1,hidden_dim+1))
w0 = np.hstack((W_H.flatten(),W_O.flatten()))
print('Initial weights =',w0)

#est_dct = json.load(open('Jordan_est_Mon_02-11-2015_06.57.json'))
#w0 = np.array(est_dct['w_opt'])

log_like = model.log_likelihood(w0,est_data['return_dm'])
print('Initial log likelihood =',log_like)

# Function for printing output during BFGS minimization
def callbackF(Xi):
    global Nfeval
    global log_like
    global w_list
    w_list.append(list(Xi))  
    loglike_est = model.log_likelihood(Xi,est_data['return_dm']) # Evaluating log like on estimation data (for printing only)
    loglike_val = model.log_likelihood(Xi,val_data['return_dm']) # Evaluating log like on valisation data (for printing only)
    log_like.loc[Nfeval,'EstData'] = loglike_est # Saving estimation data log like value for future plotting
    log_like.loc[Nfeval,'ValData'] = loglike_val # Saving validation data log like value for future plotting
    if (Nfeval % 1) == 0:
    	# Printing number of evaluations and log like values 
    	print('{0:4d} {1: 3.6f} {2: 3.6f}'.format(Nfeval, loglike_est,loglike_val))
    	#print('{0:4d}'.format(Nfeval))
    Nfeval += 1

#min_error = []
#lam_range = np.arange(0,2,0.1)
#i = 1
#for lam in lam_range:
w_list = []
log_like = pd.DataFrame()
Nfeval = 0

# Printing header for minimization procedure
print('{0:4s}  {1:9s}  {2:9s}'.format('Iter','EstData','ValData')) 
res = op.minimize(	model.log_likelihood,			# pointer to function to minimize
					x0=w0,                          # numpy array of intial weights
					args=(est_data['return_dm'],0), # arguments for function (y, lambda)
					method='BFGS',					# minimization method
					callback=callbackF,				# Function evaluated during each iteration, here used for printing
					options={'maxiter':100,
							 'norm':1,
							 'gtol':1e-3,
							 'eps':1e-4})
			
	
# Used for optimizing weight decay rate (lambda) value
# min_error.append(min(log_like['ValData']))
# i +=1
# print(i)
# print(lam)

print(model.W_O)

import matplotlib.pyplot as plt
#plt.plot(lam_range,min_error)
#plt.show()

# Smallest loss value on validation sample
argmin_val = np.argmin(log_like['ValData'])
argmin_est = np.argmin(log_like['EstData'])
print('Smallest loss function value at iteration on estimation data:',argmin_est)
print('Smallest loss function value at iteration on validation data:',argmin_val)

# Plotting average
log_like.plot()
plt.show()

# Saving results to text-file
est_dct = {'timestamp':str(time.strftime("%a %d %m %Y %H:%M", time.gmtime())),
		   'period':[str(min(est_index)),str(max(est_index))],
		   'n_obs':len(est_data['return']),
		   'mean':mu,
		   'variance':variance,
		   'D-M-K':[model.input_dim,model.hidden_dim,model.output_dim],
		   'model_type': model.model_type,
		   'w_opt':list(res.x),
		   'w_est_min':list(w_list[argmin_est]),
		   'w_cross_val':list(w_list[argmin_val]),
		   'w_all':list(w_list),
		   'log_like':res.fun}

file_name = 'Jordan_est_' + time.strftime("%a_%d-%m-%Y_%H.%M", time.gmtime())+'.json'
json.dump(est_dct,open(file_name,'w'))

# Evaluating model after minimization of loss function and saving in DataFrame for plotting
est_data['sigma2_est'] = model.forward_prop(est_data['return_dm'])
model_cv = rnn(1,hidden_dim,1,w=w_list[argmin_val],variance=variance,model_type='Jordan')
est_data['sigma2_val'] = model_cv.forward_prop(est_data['return_dm'])

import matplotlib.pyplot as plt
plt.subplot(211)
est_data['return'].plot()

plt.subplot(212)
np.sqrt(est_data['sigma2_est']).plot()
np.sqrt(est_data['sigma2_val']).plot()
np.sqrt(est_data['return_dm2']).plot()
plt.show()