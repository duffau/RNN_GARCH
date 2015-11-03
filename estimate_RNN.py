import pandas as pd
import numpy as np
from RNNnumpy import RNNnumpy as rnn
import scipy.optimize as op
import json
import time

df = pd.read_csv('./data/sp500.csv',index_col='Date',parse_dates=True).sort_index()
df['return'] = np.log(df['Close']).diff()
df = df.dropna()

print(df.shape)

T = 1500
#T = 100
est_index = df.index[0:T]
val_index = df.index[T:(T+500)]

est_data = pd.DataFrame()
val_data = pd.DataFrame()
est_data['return']  = df.loc[est_index,'return']
val_data['return']  = df.loc[val_index,'return']
mu = np.mean(est_data['return'])  
est_data['return_dm']  = est_data['return'] - mu
est_data['return_dm_abs'] = np.abs(est_data['return_dm'])
val_data['return_dm']  = val_data['return'] - mu
val_data['return_dm_abs']  = np.abs(val_data['return_dm'])
variance = np.var(est_data['return'])
print('average  = ', mu)
print('variance = ', variance)

hidden_dim = 8
model = rnn(1,hidden_dim,1,variance=variance,model_type='Jordan')
print('w dimesnsion =',model.w_dim)
#state, est_data['sigma2'] = model.foward_prop(est_data['return'])

W_H = np.random.uniform(-np.sqrt(1./3),np.sqrt(1./3),(hidden_dim,3))
W_O = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(1,hidden_dim+1))
w0 = np.hstack((W_H.flatten(),W_O.flatten()))

#est_dct = json.load(open('Jordan_est_Mon_02-11-2015_02.00.json'))
#w0 = est_dct['w_opt']
print(w0)

log_like = model.log_likelihood(w0,est_data['return_dm_abs'])
print('log_like',log_like)

def callbackF(Xi):
    global Nfeval
    global log_like
    global w_list
    w_list.append(Xi)  
    loglike_est = model.log_likelihood(Xi,est_data['return_dm_abs'])
    loglike_val = model.log_likelihood(Xi,val_data['return_dm_abs'])
    log_like.loc[Nfeval,'EstData'] = loglike_est
    log_like.loc[Nfeval,'ValData'] = loglike_val
    if (Nfeval % 1) == 0: 
    	print('{0:4d} {1: 3.6f} {2: 3.6f}'.format(Nfeval, loglike_est,loglike_val))
    	#print('{0:4d}'.format(Nfeval))
    Nfeval += 1

#min_error = []
#lam_range = np.arange(0,2,0.1)
#i = 1
#for lam in lam_range:
w_list = []
log_like = pd.DataFrame()
#log_like[['EstData','ValData']] =[0,0]
Nfeval = 0
print('{0:4s}  {1:9s}  {2:9s}'.format('Iter','EstData','ValData')) 
res = op.minimize(	model.log_likelihood,
					x0=w0,
					args=(est_data['return_dm_abs'],0),
					method='BFGS',
					callback=callbackF,
					options={'maxiter':50})

	# min_error.append(min(log_like['ValData']))
	# i +=1
	# print(i)
	# print(lam)


import matplotlib.pyplot as plt
#plt.plot(lam_range,min_error)
#plt.show()

argmin = np.argmin(log_like['ValData'])
print('argmin =',argmin)

#log_like['EstData'] = log_like['EstData']/T
#log_like['ValData'] = log_like['ValData']/300
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
		   'w_cross_val':list(w_list[argmin]),
		   'w_all':w_list,
		   'log_like':res.fun}

file_name = 'Jordan_est_' + time.strftime("%a_%d-%m-%Y_%H.%M", time.gmtime())+'.json'
json.dump(est_dct,open(file_name,'w'))

# import matplotlib.pyplot as plt
# plt.subplot(211)
# est_data['return'].plot()

# plt.subplot(212)
# est_data['sigma2'].plot()
# plt.show()