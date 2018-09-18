import pandas as pd
import numpy as np
import scipy.optimize as opt
from GARCH import GARCH
import utils

ALPHA_INIT = 0.088
BETA_INIT = 0.86

# Data load and manipulation
# --------------------------
df = pd.read_csv('./data/sp500.csv', index_col='Date', parse_dates=True).sort_index()  # Load data from CSV
df['return'] = np.log(df['Close']).diff() * 100  # Calculate returns
df = df.dropna()  # Remove missings
print('Dimension od loaded CSV file:', df.shape)

# Choosing estimation period
# --------------------------
n_validation = 500
df_train, df_val = utils.split_in_training_and_out_of_sample_validation(df, n_validation=n_validation)

# Calculating demeaned returns and variance
# -----------------------------------------
mean_train = df_train['return'].mean()
df_train['return_dm'] = df_train['return'] - mean_train
df_val['return_dm'] = df_val['return'] - mean_train

variance = np.var(df_train['return'])
omega_init = variance*(1 - ALPHA_INIT - BETA_INIT)
print('average  = {:.3g}\nvariance = {:.3g}\nomega = {:.3g}'.format(mean_train, variance, omega_init))

model = GARCH()
theta0 = np.array([omega_init, ALPHA_INIT, BETA_INIT])
gamma0 = np.log(theta0)
log_like, df_train['sigma2_non-opt'] = model.log_likelihood(gamma=gamma0, y=df_train['return_dm'], fmin=False)
print('Initial log likelihood =', log_like)


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

