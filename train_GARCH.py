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
callback_func = utils.CallbackFunc(df_train, df_val, 'return_dm', model, )
res = opt.minimize(model.log_likelihood,
                   x0=gamma0,
                   args=(df_train['return_dm'], True),  # arguments for function to be minimized (y, fmin=True)
                   method='BFGS',
                   callback=callback_func,
                   options={'maxiter': 50})

print('Results of BFGS minimizartion\n', res)
print('theta0    =', theta0)
print('theta_opt =', np.exp(res.x))


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

