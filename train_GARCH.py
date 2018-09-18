import pandas as pd
import numpy as np
from vol_model.GARCH import GARCH
import utils
import time

ALPHA_INIT = 0.08
BETA_INIT = 0.9

# Data load
# ---------
df = pd.read_csv('./data/sp500.csv', index_col='Date', parse_dates=True).sort_index()
df = df.sample(n=100)
df['return'] = np.log(df['Close']).diff() * 100
df = df.dropna()
print('Dimension of DataFrame loaded from  CSV file:', df.shape)

n_validation = 10
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
gamma0 = model.inv_repam(theta0)
log_like = model.log_likelihood(params_repam=gamma0, y=df_train['return_dm'], fmin=True)
print('Initial log likelihood =', log_like)

# Minimize loss function
# ----------------------
callback_func = utils.CallbackFunc(df_train, df_val, 'return_dm', model)
model.train(theta0, df_train['return_dm'], callback_func)


filename = 'GARCH_est_' + time.strftime("%a_%d-%m-%Y_%H.%M", time.gmtime()) + '.json'
model.to_json(filename)
