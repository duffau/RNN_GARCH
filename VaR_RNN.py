# Calculate VaR and Kupiec statistics
import pandas as pd
import numpy as np
from RNNnumpy import RNNnumpy as rnn
import json
from pprint import pprint
import scipy.stats as sps
import matplotlib.pyplot as plt
from pprint import pprint

#est_dct = json.load(open('Jordan_est_Mon_02-11-2015_06.57.json')) # Bedste bud kl 8:00
est_dct = json.load(open('Jordan_est_Tue_03-11-2015_18.35.json'))
#pprint(est_dct)
model = rnn(1,est_dct['D-M-K'][1],1,
			#w = est_dct['w_opt'],
			w = est_dct['w_cross_val'],
			#w = est_dct['w_all'][40],
			variance=est_dct['variance'],
			mu = est_dct['mean'],
			model_type=est_dct['model_type'])
print(model)
print(est_dct['variance'])
est_dct_garch = json.load(open('GARCH_est_Sun_01-11-2015_21.06.57.json'))
pprint(est_dct_garch)

# Plot News impact curve
# ----------------------
col_lst = plt.cm.Set1(np.linspace(0, 1, 9))
y      = np.arange(-4,4,0.001)
sigma2 = np.array([model.forward_prop([yi]) for yi in y])

# GARCH par
theta = np.exp(est_dct_garch['theta_opt'])
omega = theta[0]
alpha = theta[1]
beta  = theta[2]
garch_var = omega/(1-alpha-beta)
garch_news = omega + alpha * y**2 + beta*est_dct['variance']

plt.figure(figsize=(6,4))
plt.plot(y,sigma2,label='Jordan NN',color=col_lst[0],lw=2)
plt.plot(y,garch_news,label='GARCH',color=col_lst[1],lw=2)
#plt.axis([-0.02, 0.02, 0, 0.0005])
plt.ylabel('$\sigma^2_t$')
plt.xlabel('$y_{t}$')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='upper right')
#plt.savefig('./plots/Jordan_newsimpact.pdf')
plt.show()

# VaR forecast
# ------------
df = pd.read_csv('./data/sp500.csv',index_col='Date',parse_dates=True).sort_index()
df['return'] = np.log(df['Close']).diff()*100
df = df.dropna()
var_index = df.index[df.index>"1997-10-10 00:00:00"]
var_data = pd.DataFrame()
var_data['return'] = df.loc[var_index,'return']

VaR = model.VaR(var_data['return'])
uncon_cov = pd.DataFrame()
N = len(var_data['return'])
i  = 0
for alpha in [0.01,0.025,0.05]:
	var_data['VaR_'+str(alpha)] = VaR[str(alpha)]
	N_ex = sum(var_data['return'] < VaR[str(alpha)])
	uncon_cov.loc[i,'alpha'] = alpha
	uncon_cov.loc[i,'coverage'] = N_ex/N 
	lam = 2*(N_ex * np.log(N_ex/N) + (N-N_ex) * np.log(1-N_ex/N) - (N_ex * np.log(alpha) + (N-N_ex) * np.log(1-alpha)))
	uncon_cov.loc[i,'Kupiec_lambda'] = lam
	uncon_cov.loc[i,'Kupiec_p']   = sps.chi2.pdf(lam,1)
	i += 1

print('Unconditional coverage test\n',uncon_cov )


col_lst = plt.cm.Set1(np.linspace(0, 1, 9))
#var_data[['return','VaR_0.01','VaR_0.025','VaR_0.05']].plot(color=col_lst[[1,0,2,3]],linewidth=1)
var_data[['return','VaR_0.01']].plot(color=col_lst[[1,0]],rot=90,linewidth=1,figsize=(6,4))
#plt.savefig('./plots/Jordan_VaR.pdf',bbox_inches='tight')
plt.show()

def acf(x, length=20,alpha=0.05):
	N = len(x)
	mu = np.mean(x)
	acv = np.array([np.dot(x[:-i]-mu, x[i: ]-mu) for i in range(1, length)])
	v1  = np.array([np.dot(x[:-i]-mu, x[:-i]-mu) for i in range(1, length)])
	v2  = np.array([np.dot(x[i: ]-mu, x[i: ]-mu) for i in range(1, length)])
	conf_l = np.ones(length-1)*1/N-1/np.sqrt(N)*sps.norm.ppf(1-alpha/2)
	conf_u = np.ones(length-1)*1/N+1/np.sqrt(N)*sps.norm.ppf(1-alpha/2)
	return {'acf':acv/np.sqrt(v1*v2),'conf_u':conf_u,'conf_l':conf_l}

# Plot of acf
# -----------
k = 10
for alpha in [0.01,0.025,0.05]:
	indic = var_data['return']<VaR[str(alpha)]
	acf_out = acf(indic,k)
	print(acf_out['acf'])
	plt.figure(figsize=(6,4))
	plt.plot(np.arange(1,k),acf_out['acf'],color=col_lst[1])
	plt.plot(np.arange(1,k),acf_out['conf_l'],color=col_lst[0],ls='--')
	plt.plot(np.arange(1,k),acf_out['conf_u'],color=col_lst[0],ls='--')
	plt.axis([0,10,-0.08,0.2])
	#plt.savefig('./plots/Jordan_indic_acf'+str(alpha)+'.pdf',bbox_inches='tight')
	plt.show()