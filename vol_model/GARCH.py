import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
from vol_model.volatility_model import VolatilityModel


class GARCH(VolatilityModel):
    '''
    This class defines the GARCH model object which contains, functions
    for estimation and VaR forecasting.
    '''

    def __init__(self, params=None, mu=None):
        super().__init__()

        # Initialize parameters
        if (params != None):
            self.params = np.array(params)
        else:
            self.params = np.array([1.e-06, 0.09, 0.9])
        if (mu == None):
            self.mu = 0
        else:
            self.mu = mu

    def __repr__(self):
        return "omega = {:.3g}\nalpha = {:.3g}\nbeta  = {:.3g}".format(*self.params)

    def train(self, init_params, y, callback_func=None):
        self.n_obs = len(y)
        self.start_date = str(y.index[0])
        self.end_date = str(y.index[-1])
        opt_result = opt.minimize(self.log_likelihood,
                           x0=self.inv_repam(init_params),
                           args=(y, True),  # arguments for function to be minimized (y, fmin=True)
                           method='BFGS',
                           callback=callback_func,
                           options={'maxiter': 100})
        self.params = self.repam(opt_result.x)
        print('\nResults of BFGS minimization\n{}\n{}'.format(''.join(['-']*28), opt_result))
        print('\nResulting params = {}'.format(self.params))

    def log_likelihood(self, params_repam, y, fmin=False):
        '''
        Takes the reparametrized 3X1 numpy array gamma = log((omega,alpha,beta))
        as input (if given or else uses the ones in self namespace).
        And returns either sum of all likelihood contributions that is a 1X1
        numpy array or both the likelihood and the (t_max,) numpy array of estimated conditional variances.
        '''
        self.params = self.repam(params_repam)
        omega = self.params[0]
        alpha = self.params[1]
        beta = self.params[2]

        t_max = len(y)
        avg_log_like = 0
        sigma2 = np.zeros(t_max + 1)
        sigma2[0] = np.var(y)
        for t in range(1, t_max):
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
            avg_log_like += (np.log(sigma2[t]) + y[t]**2 / sigma2[t]) / t_max
        if fmin:
            return avg_log_like
        else:
            return [avg_log_like, sigma2]

    def filter(self, y):
        omega = self.params[0]
        alpha = self.params[1]
        beta = self.params[2]

        t_max = len(y)
        sigma2 = np.zeros(t_max + 1)
        sigma2[0] = np.var(y)
        for t in range(1, t_max):
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
        return sigma2

    def repam(self, params_repam):
        return np.exp(params_repam)

    def inv_repam(self, params):
        return np.log(params)

    def VaR(self, y, pct=(0.01, 0.025, 0.05)):
        est_variance = self.log_likelihood(y=y, fmin=False)[1]
        VaR = {}
        for alpha in pct:
            VaR[str(alpha)] = self.mu + norm.ppf(alpha) * np.sqrt(est_variance)
        return VaR
