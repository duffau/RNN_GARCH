import numpy as np
from scipy.stats import norm


class JordanRNN:
    '''
    This class defines a single layer recurrent neural net of Jordan type.
    The class contains:
    - Methods for forward propagation, that is evaluation of a network given a set of weights.
    - Some relevant activation functions
    - The loss function (negative log likelihood) for estimation with scipy.stats.min_bfgs
    - VaR forecasting function
    '''

    def __init__(self, input_dim, hidden_dim, output_dim, w=None, mu=0, variance=0.1, q=2):
        '''
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param w:
        :param mu:
        :param variance:
        :param q:
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.q = q
        self.mu = mu
        self.variance = variance
        self.w_dim = self.hidden_dim * (self.input_dim + 2) + self.output_dim * (self.hidden_dim + 1)
        self.__init_weights(w, input_dim, hidden_dim, output_dim)
        self.__init_gradient(input_dim, hidden_dim, output_dim)

    def __init_weights(self, w, input_dim, hidden_dim, output_dim):
        if (w == None):
            self.W_H = np.random.uniform(-np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), (hidden_dim, input_dim + 2))
            self.W_O = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (output_dim, hidden_dim + 1))
        elif (w != None):
            self.W_H, self.W_O = self.w_vec2mat(w, input_dim, hidden_dim, output_dim)

    def __init_gradient(self, input_dim, hidden_dim, output_dim):
        self.dW_H = np.zeros((hidden_dim, input_dim + 2))
        self.dW_O = np.zeros((output_dim, hidden_dim + 1))

    def w_vec2mat(self, w):
        '''
        Transforms a flat numpy array of dimension
        ((input_dim+2)*hidden_dim + output_dim*(hidden_dim+1),)
        into two 2-dimensional numpy arrays containing
        weights for the hidden layer (W_H) and weights for the output layer (W_O).
        '''
        w_H = w[0:(self.hidden_dim * (self.input_dim + 2))]
        id_H = self.hidden_dim * (self.input_dim + 2)
        w_O = w[id_H:(id_H + 1 + self.output_dim * (self.hidden_dim + 1))]
        W_H = np.reshape(w_H, (self.hidden_dim, self.input_dim + 2))
        W_O = np.reshape(w_O, (self.output_dim, self.hidden_dim + 1))
        return W_H, W_O

    def w_vec2vecs(self, w):
        '''
        Transforms a flat numpy array of dimension
        ((input_dim+2)*hidden_dim + output_dim*(hidden_dim+1),)
        into two 1-dimensional numpy arrays containing
        weights for the hidden layer (w_H) and weights for the output layer (w_O).
        '''
        w_H = w[0:(self.hidden_dim * (self.input_dim + 2))]
        id_H = self.hidden_dim * (self.input_dim + 2)
        w_O = w[id_H:(id_H + 1 + self.output_dim * (self.hidden_dim + 1))]
        return (w_H, w_O)

    def logi_fun(self, x):
        return 1 / (1 + np.exp(-x))

    def rect_fun(self, x):
        return np.log(1 + np.exp(x))

    def forward_prop(self, y):
        '''
        Forward propagate x_t
        through the neural network.

        Returns sigma^2 numpy array of dimension (T,)
        '''
        self.T = len(y)
        x = np.ones((self.T + 1, self.input_dim + 2))
        x[:, 1] = np.hstack((np.array(y), [self.variance]))

        # Initialize with two numpy arrays
        self.state = np.ones((self.T + 1, self.hidden_dim + 1))  # hidden neurons of latent state
        self.sigma2 = np.ones(self.T + 1) * self.variance  # variance array where all values are initialized with the sample variance
        for t in range(self.T):
            x[t, 2] = self.sigma2[t - 1]  # The estimated variance of the previous period is input at the current period
            self.state[t, 1:] = self.logi_fun(self.W_H.dot(x[t])).reshape(self.hidden_dim, )  # Need to reshape to make 2nd dimesion empty
            self.sigma2[t] = np.exp(self.W_O.dot(self.state[t])).reshape(self.output_dim, )  # Need to reshape to make 2nd dimesion empty
        # sigma2[t] = self.rect_fun( self.W_O.dot(state[t]) ).reshape(self.output_dim,)  # Need to reshape to make 2nd dimesion empty

        return self.sigma2[0:self.T]

    def backprop(self, w, y):
        '''
        Calculates exact gradient of the log likelihood function through backpropagation.
        '''
        y = np.array(y)

        w_H, w_O = self.w_vec2vecs(w)
        for t in range(0, self.T):
            self.dW_O[t] = (1 - y[t] ** 2 / sigma2[t]) * self.state[t] + self.lam / 2 * self.q * w_O**(self.q - 1)
        return self.dW_O.sum(axis=0)

    def log_likelihood(self, w, y, lam=1):
        '''
        Takes numpy arrays of weights and returns as input.
        Return the average of reguralized log likelihood function in a 1X1 numpy array.
        '''
        self.W_H, self.W_O = self.w_vec2mat(w)
        y = np.array(y)
        self.T = len(y)
        sigma2 = self.forward_prop(y)
        self.lam = lam
        log_like = 1 / 2 * self.T * np.log(2 * np.pi) + 1 / 2 * sum(np.log(sigma2) + y ** 2 / sigma2) + self.lam / 2 * (w ** (self.q / 2)).T.dot(w ** (self.q / 2))
        # log_like =  1/2*sum((y**2-sigma2)**2) + lam/2*w.T.dot(w)

        return log_like

    def num_gradient(self, w, y, lam):
        '''
        Numerical gradient for checking backprop function.
        '''
        eps = 0.001
        w_plus = w
        w_minus = w
        num_dw = w
        for i in range(self.w_dim):
            w_minus[i] = w[i] - eps
            w_plus[i] = w[i] + eps
            num_dw[i] = (self.log_likelihood(w_plus, y, lam) - self.log_likelihood(w_minus, y, lam)) / (2 * eps)
        num_dW_H, num_dW_O = self.w_vec2mat(num_dw)
        return num_dW_H, num_dW_O

    def VaR(self, y, pct=(0.01, 0.025, 0.05)):
        est_variance = self.forward_prop(y)
        VaR = {}
        for alpha in pct:
            VaR[str(alpha)] = self.mu + norm.ppf(alpha) * np.sqrt(est_variance)
        return VaR
