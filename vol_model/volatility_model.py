import time
import json


class VolatilityModel:
    '''
    Base class for volatility models.
    '''
    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.n_obs = None
        self.mean = None
        self.params = None

    def filter(self, y):
        raise NotImplementedError

    def train(self, init_params, y, callback_func=None):
        raise NotImplementedError

    def loglikelihood(self):
        raise NotImplementedError

    def to_json(self, filename):
        est_dct = {'timestamp': str(time.strftime("%a %d %m %Y %H:%M", time.gmtime())),
                   'period': [self.start_date, self.end_date],
                   'n_obs': self.n_obs,
                   'mean': self.mean,
                   'params': list(self.params)}
        with open(filename, 'w') as json_file:
            json.dump(est_dct, json_file)
