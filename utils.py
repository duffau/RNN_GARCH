
def split_in_training_and_out_of_sample_validation(df, n_validation):
    n_total = df.shape[0]
    n_training = n_total - n_validation
    if n_training < 1:
        raise ValueError('Number of training examples:{}. Must be greater than 1. Use smaller n_validation.'.format(n_training))
    training_index = df.index[0:n_training]
    validation_index = df.index[n_training:(n_training + n_validation)]
    print('Training data from {} to {}'.format(min(training_index), max(training_index)))
    print('Validation data from {} to {}'.format(min(validation_index), max(validation_index)))
    return df.loc[training_index], df.loc[validation_index]


class CallbackFunc:

    def __init__(self, df_train, df_val, variable, model):
        self.n_func_evals  = 0
        self.df_train = df_train
        self.df_val = df_val
        self.model = model
        self.variable = variable

    def __call__(self, xi):
        if self.n_func_evals == 0:
            print('{:5s} {:>18s} {:>18s}'.format('Iter.', 'Loglike training', 'Loglike validation'))

        if (self.n_func_evals % 5) == 0:
            loglike_train = self.model.log_likelihood(xi, self.df_train[self.variable], fmin=True)
            loglike_val = self.model.log_likelihood(xi, self.df_val[self.variable], fmin=True)
            print('{:5d} {:18.3f} {:18.3f}'.format(self.n_func_evals, loglike_train, loglike_val))

        self.n_func_evals += 1