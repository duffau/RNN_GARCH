
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
