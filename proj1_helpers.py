# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import csv


def load_data(path_dataset,sub_sample=False):
    """Load data and convert it to the metric system."""
    X = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=list(range(2, 32)))
    Y = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=(1), dtype=str)

    # check data 
    if X.shape[0] != Y.shape[0]:
        print('Inconsisent data number')
    print(np.unique(Y))
    
    Y_int = np.ones(Y.size)
    Y_int[np.where(Y=='b')] = -1
    # sub-sample
    if sub_sample:
        X = X[::50]
        Y = Y[::50]

    return X, Y_int 


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


