import numpy as np
from utils import *
from proj1_helpers import *


def mean_squared_error_gd(y, tx, w_init, max_iters, gamma):
    ws = [w_init]
    losses = []
    w = w_init
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = MSE(err)
        w = w - gamma * grad

        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws[-1]


def mean_squared_error_sgd(y, tx, w_init, max_iters, gamma):
    ws = [w_init]
    losses = []
    w = w_init

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            loss = MSE(y - tx @ w)
            w = w - gamma * grad
            # store w and loss
            ws.append(w)
            losses.append(loss)
        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws[-1]


def least_squares(y, tx):
    return


def ridge_regression(y, tx, l):
    return


def logistic_regression(y, tx, w_init, max_iters, gamma):
    return


def reg_logistic_regression(y, tx, l, w_init, max_iters, gamma):
    return
