import numpy as np
from utils import *
from proj1_helpers import *


# Todo add comment， maybe add early stop

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad, err = linear_gradient(y, tx, w)
        loss = MSE(err)
        w = w - gamma * grad

        ws.append(w)
        losses.append(loss)
        print("iteration {i}/{n}: loss={l}, w={w}".format(
            i=n_iter, n=max_iters - 1, l=loss, w=w))
        # might add early stop
    return ws[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            grad, _ = linear_gradient(y_batch, tx_batch, w)
            loss = MSE(y - tx @ w)
            w = w - gamma * grad
            # store w and loss
        ws.append(w)
        losses.append(loss)
        print("SGD iteration {i}/{n}: loss={l}, w={w}".format(
            i=n_iter, n=max_iters - 1, l=loss, w=w))
        # might add early stop
    return ws[-1], losses[-1]


def least_squares(y, tx):
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = MSE(y - tx @ w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    a = tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = MSE(y - tx @ w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            grad, _ = logo_gradient(y_batch, tx_batch, w)
            loss = cross_entropy(y, tx, w)
            w = w - gamma * grad
            # store w and loss
        ws.append(w)
        losses.append(loss)
        print("SGD iteration {i}/{n}: loss={l}, w={w}".format(
            i=n_iter, n=max_iters - 1, l=loss, w=w))
        # might add early stop
    return ws[-1], losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return
