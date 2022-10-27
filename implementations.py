import numpy as np
from utils import *
from helpers import *


# Todo add comment， maybe add early stop

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
    """
    w, ws, losses = initial_w, [initial_w], []
    for n_iter in range(max_iters):
        loss = mean_square_error(y, tx, w)
        w = w - gamma * linear_regression_gradient(y, tx, w)

        ws.append(w)
        losses.append(loss)
        print("iteration {i}/{n}: loss={l}, w={w}".format(
            i=n_iter, n=max_iters - 1, l=loss, w=w))
        # might add early stop

    return ws[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """
    w, ws, losses = initial_w, [initial_w], []
    for n_iter in range(max_iters):
        loss = mean_square_error(y, tx, w)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            w = w - gamma * \
                linear_regression_gradient(minibatch_y, minibatch_tx, w)

        ws.append(w)
        losses.append(loss)
        print("SGD iteration {i}/{n}: loss={l}, w={w}".format(
            i=n_iter, n=max_iters - 1, l=loss, w=w))
        # might add early stop

    return ws[-1], losses[-1]


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        weights: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mean square error, scalar.
    """
    loss = mean_square_error(y, tx, weights)
    weights = np.linalg.solve(tx.T @ tx, tx.T @ y)

    return weights, loss


def ridge_regression(y, tx, lambda_):
    """Calculate ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: scalar.
        bias_term: true if tx has a bias term

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mean square error, scalar.
    """
    n, d = tx.shape
    weights = np.linalg.solve(
        tx.T @ tx + 2 * n * lambda_ * np.eye(d), tx.T @ y)
    loss = mean_square_error(y, tx, weights)

    return weights, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent algorithm (GD).

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        bias_term: true if tx has a bias term

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The Gradient Descent algorithm (GD).

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        bias_term: true if tx has a bias term

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """
    m1 = 0.1  # Parameters for Goldstein-Price
    m2 = 0.9
    tol = 1e-8

    w, ws, losses = initial_w, [initial_w], []
    for n_iter in range(max_iters):
        loss = cross_entropy_loss(y, tx, w, lambda_=lambda_)

        gradient = -logistic_regression_gradient(y, tx, w, lambda_=lambda_)

        tl = 0
        tr = 0
        t = 1
        while True:
            qt = cross_entropy_loss(y, tx, w + t * gradient, lambda_=lambda_)
            qp = -gradient.T @ gradient
            gpt = logistic_regression_gradient(
                y, tx, w + t * gradient, lambda_=lambda_).T @ gradient
            if ((qt - loss) / t <= (m1 * qp)) and (gpt >= (m2 * qp)):
                gamma = t   # we found a good step
                break
            if ((qt - loss) / t > (m1 * qp)):
                # step too big
                tr = t
            if ((qt - loss) / t <= (m1 * qp)) and (gpt < (m2 * qp)):
                # step too small
                tl = t
            if(tr == 0):
                t = 2 * tl
            else:
                t = 0.5 * (tl + tr)
            if abs(tr - tl) <= tol:
                break

        w = w + gamma * gradient

        ws.append(w)
        losses.append(loss)

        if n_iter % 200 == 0:
            print(f"Iteration {n_iter + 1}/{max_iters}: loss={loss}, w={w}")

        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < tol:
            break

    return ws[-1], losses
