import numpy as np
from utils import *
from helpers import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the step size

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
    """
    w, ws, losses = initial_w, [initial_w], []
    if max_iters == 0:
        loss = mean_square_error(y, tx, w)
        return w, loss
    for n_iter in range(max_iters):
        w = w - gamma * linear_regression_gradient(y, tx, w)
        loss = mean_square_error(y, tx, w)

        ws.append(w)
        losses.append(loss)
        print("iteration {i}/{n}: loss={l}, w={w}".format(
            i=n_iter, n=max_iters - 1, l=loss, w=w))
        # might add early stop
    return ws[-1], loss[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the step size

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """
    w, ws, losses = initial_w, [initial_w], []
    if max_iters == 0:
        loss = mean_square_error(y, tx, w)
        return w, loss
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            w = w - gamma * \
                linear_regression_gradient(minibatch_y, minibatch_tx, w)
        loss = mean_square_error(y, tx, w)
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
    weights = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = mean_square_error(y, tx, weights)

    return weights, loss


def ridge_regression(y, tx, lambda_):
    """Calculate ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: scalar, ridge regression parameter

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mean square error, scalar.
    """
    n, d = tx.shape
    weights = np.linalg.solve(
        tx.T @ tx + 2 * n * lambda_ * np.eye(d), tx.T @ y)
    loss = mean_square_error(y, tx, weights)

    return weights, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, balanced=False):
    """The Gradient Descent algorithm (GD).

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the step size
        balanced: bool, use the weighted cross-entropy loss if True

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma, balanced)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, balanced=False):
    """The Gradient Descent algorithm (GD).

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        lambda_: scalar, ridge regression parameter
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the step size
        balanced: bool, use the weighted cross-entropy loss if True

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
    """
    m1 = 0.1  # Parameters for Goldstein-Price
    m2 = 0.9
    tol = 1e-8

    w, ws, losses = initial_w, [initial_w], []
    if max_iters == 0:
        loss = cross_entropy_loss(y, tx, w)
        return w, loss
    for n_iter in range(max_iters):
        gradient = -logistic_regression_gradient(y, tx, w, lambda_=lambda_)

        tl = 0
        tr = 0
        t = 1
        # while True:
        #     qt = cross_entropy_loss(y, tx, w + t * gradient, lambda_=lambda_, balanced=balanced)
        #     qp = -gradient.T @ gradient
        #     gpt = logistic_regression_gradient(
        #         y, tx, w + t * gradient, lambda_=lambda_).T @ gradient
        #     if ((qt - loss) / t <= (m1 * qp)) and (gpt >= (m2 * qp)):
        #         gamma = t  # we found a good step
        #         break
        #     if (qt - loss) / t > (m1 * qp):
        #         # step too big
        #         tr = t
        #     if ((qt - loss) / t <= (m1 * qp)) and (gpt < (m2 * qp)):
        #         # step too small
        #         tl = t
        #     if tr == 0:
        #         t = 2 * tl
        #     else:
        #         t = 0.5 * (tl + tr)
        #     if abs(tr - tl) <= tol:
        #         break

        w = w + gamma * gradient
        loss = cross_entropy_loss(y, tx, w, lambda_=lambda_, balanced=balanced)
        ws.append(w)
        losses.append(loss)
        if n_iter % 200 == 0:
            print(f"Iteration {n_iter + 1}/{max_iters}: loss={loss}")

        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < tol:
            break

    return ws[-1], losses[-1]
