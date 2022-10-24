import numpy as np


# loss function
def mean_square_error(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - tx @ w
    return 0.5 * error.T @ error / y.shape[0]


def linear_regression_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    return -tx.T @ (y - tx @ w) / y.shape[0]


def sigmoid(x):
    """Computes the value of the sigmoid function on x.
    
    Args:
        x: scalar or numpy array
    
    Returns:
        A scalar or an array of shape (N, ) of respective values of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def cross_entropy_loss(y, tx, w, lambda_=0, bias_term=False):
    """Compute the cross entropy loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        lambda_: ridge regression parameter
        bias_term: true if tx has a bias term

    Returns:
        a non-negative loss
    """
    n = y.shape[0]
    start_2_norm = 1 if bias_term else 0

    sigmoid_values = sigmoid(tx @ w)
    return -(y.T @ np.log(sigmoid_values) + (1 - y.T) @ np.log(
        1 - sigmoid_values)) / n + lambda_ / n * np.linalg.norm(w[start_2_norm:], 2)


def logistic_regression_gradient(y, tx, w, lambda_=0, bias_term=False):
    """Compute the gradient of loss for logistic regression.
    
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        bias_term: true if tx has a bias term

    Returns:
        a vector of shape (D, 1)
    """
    n, d = tx.shape
    ridge_matrix = np.eye(d)

    if bias_term:
        ridge_matrix[0, 0] = 0

    return tx.T @ (sigmoid(tx @ w) - y) / n + lambda_ / n * ridge_matrix @ w


def split_data(x, y, ratio):
    """
    Split the dataset based on the split ratio.
    
    Args:
        x: numpy array of shape (N, D), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    N = x.shape[0]
    div = int(N * ratio)

    perm = np.random.permutation(N)

    perm_tr, perm_te = perm[:div], perm[div:]

    return x[perm_tr], x[perm_te], y[perm_tr], y[perm_te]


def predictions(x, weights):
    if sigmoid(x @ weights) > 0.5:
        return 1
    else:
        return 0


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def accuracy(y, y_hat):
    """Calculate the accuracy"""
    return (y == y_hat).sum() / y.shape[0]


def outliers_map(x):
    """get outliers position per column"""
    p3, p1 = np.percentile(x[(x != -999)], [95, 5])
    iqr = p3 - p1
    low = p1 - 1.5 * iqr
    high = p3 + 1.5 * iqr
    # print(low, high)
    mask1 = (x < high) & (x > low)
    mask2 = (x == -999)
    return mask1 | mask2


def remove_outliers(x, y):
    """remove outliers"""
    x_copy = x.copy()
    y_copy = y.copy()
    outliers = np.ones(x.shape, dtype=bool)

    for i in range(x.shape[1]):
        outliers[:, i] = outliers_map(x[:, i])
    outliers = (np.sum(outliers, axis=1) == 30)
    x_copy = x_copy[outliers]
    y_copy = y_copy[outliers]

    return x_copy, y_copy
