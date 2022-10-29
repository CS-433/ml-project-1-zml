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


def cross_entropy_loss(y, tx, w, lambda_=0, balanced=False):
    """Compute the cross entropy loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        lambda_: ridge regression parameter
        balanced: true if balanced loss

    Returns:
        a non-negative loss
    """
    if not balanced:
        return np.mean(np.log(1 + np.exp(tx @ w)) - y * (tx @ w)) + 0.5 * lambda_ * np.linalg.norm(w) ** 2
    else:
        y_0 = y[np.where(y == 0)]
        beta = y_0.shape[0] / y.shape[0]
        y_hat = sigmoid(tx @ w)
        return -np.mean(
            (beta * y * np.log(y_hat) + (1 - beta) * (1 - y) * np.log(1 - y_hat))) + 0.5 * lambda_ * np.linalg.norm(
            w) ** 2


def logistic_regression_gradient(y, tx, w, lambda_=0):
    """Compute the gradient of loss for logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        lambda_: true if tx has a bias term

    Returns:
        a vector of shape (D, 1)
    """
    return tx.T @ (sigmoid(tx @ w) - y) / y.shape[0] + lambda_ * w


def linear(x, weights):
    if x @ weights < 0:
        return -1
    else:
        return 1

def logistic(x, weights):
    if sigmoid(x @ weights) >= 0.5:
        return 1
    else:
        return 0


def compute_score(y, tx, weights, f=logistic):
    y_pred = np.array([f(x, weights) for x in tx])
    return (y_pred == y).sum() / len(y)


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
    n = x.shape[0]
    div = int(n * ratio)

    perm = np.random.permutation(n)

    perm_tr, perm_te = perm[:div], perm[div:]

    return x[perm_tr], x[perm_te], y[perm_tr], y[perm_te]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N, D), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N, degree + 1, D)
    """
    res = []
    for x_i in x:
        res.append(np.array([x_i ** j for j in range(degree + 1)]))
    return np.array(res)


def logistic(x, weights):
    if sigmoid(x @ weights) >= 0.5:
        return 1
    else:
        return 0


def linear(x, weights):
    if x @ weights >= 0.5:
        return 1
    else:
        return 0


def compute_score(y, tx, weights, f='log'):
    """Calculate the accuracy"""
    if f =='log':
        y_pred = np.array([logistic(x, weights) for x in tx])
    if f =='linear':
        y_pred = np.array([linear(x, weights) for x in tx])
    return (y_pred == y).sum() / len(y)


def f1_score(actual, tx, weights, label=1, f='log'):
    """ calculate f1-score for the given `label` """
    if f == 'log':
        predicted = np.array([logistic(x, weights) for x in tx])
    if f == 'linear':
        predicted = np.array([linear(x, weights) for x in tx])

    tp = np.sum((actual == label) & (predicted == label))
    fp = np.sum((actual != label) & (predicted == label))
    fn = np.sum((predicted != label) & (actual == label))

    # F1 = 2 * (precision * recall) / (precision + recall)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis=0)
    x = x - mean_x
    std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x


def add_bias_term(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)


def kfolds(n, kfold=10, shuffle=True):
    if shuffle:
        perm = np.random.permutation(n)
    else:
        perm = np.arange(n)

    kfoldsShuffle, div = [], int(n / kfold)
    for i in range(kfold):
        test_indices = perm[div * i: div * (i + 1)]
        train_indices = np.concatenate((perm[:div * i], perm[div * (i + 1):]))
        kfoldsShuffle.append((train_indices, test_indices))

    return kfoldsShuffle

def outliers_map(x):
    """get outliers position per column"""
    p3, p1 = np.percentile(x[~np.isnan(x)], [98, 2])
    iqr = p3 - p1
    low = p1 - 1.5 * iqr
    high = p3 + 1.5 * iqr
    mask1 = (x < high) & (x > low)
    mask2 = np.isnan(x)
    return mask1 | mask2


def remove_outliers(x, y):
    """remove outliers"""
    x_copy = x.copy()
    y_copy = y.copy()
    outliers = np.ones(x.shape, dtype=bool)

    for i in range(x.shape[1]):
        outliers[:, i] = outliers_map(x[:, i])

    outliers = np.sum(outliers, axis=1) == x.shape[1]
    x_copy = x_copy[outliers]
    y_copy = y_copy[outliers]

    return x_copy, y_copy, outliers

def f1_score(actual, tx, weights, label=1, f=logistic):
    """ calculate f1-score for the given `label` """
    predicted = np.array([f(x, weights) for x in tx])

    tp = np.sum((actual == label) & (predicted == label))
    fp = np.sum((actual != label) & (predicted == label))
    fn = np.sum((predicted != label) & (actual == label))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def group_by_categories(X, column):
    """
    Group by data samples by categories of feature in column 'column'.
    Note that column must represent a categorical feature, otherwise
    the devision makes no sense.
    """
    categories = np.unique(X[:, column])
    groups = [np.where((X[:, column] == category))[0] for category in [0, 1]]
    groups.append(np.where(np.logical_or(X[:, column] == 2, X[:, column] == 3))[0])

    return groups