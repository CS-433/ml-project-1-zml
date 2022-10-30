import utils
import implementations

import numpy as np


def preprocess_data(x_tr, x_test, y_tr, y_test, degree=None):
    """
    Args:
        x_tr: numpy array of shape (N, D), N is number of samples, D is number of features, representing training set
        x_test: numpy array of shape (N', D), N' is number of samples, D is number of features, representing testing/validation set
        y_tr: numpy array of shape (N, 1), N is number of samples,
        y_test:
        degree:
    Preprocess the input resources.
    1. Replace -999.0 with NaN
    [2. One-hot encoding of column 22]
    3. Add binary feature isNaN for column 0
    4. Fill up missing values -> median or learn
    5. Log positive columns 0, 1, 2, 3, 5, 8, 9, 10, 12, 13, 16, 19, 21, 22, 23, 26, 29
    6. Drop some columns
    7. Delete big outliers rows
    8. Standardize
    9. Add bias or build polynomial features
    10. Add angle features
    Returns:
        x_train': numpy array of preprocessed x_train
        x_test': numpy array of preprocessed x_test

    """
    # step 1 replacing non-useful values with None
    x_tr[x_tr == -999.0] = np.NaN
    x_tr[:, -1][
        x_tr[:, -1] == 0
    ] = np.NaN  # for last column "PRI_jet_all_pt", we recognize that 0.0 represent NaNs
    x_test[x_test == -999.0] = np.NaN
    x_test[:, -1][x_test[:, -1] == 0] = np.NaN

    # step 3
    isNan0_tr = (np.isnan(x_tr[:, 0]) * 1).reshape(-1, 1)
    isNan0_test = (np.isnan(x_test[:, 0]) * 1).reshape(-1, 1)

    # step 4 filling up missing values
    columns_to_drop, columns_to_fill, feature_medians = calculate_feature_medians(x_tr)
    x_tr = fill_features_with_median(x_tr, columns_to_fill, feature_medians)
    x_test = fill_features_with_median(x_test, columns_to_fill, feature_medians)

    # columns_to_drop, columns_to_fill, feature_weights, learning_features = calculate_feature_weights(x_tr)
    # x_tr = fill_features_with_weights(x_tr, columns_to_fill, feature_weights, learning_features)
    # x_test = fill_features_with_weights(x_test, columns_to_fill, feature_weights, learning_features)

    # columns_to_drop = colums_with_missing_features(x_tr)

    # step 5 log heavy tailed features
    log_columns = [0, 1, 2, 3, 5, 8, 9, 10, 12, 13, 16, 19, 21, 23, 26, 29]
    log_columns = list(
        set(log_columns) - set(columns_to_drop)
    )  # we don't wish to transform columns that we are going to drop anyway
    x_tr = log_transform(x_tr, columns=log_columns)
    x_test = log_transform(x_test, columns=log_columns)

    # step 6 dropping features
    """if 0 not in columns_to_drop:
        columns_to_drop.append(0)"""

    unique = True
    if 22 not in columns_to_drop:
        unique = len(np.unique(x_tr[:, 22])) == 1
        if not unique:
            x_tr[:, 22][x_tr[:, 22] == 2] == 0
            x_tr[:, 22][x_tr[:, 22] == 3] == 1
            jet_num_tr = x_tr[:, 22].copy()

            x_test[:, 22][x_test[:, 22] == 2] == 0
            x_test[:, 22][x_test[:, 22] == 3] == 1
            jet_num_test = x_test[:, 22].copy()

        columns_to_drop.append(22)

    columns_angle = []
    for col in [15, 18, 20, 25, 28]:
        if col not in columns_to_drop:
            columns_angle.append(col)
            columns_to_drop.append(col)

    x_angle_tr = x_tr[:, columns_angle].copy()
    x_angle_test = x_test[:, columns_angle].copy()

    columns_to_drop.sort()
    x_tr = np.delete(x_tr, columns_to_drop, axis=1)
    x_test = np.delete(x_test, columns_to_drop, axis=1)

    # step 7 remove outliers
    """x_tr, y_tr, non_outliers_index_tr = utils.remove_outliers(x_tr, y_tr)
    x_angle_tr = x_angle_tr[non_outliers_index_tr]
    isNan0_tr = isNan0_tr[non_outliers_index_tr]
    if not unique:
        jet_num_tr = jet_num_tr[non_outliers_index_tr]"""
    print("Shape", x_tr.shape)

    x_tr = np.hstack((x_tr, np.sin(x_angle_tr), np.cos(x_angle_tr)))
    x_test = np.hstack((x_test, np.sin(x_angle_test), np.cos(x_angle_test)))

    # setp 8 standardization of features
    x_tr, mean_x, std_x = standardize(x_tr)
    x_test, _, _ = standardize(x_test, mean_x, std_x)

    # step 9 create polynomial regression or juts add bias term
    if degree is None:
        x_tr = add_bias_term(x_tr)
        x_test = add_bias_term(x_test)
    else:
        x_tr = build_poly_feature(x_tr, degree)
        x_test = build_poly_feature(x_test, degree)

    # step 10 merge all features
    # x_tr = np.hstack((x_tr, isNan0_tr, np.sin(x_angle_tr), np.cos(x_angle_tr)))
    # x_test = np.hstack((x_test, isNan0_test, np.sin(x_angle_test), np.cos(x_angle_test)))
    if not unique:
        print("Shape3", x_tr.shape, jet_num_tr.shape)
        x_tr = np.hstack((x_tr, jet_num_tr.reshape(-1, 1)))
        x_test = np.hstack((x_test, jet_num_test.reshape(-1, 1)))

    x_tr = np.hstack(
        (
            x_tr,
            isNan0_tr,
            # np.sin(x_angle_tr), np.cos(x_angle_tr)
        )
    )
    x_test = np.hstack(
        (
            x_test,
            isNan0_test,
            # np.sin(x_angle_test), np.cos(x_angle_test)
        )
    )

    return x_tr, x_test, y_tr, y_test


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original resources set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x

    if std_x is None:
        std_x = np.std(x, axis=0)
    x = x[:, std_x > 0] / std_x[std_x > 0]

    return x, mean_x, std_x


def add_bias_term(x):
    """
    Add a bais/constant term as the first column of a dataset.

    Args:
        x: numpy array of shape (N, D), N is number of samples, D is number of features

    Returns:
        x': numpy array of shape (N, D + 1)
    """
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)


def calculate_feature_medians(X):
    """
    Calculate median for each feature that has at least one NaN value.

    Args:
        X: numpy array of shape (N, D), N is number of samples, D is number of features

    Returns:
        collumns_to_drop: list of indices of columns to drop due to high volume oof missing values (over 80%)
        columns_to_fill: list of indices of columns to be filled with their median
        feature_medians: corresponding medians of each feature/column with respect to the given indices columns_to_fill
    """
    columns_with_missing_features = np.where(np.any(np.isnan(X), axis=0))[0]

    columns_to_drop, columns_to_fill = [], []
    for column in columns_with_missing_features:
        if np.isnan(X[:, column]).mean() > 0.8:
            columns_to_drop.append(column)
        else:
            columns_to_fill.append(column)

    feature_medians = np.nanmedian(X[:, columns_to_fill], axis=0)
    return columns_to_drop, columns_to_fill, feature_medians


def fill_features_with_median(
    X, columns_with_missing_features=None, feature_medians=None
):
    """
    Fill missing values of given columns and with respective festure medians. If they are not given,
    calculate them manulally.

    Args:
        X: numpy array of shape (N, D), N is number of samples, D is number of features
        columns_with_missing_features: numpy array of indices of columns containing at least one NaN value
        feature_medians: corresponding medians of each feature/column with respect to the indices returned

    Returns:
        X: numpy array of shape (N, D), with NaN values filled with given medians
    """
    if columns_with_missing_features is None or feature_medians is None:
        _, columns_with_missing_features, feature_medians = calculate_feature_medians(X)

    for column, median in zip(columns_with_missing_features, feature_medians):
        X_i = X[:, column]
        X_i[np.isnan(X_i)] = median

    return X


def calculate_feature_weights(X):
    """
    Calculate median for each feature that has at least one NaN value.

    Args:
        X: numpy array of shape (N, D), N is number of samples, D is number of features

    Returns:
        collumns_to_drop: list of indices of columns to drop due to high volume oof missing values (over 80%)
        columns_to_fill: list of indices of columns to be filled with their median
        feature_medians: corresponding medians of each feature/column with respect to the given indices columns_to_fill
    """
    columns_with_missing_features = np.where(np.any(np.isnan(X), axis=0))[0]
    columns_wih_full_features = np.where(np.all(~np.isnan(X), axis=0))[0]

    collumns_to_drop, columns_to_fill = [], []
    for column in columns_with_missing_features:
        if np.isnan(X[:, column]).mean() > 0.8:
            collumns_to_drop.append(column)
        else:
            columns_to_fill.append(column)

    learning_matrix = X[:, columns_wih_full_features]
    feature_weights, learning_features = [], []
    for column in columns_to_fill:
        x_i = X[:, column]
        x_values, y_values = learning_matrix[~np.isnan(x_i)], x_i[~np.isnan(x_i)]

        non_zero_cols = np.where(np.any(x_values != 0, axis=0))[0]
        learning_features.append(non_zero_cols)
        x_values = x_values[:, non_zero_cols]

        weights, _ = implementations.least_squares(y_values, x_values)
        feature_weights.append(weights)

    return collumns_to_drop, columns_to_fill, feature_weights, learning_features


def fill_features_with_weights(
    X, columns_with_missing_features=None, feature_weights=None, learning_features=None
):
    """
    Fill missing values of given columns and with respective festure medians. If they are not given,
    calculate them manulally.

    Args:
        X: numpy array of shape (N, D), N is number of samples, D is number of features
        columns_with_missing_features: numpy array of indices of columns containing at least one NaN value
        feature_medians: corresponding medians of each feature/column with respect to the indices returned

    Returns:
        X: numpy array of shape (N, D), with NaN values filled with given medians
    """
    if columns_with_missing_features is None or feature_weights is None:
        (
            _,
            columns_with_missing_features,
            feature_weights,
            learning_features,
        ) = calculate_feature_weights(X)

    columns_wih_full_features = np.where(np.all(~np.isnan(X), axis=0))[0]
    learning_matrix = X[:, columns_wih_full_features]
    for column, weights, feature in zip(
        columns_with_missing_features, feature_weights, learning_features
    ):
        X_i = X[:, column]
        X_learning = learning_matrix[:, feature]
        X_i[np.isnan(X_i)] = X_learning[np.isnan(X_i)] @ weights

    return X


def colums_with_missing_features(X):
    """
    Return indices of columns of X that have at least one missing value.

    Args:
        X: numpy array of shape (N, D), N is number of samples, D is number of features

    Returns:
        not_full_columns: list of indices of columns with at least one missing value
    """
    not_full_columns = np.where(np.any(np.isnan(X), axis=0))[0]
    return list(not_full_columns)


def log_transform(x, columns):
    """
    Transforms heavy taild non-negative features with log transform.
    Fetures to be transfored were found through process of resources analysis.

    Args:
        x: numpy array of shape (N, D), N is number of samples, D is number of features

    Returns:
        x': numpy array of shape (N, D), with log-transformed features where appropiate
    """
    x[:, columns] = np.log(1 + x[:, columns])
    return x


def build_poly_feature(tx, degree):
    """
    Polynomial basis functions for input resources x up to degree 'degree'.
    In addition, perform coupling between each pair of x features.

    Args:
        x: numpy array of shape (N, D), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N, degree + 1)
    """
    N, D = tx.shape
    poly = tx.copy()
    # powers
    if degree > 1:
        for d in range(2, degree + 1):
            poly = np.hstack((poly, np.power(tx, d)))
    # product
    for i in range(D):
        for j in range(i + 1, D):
            product = tx[:, i] * tx[:, j]
            poly = np.c_[poly, product]
    return add_bias_term(poly)
