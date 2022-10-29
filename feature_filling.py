from statistics import median
import numpy as np
from implementations import *
from helpers import *
from utils import *


def standardizeIgnoreNaNs(x):
    """Standardize the data ignoring the NaN values"""
    mean_x = np.nanmean(x)
    x -= mean_x
    std_x = np.nanstd(x)
    x /= std_x
    return x, mean_x, std_x


def calculate_feature_weight(X):
    """returns the weight matrix for missing feature columns"""

    Columns_With_All_Features = np.where(np.all(~np.isnan(X), axis=0))[0]
    columns_with_missing_features = np.setdiff1d(
        np.arange(X.shape[1]), Columns_With_All_Features)

    Feature_Weights = []

    for Feature_Index in range(X.shape[1]):
        # print(Columns_With_All_Features)
        if Feature_Index in Columns_With_All_Features:
            continue
        # print(~np.isnan(X[:, Feature_Index]))
        # print(np.nonzero(~np.isnan(X[:, Feature_Index]))[0])

        Rows_With_Feature_Index = np.nonzero(~np.isnan(X[:, Feature_Index]))[0]
        # rows_without_feature_index = np.setdiff1d(np.arange(X.shape[0]), Rows_With_Feature_Index)

        # print(np.take(X, [Rows_With_Feature_Index, Columns_With_All_Features]))
        # print(Rows_With_Feature_Index.shape[0] + rows_without_feature_index.shape[0])
        # print("##########")
        weights, loss = least_squares(X[np.ix_(Rows_With_Feature_Index, Columns_With_All_Features)], X[np.ix_(
            Rows_With_Feature_Index, [Feature_Index])])
        Feature_Weights.append(weights)

        # weights = weights.T
        # X[np.ix_(rows_without_feature_index, [Feature_Index])] = X[np.ix_(rows_without_feature_index, Columns_With_All_Features)] @ weights
    return columns_with_missing_features, Feature_Weights


def fill_features_with_linear_regression(X, Columns_With_Missing_Features=None, Feature_Weights=None):
    if Columns_With_Missing_Features is None or Feature_Weights is None:
        Columns_With_Missing_Features, Feature_Weights = calculate_feature_weight(
            X)

    Columns_With_All_Features = np.setdiff1d(
        np.arange(X.shape[1]), columns_with_missing_features)

    for missing_feature_column in columns_with_missing_features:
        Rows_With_Feature_Index = np.nonzero(
            ~np.isnan(X[:, missing_feature_column]))[0]
        rows_without_feature_index = np.setdiff1d(
            np.arange(X.shape[0]), Rows_With_Feature_Index)

        weights = np.array(Feature_Weights[missing_feature_column]).T
        X[np.ix_(rows_without_feature_index, [missing_feature_column])] = X[np.ix_(
            rows_without_feature_index, Columns_With_All_Features)] @ weights

    return X


def calculate_feature_medians(X):
    columns_with_missing_features = np.where(np.any(np.isnan(X), axis=0))[0]
    feature_medians = np.nanmedian(X[:, columns_with_missing_features], axis=0)
    return columns_with_missing_features, feature_medians


def fill_features_with_median(X, columns_with_missing_features=None, feature_medians=None):
    if columns_with_missing_features is None or feature_medians is None:
        columns_with_missing_features, feature_medians = calculate_feature_medians(X)

    for column, median in zip(columns_with_missing_features, feature_medians):
        X_i = X[:, column]
        X_i[np.isnan(X_i)] = median

    return X
