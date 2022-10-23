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
    Columns_With_Missing_Features = np.setdiff1d(
        np.arange(X.shape[1]), Columns_With_All_Features)

    Feature_Weights = []

    for Feature_Index in range(X.shape[1]):
        # print(Columns_With_All_Features)
        if Feature_Index in Columns_With_All_Features:
            continue
        # print(~np.isnan(X[:, Feature_Index]))
        # print(np.nonzero(~np.isnan(X[:, Feature_Index]))[0])

        Rows_With_Feature_Index = np.nonzero(~np.isnan(X[:, Feature_Index]))[0]
        # Rows_Without_Feature_Index = np.setdiff1d(np.arange(X.shape[0]), Rows_With_Feature_Index)

        # print(np.take(X, [Rows_With_Feature_Index, Columns_With_All_Features]))
        # print(Rows_With_Feature_Index.shape[0] + Rows_Without_Feature_Index.shape[0])
        # print("##########")
        weights, loss = least_squares(X[np.ix_(Rows_With_Feature_Index, Columns_With_All_Features)], X[np.ix_(
            Rows_With_Feature_Index, [Feature_Index])])
        Feature_Weights.append(weights)

        # weights = weights.T
        # X[np.ix_(Rows_Without_Feature_Index, [Feature_Index])] = X[np.ix_(Rows_Without_Feature_Index, Columns_With_All_Features)] @ weights
    return Columns_With_Missing_Features, Feature_Weights


def fill_features_with_linear_regression(X, Columns_With_Missing_Features=None, Feature_Weights=None):
    if(Columns_With_Missing_Features is None or Feature_Weights is None):
        Columns_With_Missing_Features, Feature_Weights = calculate_feature_weight(
            X)

    Columns_With_All_Features = np.setdiff1d(
        np.arange(X.shape[1]), Columns_With_Missing_Features)

    for Missing_Feature_Index in Columns_With_Missing_Features:
        Rows_With_Feature_Index = np.nonzero(
            ~np.isnan(X[:, Missing_Feature_Index]))[0]
        Rows_Without_Feature_Index = np.setdiff1d(
            np.arange(X.shape[0]), Rows_With_Feature_Index)

        weights = np.array(Feature_Weights[Missing_Feature_Index]).T
        X[np.ix_(Rows_Without_Feature_Index, [Missing_Feature_Index])] = X[np.ix_(
            Rows_Without_Feature_Index, Columns_With_All_Features)] @ weights

    return X


def calculate_feature_medians(X):
    Columns_With_Missing_Features = np.where(np.all(np.isnan(X), axis=0))[0]

    Feature_Median = np.nanmedian(
        X[np.ix_(np.arange(X.shape[0]), Columns_With_Missing_Features)], axis=1)
    return Columns_With_Missing_Features, Feature_Median


def fill_features_with_median(X, Columns_With_Missing_Features=None, Feature_Median=None):
    if(Columns_With_Missing_Features is None or Feature_Median is None):
        Columns_With_Missing_Features, Feature_Median = calculate_feature_medians(
            X)

    for Missing_Feature_Index in Columns_With_Missing_Features:
        Rows_Without_Feature_Index = np.nonzero(
            np.isnan(X[:, Missing_Feature_Index]))[0]
        X[np.ix_(Rows_Without_Feature_Index, [Missing_Feature_Index])
          ] = Feature_Median[Missing_Feature_Index]

    return X
