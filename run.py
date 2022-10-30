import preprocessing
import utils
import implementations
import helpers

import numpy as np


if __name__ == "__main__":
    # load resources

    # Don't forget to shuffle resources testing process
    y, X, _ = helpers.load_csv_data(path="resources/train.csv")

    # convert labels to be only 0 and 1
    # y[np.where(y == -1)] = 0

    # split resources for validation
    x_tr, x_val, y_tr, y_val = utils.split_data(X, y, 0.8)
    # y_val, x_val, ids = helpers.load_csv_data(path="resources/test.csv")

    # split by jet
    training_groups = utils.group_by_categories(x_tr, column=22)
    validation_groups = utils.group_by_categories(x_val, column=22)
    degrees = [12, 12, 12, 12]
    lambdas = [0.0001, 0.0, 0.0, 0.0]

    total_correct = 0
    for i, (training_group_idx, validation_group_idx, degree, lambda_) in enumerate(
        zip(training_groups, validation_groups, degrees, lambdas)
    ):
        print("PRI_jet_num =", i)
        print("-" * 30)

        x_tr_i, y_tr_i = x_tr[training_group_idx], y_tr[training_group_idx]
        x_val_i, y_val_i = x_val[validation_group_idx], y_val[validation_group_idx]

        x_tr_i, x_val_i, y_tr_i, y_val_i = preprocessing.preprocess_data(
            x_tr_i, x_val_i, y_tr_i, y_val_i, degree
        )

        """weights, _ = implementations.reg_logistic_regression(
            y_tr_i, x_tr_i, 0, np.zeros(x_tr_i.shape[1]), max_iters=10000, gamma='adaptive')"""
        weights, _ = implementations.ridge_regression(y_tr_i, x_tr_i, lambda_)

        # y_pred = np.array([utils.predictions(x, weights) for x in x_val_i])
        y_pred = np.array([-1 if x @ weights < 0 else 1 for x in x_val_i])
        correct_predict = (y_pred == y_val_i).sum()
        print("Group precision", correct_predict / len(y_val_i))

        total_correct += correct_predict
        # y_val[validation_group_idx] = y_pred
        print("*" * 30)

    print("Validation accuracy", total_correct / len(y_val))
    # print("Ratio", (y_val == 1).sum() / len(y_val))
    # print(y_val)

    # helpers.create_csv_submission(ids, y_val, "lm6.csv")
