import preprocessing
import utils
import implementations
import helpers

import numpy as np

if __name__ == "__main__":
    y, X, ids = helpers.load_csv_data(path="resources/train.csv")

    lambdas = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    degrees = [i for i in range(2, 13)]

    traning_groups = utils.group_by_categories(X, column=22)
    for i, idx in enumerate(traning_groups):
        print(f"PRI_jet_num =", i)
        x_i, y_i = X[idx], y[idx]

        kfold = 10
        best_accuracy, best_degree, best_lambda = 0, -1, -1
        best_f1, f1_degree, f1_lambda = 0, -1, -1
        for degree in degrees:
            for lambda_ in lambdas:
                rmse, accuracy, f1 = 0, 0, 0
                for train, test in utils.kfolds(y_i.shape[0], kfold=kfold):
                    x_tr, x_test, y_tr, y_test = preprocessing.preprocess_data(
                        x_i[train], x_i[test], y_i[train], y_i[test], degree
                    )

                    weights, _ = implementations.ridge_regression(y_tr, x_tr, lambda_)

                    rmse_test = np.sqrt(
                        2 * utils.mean_square_error(y_test, x_test, weights)
                    )
                    rmse += rmse_test

                    y_pred = np.array([-1 if x.T @ weights < 0 else 1 for x in x_test])
                    accuracy += (y_pred == y_test).mean()
                    f1 += utils.f1_score(y_test, x_test, weights, f=utils.linear)

                rmse /= kfold
                accuracy /= kfold
                f1 /= kfold
                print(
                    f"For degree={degree} and lambda={lambda_}: rmse={rmse}, score={accuracy}, f1={f1}"
                )

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_degree = degree
                    best_lambda = lambda_

                if f1 > best_f1:
                    best_f1 = f1
                    f1_degree = degree
                    f1_lambda = lambda_

        print("-" * 50)
        print(
            f"For PRI_jet_num={i}, the best score={best_accuracy} is given for degree={best_degree} and lambda={lambda_}"
        )
        print(
            f"For PRI_jet_num={i}, the best f1={best_f1} is given for degree={f1_degree} and lambda={f1_lambda}"
        )
        print("*" * 50)
