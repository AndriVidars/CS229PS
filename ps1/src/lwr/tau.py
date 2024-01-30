import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    tests = []
    for t in tau_values:
        lwr = LocallyWeightedLinearRegression(t)
        lwr.fit(x_train, y_train)
        preds = lwr.predict(x_eval)
        mse = util.mse(preds, y_eval)
        print(f'tau: {t}, MSE: {mse}')
        lwr.plot_lwr(x_eval, preds, f'tau={t}.png', t)
        tests.append((t, lwr, preds, mse))


    lwr_best = min(tests, key = lambda x: x[3])
    test_preds = lwr_best[1].predict(x_test)
    lwr_best[1].plot_lwr(x_test, test_preds, 'optimal_lwr_test.png')

    mse_test = util.mse(test_preds, y_test)
    print(f'Test MSE for optimal tau: {mse_test}')
    with open(pred_path, 'w') as outfile:
        outfile.writelines(list(map(lambda x: str(x), list(test_preds))))

    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
