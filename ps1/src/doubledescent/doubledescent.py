import numpy as np
import util

# Dimension of x
d = 500
# List for lambda to plot
reg_list = [0, 1, 5, 10, 50, 250, 500, 1000]
# List of dataset sizes
n_list = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

def regression(train_path, test_path):
    """Part (b): Double descent for unregularized linear regression.
    For a specific training set, obtain beta_hat and return test error.

    Args:
        train_path: Path to CSV file containing training set.
        test_path: Path to CSV file containing test set.

    Return:
        test_err: test error
    """
    x_train, y_train = util.load_dataset(train_path)
    x_test, y_test = util.load_dataset(test_path)

    test_err = 0
    # *** START CODE HERE ***
    beta = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
    #beta_ = np.linalg.solve(x_train.T @ x_train, x_train.T @ y_train)
    #print(np.isclose(beta, beta_))

    mse = (1/(2*x_test.shape[0])) * np.linalg.norm(x_test@beta - y_test, ord = 2)**2

    #preds = [x_test[i, :] @ beta for i in range(x_test.shape[0])]
    #mse_ = np.mean([(preds[i] - y_test[i])**2 for i in range(len(preds))])
    #print(mse, mse_)
    test_err = mse
    # *** END CODE HERE
    return test_err

def ridge_regression(train_path, test_path):
    """Part (c): Double descent for regularized linear regression.
    For a specific training set, obtain beta_hat under different l2 regularization strengths
    and return test error.

    Args:
        train_path: Path to CSV file containing training set.
        test_path: Path to CSV file containing test set.

    Return:
        test_err: List of test errors for different scaling factors of lambda in reg_list.
    """
    x_train, y_train = util.load_dataset(train_path)
    x_test, y_test = util.load_dataset(test_path)

    test_err = []
    # *** START CODE HERE ***
    for l in reg_list:
        beta = np.linalg.pinv(x_train.T @ x_train + l * np.identity(x_train.shape[1])) @ x_train.T @ y_train # pinv for all
        mse = (1/(2*x_test.shape[0])) * np.linalg.norm(x_test@beta - y_test, ord = 2)**2
        test_err.append(mse)
    #beta_ = np.linalg.solve(x_train.T @ x_train, x_train.T @ y_train)
    #print(np.isclose(beta, beta_))
    # *** END CODE HERE
    return test_err

if __name__ == '__main__':
    test_err = []
    for n in n_list:
        test_err.append(regression(train_path='train%d.csv' % n, test_path='test.csv'))
    util.plot(test_err, 'unreg.png', n_list)

    test_errs = []
    for n in n_list:
        test_errs.append(ridge_regression(train_path='train%d.csv' % n, test_path='test.csv'))
    test_errs = np.asarray(test_errs).T
    util.plot_all(test_errs, 'reg.png', n_list)
