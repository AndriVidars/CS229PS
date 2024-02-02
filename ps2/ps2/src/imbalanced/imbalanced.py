import numpy as np
import util
import sys
from random import random

sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def accuracy(preds, y):
    return sum([1 if preds[i] == y[i] else 0 for i in range(len(preds))])/len(preds)

def sub_accuracy(preds, y, class_):
    n_cls = np.count_nonzero(y == class_)
    cnt = 0
    for i in range(len(preds)):
        if y[i] == class_ and preds[i] == class_:
            cnt += 1
    
    return cnt / n_cls

def runProblem(x_train, y_train, x_val, y_val, plotPath):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred_prob = model.predict(x_val)
    preds = [0 if p < 0.5 else 1 for p in pred_prob]

    acc = accuracy(preds, y_val)
    acc_1 = sub_accuracy(preds, y_val, 1)
    acc_0 = sub_accuracy(preds, y_val, 0)
    balanced_acc = 0.5 * (acc_1 + acc_0)

    print(f'Accuracy: {acc}')
    print(f'A_1: {acc_1}')
    print(f'A_0: {acc_0}')
    print(f'Balanced accuracy: {balanced_acc}\n')

    util.plot(x_val, y_val, model.theta, plotPath)
    return pred_prob


def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    print(f'Part b:\n')
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(validation_path, add_intercept=True)
    pred_probs = runProblem(x_train, y_train, x_val, y_val, 'naive.png')
    np.savetxt(output_path_naive, pred_probs)

    
    # Part (d): Upsampling minority class
    print(f'Part d:\n')

    p = np.count_nonzero(y_train == 1) / y_train.shape[0]
    k = p/(1-p)
    k_ = int(1/k)

    # Extract rows where y = 1
    rep_rows = np.where(y_train == 1)
    y_1 = y_train[rep_rows]
    x_1 = x_train[rep_rows, :]

    # Repeat these rows
    rep_y = np.repeat(y_1, repeats=k_-1, axis=0)
    rep_x = np.repeat(x_1, repeats=k_-1, axis=1)

    y_t = np.concatenate((y_train, rep_y), axis = 0)
    x_t = np.concatenate((x_train, rep_x[0, :, :]), axis = 0)

    pred_probs = runProblem(x_t, y_t, x_val, y_val, 'upsampling.png')
    np.savetxt(output_path_upsampling, pred_probs)
    
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
