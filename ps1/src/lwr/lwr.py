import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train, y_train)

    preds = lwr.predict(x_eval)
    mse = util.mse(preds, y_eval)
    print(f'MSE: {mse}')

    lwr.plot_lwr(x_eval, preds, 'lwr_plot.png')

    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.
        """

        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        preds = []
        for x_i in x:
            W = np.diag([0.5*np.exp(-(np.linalg.norm((x_i - x_j), 2)**2)/(2*self.tau**2)) for x_j in self.x])
            theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y
            pred = theta.T @ x_i
            preds.append(pred)
        
        return preds
    
    def plot_lwr(self, x_eval, preds, plot_name, tau = -1):
        x_t = [x[1] for x in self.x]
        x_e = [x[1] for x in x_eval]

        y_t = [y for _,y in sorted(zip(x_t, self.y))]
        preds_ = [y for _,y in sorted(zip(x_e, preds))]

        x_t.sort()
        x_e.sort()

        plt.figure()
        plt.scatter(x_t, y_t, label = 'Train', c='blue', marker='x', s = 15)
        plt.scatter(x_e, preds_, label = 'Preds', c='red', marker='o', s = 15)
        plt.xlabel('x')
        plt.legend(loc = 'best')
        if tau != -1:
            plt.title(r'$\tau$ = {:.2f}'.format(tau))
        
        plt.savefig(plot_name)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
