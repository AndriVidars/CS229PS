import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(X.T @ X, X.T @ y)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples,) # edited input shape
        """
        # *** START CODE HERE ***
        return np.array([[x**i for i in range(k+1)] for x in X]).reshape(X.shape[0], k+1)
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples). # edited input shape
        """
        # *** START CODE HERE ***
        return np.asarray([[x**i for i in range(k+1)] + [np.sin(x)] for x in X]).reshape(X.shape[0], k+2)
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.array([self.theta.T @ x for x in X]).reshape(-1)
        # *** END CODE HERE ***

# run part b
def part_b(train_path):
    train_x, train_y = util.load_dataset(train_path,add_intercept=False)
    lm = LinearModel()
    train_x_feat = lm.create_poly(3, train_x)
    lm.fit(train_x_feat, train_y)
    
    x_pred_line = np.linspace(min(train_x), max(train_x), 1000)
    x_pred_line_feat = lm.create_poly(3, x_pred_line)
    y_pred_line = lm.predict(x_pred_line_feat)

    plt.figure()
    plt.scatter(train_x, train_y, label = 'Train')
    plt.plot(x_pred_line, y_pred_line, label = 'Pred', c='red')
    plt.legend(loc = 'best')
    plt.savefig('Prob_2_b.png')

# run part c
def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=False)
    plot_x = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x, train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        lm = LinearModel()
        train_x_feat = lm.create_sin(k, train_x) if sine else lm.create_poly(k, train_x)
        lm.fit(train_x_feat, train_y)
        x_pred_line_feat = lm.create_sin(k, plot_x) if sine else  lm.create_poly(k, plot_x)
        plot_y = lm.predict(x_pred_line_feat)

        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x, plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()

def run_exp_e(train_path, eval_path, ks=[1, 2, 3, 5, 10, 20], filename='overfitting.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=False)
    eval_x, eval_y=util.load_dataset(eval_path,add_intercept=False)
    plot_x = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x, train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        lm = LinearModel()
        train_x_feat = lm.create_poly(k, train_x)
        lm.fit(train_x_feat, train_y)
        x_pred_line_feat = lm.create_poly(k, plot_x)
        plot_y = lm.predict(x_pred_line_feat)

        eval_x_feat = lm.create_poly(k, eval_x)
        eval_pred_y = lm.predict(eval_x_feat)
        mse = util.mse(eval_y, eval_pred_y)
        print(f'K = {k}, MSE:{mse}')

        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(plt.ylim(-5, 5))
        plt.plot(plot_x, plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    part_b(train_path) # part b
    run_exp(train_path) # part c
    run_exp(train_path, ks=[0, 1, 2, 3, 5, 10, 20], sine = True, filename = 'sine.png') # part d
    run_exp_e(small_path, eval_path) # part e

    # explination for part # e, for i datapoints, a polynomial of degree i-1 does completely interpolate the data
    # thus for k>= 5 there is 0 training loss and the models fits the data perfectly. However those models overfit the data by a large margin
    # as we can see with the increasing MSE on the evaluation set.
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
