import numpy as np
import util
import matplotlib.pylab as plt

def main(train_path, save_path, reg = False, l_decay=False):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)
    model = LogisticRegression()
    losses, thetas = model.fit(x_train, y_train, reg, l_decay)
    preds = model.predict(x_train)
    np.savetxt(f'{save_path}.txt', preds)
    
    util.plot(x_train, y_train, model.theta, f'{save_path}.png')

    plt.figure()
    plt.plot(losses)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.savefig(f'{save_path}_loss.png')

    theta_norms = [np.linalg.norm(theta) for theta in thetas]
    plt.figure()
    plt.plot(theta_norms)
    plt.xlabel('iter')
    plt.ylabel('parameter magnitude')
    plt.savefig(f'{save_path}_param_magnitude.png')

    print(thetas[-1])

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5, lbda = 0.01,
                 theta_0=None, verbose=True):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            lbda: regularization rate
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.lbda = lbda
        self.verbose = verbose

        # *** START CODE HERE ***
        # *** END CODE HERE ***
    
    def loss(self, x, y, reg = False):
        e = 1e-5 
        n = x.shape[0]
        loss = (-1/n)*sum([(y[i]*np.log(self.sigmoid(x[i,:] @ self.theta) + e)) + ((1-y[i])*np.log(1-self.sigmoid(x[i,:] @ self.theta) + e)) for i in range(n)])
        if reg:
            loss += 0.5 * self.lbda * np.linalg.norm(self.theta, ord = 2)**2
        
        return loss
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, x, y, reg = False, l_decay=False):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])
        n = x.shape[0]
        ls = [] # loss over iteations
        thetas = []
        for i in range(self.max_iter):
            grad = (-1/n)*sum([(y[i]-self.sigmoid(x[i, :] @ self.theta))*x[i,:].T for i in range(n)])
            if reg:
                grad += self.lbda*self.theta
            
            old_theta = self.theta.copy()
            if l_decay:
                self.theta -= (1/(i+1)**2) * self.learning_rate * grad
            else:
                self.theta -= self.learning_rate * grad
            
            thetas.append(self.theta.copy())
            
            ls.append(self.loss(x, y, reg))
            if np.linalg.norm(old_theta - self.theta, ord = 1) < self.eps:
                break
        
        return ls, thetas
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    print('\n==== Training model on data set A ====')
    main(train_path='ds1_a.csv',
         save_path='logreg_pred_a')
    
    print('\n==== Training model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b')
    
    
    print('\n==== Training regularized model on data set A ====')
    main(train_path='ds1_a.csv',
         save_path='logreg_pred_a_reg', reg=True)
    
    print('\n==== Training regularized model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b_reg', reg=True)
