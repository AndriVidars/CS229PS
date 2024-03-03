import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    n = x.shape[0]

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices) 
    x_shuffled = x[indices]
    x_split = np.array_split(x_shuffled, K, axis=0)
    mu = [np.mean(x_, axis = 0) for x_ in x_split]
    sigma = [np.cov(x_.T) for x_ in x_split]

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(K, 1/K)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full((n, K), 1/K)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):

    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000
    
    n = x.shape[0]
    d = x.shape[1]
    k = w.shape[1]

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        
        # (1) E-step: Update your estimates in w
        for i in range(n):
            # maybe x[i, :] needs to be transpose to match up with mu ?
            terms = [(np.linalg.det(sigma[j])**(-0.5))*np.exp(-0.5*(x[i, :] - mu[j]).T @ np.linalg.inv(sigma[j]) @ (x[i, :] - mu[j])) * phi[j] for j in range(k)]
            denom = sum(terms)
            
            for j in range(k):
                w[i, j] = terms[j] / denom

        # (2) M-step: Update the model parameters phi, mu, and sigma
        
        # update mus, should we calculate all grads with prev data and then, yes prob
        mu_new = [np.zeros(d) for _ in range(k)]
        phi_new = np.zeros(k)
        sigma_new = [np.zeros((d, d)) for _ in range(k)]
        
        for j in range(k):
            mu_new[j] = sum([w[i, j] * x[i, :] for i in range(n)]) / sum([w[i, j] for i in range(n)])
            
            phi_new[j] = (1/n) * sum([w[i, j] for i in range(n)])

            sigma_new[j] = sum([w[i, j] * np.outer(x[i, :] - mu[j], x[i, :] - mu[j]) for i in range(n)]) / sum([w[i, j] for i in range(n)])


        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
            
        prev_ll = ll_unsup(x, phi, mu, sigma)
        ll = ll_unsup(x, phi_new, mu_new, sigma_new)

        assert(prev_ll < ll)

        mu, phi, sigma = mu_new, phi_new, sigma_new

        it += 1


        # *** END CODE HERE ***
    print(it)
    return w

def ll_unsup(x, phi, mu, sigma):
    out = 0
    n, d = x.shape
    k = len(mu)
    for i in range(n):
        out += np.log(sum([(1/((2*np.pi)**(d/2)*np.linalg.det(sigma[j])**(0.5)))*
                           np.exp(-0.5*(x[i, :] - mu[j]).T @ np.linalg.inv(sigma[j]) @ (x[i, :] - mu[j])) * phi[j] for j in range(k)]))
    
    
    return out

def ll_semisup(x, x_tilde, phi, mu, sigma, alpha):
    out = 0
    n, d = x.shape
    n_tilde = x_tilde.shape[0]
    k = len(mu)
    for i in range(n):
        out += np.log(sum([(1/((2*np.pi)**(d/2)*np.linalg.det(sigma[j])**(0.5)))*
                           np.exp(-0.5*(x[i, :] - mu[j]).T @ np.linalg.inv(sigma[j]) @ (x[i, :] - mu[j])) * phi[j] for j in range(k)]))

    for i in range(n_tilde):
        out += alpha * np.log(sum([(1/((2*np.pi)**(d/2)*np.linalg.det(sigma[j])**(0.5)))*
                           np.exp(-0.5*(x_tilde[i, :] - mu[j]).T @ np.linalg.inv(sigma[j]) @ (x_tilde[i, :] - mu[j])) * phi[j] for j in range(k)]))
    
    
    return out




def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    n = x.shape[0]
    n_tilde = x_tilde.shape[0]
    d = x.shape[1]
    k = w.shape[1]

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        for i in range(n):
            # maybe x[i, :] needs to be transpose to match up with mu ?
            terms = [(np.linalg.det(sigma[j])**(-0.5))*np.exp(-0.5*(x[i, :] - mu[j]).T @ np.linalg.inv(sigma[j]) @ (x[i, :] - mu[j])) * phi[j] for j in range(k)]
            denom = sum(terms)
            
            for j in range(k):
                w[i, j] = terms[j] / denom

        # (2) M-step: Update the model parameters phi, mu, and sigma
        
        # update mus, should we calculate all grads with prev data and then, yes prob
        mu_new = [np.zeros(d) for _ in range(k)]
        phi_new = np.zeros(k)
        sigma_new = [np.zeros((d, d)) for _ in range(k)]
        
        for j in range(k):
            mu_new[j] = (sum([w[i, j] * x[i, :] for i in range(n)]) + sum([alpha* x_tilde[i, :] for i in range(n_tilde) if z_tilde[i] == j])) \
                  / (sum([w[i, j] for i in range(n)]) + sum([alpha for i in range(n_tilde) if z_tilde[i] == j]))
            
            phi_new[j] = (sum([w[i, j] for i in range(n)]) + sum([alpha for i in range(n_tilde) if z_tilde[i] == j])) / (n + alpha*n_tilde)

            sigma_new[j] = (sum([w[i, j] * np.outer(x[i, :] - mu[j], x[i, :] - mu[j]) for i in range(n)])\
                             + sum([alpha * np.outer(x_tilde[i, :] - mu[j], x_tilde[i, :] - mu[j]) for i in range(n_tilde) if z_tilde[i] == j]))\
                  / (sum([w[i, j] for i in range(n)]) + sum([alpha for i in range(n_tilde) if z_tilde[i] == j]))


        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
            
        prev_ll = ll_semisup(x, x_tilde, phi, mu, sigma, alpha)
        ll = ll_semisup(x, x_tilde, phi_new, mu_new, sigma_new, alpha)

        assert( prev_ll < ll)

        mu, phi, sigma = mu_new, phi_new, sigma_new

        it += 1

    print(it)
    return w


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        #main(is_semi_supervised=False, trial_num=t)
        

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
