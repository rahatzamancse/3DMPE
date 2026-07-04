"""t-SNE objective and the :class:`TSNE` class (alternative per-view objective)."""

import numbers
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance

from . import misc, gd, plots, setup

MACHINE_EPSILON = np.finfo(np.double).eps


def joint_probabilities(distances, perplexity):
    """Compute condensed joint probabilities ``p_ij`` for the given perplexity."""
    distances = scipy.spatial.distance.squareform(distances)
    n_samples = len(distances)

    lower_bound = 1e-1
    upper_bound = 1e1
    iters = 10
    sigma = np.empty(n_samples)
    for i in range(n_samples):
        D_i = np.delete(distances[i], i)
        estimate = np.sum(D_i) / (n_samples - 1) / 5
        lower_bound_i = lower_bound * estimate
        upper_bound_i = upper_bound * estimate
        for _ in range(iters):
            sigma_i = (lower_bound_i * upper_bound_i) ** (1 / 2)
            P_i = np.exp(-D_i ** 2 / (2 * sigma_i ** 2))
            P_i /= np.sum(P_i)
            HP_i = -np.dot(P_i, np.log2(P_i + MACHINE_EPSILON))
            PerpP_i = 2 ** (HP_i)
            if PerpP_i > perplexity:
                upper_bound_i = sigma_i
            else:
                lower_bound_i = sigma_i
        sigma[i] = (lower_bound_i * upper_bound_i) ** (1 / 2)

    conditional_P = np.exp(-distances ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(conditional_P, 0)
    conditional_P /= np.sum(conditional_P, axis=0)

    P = (conditional_P + conditional_P.T)
    P = scipy.spatial.distance.squareform(P, checks=False)
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(P / sum_P, MACHINE_EPSILON)

    return P


def inverse_square_law_distances(embedding):
    """Return the condensed Student-t (inverse square law) affinities of ``embedding``."""
    dist = scipy.spatial.distance.pdist(embedding, metric='sqeuclidean')
    dist += 1.0
    dist **= -1.0
    return dist


def KL(P, embedding):
    """Return the KL divergence ``KL(P || Q)`` for the current embedding."""
    dist = scipy.spatial.distance.pdist(embedding, metric='sqeuclidean')
    dist += 1.0
    dist **= -1.0
    Q = np.maximum(dist / (np.sum(dist)), MACHINE_EPSILON)
    return 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))


def grad_KL(P, embedding, dist=None, Q=None):
    """Return the KL-divergence gradient (and value) at the current embedding."""
    if dist is None or Q is None:
        dist = scipy.spatial.distance.pdist(embedding, metric='sqeuclidean')
        dist += 1.0
        dist **= -1.0
        Q = np.maximum(dist / (np.sum(dist)), MACHINE_EPSILON)

    kl_divergence = 2.0 * np.dot(
        P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    grad = np.ndarray(embedding.shape)
    PQd = scipy.spatial.distance.squareform((P - Q) * dist)
    for i in range(len(embedding)):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'), embedding[i] - embedding)
    grad *= 4

    return grad, kl_divergence


def batch_gradient(P, embedding, batch_size=10, indices=None):
    """Return a stochastic (mini-batch) estimate of the KL gradient."""
    n_samples = len(embedding)
    if indices is None:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
    else:
        assert len(indices) == n_samples

    grad = np.empty(embedding.shape)
    stress = 0
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = np.sort(indices[start:end])
        embedding_batch = embedding[batch_idx]
        P_batch = P[setup.batch_indices(batch_idx, n_samples)]
        dist = inverse_square_law_distances(embedding_batch)
        Q_batch = dist / (np.sum(dist)) / (n_samples / len(batch_idx)) ** 2
        grad[batch_idx], st0 = grad_KL(P_batch, embedding_batch,
                                       dist=dist, Q=Q_batch)
        stress += st0
    grad *= n_samples / batch_size
    stress *= n_samples / batch_size
    return grad, stress


class TSNE(object):
    """Solve a t-SNE problem by gradient descent on the KL divergence."""

    def __init__(self, data, dim=2, perplexity=30.0,
                 sample_labels=None, sample_classes=None, sample_colors=None,
                 verbose=0, indent='', **kwargs):
        """Initialize the TSNE object from distances/dissimilarities ``data``."""
        self.verbose = verbose
        self.indent = indent
        if self.verbose > 0:
            print(self.indent + 'mview.TSNE():')

        self.distances = setup.setup_distances(data, **kwargs)
        self.n_samples = scipy.spatial.distance.num_obs_y(self.distances)

        self.sample_labels = sample_labels
        self.sample_classes = sample_classes
        self.sample_colors = sample_colors

        self.N = self.n_samples
        self.D = self.distances

        assert isinstance(dim, int)
        assert dim > 0
        self.dim = dim

        if verbose > 0:
            print(indent + '  data details:')
            print(indent + f'    number of samples : {self.n_samples}')
            print(indent + '  embedding details:')
            print(indent + f'    embedding dimension : {dim}')
            print(indent + f'    perplexity : {perplexity:0.2f}')

        self.P = joint_probabilities(self.D, perplexity)

        self.objective = lambda X, P=self.P, **kwargs: KL(P, X)

        def gradient(embedding, batch_size=None, indices=None, **kwargs):
            if batch_size is None or batch_size >= self.n_samples:
                return grad_KL(self.P, embedding)
            else:
                return batch_gradient(self.P, embedding, batch_size, indices)
        self.gradient = gradient

        self.computation_history = []
        self.initialize()

    def initialize(self, X0=None, **kwargs):
        """Set the initial embedding (random by default)."""
        if self.verbose > 0:
            print(self.indent + '  TSNE.initialize():')

        if X0 is None:
            X0 = misc.initial_embedding(self.N, dim=self.dim, radius=1, **kwargs)
            if self.verbose > 0:
                print(self.indent + '    method : random')
        else:
            assert isinstance(X0, np.ndarray)
            assert X0.shape == (self.N, self.dim)
            if self.verbose > 0:
                print(self.indent + '    method : initialization given')

        self.update(X0)
        self.embedding0 = self.embedding.copy()

        if self.verbose > 0:
            print(self.indent + f'    initial cost : {self.cost:0.2e}')

    def update(self, X, H=None):
        self.embedding = X
        self.cost = self.objective(self.embedding)
        if H is not None:
            self.computation_history.append(H)

    def gd(self, batch_size=None, lr=None, **kwargs):
        """Run gradient descent to minimize the KL divergence."""
        if self.verbose > 0:
            print(self.indent + '  TSNE.gd():')
            print(self.indent + '    specs:')

        if lr is None:
            if len(self.computation_history) != 0:
                lr = self.computation_history[-1]['lr']
            else:
                lr = 100

        if batch_size is None or batch_size >= self.n_samples:
            Xi = None
            F = lambda embedding: self.gradient(embedding)
            if self.verbose > 0:
                print(self.indent + '      gradient type : full')
        else:
            def Xi():
                indices = np.arange(self.n_samples)
                np.random.shuffle(indices)
                return {'indices': indices}
            F = lambda X, indices: self.gradient(X, batch_size, indices)
            if self.verbose > 0:
                print(self.indent + '      gradient type : batch')
                print(self.indent + '      batch size :', batch_size)

        X, H = gd.single(self.embedding, F, Xi=Xi, lr=lr, verbose=self.verbose,
                         indent=self.indent + '    ', **kwargs)
        self.update(X, H)
        if self.verbose > 0:
            print(self.indent + f'    final stress : {self.cost:0.2e}')

    def optimized(self, iters=[20, 20, 20, 100], **kwargs):
        """Run a multi-stage schedule of mini-batch then full-batch descent."""
        if self.verbose > 0:
            print(self.indent + '  TSNE.optimized():')
        self.gd(batch_size=self.n_samples // 50, max_iter=iters[0], scheme='mm')
        self.gd(batch_size=self.n_samples // 10, max_iter=iters[1], scheme='mm')
        self.gd(batch_size=self.n_samples // 5, max_iter=iters[2], scheme='mm')
        self.gd(max_iter=iters[3], scheme='mm')

    def plot_embedding(self, title='', edges=False, colors=None, labels=None,
                       axis=True, plot=True, ax=None, **kwargs):
        """Plot the current 2D embedding."""
        assert self.dim >= 2
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plot = False
        if edges is True:
            edges = self.D['edge_list']
        elif edges is False:
            edges = None
        if colors is True:
            colors = self.sample_colors
        plots.plot2D(self.embedding, edges=edges, colors=colors, labels=labels,
                     axis=axis, ax=ax, title=title, **kwargs)
        if plot is True:
            plt.draw()
            plt.pause(1)
