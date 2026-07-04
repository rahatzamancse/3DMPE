"""Multidimensional scaling (MDS): stress, gradients, and the :class:`MDS` class."""

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import scipy.spatial.distance

from . import misc, setup, gd, plots


def stress(distances, embedding, weights=None, normalize=True):
    """Return (optionally normalized) MDS stress for a distance/embedding pair.

    Parameters
    ----------
    distances : ndarray, shape ``(n*(n-1)/2,)``
        Condensed target distances.
    embedding : ndarray, shape ``(n, dim)``
        Current embedding.
    weights : ndarray or None
        Optional per-pair weights.
    """
    dist = scipy.spatial.distance.pdist(embedding)
    diff = distances - dist
    if weights is None:
        stress = np.linalg.norm(diff) ** 2
        if normalize:
            stress = math.sqrt(stress / len(distances))
    else:
        stress = np.dot(weights, diff ** 2)
        if normalize:
            stress = math.sqrt(stress / np.sum(weights))
    return stress


def full_gradient(distances, embedding, weights=None, normalize=True,
                  minimum_distance=None, return_objective=True):
    """Return the MDS stress gradient (and stress) for the full distance set."""
    grad = np.zeros(embedding.shape)
    stress = 0
    dist = scipy.spatial.distance.pdist(embedding)
    if minimum_distance is not None:
        dist = np.maximum(minimum_distance, dist)
    diff = dist - distances

    if weights is None:
        constants = 2 * diff / dist
    else:
        constants = 2 * weights * diff / dist

    grad_terms = scipy.spatial.distance.squareform(constants)
    for i in range(len(embedding)):
        grad[i] = np.dot(np.ravel(grad_terms[i], order='K'),
                         embedding[i] - embedding)

    if normalize:
        if weights is None:
            grad /= np.linalg.norm(distances)
        else:
            grad /= np.sqrt(np.dot(weights, distances ** 2))

    if return_objective:
        if weights is None:
            stress = np.linalg.norm(diff) ** 2
        else:
            stress = np.dot(weights, diff ** 2)
        if normalize:
            if weights is None:
                stress = math.sqrt(stress / len(distances))
            else:
                stress = math.sqrt(stress / np.sum(weights))
        return grad, stress
    else:
        return grad


def batch_gradient(distances, embedding, batch_size=10, indices=None,
                   weights=None, normalize=True, minimum_distance=None,
                   return_objective=True):
    """Return a stochastic (mini-batch) estimate of the MDS stress gradient."""
    n_samples = len(embedding)
    if indices is None:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
    else:
        assert len(indices) == n_samples
    grad = np.empty(embedding.shape)
    stress = 0
    weights_batch = None
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = np.sort(indices[start:end])
        embedding_batch = embedding[batch_idx]
        batch_indices = setup.batch_indices(batch_idx, n_samples)
        distances_batch = distances[batch_indices]
        if weights is not None:
            weights_batch = weights[batch_indices]
        grad[batch_idx], st0 = full_gradient(distances_batch, embedding_batch,
                                             weights=weights_batch,
                                             minimum_distance=minimum_distance)
        stress += st0 ** 2
    if normalize:
        n_batches = math.ceil(n_samples / batch_size)
        grad /= np.linalg.norm(distances) / n_samples * batch_size
        stress = math.sqrt(stress / n_batches)

    if return_objective:
        return grad, stress
    else:
        return grad


class MDS(object):
    """Solve a (weighted) multidimensional scaling problem by gradient descent."""

    def __init__(self, data, dim=2, weights=None, estimate=True, safety=1e-4,
                 normalize=True, initial_embedding='random',
                 sample_colors=None, verbose=0, indent='', **kwargs):
        """Initialize the MDS object.

        Parameters
        ----------
        data : ndarray
            Condensed distances, a square distance matrix, or a feature array.
        dim : int
            Embedding dimension.
        weights : None or str or callable or ndarray
            Weights used in the stress function.
        """
        self.verbose = verbose
        self.indent = indent
        if self.verbose > 0:
            print(self.indent + 'mview.MDS():')

        self.distances = setup.setup_distances(data, **kwargs)
        self.n_samples = scipy.spatial.distance.num_obs_y(self.distances)

        if safety is None:
            self.minimum_distance = None
        else:
            assert safety > 0 and safety <= 1e-2
            self.minimum_distance = np.max(self.distances) * safety
            self.distances = np.maximum(self.distances, self.minimum_distance)

        self.weights = setup.setup_weights(self.distances, weights=weights)
        self.normalize = normalize

        if sample_colors is None:
            self.sample_colors = self.distances[0:self.n_samples]
        else:
            self.sample_colors = sample_colors

        assert isinstance(dim, int)
        assert dim > 0
        self.dim = dim

        assert isinstance(estimate, bool)
        self.estimate = estimate

        self.objective = lambda X, **kwargs: stress(
            self.distances, X, weights=self.weights, normalize=self.normalize)

        def gradient(embedding, batch_size=None, indices=None, **kwargs):
            if batch_size is None or batch_size >= self.n_samples:
                return full_gradient(
                    self.distances, embedding,
                    weights=self.weights, normalize=self.normalize,
                    minimum_distance=self.minimum_distance)
            else:
                return batch_gradient(
                    self.distances, embedding, batch_size, indices,
                    weights=self.weights, normalize=self.normalize,
                    minimum_distance=self.minimum_distance)
        self.gradient = gradient

        if verbose > 0:
            print(indent + '  data details:')
            print(indent + f'    number of samples : {self.n_samples}')
            print(indent + f'    weighted : {self.weights is not None}')
            print(indent + '  embedding details:')
            print(indent + f'    embedding dimension : {self.dim}')

        if isinstance(initial_embedding, np.ndarray):
            assert initial_embedding.shape == (self.n_samples, self.dim)
            if self.verbose > 0:
                print('    initial embedding : given')
            self.X0 = initial_embedding
            self.X = self.X0
        elif initial_embedding == 'random':
            self.X0 = misc.initial_embedding(self.n_samples, dim=self.dim,
                                             radius=1, **kwargs)
            self.X = self.X0
            if self.verbose > 0:
                print('    initial embedding : random')
        else:
            assert initial_embedding is None

        if initial_embedding is not None:
            self.initial_cost = self.objective(self.X0, **kwargs)
            self.cost = self.initial_cost
            if self.verbose > 0:
                print(f'    initial stress : {self.cost:0.2e}')

        self.computation_history = []

    def update(self, X, H, **kwargs):
        self.X = X
        self.cost = self.objective(self.X, **kwargs)
        self.computation_history.append(H)

    def gd(self, batch_size=None, **kwargs):
        """Run gradient descent to minimize the MDS stress."""
        if self.verbose > 0:
            print(self.indent + '  MDS.gd():')
            print(self.indent + '    specs:')

        if batch_size is None or batch_size >= self.n_samples:
            Xi = None
            F = lambda X: self.gradient(X)
            if self.verbose > 0:
                print(self.indent + '      gradient type : full')
        else:
            def Xi():
                indices = np.arange(self.n_samples)
                np.random.shuffle(indices)
                return {'indices': indices}
            F = lambda X, indices: self.gradient(X, batch_size=batch_size,
                                                 indices=indices)
            if self.verbose > 0:
                print(self.indent + '      gradient type : batch')
                print(self.indent + '      batch size :', batch_size)
        X, H = gd.single(self.X, F, Xi=Xi, verbose=self.verbose,
                         indent=self.indent + '    ', **kwargs)
        self.update(X, H, **kwargs)
        if self.verbose > 0:
            print(self.indent + f'    final stress : {self.cost:0.2e}')

    def plot_embedding(self, title='embedding', edges=False, colors='default',
                       labels=None, axis=True, plot=True, ax=None, **kwargs):
        """Plot the current 2D or 3D embedding."""
        assert self.dim >= 2
        if edges is True:
            edges = self.distances['edge_list']
        elif edges is False:
            edges = None
        if colors == 'default':
            colors = self.sample_colors

        if self.dim == 2:
            plots.plot2D(self.X, edges=edges, colors=colors, labels=labels,
                         axis=axis, ax=ax, title=title, **kwargs)
        else:
            plots.plot3D(self.X, edges=edges, colors=colors, title=title,
                         ax=ax, **kwargs)
        if plot is True:
            plt.draw()
            plt.pause(1)
