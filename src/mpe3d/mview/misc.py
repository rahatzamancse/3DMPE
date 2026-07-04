"""Miscellaneous helpers: random initial embeddings and triangular indexing."""

import numpy as np


def box(number, dim=2, center=0, radius=1.0, **kwargs):
    """Sample ``number`` points uniformly from a ``dim``-dimensional box."""
    return 2 * (np.random.rand(number, dim) - 0.5) * radius + center


def disk(number, dim=2, center=0, radius=1.0, **kwargs):
    """Sample ``number`` points uniformly from a ``dim``-dimensional ball."""
    r = np.random.rand(number)
    X0 = np.random.randn(number, dim)
    return (X0.T / np.linalg.norm(X0, axis=1) * r ** (1.0 / dim)).T * radius + center


initial_embedding_methods = {
    'box': box,
    'disk': disk,
}


def initial_embedding(number, method='disk', **kwargs):
    """Produce an initial embedding using one of :data:`initial_embedding_methods`."""
    algorithm = initial_embedding_methods[method]
    return algorithm(number, **kwargs)


def labels(X, function=None, axis=0):
    """Return rank labels of the points ``X`` along the given ``axis``."""
    if function is None:
        temp = sorted(X[:, axis])
        labels = [temp.index(i) for i in X[:, axis]]
    return labels


def list_to_triangular(N, index_list):
    """Map condensed upper-triangular indices back to ``(i, j)`` edge pairs."""
    edges = np.empty((len(index_list), 2), dtype=int)
    i = N - 2 - np.floor(np.sqrt(-8 * index_list + 4 * N * (N - 1) - 7) / 2.0 - 0.5)
    j = index_list + i + 1 - N * (N - 1) / 2 + (N - i) * ((N - i) - 1) / 2
    edges[:, 0] = i
    edges[:, 1] = j
    return edges


def random_triangular(N, number, replace=False):
    """Return ``number`` random ``(i, j)`` edges of the complete graph on ``N`` nodes."""
    k = np.random.choice(round(N * (N - 1) / 2), number, replace=replace)
    return list_to_triangular(N, k)
