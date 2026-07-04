"""Set up condensed distances and weights used by MDS/MPSE."""

import numpy as np
import scipy.spatial.distance as distance
import scipy.sparse.csgraph as csgraph
import itertools


def setup_distances(data, shortest_path=False, min_distance=1e-4, **kwargs):
    """Return condensed distances from a data array.

    Parameters
    ----------
    data : ndarray
        Either a condensed 1D distance array, a square distance matrix, or a
        feature array (pairwise Euclidean distances are computed in that case).
    shortest_path : bool
        If ``True``, replace distances with graph shortest-path distances.
    min_distance : float or None
        If given, clamp distances to at least ``min_distance * max(distances)``.
    """
    assert isinstance(data, np.ndarray)
    if len(data.shape) == 1:
        assert distance.is_valid_y(data)
        distances = data
    else:
        assert len(data.shape) == 2
        a, b = data.shape
        if b == a:
            distances = distance.squareform(data, checks=False)
        else:
            distances = distance.pdist(data)
    if shortest_path:
        distances = distance.squareform(distances)
        distances = csgraph.shortest_path(distances)
        distances = distance.squareform(distances, checks=False)
    if min_distance is not None:
        distances = np.maximum(distances, min_distance * np.max(distances))
    return distances


def setup_weights(distances, weights, max_weight=2.0, min_weight=1e-4):
    """Return a condensed weights array matching ``distances``.

    ``weights`` may be ``None`` (unweighted), the string ``'reciprocal'``, a
    callable of distance, or an array of per-node or per-pair weights.
    """
    if isinstance(weights, str):
        if weights == 'reciprocal':
            weights = 1.0 / distances
        else:
            raise ValueError(f'unknown weight type: {weights!r}')
    elif callable(weights):
        try:
            weights = weights(distances)
        except Exception:
            weights = np.array([weights(dist) for dist in distances])
    elif isinstance(weights, np.ndarray):
        if len(weights) == distance.num_obs_y(distances):
            assert np.min(weights) >= 0
            weights = np.array(weights)
            weights = weights.T * weights
            weights = distance.squareform(weights, checks=False)
        assert distances.shape == weights.shape
    else:
        assert weights is None

    if weights is not None:
        if max_weight is not None:
            weights = np.minimum(weights, max_weight)
        if min_weight is not None:
            weights = np.maximum(weights, min_weight)

    return weights


def setup_distances_from_multiple_perspectives(data, data_args=None):
    """Return a list of condensed distances, one per perspective in ``data``."""
    n_perspectives = len(data)

    if data_args is None:
        data_args = [{}] * n_perspectives
    elif isinstance(data_args, dict):
        data_args = [data_args] * n_perspectives
    else:
        assert isinstance(data_args, list)
        assert len(data_args) == n_perspectives
        for i in range(n_perspectives):
            if data_args[i] is None:
                data_args[i] = {}
            else:
                assert isinstance(data_args[i], dict)

    condensed_distances = []
    for i in range(n_perspectives):
        condensed_distances.append(setup_distances(data[i], **data_args[i]))

    return condensed_distances


def batch_indices(samples, n_samples):
    """Return condensed-distance indices for all pairs within a batch of samples."""
    pairs = np.array(list(itertools.combinations(samples, 2)))
    indices = n_samples * pairs[:, 0] - pairs[:, 0] * (pairs[:, 0] + 1) // 2 + \
        pairs[:, 1] - 1 - pairs[:, 0]
    return indices


def distance_to_sample(distances, sample):
    """Return the row of the square distance matrix for a given sample index."""
    return distance.squareform(distances)[sample]
