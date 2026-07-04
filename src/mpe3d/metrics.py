"""Reconstruction quality metrics and a simple MDS baseline.

The two point-cloud similarity measures are Chamfer distance
(:func:`chamfer_distance`) and Earth Mover's Distance (:func:`earth_movers_distance`).
:func:`baseline_metrics` provides a reference embedding from plain MDS on the
averaged distance matrix, so MPSE results can be compared against a naive method.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .alignment import apply_transformation, four_point_sample_transform
from .mview import MDS


def _mean_min_distance(array1, array2):
    """Mean over ``array1`` of the distance from each point to its nearest in ``array2``."""
    d = cdist(array1, array2)
    return np.mean(np.min(d, axis=1))


def chamfer_distance(array1, array2, scale=100):
    """Symmetric Chamfer distance between two ``(N, 3)`` point clouds.

    Both directions of nearest-neighbour distances are averaged and multiplied by
    ``scale`` (matching the historical reporting convention).
    """
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    dist = _mean_min_distance(array1, array2) + _mean_min_distance(array2, array1)
    return dist * scale


def earth_movers_distance(X, Y, scale=100):
    """Earth Mover's Distance via optimal 1-to-1 assignment (Hungarian algorithm)."""
    d = cdist(X, Y)
    row, col = linear_sum_assignment(d)
    return d[row, col].sum() * scale / min(len(X), len(Y))


def baseline_metrics(points, dist_mats, max_iter=200, rng=None):
    """Reconstruct with plain MDS on the averaged distance matrix, then score it.

    This is the naive reference the MPSE pipeline is compared against: it ignores
    the per-view structure and simply embeds the mean distance matrix in 3D.

    Returns a dict with ``chamfer``, ``emd``, ``transform`` and ``embedding``.
    """
    mean_distances = np.array(dist_mats).mean(axis=0)
    mds = MDS(mean_distances, dim=3, initial_embedding='random')
    mds.gd(max_iter=max_iter)
    embedding = mds.X

    transform, _ = four_point_sample_transform(embedding, points, rng=rng)
    aligned = apply_transformation(embedding, transform)
    return {
        'chamfer': chamfer_distance(aligned, points),
        'emd': earth_movers_distance(aligned, points),
        'transform': transform,
        'embedding': embedding,
    }
