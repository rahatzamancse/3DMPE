"""Reconstruction quality metrics and a simple MDS baseline.

Three point-cloud similarity measures, following the 3DMPE paper:

* :func:`chamfer_distance` -- Chamfer Distance (paper Eq. 5).
* :func:`earth_movers_distance` -- Earth Mover's Distance (paper Eq. 4).
* :func:`roa` -- RMSE-Optimize-Align (paper Eq. 8), the RMSE over corresponding
  point pairs after the optimal rigid (Kabsch) alignment.

:func:`baseline_metrics` provides the paper's baseline: plain MDS on the
averaged distance matrix, so 3DMPE results can be compared against a naive
method.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .alignment import (apply_transformation, four_point_sample_transform,
                        kabsch_transform)
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


def roa(reconstruction, ground_truth):
    """RMSE-Optimize-Align (ROA) metric between corresponding point clouds.

    Unlike Chamfer and EMD, ROA exploits the known point-to-point
    correspondence: the reconstruction is aligned onto the ground truth with
    the optimal rigid transform (closed-form Kabsch/SVD solution, paper
    Eq. 13-16) and the metric is the mean squared distance over corresponding
    pairs (paper Eq. 8).
    """
    reconstruction = np.asarray(reconstruction, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    T = kabsch_transform(reconstruction, ground_truth)
    aligned = apply_transformation(reconstruction, T)
    return float(np.mean(np.sum((aligned - ground_truth) ** 2, axis=1)))


def baseline_metrics(points, dist_mats, max_iter=200, rng=None):
    """Reconstruct with plain MDS on the averaged distance matrix, then score it.

    This is the naive reference the 3DMPE pipeline is compared against: it
    ignores the per-view structure and simply embeds the mean distance matrix
    in 3D. It also serves as the paper's "smart initialization" for SGD.

    Returns a dict with ``chamfer``, ``emd``, ``roa``, ``transform`` and
    ``embedding``.
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
        'roa': roa(embedding, points),
        'transform': transform,
        'embedding': embedding,
    }
