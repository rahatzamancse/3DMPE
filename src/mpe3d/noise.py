"""Inject noise into the pipeline to study reconstruction robustness.

Two noise models are provided:

* :func:`add_distance_noise` perturbs the entries of the per-view distance
  matrices directly (measurement noise).
* :func:`add_matching_noise` swaps the identities of nearby points within a
  perspective (correspondence/matching noise).
"""

import numpy as np


def add_distance_noise(distance_matrices, noise_amount=0.01, noise_level=0.01,
                       in_place=True, noise_dist='gaussian', rng=None):
    """Add noise to a fraction of points across all distance matrices.

    Parameters
    ----------
    distance_matrices : list of ndarray
        Per-perspective square distance matrices.
    noise_amount : float
        Fraction of points (rows/columns) to perturb.
    noise_level : float
        Noise magnitude, as a fraction of each matrix's value range.
    in_place : bool
        If ``False``, operate on copies and leave the inputs untouched.
    noise_dist : str
        ``'gaussian'`` or ``'uniform'``.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(distance_matrices[0])
    noise_points = rng.choice(n, int(noise_amount * n), replace=False)
    if not in_place:
        distance_matrices = [persp.copy() for persp in distance_matrices]

    for persp in range(len(distance_matrices)):
        matrix = distance_matrices[persp]
        dist_range = (matrix.max() - matrix.min()) * noise_level
        for i in noise_points:
            if noise_dist == 'gaussian':
                noise = rng.normal(loc=0, scale=dist_range, size=n)
            elif noise_dist == 'uniform':
                noise = rng.uniform(low=-dist_range / 2, high=dist_range / 2, size=n)
            else:
                raise ValueError(f"unknown noise_dist: {noise_dist!r}")
            matrix[i, :] += noise
            matrix[:, i] += noise
            matrix[i, i] = 0
    return [np.maximum(persp, 0) for persp in distance_matrices]


def add_matching_noise(perspectives, p=0.01, q=0.01, rng=None):
    """Swap the identities of nearby points within random perspectives.

    For a fraction ``p`` of points, one perspective is chosen at random and the
    point is swapped with one of its ``q``-nearest neighbours in that view. This
    corrupts the cross-view correspondence rather than the distances themselves.

    Parameters
    ----------
    perspectives : sequence of ndarray
        One ``(N, d)`` array of point coordinates per perspective.
    p : float
        Fraction of points to corrupt.
    q : float
        Neighbourhood size (fraction of points) to swap within.
    """
    rng = np.random.default_rng() if rng is None else rng
    perspectives = np.array(perspectives)
    n = perspectives.shape[1]
    noise_points = rng.choice(n, int(p * n), replace=False)

    for point1 in noise_points:
        persp = rng.choice(perspectives.shape[0])
        dists = np.linalg.norm(
            perspectives[persp, :, :2] - perspectives[persp, point1, :2], axis=1)
        neighbours = np.argsort(dists)[:max(int(q * n), 1)]
        point2 = rng.choice(neighbours)
        perspectives[persp, [point1, point2]] = perspectives[persp, [point2, point1]]

    return perspectives
