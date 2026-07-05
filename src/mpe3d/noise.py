"""Inject noise into the pipeline to study reconstruction robustness.

Two noise regimes from the 3DMPE paper (Section 3.1, Figure 5) are provided:

* :func:`add_distance_noise` -- additive Gaussian noise on the per-view
  pairwise distance matrices (measurement noise). For amplitude ``p``, a
  fraction ``q`` of the rows receive zero-mean Gaussian noise with variance
  ``p * d`` where ``d`` is the diameter of the point cloud; negative entries
  are clipped to 0 and the matrices are kept symmetric.
* :func:`add_matching_noise` -- erroneous correspondences. A fraction ``q`` of
  the points have their identity reassigned to a random point at most
  ``p * d`` away in 3D space.
"""

import numpy as np


def add_distance_noise(distance_matrices, noise_amount=0.01, noise_level=0.05,
                       in_place=True, noise_dist='gaussian', rng=None):
    """Add noise to a fraction of points across all distance matrices.

    Parameters
    ----------
    distance_matrices : list of ndarray
        Per-perspective square distance matrices.
    noise_amount : float
        Fraction ``q`` of points (rows/columns) to perturb.
    noise_level : float
        Noise amplitude ``p``: the Gaussian variance is ``p * d`` where ``d``
        is the diameter of the point cloud (the largest observed pairwise
        distance).
    in_place : bool
        If ``False``, operate on copies and leave the inputs untouched.
    noise_dist : str
        ``'gaussian'`` (paper model) or ``'uniform'``.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(distance_matrices[0])
    noise_points = rng.choice(n, int(noise_amount * n), replace=False)
    if not in_place:
        distance_matrices = [persp.copy() for persp in distance_matrices]

    diameter = max(persp.max() for persp in distance_matrices)
    sigma = np.sqrt(noise_level * diameter)

    for matrix in distance_matrices:
        for i in noise_points:
            if noise_dist == 'gaussian':
                noise = rng.normal(loc=0, scale=sigma, size=n)
            elif noise_dist == 'uniform':
                noise = rng.uniform(low=-sigma, high=sigma, size=n)
            else:
                raise ValueError(f"unknown noise_dist: {noise_dist!r}")
            # Adding the same vector to row i and column i keeps the matrix
            # symmetric.
            matrix[i, :] += noise
            matrix[:, i] += noise
            matrix[i, i] = 0
    return [np.maximum(persp, 0) for persp in distance_matrices]


def add_matching_noise(perspectives, noise_amount=0.01, noise_level=0.05,
                       rng=None):
    """Corrupt cross-view point correspondences (paper Section 3.1).

    For a fraction ``noise_amount`` (``q``) of the points, one perspective is
    chosen at random and the point's identity is swapped with a random point
    at most ``noise_level * d`` (``p * d``) away, where ``d`` is the diameter
    of the point cloud. This corrupts the correspondence rather than the
    distances themselves.

    Parameters
    ----------
    perspectives : sequence of ndarray
        One ``(N, 3)`` array of point coordinates per perspective.
    noise_amount : float
        Fraction ``q`` of points to corrupt.
    noise_level : float
        Amplitude ``p``: maximum reassignment distance as a fraction of the
        point-cloud diameter.
    """
    rng = np.random.default_rng() if rng is None else rng
    perspectives = np.array(perspectives)
    n = perspectives.shape[1]
    noise_points = rng.choice(n, int(noise_amount * n), replace=False)

    extent = perspectives[0].max(axis=0) - perspectives[0].min(axis=0)
    radius = noise_level * np.linalg.norm(extent)

    for point1 in noise_points:
        persp = rng.choice(perspectives.shape[0])
        dists = np.linalg.norm(
            perspectives[persp] - perspectives[persp, point1], axis=1)
        dists[point1] = np.inf
        candidates = np.flatnonzero(dists <= radius)
        if len(candidates) == 0:
            candidates = [np.argmin(dists)]
        point2 = rng.choice(candidates)
        perspectives[persp, [point1, point2]] = perspectives[persp, [point2, point1]]

    return perspectives
