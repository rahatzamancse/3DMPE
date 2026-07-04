"""Align a recovered embedding to the ground-truth point cloud.

MPSE recovers a point cloud only up to a rigid transformation (and reflection),
so before measuring reconstruction error the embedding must be aligned to the
ground truth. Two strategies are provided: a random 4-point sampling search
(:func:`four_point_sample_transform`) and a closed-form SVD alignment
(:func:`svd_transform`).
"""

import numpy as np


def to_homogeneous_transform(mat):
    """Promote a 3x3 rotation (or 3x4) to a 4x4 homogeneous transform."""
    if mat.shape[0] == 3:
        mat = np.append(mat, [[0, 0, 0]], axis=0)
        mat = np.insert(mat, 3, [0, 0, 0, 1], axis=1)
    return mat


def to_homogeneous_points(points):
    """Append a ones column to ``(N, 3)`` points, giving ``(N, 4)``."""
    if points.shape[1] == 4:
        return points
    return np.append(points, np.ones((len(points), 1)), axis=1)


def to_cartesian_points(points):
    """Drop the homogeneous coordinate, giving ``(N, 3)``."""
    if points.shape[1] == 3:
        return points
    return points[:, :3]


def apply_rotation(points, rotation):
    """Apply a 3x3 rotation matrix to ``(N, 3)`` points."""
    return np.dot(points, rotation)


def apply_transformation(points, transform):
    """Apply a 3x3 or 4x4 ``transform`` to ``(N, 3)`` points, returning ``(N, 3)``."""
    initial_shape = points.shape
    if transform.shape[0] == 3:
        transform = to_homogeneous_transform(transform)
    if initial_shape[1] == 3:
        points = to_homogeneous_points(points)
    result = np.dot(transform, points.T).T
    if initial_shape[1] == 3:
        result = to_cartesian_points(result)
    return result


def transform_from_four_points(points_a, points_b):
    """Return the affine transform mapping 4 source points onto 4 target points."""
    points_a = np.array(points_a)
    points_b = np.array(points_b)
    assert points_a.shape[0] == 4 and points_b.shape[0] == 4, \
        "exactly 4 points are required"
    return np.dot(np.linalg.inv(to_homogeneous_points(points_a)),
                  to_homogeneous_points(points_b)).T


def four_point_sample_transform(points, gt_points, dist_agg_fn=None,
                                n_samples=10000, rng=None):
    """Search for the best alignment by random 4-point correspondences.

    Repeatedly picks 4 index-matched point pairs, solves for the transform they
    imply, and keeps whichever transform minimizes the aggregated distance to the
    ground truth over all points.

    Returns ``(best_transform, best_distance)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    if dist_agg_fn is None:
        dist_agg_fn = np.mean

    points = np.array(points)
    gt_points = np.array(gt_points)
    min_dist = np.inf
    best_t = np.identity(4)
    for _ in range(n_samples):
        ids = rng.choice(len(points), 4)
        try:
            transform = transform_from_four_points(points[ids], gt_points[ids])
        except np.linalg.LinAlgError:
            continue
        transformed = apply_transformation(points, transform)
        dist = dist_agg_fn(np.linalg.norm(transformed - gt_points, axis=1))
        if dist < min_dist:
            min_dist = dist
            best_t = transform
    return best_t, min_dist


def svd_transform(X, Y, dist_agg_fn=None):
    """Closed-form (orthogonal Procrustes) alignment of ``X`` onto ``Y``.

    Returns ``(transform, aggregated_distance)``.
    """
    if dist_agg_fn is None:
        dist_agg_fn = np.mean
    T = np.linalg.inv(X.T @ X) @ Y.T @ X
    U, _, V = np.linalg.svd(T)
    T = U @ V

    T = to_homogeneous_transform(T)
    aligned = apply_transformation(X, T)
    offset = Y[0] - aligned[0]
    for i in range(3):
        T[i, 3] = offset[i]

    return T, dist_agg_fn(np.linalg.norm(apply_transformation(X, T) - Y, axis=1))
