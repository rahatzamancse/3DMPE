"""Generate 2D perspectives of a 3D point cloud and their per-view distance matrices.

The MPSE pipeline never sees the 3D points directly. Instead it receives, for
each simulated camera, a matrix of pairwise 2D distances between the points that
are visible in that view. This module produces those perspectives (by rotating
the cloud), models which points each view can see, and assembles the distance
and weight matrices that :func:`mpe3d.mview.basic` consumes.

Points are carried around as "labeled" points: lists of ``{'data': ndarray,
'id': int}`` dicts, so that a point keeps its identity even when it is dropped
from some views.
"""

import itertools
from functools import lru_cache

import numpy as np


def rotation_matrix(angle_deg, axis):
    """Return the 3x3 rotation matrix for ``angle_deg`` degrees about ``axis``.

    ``axis`` is one of ``'x'``, ``'y'``, ``'z'``.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")


def rotation_matrix_xyz(alpha_deg, beta_deg, gamma_deg):
    """General 3-axis rotation matrix from the paper (Eq. 9).

    ``alpha``, ``beta`` and ``gamma`` are the rotation angles (in degrees)
    about the x, y and z axes, composed as ``Rz(alpha) @ Ry(beta) @ Rx(gamma)``::

        R = | ca*cb   ca*sb*sg - sa*cg   ca*sb*cg + sa*sg |
            | sa*cb   sa*sb*sg + ca*cg   sa*sb*cg - ca*sg |
            | -sb     cb*sg              cb*cg            |
    """
    a, b, g = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    return np.array([
        [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
        [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
        [-sb, cb * sg, cb * cg],
    ])


def equiangle_perspectives(points, n):
    """Rotate ``points`` about the Y-axis into ``n`` equally spaced perspectives.

    Returns ``(perspectives, projection_matrices)`` where each perspective is
    the rotated ``(N, 3)`` point cloud and each projection matrix ``P``
    satisfies ``view = points @ P.T`` (i.e. ``P`` maps ground-truth coordinates
    to that view's coordinates, as required for fixed-projection 3DMPE; the
    paper records the *negated* rotation angles for the same reason).
    """
    if n == 1:
        return [points], [np.identity(3)]
    perspectives, projection_mats = [], []
    if n % 2 == 1:
        perspectives.append(points)
        projection_mats.append(np.identity(3))
        n -= 1

    add_angle = 360 / n
    for i in range(1, n // 2 + 1):
        angle = i * add_angle
        for signed_angle in (angle, -angle):
            rot = rotation_matrix(signed_angle, 'y')
            perspectives.append(np.dot(points, rot))
            projection_mats.append(rot.T)
    return perspectives, projection_mats


def randomized_perspectives(points, n, angle_range=(0, 360), axes='y',
                            rng=None):
    """Rotate ``points`` into ``n`` randomly jittered perspectives.

    The angular band ``theta_r = end - start`` is split into ``n`` bins; one
    random angle is drawn per bin so the perspectives are spread out but not
    perfectly regular, and half of the views use negated angles (paper,
    Section 3 "Dataset"). The returned projection matrices map ground-truth
    coordinates to view coordinates (``view = points @ P.T``), which is the
    form fixed-projection 3DMPE expects.

    Parameters
    ----------
    points : ndarray, shape ``(N, 3)``
    n : int
        Number of perspectives.
    angle_range : tuple of float
        ``(start, end)`` rotation band in degrees.
    axes : str
        ``'y'`` rotates about the Y axis only; ``'xyz'`` draws an independent
        angle per axis and composes them with the general rotation matrix of
        the paper's Eq. 9 (:func:`rotation_matrix_xyz`).

    Returns ``(perspectives, projection_matrices)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    if axes not in ('y', 'xyz'):
        raise ValueError(f"axes must be 'y' or 'xyz', got {axes!r}")
    if n == 1:
        return [points.copy()], [np.identity(3)]
    perspectives, projection_mats = [], []
    if n % 2 == 1:
        perspectives.append(points.copy())
        projection_mats.append(np.identity(3))
        n -= 1

    start, end = angle_range
    add_angle = (end - start) / n
    n_angles = 3 if axes == 'xyz' else 1
    for i in range(1, n // 2 + 1):
        low = (i - 1) * add_angle + start
        high = i * add_angle + start
        angles1 = rng.uniform(low, high, size=n_angles)
        angles2 = rng.uniform(low, high, size=n_angles)
        for signed_angles in (angles1, -angles2):
            if axes == 'xyz':
                rot = rotation_matrix_xyz(*signed_angles)
            else:
                rot = rotation_matrix(signed_angles[0], 'y')
            perspectives.append(np.dot(points, rot))
            projection_mats.append(rot.T)
    return perspectives, projection_mats


def add_zero_column(points):
    """Append a zero column, turning ``(N, k)`` points into ``(N, k+1)``."""
    return np.append(points, np.zeros((len(points), 1)), axis=1)


def label_points(points):
    """Wrap an ``(N, d)`` array as a list of ``{'data', 'id'}`` labeled points."""
    return [{'data': point, 'id': i} for i, point in enumerate(points)]


def unlabel_points(labeled_points):
    """Return the ``(N, d)`` array of coordinates from labeled points."""
    return np.array([point['data'] for point in labeled_points])


def visible_in_n_perspectives(labeled_perspectives, points_in_each, rng=None):
    """Keep each point in only ``points_in_each`` randomly chosen perspectives.

    This simulates limited visibility: every point is seen by exactly
    ``points_in_each`` of the cameras (chosen independently per point).
    """
    rng = np.random.default_rng() if rng is None else rng
    n_points = len(labeled_perspectives[0])
    n_persp = len(labeled_perspectives)

    chosen = [rng.choice(n_persp, size=points_in_each, replace=False)
              for _ in range(n_points)]

    new_perspectives = [[] for _ in range(n_persp)]
    for point_idx, persp_indices in enumerate(chosen):
        for persp in persp_indices:
            new_perspectives[persp].append(labeled_perspectives[persp][point_idx])
    return new_perspectives


def raytrace_z(points, ray_radius=None, n_rays_x=None, n_rays_y=None):
    """Keep only points visible from ``+z`` via a coarse ray-cast (Z-buffer).

    A grid of rays is cast along the Z axis; for each ray the closest (smallest
    Z) point within ``ray_radius`` is kept. This approximates self-occlusion.

    ``points`` is a list of labeled points; the return value is the visible
    subset (also labeled).
    """
    @lru_cache(maxsize=1_000_000)
    def ray_hit(point_xy, ray_xy, r):
        return ((point_xy[0] - ray_xy[0]) ** 2
                + (point_xy[1] - ray_xy[1]) ** 2) <= r * r

    coords = np.array([p['data'] for p in points])

    if not n_rays_x:
        n_rays_x = len(points) // 3
    if not n_rays_y:
        n_rays_y = len(points) // 3

    x_rays = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_rays_x)
    y_rays = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_rays_y)

    if not ray_radius:
        ray_radius = np.min([x_rays[1] - x_rays[0], y_rays[1] - y_rays[0]]) * 2

    closest = [[{'data': [np.inf, np.inf, np.inf], 'id': -1}
                for _ in range(len(x_rays))] for _ in range(len(y_rays))]

    for i, x in enumerate(x_rays):
        for j, y in enumerate(y_rays):
            for point in points:
                if ray_hit(tuple(point['data'][:2]), (x, y), ray_radius):
                    if closest[i][j]['data'][2] > point['data'][2]:
                        closest[i][j] = point

    visible, taken = [], {-1}
    for row in closest:
        for point in row:
            if point['id'] not in taken and point['data'][0] != np.inf:
                visible.append(point)
                taken.add(point['id'])
    return visible


def distance_weight_matrices(labeled_perspectives, n_points, ndim=None):
    """Build per-perspective distance and visibility-weight matrices.

    For each perspective, ``dist[i, j]`` is the 2D distance between points ``i``
    and ``j`` (using the first ``ndim`` coordinates) when both are visible, and
    ``weight[i, j]`` is ``1`` for visible pairs and ``0`` otherwise. Points not
    seen in a perspective get zero weight there, so they are ignored by MPSE.

    Returns ``(distance_matrices, weight_matrices)``, each a list of
    ``(n_points, n_points)`` arrays.
    """
    if ndim is None:
        ndim = len(labeled_perspectives[0][0]['data'])
    dist_mats = [np.zeros((n_points, n_points)) for _ in labeled_perspectives]
    weight_mats = [np.zeros((n_points, n_points)) for _ in labeled_perspectives]

    for persp, dist_mat, weight_mat in zip(labeled_perspectives, dist_mats,
                                           weight_mats):
        for p1, p2 in itertools.combinations_with_replacement(persp, 2):
            dist = np.linalg.norm(p1['data'][:ndim] - p2['data'][:ndim])
            i, j = p1['id'], p2['id']
            dist_mat[i, j] = dist_mat[j, i] = dist
            weight_mat[i, j] = weight_mat[j, i] = 1
    return dist_mats, weight_mats


def points_per_min_perspectives(weight_mats):
    """Return how many points are visible in at least ``k`` perspectives.

    Result index ``k-1`` is the count of points that appear in at least ``k`` of
    the perspectives (for ``k = 1 .. n_perspectives``).
    """
    def visible_in_at_least(n):
        ands = []
        for combo in itertools.combinations(weight_mats, n):
            tmp = np.ones_like(weight_mats[0])
            for w in combo:
                tmp = np.logical_and(tmp, w)
            ands.append(tmp)
        combined = np.zeros_like(weight_mats[0])
        for a in ands:
            combined = np.logical_or(combined, a)
        return int(sum(1 for row in combined if row.sum() > 0))

    return [visible_in_at_least(n) for n in range(1, len(weight_mats) + 1)]
