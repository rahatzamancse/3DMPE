"""End-to-end 3D reconstruction pipeline built on the vendored MPSE core.

This module wires the individual stages together into a single call,
:func:`reconstruct`, which takes a ground-truth point cloud and returns the MPSE
reconstruction along with alignment and error metrics.

The stages are::

    points -> perspectives -> visibility -> distance matrices
           -> (optional noise) -> MPSE -> align to ground truth -> metrics
"""

from dataclasses import dataclass, field

import numpy as np

from . import mview
from .alignment import apply_transformation, four_point_sample_transform
from .metrics import baseline_metrics, chamfer_distance, earth_movers_distance
from .noise import add_distance_noise, add_matching_noise
from .views import (
    distance_weight_matrices,
    label_points,
    points_per_min_perspectives,
    randomized_perspectives,
    raytrace_z,
    visible_in_n_perspectives,
)


@dataclass
class ReconstructionResult:
    """Container for the outputs of :func:`reconstruct`."""

    ground_truth: np.ndarray
    embedding: np.ndarray
    aligned_embedding: np.ndarray
    projections: list
    transform: np.ndarray
    chamfer: float
    emd: float
    alignment_error: float
    cost: float
    cost_history: np.ndarray
    baseline: dict
    dist_mats: list = field(repr=False, default=None)
    weight_mats: list = field(repr=False, default=None)
    points_per_perspective: list = None

    def summary(self):
        """Return a compact dict of the headline metrics."""
        return {
            'chamfer': self.chamfer,
            'emd': self.emd,
            'final_cost': self.cost,
            'baseline_chamfer': self.baseline['chamfer'],
            'baseline_emd': self.baseline['emd'],
        }


def build_distance_matrices(points, n_perspectives=5, angle_range=(0, 360),
                            projection='atleast', points_in_at_least=4,
                            n_rays=None, noise_type='none', noise_amount=0.0,
                            noise_level=0.0, noise_dist='gaussian', rng=None):
    """Turn a 3D point cloud into per-view distance and weight matrices.

    See :func:`reconstruct` for the meaning of the parameters. Returns
    ``(dist_mats, weight_mats, projection_mats)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_points = len(points)

    perspectives, projection_mats = randomized_perspectives(
        points, n_perspectives, angle_range, rng=rng)

    if noise_type == 'matching':
        perspectives = add_matching_noise(perspectives, noise_amount,
                                          noise_level, rng=rng)

    labeled = [label_points(p) for p in perspectives]

    if projection == 'atleast':
        assert 1 <= points_in_at_least <= n_perspectives, \
            "points_in_at_least must be between 1 and n_perspectives"
        labeled = visible_in_n_perspectives(labeled, points_in_at_least, rng=rng)
    elif projection == 'raytracing':
        labeled = [raytrace_z(p, n_rays_x=n_rays, n_rays_y=n_rays)
                   for p in labeled]
    else:
        raise ValueError(f"unknown projection: {projection!r} "
                         "(expected 'atleast' or 'raytracing')")

    # Flatten each perspective onto its image plane (drop the depth coordinate).
    for perspective in labeled:
        for point in perspective:
            point['data'][2] = 0

    dist_mats, weight_mats = distance_weight_matrices(labeled, n_points, ndim=2)

    if noise_type == 'distance':
        dist_mats = add_distance_noise(dist_mats, noise_amount, noise_level,
                                       noise_dist=noise_dist, rng=rng)

    return dist_mats, weight_mats, projection_mats


def reconstruct(points, n_perspectives=5, angle_range=(0, 360),
                projection='atleast', points_in_at_least=4, n_rays=None,
                variable_projection=True, initial_projections='cylinder',
                noise_type='none', noise_amount=0.0, noise_level=0.0,
                noise_dist='gaussian', batch_size=None, max_iter=200,
                min_grad=1e-4, min_cost=1e-4, smart_initialize=True,
                random_initial_embedding=False, verbose=0, rng=None,
                compute_baseline=True):
    """Reconstruct a 3D point cloud from simulated multi-view distance matrices.

    Parameters
    ----------
    points : ndarray, shape ``(N, 3)``
        Ground-truth point cloud.
    n_perspectives : int
        Number of simulated camera views.
    angle_range : tuple of float
        ``(start, end)`` Y-axis rotation range, in degrees, over which the views
        are spread.
    projection : str
        Visibility model: ``'atleast'`` (each point visible in exactly
        ``points_in_at_least`` views) or ``'raytracing'`` (Z-buffer occlusion).
    points_in_at_least : int
        Used when ``projection='atleast'``.
    n_rays : int or None
        Ray grid resolution used when ``projection='raytracing'``.
    variable_projection : bool
        If ``True``, MPSE optimizes the projections starting from
        ``initial_projections``. If ``False``, the true rotation matrices are
        supplied as fixed projections.
    initial_projections : str
        Initial projection scheme when ``variable_projection`` is ``True``
        (e.g. ``'cylinder'``).
    noise_type : str
        ``'none'``, ``'distance'`` or ``'matching'``.
    noise_amount, noise_level : float
        Noise parameters (see :mod:`mpe3d.noise`).
    batch_size, max_iter, min_grad, min_cost : optimization controls
        Forwarded to :func:`mpe3d.mview.basic`.
    smart_initialize : bool
        Warm-start MPSE from a combined MDS embedding.
    random_initial_embedding : bool
        If ``True``, start MPSE from a random embedding spanning the data range.
    compute_baseline : bool
        If ``True``, also compute the plain-MDS baseline for comparison.

    Returns
    -------
    ReconstructionResult
    """
    rng = np.random.default_rng() if rng is None else rng
    points = np.asarray(points, dtype=float)

    dist_mats, weight_mats, projection_mats = build_distance_matrices(
        points, n_perspectives=n_perspectives, angle_range=angle_range,
        projection=projection, points_in_at_least=points_in_at_least,
        n_rays=n_rays, noise_type=noise_type, noise_amount=noise_amount,
        noise_level=noise_level, noise_dist=noise_dist, rng=rng)

    baseline = (baseline_metrics(points, dist_mats, rng=rng)
                if compute_baseline else {'chamfer': None, 'emd': None})

    if variable_projection:
        projection_kwargs = dict(initial_projections=initial_projections,
                                 fixed_projections=None)
    else:
        projection_kwargs = dict(fixed_projections=projection_mats)

    initial_embedding = None
    if random_initial_embedding:
        initial_embedding = rng.uniform(points.min(), points.max(),
                                        (len(points), 3))

    mv = mview.basic(
        [d.copy() for d in dist_mats],
        weights=[w.copy() for w in weight_mats],
        batch_size=batch_size,
        max_iter=max_iter,
        min_grad=min_grad,
        min_cost=min_cost,
        smart_initialize=smart_initialize,
        initial_embedding=initial_embedding,
        verbose=verbose,
        **projection_kwargs,
    )

    embedding = np.asarray(mv.X)
    transform, alignment_error = four_point_sample_transform(embedding, points,
                                                             rng=rng)
    aligned = apply_transformation(embedding, transform)

    return ReconstructionResult(
        ground_truth=points,
        embedding=embedding,
        aligned_embedding=aligned,
        projections=mv.projections,
        transform=transform,
        chamfer=chamfer_distance(aligned, points),
        emd=earth_movers_distance(aligned, points),
        alignment_error=float(alignment_error),
        cost=float(mv.cost),
        cost_history=np.asarray(mv.computation_history[-1]['costs']),
        baseline=baseline,
        dist_mats=dist_mats,
        weight_mats=weight_mats,
        points_per_perspective=points_per_min_perspectives(weight_mats),
    )
