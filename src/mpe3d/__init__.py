"""3DMPE: 3D point-cloud reconstruction from multi-view distance matrices.

This package reconstructs a 3D point cloud from several 2D "views" of it, using
Multi-Perspective Simultaneous Embedding (MPSE). Rather than pixels, each view is
summarized by a matrix of pairwise 2D distances between the visible points; MPSE
jointly recovers a single 3D embedding and one projection per view that together
reproduce all of those distance matrices.

Typical usage::

    from mpe3d import datasets, reconstruct, visualization

    points = datasets.get_dataset_points("demo:torus", n_points=300)
    result = reconstruct(points, n_perspectives=5, max_iter=200)
    print(result.summary())
    fig = visualization.plot_point_clouds(
        [result.ground_truth, result.aligned_embedding],
        names=["ground truth", "reconstruction"],
    )

Module map:

* :mod:`mpe3d.datasets` - load/generate ground-truth point clouds.
* :mod:`mpe3d.views` - simulate perspectives and build distance matrices.
* :mod:`mpe3d.noise` - inject distance/matching noise.
* :mod:`mpe3d.mview` - vendored MPSE optimizer (the numerical core).
* :mod:`mpe3d.alignment` - align a reconstruction to the ground truth.
* :mod:`mpe3d.metrics` - Chamfer/EMD/ROA metrics and an MDS baseline.
* :mod:`mpe3d.visualization` - interactive Plotly plots and benchmark curves.
* :mod:`mpe3d.pipeline` - the end-to-end :func:`reconstruct` helper.
* :mod:`mpe3d.benchmark` - LMNet/ShapeNet parameter sweeps as tidy DataFrames.
"""

from . import (  # noqa: F401
    alignment,
    benchmark,
    datasets,
    metrics,
    mview,
    noise,
    views,
    visualization,
)
from .pipeline import ReconstructionResult, build_distance_matrices, reconstruct

__all__ = [
    "reconstruct",
    "build_distance_matrices",
    "ReconstructionResult",
    "datasets",
    "views",
    "noise",
    "mview",
    "alignment",
    "metrics",
    "visualization",
    "benchmark",
]
