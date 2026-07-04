"""Vendored MPSE core (Multi-Perspective Simultaneous Embedding).

This package is a trimmed-down, lightly modernized copy of the ``mview`` library
from the original MPSE project by Rahat Zaman et al.
(https://github.com/rahatzamancse/MPSE), which implements the algorithm from:

    Hossain, Md I., et al. "Multi-Perspective, Simultaneous Embedding."
    IEEE Transactions on Visualization and Computer Graphics (2021).
    arXiv:1909.06485.

Only the code required to run :func:`basic` on precomputed distance matrices has
been kept. Flat ``import module`` statements were converted to package-relative
imports, ``sys.path`` hacks removed, and a couple of removed NumPy aliases
(``np.Inf``/``np.product``) were updated for NumPy 2.x. The numerical algorithm
itself is unchanged.

The single entry point used by the rest of ``mpe3d`` is :func:`basic`.
"""

from . import projections, mds, mpse, misc  # noqa: F401
from .projections import PROJ
from .mds import MDS
from .tsne import TSNE
from .mpse import MPSE

__all__ = ["basic", "MPSE", "MDS", "TSNE", "PROJ"]


def basic(data, data_args=None, fixed_projections=None,
          visualization_method='mds', smart_initialize=True,
          verbose=0, **kwargs):
    """Compute an MPSE embedding (and projections) from multi-view data.

    Parameters
    ----------
    data : list of ndarray, length ``n_perspectives``
        One dissimilarity/distance array per perspective. Each element may be a
        condensed 1D distance array, a square distance matrix, or a feature
        array.
    fixed_projections : None or list or ndarray or str
        If ``None`` (default), the projections are optimized. Otherwise the
        projections are held fixed at the supplied value.
    visualization_method : str
        Per-perspective objective, either ``'mds'`` or ``'tsne'``.
    smart_initialize : bool
        If ``True``, warm-start from a combined MDS embedding before the joint
        optimization (only when the projections are not fixed).
    verbose : int
        Verbosity level (``0`` is silent).
    **kwargs
        Forwarded to :class:`~mpe3d.mview.mpse.MPSE` and its ``gd`` method
        (e.g. ``weights``, ``initial_projections``, ``initial_embedding``,
        ``batch_size``, ``max_iter``, ``min_grad``, ``min_cost``, ``lr``).

    Returns
    -------
    MPSE
        The fitted MPSE object. Notable attributes: ``embedding`` (the recovered
        3D point cloud, also aliased as ``X``), ``projections`` (also ``Q``),
        ``cost``, and ``computation_history`` (also ``H``).
    """
    vis = mpse.MPSE(data, verbose=verbose, fixed_projections=fixed_projections,
                    visualization_method=visualization_method, **kwargs)
    if smart_initialize is True and fixed_projections is None:
        vis.smart_initialize()
    if visualization_method == 'mds' and 'batch_size' not in kwargs:
        kwargs['batch_size'] = 10
    vis.gd(**kwargs)

    vis.X = vis.embedding
    vis.Q = vis.projections
    vis.H = vis.computation_history

    return vis
