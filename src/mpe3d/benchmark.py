"""Parameter sweeps over the LMNet/ShapeNet benchmark, as tidy DataFrames.

The 3DMPE paper studies how reconstruction quality varies with the number of
input points, the number of viewpoints, and how many viewpoints each point is
visible in (Figures 3-7, 9-10). This module reproduces those sweeps directly
from the reconstruction pipeline: it runs :func:`mpe3d.reconstruct` over a grid
of parameters and a set of datasets, and returns a long-form
:class:`pandas.DataFrame` (one row per run) that is easy to aggregate and plot
with :mod:`mpe3d.visualization`.

Unlike the original paper-plot notebooks, results here are computed, not
hand-adjusted: every number comes straight from the pipeline.

Typical use::

    from mpe3d import benchmark, datasets

    df = benchmark.run_sweep(
        datasets.lmnet_datasets(per_category=1),
        benchmark.SWEEPS["n_perspectives"],
        datadir="/path/to/ShapeNetCore.v2",
        repeats=2,
    )

The demo datasets need no download, which is handy for quick tests::

    df = benchmark.run_sweep(["demo:torus", "demo:sphere"],
                             benchmark.SWEEPS["n_points"])
"""

import time

import numpy as np
import pandas as pd

from . import datasets as _datasets
from .pipeline import reconstruct

# Ready-made sweeps mirroring the paper's experiments. Each entry maps a swept
# parameter name to the list of values to try; everything else uses the
# per-run defaults passed to :func:`run_sweep`.
SWEEPS = {
    # Figure 4 / 14: reconstruction quality vs. number of points.
    "n_points": {"n_points": [128, 256, 512, 1024, 2048]},
    # Figure 6 / 12: quality and runtime vs. number of viewpoints.
    "n_perspectives": {"n_perspectives": [2, 3, 4, 5, 6, 7, 8]},
    # Figure 7 / 9: quality vs. how many viewpoints each point is visible in.
    "points_in_at_least": {"points_in_at_least": [1, 2, 3, 4, 5, 6, 7, 8]},
}

# Columns produced for every run.
_RESULT_COLUMNS = [
    "dataset", "category", "model",
    "n_points", "n_perspectives", "points_in_at_least", "projection",
    "variable_projection", "rotation_axes", "repeat", "seed",
    "chamfer", "emd", "roa",
    "baseline_chamfer", "baseline_emd", "baseline_roa",
    "final_cost", "runtime_seconds",
]


def _split_dataset(dataset):
    """Return ``(category, model)`` for a dataset selector (best effort)."""
    parts = dataset.split(":")
    if len(parts) >= 3:
        return parts[1], parts[2]
    if len(parts) == 2:
        return parts[1], ""
    return parts[0], ""


def run_single(dataset, datadir=None, n_points=512, n_perspectives=4,
               points_in_at_least=3, projection="atleast",
               variable_projection=True, rotation_axes="y", max_iter=300,
               normalize=True, repeat=0, seed=None, **reconstruct_kwargs):
    """Run one reconstruction and return a flat dict of parameters + metrics.

    Extra keyword arguments are forwarded to :func:`mpe3d.reconstruct`.
    """
    rng = np.random.default_rng(seed)
    points = _datasets.get_dataset_points(
        dataset, datadir=datadir, n_points=n_points, normalize=normalize)

    # When sweeping the number of viewpoints, a fixed target visibility can
    # exceed the available views; cap it so every point stays visible in all
    # views (as the paper does for small view counts).
    effective_visibility = min(points_in_at_least, n_perspectives)

    start = time.perf_counter()
    result = reconstruct(
        points,
        n_perspectives=n_perspectives,
        projection=projection,
        points_in_at_least=effective_visibility,
        variable_projection=variable_projection,
        rotation_axes=rotation_axes,
        max_iter=max_iter,
        rng=rng,
        **reconstruct_kwargs,
    )
    runtime = time.perf_counter() - start

    category, model = _split_dataset(dataset)
    return {
        "dataset": dataset,
        "category": category,
        "model": model,
        "n_points": len(points),
        "n_perspectives": n_perspectives,
        "points_in_at_least": effective_visibility,
        "projection": projection,
        "variable_projection": variable_projection,
        "rotation_axes": rotation_axes,
        "repeat": repeat,
        "seed": seed,
        "chamfer": result.chamfer,
        "emd": result.emd,
        "roa": result.roa,
        "baseline_chamfer": result.baseline["chamfer"],
        "baseline_emd": result.baseline["emd"],
        "baseline_roa": result.baseline["roa"],
        "final_cost": result.cost,
        "runtime_seconds": runtime,
    }


def run_sweep(datasets_list, sweep, datadir=None, repeats=1, base_seed=0,
              progress=True, **defaults):
    """Run a parameter sweep across datasets and return a tidy DataFrame.

    Parameters
    ----------
    datasets_list : sequence of str
        Dataset selectors (e.g. from :func:`mpe3d.datasets.lmnet_datasets`).
    sweep : dict
        Maps one parameter name to a list of values (e.g. an entry of
        :data:`SWEEPS`). Every combination of dataset x value x repeat is run.
    datadir : str or None
        Root directory for on-disk datasets (ShapeNet/Pix3D).
    repeats : int
        Number of repeats per configuration (with distinct seeds), averaged
        later during plotting/aggregation.
    base_seed : int
        Seeds are derived deterministically from this base.
    **defaults
        Default parameters forwarded to :func:`run_single` (e.g.
        ``variable_projection=False``, ``n_points=512``).

    Returns
    -------
    pandas.DataFrame
        One row per run, with the columns in :data:`_RESULT_COLUMNS`.
    """
    if len(sweep) != 1:
        raise ValueError("sweep must map exactly one parameter to its values")
    (param, values), = sweep.items()

    configs = [(ds, val, rep)
               for ds in datasets_list
               for val in values
               for rep in range(repeats)]

    iterator = configs
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(configs, desc=f"sweep over {param}")
        except ImportError:
            pass

    rows = []
    for i, (dataset, value, repeat) in enumerate(iterator):
        kwargs = dict(defaults)
        kwargs[param] = value
        kwargs["repeat"] = repeat
        kwargs["seed"] = base_seed + i
        rows.append(run_single(dataset, datadir=datadir, **kwargs))

    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)


def aggregate(df, x, y, group="category", agg=("mean", "std", "min", "max")):
    """Aggregate a sweep DataFrame for plotting.

    Groups by ``group`` and ``x`` and reduces column ``y`` with the requested
    aggregation functions, returning a flat DataFrame with columns
    ``[group, x, f"{y}_mean", ...]``.
    """
    grouped = df.groupby([group, x])[y].agg(list(agg))
    grouped.columns = [f"{y}_{name}" for name in agg]
    return grouped.reset_index()
