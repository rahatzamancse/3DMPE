"""Visualization helpers for point clouds and benchmark results.

* :func:`plot_point_clouds` overlays one or more 3D point clouds (e.g. ground
  truth vs. reconstruction) in a single interactive Plotly figure.
* :func:`plot_cost_history` plots the MPSE stress over optimization.
* :func:`plot_points_per_perspective` shows how many points are visible in at
  least ``k`` views.
* :func:`plot_metric_curve` renders the paper-style seaborn line plots of a
  metric against a swept parameter, from a :mod:`mpe3d.benchmark` DataFrame.
"""

import numpy as np
import plotly.graph_objs as go

_DEFAULT_COLORS = ['green', 'red', 'blue', 'black', 'orange', 'purple']


def plot_point_clouds(clouds, names=None, colors=None, point_size=2,
                      opacity=0.6, show_axes=True, cubic=True, pad=0.4,
                      proj_type='perspective'):
    """Return a Plotly figure overlaying several 3D point clouds.

    Parameters
    ----------
    clouds : sequence of ndarray
        Each an ``(N, 3)`` point cloud to draw.
    names : sequence of str or None
        Legend label for each cloud.
    colors : sequence of str or None
        Marker color for each cloud. The special value ``'fancy'`` colors points
        by their Z coordinate.
    point_size, opacity : float
        Marker styling.
    show_axes : bool
        If ``True``, draw reference X/Y/Z axis lines and a cubic bounding box.
    cubic : bool
        If ``True``, force an equal aspect ratio spanning all clouds.
    pad : float
        Fractional padding applied to the cubic range.
    proj_type : str
        Plotly 3D projection: ``'perspective'`` or ``'orthographic'``.
    """
    clouds = [np.asarray(c) for c in clouds]
    if names is None:
        names = [f'cloud {i}' for i in range(len(clouds))]
    if colors is None:
        colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
                  for i in range(len(clouds))]

    traces = []
    for cloud, color, name in zip(clouds, colors, names):
        traces.append(go.Scatter3d(
            x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                opacity=opacity,
                color=cloud[:, 2] if color == 'fancy' else color,
                colorscale='hsv',
                symbol='circle',
            ),
            name=name,
        ))

    scene = dict(aspectmode='cube' if cubic else 'auto')
    if show_axes:
        all_points = np.concatenate(clouds, axis=0)
        lo, hi = all_points.min(), all_points.max()
        span = hi - lo
        rng = [lo - pad * abs(span), hi + pad * abs(span)]
        traces += [
            go.Scatter3d(x=rng, y=[0, 0], z=[0, 0], mode='lines',
                         name='axis', legendgroup='axis', showlegend=True),
            go.Scatter3d(x=[0, 0], y=rng, z=[0, 0], mode='lines',
                         name='axis', legendgroup='axis', showlegend=False),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=rng, mode='lines',
                         name='axis', legendgroup='axis', showlegend=False),
        ]
        scene.update(
            xaxis=dict(range=rng), yaxis=dict(range=rng), zaxis=dict(range=rng))

    scene['camera'] = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.8),
        projection=dict(type=proj_type),
    )

    layout = go.Layout(
        margin=dict(l=10, r=10, b=10, t=30),
        scene=scene,
    )
    return go.Figure(data=traces, layout=layout)


def plot_cost_history(costs, title='MPSE stress'):
    """Return a Plotly figure of the optimization cost history (log scale)."""
    fig = go.Figure(
        data=[go.Scatter(y=list(costs), mode='lines', name='stress')],
        layout=go.Layout(title=title, xaxis_title='iteration',
                         yaxis_title='stress', yaxis_type='log'),
    )
    return fig


def plot_points_per_perspective(points_per_perspective, title=None):
    """Return a Plotly bar chart of point visibility across perspectives.

    ``points_per_perspective[k-1]`` is the number of points visible in at least
    ``k`` perspectives (as produced by
    :func:`mpe3d.views.points_per_min_perspectives`).
    """
    counts = np.asarray(points_per_perspective, dtype=float)
    x = list(range(1, len(counts) + 1))
    fig = go.Figure(
        data=[go.Bar(x=x, y=counts)],
        layout=go.Layout(
            title=title or 'Points visible in at least k perspectives',
            xaxis_title='visible in >= k perspectives',
            yaxis_title='number of points',
        ),
    )
    return fig


def plot_metric_curve(df, x, y, group='category', ax=None, errorbar='sd',
                      logx=False, ylabel=None, xlabel=None, title=None,
                      xticks=None):
    """Line plot of a benchmark metric vs. a swept parameter (seaborn).

    Parameters
    ----------
    df : pandas.DataFrame
        A sweep result from :func:`mpe3d.benchmark.run_sweep` (one row per run).
    x : str
        Swept parameter column (e.g. ``'n_perspectives'``, ``'n_points'``,
        ``'points_in_at_least'``).
    y : str
        Metric column (e.g. ``'chamfer'``, ``'emd'``, ``'roa'``,
        ``'runtime_seconds'``).
    group : str
        Column used for the hue / one line per value (default ``'category'``).
    errorbar : str or None
        Passed to :func:`seaborn.lineplot` (e.g. ``'sd'``, ``'se'`` or ``None``)
        to draw a spread band across repeats/models.
    logx : bool
        Use a logarithmic x scale (useful for the points sweep).
    xticks : sequence or None
        Explicit x tick positions.

    Returns the Matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    sns.lineplot(data=df, x=x, y=y, hue=group, marker='o',
                 errorbar=errorbar, ax=ax)
    if logx:
        ax.set_xscale('log')
    if xticks is not None:
        ax.set_xticks(list(xticks))
        ax.set_xticklabels([str(t) for t in xticks])
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.legend(ncol=2, fontsize='small')
    return ax
