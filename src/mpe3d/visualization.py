"""Interactive 3D visualization of point clouds with Plotly.

:func:`plot_point_clouds` overlays one or more point clouds (e.g. ground truth
vs. reconstruction) in a single interactive figure, which can be shown in a
notebook or saved to a self-contained HTML file.
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
