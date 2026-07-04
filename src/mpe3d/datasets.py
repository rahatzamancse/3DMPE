"""Load or generate the ground-truth 3D point clouds used as reconstruction targets.

A dataset is referenced by a colon-separated string:

* ``"demo"`` or ``"demo:<shape>"`` - a procedurally generated primitive
  (``torus`` by default; also ``sphere``, ``box``, ``cone``). Needs no download.
* ``"ModelNet10:<category>:<index>"`` - a mesh from the ModelNet10 dataset,
  downloaded and cached automatically on first use.
* ``"ShapeNet:<category>:<model_hash>"`` - a mesh from a local ShapeNetCore.v2
  tree (pass its path as ``datadir``).
* ``"Pix3D:<category>:<object_id>"`` - a mesh from a local Pix3D tree
  (pass its path as ``datadir``).
* ``"toy:<path-to-csv>"`` - a point cloud read from a headerless CSV where each
  column is a point (the historical "toy_points" format).
"""

import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import trimesh

MODELNET10_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"

shapenet_category_to_id = {
    'airplane': '02691156',
    'bench': '02828884',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'monitor': '03211117',
    'rifle': '04090263',
    'sofa': '04256520',
    'speaker': '03691459',
    'table': '04379243',
    'telephone': '04401088',
    'vessel': '04530566',
}
shapenet_id_to_category = {v: k for k, v in shapenet_category_to_id.items()}


def default_cache_dir():
    """Return (and create) the directory used to cache downloaded datasets."""
    cache = os.environ.get(
        "MPE3D_DATA_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "mpe3d"),
    )
    os.makedirs(cache, exist_ok=True)
    return cache


def as_mesh(scene_or_mesh):
    """Collapse a :class:`trimesh.Scene` into a single :class:`trimesh.Trimesh`."""
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        return trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                  for g in scene_or_mesh.geometry.values()))
    assert isinstance(scene_or_mesh, trimesh.Trimesh)
    return scene_or_mesh


def demo_mesh(shape='torus'):
    """Return a procedurally generated primitive mesh for offline demos."""
    if shape == 'torus':
        return trimesh.creation.torus(major_radius=1.0, minor_radius=0.35)
    if shape == 'sphere':
        return trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    if shape == 'box':
        return trimesh.creation.box(extents=(1.5, 1.0, 0.6))
    if shape == 'cone':
        return trimesh.creation.cone(radius=1.0, height=2.0)
    raise ValueError(f"unknown demo shape: {shape!r} "
                     "(expected one of: torus, sphere, box, cone)")


def _download_modelnet(cache_dir):
    """Download and extract ModelNet10 into ``cache_dir``; return the root path."""
    root = os.path.join(cache_dir, "ModelNet10")
    if os.path.isdir(root):
        return root
    print(f"Downloading ModelNet10 (~450 MB) to {cache_dir} ...")
    with urllib.request.urlopen(MODELNET10_URL) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(cache_dir)
    return root


def _sample_mesh(mesh, n_points):
    """Sample ``n_points`` surface points from a mesh (or scene)."""
    return as_mesh(mesh).sample(n_points)


def get_dataset_points(dataset, datadir=None, n_points=1024, normalize=False):
    """Return an ``(n_points, 3)`` point cloud sampled from ``dataset``.

    Parameters
    ----------
    dataset : str
        Dataset selector (see the module docstring for the accepted formats).
    datadir : str or None
        Root directory for datasets that live on disk (ShapeNet, Pix3D). Ignored
        for ``demo`` and ``ModelNet10``.
    n_points : int
        Number of surface points to sample.
    normalize : bool
        If ``True``, rescale each axis independently into ``[0, 1]``.
    """
    kind = dataset.split(':')[0]

    if kind == 'demo':
        parts = dataset.split(':')
        shape = parts[1] if len(parts) > 1 else 'torus'
        points = _sample_mesh(demo_mesh(shape), n_points)

    elif kind == 'toy':
        path = dataset.split(':', 1)[1]
        points = np.loadtxt(path, delimiter=',').T

    elif kind == 'ModelNet10':
        _, category, index = dataset.split(':')
        root = _download_modelnet(default_cache_dir())
        obj_path = os.path.join(root, category, 'train',
                                f'{category}_{index}.off')
        points = _sample_mesh(trimesh.load(obj_path), n_points)

    elif kind == 'ShapeNet':
        assert datadir is not None, "ShapeNet requires a `datadir` path"
        _, category, model_hash = dataset.split(':')
        category_id = shapenet_category_to_id[category]
        obj_path = os.path.join(datadir, category_id, model_hash,
                                'models', 'model_normalized.obj')
        points = _sample_mesh(trimesh.load(obj_path), n_points)

    elif kind == 'Pix3D':
        assert datadir is not None, "Pix3D requires a `datadir` path"
        _, category, obj_id = dataset.split(':')
        with open(os.path.join(datadir, 'pix3d.json')) as f:
            metadata = json.load(f)
        candidates = {f'img/{category}/{obj_id}.png',
                      f'img/{category}/{obj_id}.jpg'}
        model_rel = next((o['model'] for o in metadata
                          if o.get('img') in candidates), None)
        if model_rel is None:
            raise ValueError(f"Pix3D object not found for {dataset!r}")
        points = _sample_mesh(trimesh.load(os.path.join(datadir, model_rel)),
                              n_points)

    else:
        raise ValueError(f"unknown dataset selector: {dataset!r}")

    points = np.asarray(points, dtype=float)
    if normalize:
        span = points.max(axis=0) - points.min(axis=0)
        span[span == 0] = 1.0
        points = (points - points.min(axis=0)) / span

    return points
