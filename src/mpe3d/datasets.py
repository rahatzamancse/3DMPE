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

# Curated ShapeNet model subset used by the 3D-LMNet benchmark (Mandikal et al.,
# BMVC 2018) and adopted by the 3DMPE paper for its ShapeNet comparisons. Keyed
# by category, each value is a list of ShapeNetCore.v2 model hashes.
LMNET_MODELS = {
    'airplane': [
        '103c9e43cdf6501c62b600da24e0965',
        '105f7f51e4140ee4b6b87e72ead132ed',
        '10e4331c34d610dacc14f1e6f4f4f49b',
        'd405b9e5f942fed5efe5d5ae25ee424e',
        '157bb84c08754307dff9b4d1071b12d7',
        '8cf06a71987992cf90a51833252023c7',
    ],
    'bench': [
        '42ffe8d78c2e8da9d40c07d3c15cc681',
        'cad0a0e60708ab662ab293e158725cf0',
        'cca18c7f8636606f51f77a6d7299806',
        '89e2eaeb437cd42f85e40cb3507a0145',
        '702870d836ea3bf32056b4bd5d870b47',
        'fc0486ec53630bdbd2b12aa6a0f050b3',
    ],
    'car': [
        '44f30f4c65c3142a16abce8cb03e7794',
        'd9034b15c7f295551a46c391b387480b',
        '35de0d0cc71179dc1a98dff5b6c5dec6',
        'd6f8cfdb1659142814fccfc8a25361e',
        'd79f66a4566ff981424db5a60837de26',
    ],
    'chair': [
        'bf91d0169eae3bfdd810b14a81e12eca',
        '6a3d2feff3783804387379bbd607d69e',
        'cd6a8020b69455dbb161f36d4e309050',
        'cd9702520ad57689bbc7a6acbd8f058b',
    ],
    'lamp': [
        '102273fdf8d1b90041fbc1e2da054acb',
        'fa0a32c4326a42fef51f77a6d7299806',
        'e6d62a37e187bde599284d844aba7576',
    ],
    'rifle': [
        '10cc9af8877d795c93c9577cd4b35faa',
        '81ba8d540499dd04834bde3f2f2e7c0c',
        '823b97177d57e5dd8e0bef156e045efe',
        'f55544d331eb019a1aca20a2bd5ca645',
    ],
    'table': [
        '105b9a03ddfaf5c5e7828dbf1991f6a4',
        'c3884d2d31ac0ac9593ebeeedbff73b',
        '16961ddf69b6e91ea8ff4f6e9563bff6',
        '86e6ef5ae3420e95963080fd7249126d',
    ],
    'sofa': [
        '79bea3f7c72e0aae490ad276cd2af3a4',
        'cff485b2c98410135dda488a4bbb1e1',
        'd5a2b159a5fbbc4c510e2ce46c1af6e',
        'd8c748ced5e5f2cc7e3820d17093b7c2',
    ],
}


def lmnet_datasets(categories=None, per_category=None):
    """Return a list of ``"ShapeNet:<category>:<hash>"`` selectors for LMNet models.

    Parameters
    ----------
    categories : sequence of str or None
        Restrict to these categories (default: all eight LMNet categories).
    per_category : int or None
        Keep at most this many models per category (default: all).

    These selectors require a local ShapeNetCore.v2 tree passed as ``datadir``
    to :func:`get_dataset_points` / :func:`get_dataset_mesh`.
    """
    if categories is None:
        categories = list(LMNET_MODELS)
    selectors = []
    for category in categories:
        hashes = LMNET_MODELS[category]
        if per_category is not None:
            hashes = hashes[:per_category]
        selectors.extend(f'ShapeNet:{category}:{h}' for h in hashes)
    return selectors


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


def get_dataset_mesh(dataset, datadir=None):
    """Return the :class:`trimesh.Trimesh` behind a mesh-based ``dataset``.

    Useful for interactive visualization (``mesh.show()``) before sampling a
    point cloud. Not supported for the ``toy`` (raw CSV) selector.
    """
    kind = dataset.split(':')[0]

    if kind == 'demo':
        parts = dataset.split(':')
        shape = parts[1] if len(parts) > 1 else 'torus'
        return as_mesh(demo_mesh(shape))

    if kind == 'ModelNet10':
        _, category, index = dataset.split(':')
        root = _download_modelnet(default_cache_dir())
        obj_path = os.path.join(root, category, 'train',
                                f'{category}_{index}.off')
        return as_mesh(trimesh.load(obj_path))

    if kind == 'ShapeNet':
        assert datadir is not None, "ShapeNet requires a `datadir` path"
        _, category, model_hash = dataset.split(':')
        category_id = shapenet_category_to_id[category]
        obj_path = os.path.join(datadir, category_id, model_hash,
                                'models', 'model_normalized.obj')
        return as_mesh(trimesh.load(obj_path))

    if kind == 'Pix3D':
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
        return as_mesh(trimesh.load(os.path.join(datadir, model_rel)))

    raise ValueError(f"{dataset!r} has no associated mesh (kind={kind!r})")


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

    if kind == 'toy':
        path = dataset.split(':', 1)[1]
        points = np.loadtxt(path, delimiter=',').T
    else:
        points = get_dataset_mesh(dataset, datadir=datadir).sample(n_points)

    points = np.asarray(points, dtype=float)
    if normalize:
        span = points.max(axis=0) - points.min(axis=0)
        span[span == 0] = 1.0
        points = (points - points.min(axis=0)) / span

    return points
