"""Command-line runner for a single 3DMPE reconstruction experiment.

Loads a dataset, runs the reconstruction pipeline, and saves all results (params,
metrics, point clouds, and interactive HTML plots) into a timestamped folder
under ``--output-dir``. Unlike the original project, nothing is uploaded to a
remote tracker: every artifact is written locally.

Examples
--------
Run the zero-download demo::

    uv run python experiments/run_experiment.py --dataset demo:torus

Run on a local ShapeNet model with ray-traced visibility::

    uv run python experiments/run_experiment.py \\
        --dataset ShapeNet:airplane:<hash> --shapenet-dir /path/to/ShapeNetCore.v2 \\
        --projection raytracing --n-rays 400
"""

import argparse
import datetime as dt
import json
import os

import numpy as np

from mpe3d import datasets, reconstruct
from mpe3d.visualization import plot_cost_history, plot_point_clouds


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a single 3DMPE reconstruction experiment.")

    data = p.add_argument_group("dataset")
    data.add_argument("--dataset", default="demo:torus",
                      help="Dataset selector, e.g. 'demo:torus', "
                           "'ModelNet10:chair:0001', 'ShapeNet:airplane:<hash>'.")
    data.add_argument("--n-points", type=int, default=300,
                      help="Number of points to sample from the shape.")
    data.add_argument("--no-normalize", action="store_true",
                      help="Do not rescale the point cloud into the unit cube.")
    data.add_argument("--shapenet-dir",
                      default=os.environ.get("SHAPENET_DIR"),
                      help="Path to ShapeNetCore.v2 (or set $SHAPENET_DIR).")
    data.add_argument("--pix3d-dir", default=os.environ.get("PIX3D_DIR"),
                      help="Path to the Pix3D root (or set $PIX3D_DIR).")

    views = p.add_argument_group("views")
    views.add_argument("--n-perspectives", type=int, default=5)
    views.add_argument("--angle-start", type=float, default=0.0)
    views.add_argument("--angle-end", type=float, default=360.0)
    views.add_argument("--rotation-axes", choices=["y", "xyz"], default="y",
                       help="'y' for single-axis viewpoints, 'xyz' for the "
                            "paper's general 3-axis rotations (Eq. 9).")
    views.add_argument("--projection", choices=["atleast", "raytracing"],
                       default="atleast")
    views.add_argument("--points-in-at-least", type=int, default=4,
                       help="For --projection atleast: views each point is in.")
    views.add_argument("--n-rays", type=int, default=None,
                       help="For --projection raytracing: ray grid resolution.")

    noise = p.add_argument_group("noise")
    noise.add_argument("--noise-type", choices=["none", "distance", "matching"],
                       default="none")
    noise.add_argument("--noise-amount", type=float, default=0.0,
                       help="Fraction q of points affected by noise.")
    noise.add_argument("--noise-level", type=float, default=0.0,
                       help="Noise amplitude p, relative to the point-cloud "
                            "diameter (paper uses 0.05 and 0.1).")
    noise.add_argument("--noise-dist", choices=["gaussian", "uniform"],
                       default="gaussian")

    mpse = p.add_argument_group("mpse")
    mpse.add_argument("--fixed-projection", action="store_true",
                      help="Use the true rotations as fixed projections instead "
                           "of optimizing them.")
    mpse.add_argument("--initial-projections", default="cylinder")
    mpse.add_argument("--batch-size", type=int, default=None)
    mpse.add_argument("--max-iter", type=int, default=300,
                      help="Paper uses at most 300 minibatch SGD iterations.")
    mpse.add_argument("--min-grad", type=float, default=1e-4)
    mpse.add_argument("--min-cost", type=float, default=1e-4)
    mpse.add_argument("--no-smart-initialize", action="store_true")
    mpse.add_argument("--verbose", type=int, default=1)

    out = p.add_argument_group("output")
    out.add_argument("--output-dir", default="results")
    out.add_argument("--name", default=None,
                     help="Experiment name (defaults to the dataset selector).")
    out.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def datadir_for(dataset, args):
    """Return the on-disk root directory required for the given dataset, if any."""
    kind = dataset.split(":")[0]
    if kind == "ShapeNet":
        return args.shapenet_dir
    if kind == "Pix3D":
        return args.pix3d_dir
    return None


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    params = vars(args).copy()

    print(f"Loading dataset {args.dataset!r} ...")
    points = datasets.get_dataset_points(
        args.dataset,
        datadir=datadir_for(args.dataset, args),
        n_points=args.n_points,
        normalize=not args.no_normalize,
    )
    params["n_points_actual"] = len(points)

    print("Running reconstruction ...")
    result = reconstruct(
        points,
        n_perspectives=args.n_perspectives,
        angle_range=(args.angle_start, args.angle_end),
        rotation_axes=args.rotation_axes,
        projection=args.projection,
        points_in_at_least=args.points_in_at_least,
        n_rays=args.n_rays,
        variable_projection=not args.fixed_projection,
        initial_projections=args.initial_projections,
        noise_type=args.noise_type,
        noise_amount=args.noise_amount,
        noise_level=args.noise_level,
        noise_dist=args.noise_dist,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        min_grad=args.min_grad,
        min_cost=args.min_cost,
        smart_initialize=not args.no_smart_initialize,
        verbose=args.verbose,
        rng=rng,
    )

    name = args.name or args.dataset.replace(":", "_")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    metrics = {
        **result.summary(),
        "alignment_error": result.alignment_error,
        "points_per_perspective": result.points_per_perspective,
    }

    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2, default=str)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    np.save(os.path.join(run_dir, "ground_truth.npy"), result.ground_truth)
    np.save(os.path.join(run_dir, "embedding.npy"), result.embedding)
    np.save(os.path.join(run_dir, "aligned_embedding.npy"),
            result.aligned_embedding)

    plot_point_clouds(
        [result.ground_truth, result.aligned_embedding],
        names=["ground truth", "reconstruction"],
        colors=["green", "red"],
    ).write_html(os.path.join(run_dir, "reconstruction.html"))

    plot_cost_history(result.cost_history).write_html(
        os.path.join(run_dir, "cost_history.html"))

    print(f"\nDone. Results written to {run_dir}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
