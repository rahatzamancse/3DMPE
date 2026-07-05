"""Run a 3DMPE parameter sweep over the LMNet/ShapeNet benchmark.

Reproduces the paper's sweeps (reconstruction quality vs. number of points,
number of viewpoints, or point visibility) by running the reconstruction
pipeline over the curated LMNet model subset and saving:

* ``results.csv``   - one row per run (all parameters and metrics),
* ``aggregated.csv``- per-category mean/std of the metric,
* one ``<metric>.png`` line plot per metric.

Examples
--------
Quick, download-free demo sweep (no ShapeNet required)::

    uv run python experiments/run_lmnet_benchmark.py \\
        --sweep n_perspectives --datasets demo:torus demo:sphere \\
        --n-points 200 --max-iter 80 --repeats 1

Full LMNet sweep over number of points (needs a local ShapeNetCore.v2)::

    uv run python experiments/run_lmnet_benchmark.py \\
        --sweep n_points --shapenet-dir /path/to/ShapeNetCore.v2 \\
        --per-category 1 --repeats 3 --variable-projection
"""

import argparse
import datetime as dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpe3d import benchmark, datasets
from mpe3d.visualization import plot_metric_curve

_METRICS = ["chamfer", "emd", "roa", "runtime_seconds"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", choices=list(benchmark.SWEEPS), default="n_perspectives",
                   help="Which parameter to sweep.")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Explicit dataset selectors. Defaults to the LMNet "
                        "subset (requires --shapenet-dir).")
    p.add_argument("--categories", nargs="+", default=None,
                   help="Restrict LMNet to these categories.")
    p.add_argument("--per-category", type=int, default=1,
                   help="Max LMNet models per category.")
    p.add_argument("--shapenet-dir", default=os.environ.get("SHAPENET_DIR"),
                   help="Path to ShapeNetCore.v2 (or set $SHAPENET_DIR).")

    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--n-points", type=int, default=512)
    p.add_argument("--n-perspectives", type=int, default=4)
    p.add_argument("--points-in-at-least", type=int, default=3)
    p.add_argument("--variable-projection", dest="variable_projection",
                   action="store_true", default=True)
    p.add_argument("--fixed-projection", dest="variable_projection",
                   action="store_false")
    p.add_argument("--rotation-axes", choices=["y", "xyz"], default="y")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--base-seed", type=int, default=0)

    p.add_argument("--output-dir", default="results")
    p.add_argument("--name", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.datasets is not None:
        selectors = args.datasets
    else:
        selectors = datasets.lmnet_datasets(categories=args.categories,
                                            per_category=args.per_category)

    # Defaults that are held fixed while one parameter is swept. Drop the swept
    # parameter so it does not clash with the sweep values.
    swept_param = next(iter(benchmark.SWEEPS[args.sweep]))
    defaults = {
        "n_points": args.n_points,
        "n_perspectives": args.n_perspectives,
        "points_in_at_least": args.points_in_at_least,
        "variable_projection": args.variable_projection,
        "rotation_axes": args.rotation_axes,
        "max_iter": args.max_iter,
    }
    defaults.pop(swept_param, None)

    print(f"Running '{args.sweep}' sweep over {len(selectors)} dataset(s), "
          f"{args.repeats} repeat(s)...")
    df = benchmark.run_sweep(
        selectors, benchmark.SWEEPS[args.sweep],
        datadir=args.shapenet_dir, repeats=args.repeats,
        base_seed=args.base_seed, **defaults,
    )

    name = args.name or f"lmnet_{args.sweep}"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    logx = swept_param == "n_points"
    xticks = sorted(df[swept_param].unique())
    agg_frames = []
    for metric in _METRICS:
        ax = plot_metric_curve(df, swept_param, metric, group="category",
                               logx=logx, xticks=xticks,
                               title=f"{metric} vs {swept_param}")
        ax.figure.savefig(os.path.join(run_dir, f"{metric}.png"),
                          bbox_inches="tight", dpi=150)
        plt.close(ax.figure)

        agg = benchmark.aggregate(df, swept_param, metric)
        agg_frames.append(agg.set_index(["category", swept_param]))

    import pandas as pd
    pd.concat(agg_frames, axis=1).reset_index().to_csv(
        os.path.join(run_dir, "aggregated.csv"), index=False)

    print(f"\nDone. Wrote results, aggregated CSV and {len(_METRICS)} plots to "
          f"{run_dir}")


if __name__ == "__main__":
    main()
