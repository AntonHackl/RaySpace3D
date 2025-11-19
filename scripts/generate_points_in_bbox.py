#!/usr/bin/env python3
"""Generate uniformly-distributed 3D points inside a bounding box using NumPy.

Writes points in WKT format (one POINT per line) to the output file. Generation
is done in chunks to avoid holding all points in memory at once.

Example:
  python scripts/generate_points_in_bbox.py --min 0 0 0 --max 50 50 50 --num_points 100000000 --output generated_files/points_100m.wkt
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np


def parse_xyz_triplet(values: list[str], name: str) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} requires three floats: x y z")
    try:
        x, y, z = map(float, values)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} values must be floats")
    return x, y, z


def iter_chunks(num_points: int, chunk_size: int):
    written = 0
    while written < num_points:
        remaining = num_points - written
        take = chunk_size if remaining >= chunk_size else remaining
        yield take
        written += take


def generate_and_write(
    out_path: str,
    num_points: int,
    min_xyz: Tuple[float, float, float],
    max_xyz: Tuple[float, float, float],
    chunk_size: int = 10_000_000,
    seed: int | None = None,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rng = np.random.default_rng(seed)
    min_arr = np.asarray(min_xyz, dtype=np.float64)
    max_arr = np.asarray(max_xyz, dtype=np.float64)

    if np.any(max_arr <= min_arr):
        raise ValueError("Each max coordinate must be greater than the corresponding min coordinate")

    total_written = 0
    with open(out_path, "w", buffering=1 * 1024 * 1024) as f:
        for take in iter_chunks(num_points, chunk_size):
            pts = rng.uniform(0.0, 1.0, size=(take, 3))
            pts = min_arr + pts * (max_arr - min_arr)

            fmt = "POINT (%.6f %.6f %.6f)"
            np.savetxt(f, pts, fmt=fmt)

            total_written += take
            print(f"Wrote {total_written}/{num_points} points", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Generate uniform points inside a 3D bounding box (WKT output)")
    p.add_argument("--min", nargs=3, metavar=("XMIN", "YMIN", "ZMIN"), help="Minimum corner of bbox (x y z)")
    p.add_argument("--max", nargs=3, metavar=("XMAX", "YMAX", "ZMAX"), help="Maximum corner of bbox (x y z)")
    p.add_argument("--num_points", type=int, default=100_000_000, help="Total number of points to generate (default: 100000000)")
    p.add_argument("--chunk-size", type=int, default=1_000_000, help="Number of points generated and written per chunk (default: 1_000_000)")
    p.add_argument("--output", type=str, default="generated_files/points.wkt", help="Output file path (WKT lines)")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")

    args = p.parse_args(argv)

    if args.min is None or args.max is None:
        p.error("--min and --max are required and must be three floats each")

    min_xyz = parse_xyz_triplet(args.min, "--min")
    max_xyz = parse_xyz_triplet(args.max, "--max")

    if args.chunk_size <= 0:
        p.error("--chunk-size must be > 0")
    if args.num_points < 0:
        p.error("--num_points must be >= 0")

    try:
        generate_and_write(
            out_path=args.output,
            num_points=args.num_points,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print(f"Finished: wrote {args.num_points} points to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
