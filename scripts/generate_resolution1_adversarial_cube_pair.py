#!/usr/bin/env python3
"""Generate a cube-pair dataset that is adversarial for grid resolution=1 selectivity.

The construction creates:
- Dense cluster A in one corner region
- Dense cluster B in a far-away corner region
- A few far anchors per dataset to inflate each dataset's world AABB

At resolution 1, selectivity estimation loses spatial structure and can overestimate.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from generate_test_cubes import generate_cube_faces, generate_cube_vertices


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _rand_in_ball(rng: random.Random, radius: float) -> Tuple[float, float, float]:
    # Rejection sampling in unit cube, then scale by radius.
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        if (x * x + y * y + z * z) <= 1.0:
            return x * radius, y * radius, z * radius


def _gen_cluster_centers(
    count: int,
    center: Tuple[float, float, float],
    radius: float,
    world_size: float,
    rng: random.Random,
) -> List[Tuple[float, float, float]]:
    cx, cy, cz = center
    out: List[Tuple[float, float, float]] = []
    for _ in range(count):
        dx, dy, dz = _rand_in_ball(rng, radius)
        out.append(
            (
                _clamp(cx + dx, 0.0, world_size),
                _clamp(cy + dy, 0.0, world_size),
                _clamp(cz + dz, 0.0, world_size),
            )
        )
    return out


def _anchor_points_for_a(world_size: float) -> List[Tuple[float, float, float]]:
    return [
        (0.0, 0.0, 0.0),
        (0.0, world_size, world_size),
    ]


def _anchor_points_for_b(world_size: float) -> List[Tuple[float, float, float]]:
    return [
        (world_size, world_size, world_size),
        (world_size, 0.0, 0.0),
    ]


def _write_obj(path: Path, centers: Sequence[Tuple[float, float, float]], cube_size: float) -> None:
    faces = generate_cube_faces()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Resolution-1 adversarial cube dataset\n")
        handle.write(f"# Number of cubes: {len(centers)}\n")
        handle.write(f"# Cube size: {cube_size}\n\n")

        vertex_offset = 0
        for cube_id, (x, y, z) in enumerate(centers):
            handle.write(f"o cube_{cube_id}\n")
            for vx, vy, vz in generate_cube_vertices(x, y, z, cube_size):
                handle.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for a, b, c in faces:
                handle.write(
                    f"f {a + vertex_offset + 1} {b + vertex_offset + 1} {c + vertex_offset + 1}\n"
                )
            handle.write("\n")
            vertex_offset += 8


def _summary(centers: Sequence[Tuple[float, float, float]]) -> dict:
    xs = [p[0] for p in centers]
    ys = [p[1] for p in centers]
    zs = [p[2] for p in centers]
    n = float(len(centers))
    mean_z = (sum(zs) / n) if n else None
    var_z = (sum((z - mean_z) ** 2 for z in zs) / n) if n and mean_z is not None else None
    return {
        "count": int(n),
        "x_range": [min(xs), max(xs)] if xs else None,
        "y_range": [min(ys), max(ys)] if ys else None,
        "z_range": [min(zs), max(zs)] if zs else None,
        "z_mean": mean_z,
        "z_std": math.sqrt(var_z) if var_z is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired cube datasets adversarial for grid-resolution=1 selectivity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-cubes-a", type=int, default=50_000, help="Cube count for dataset A")
    parser.add_argument("--num-cubes-b", type=int, default=50_000, help="Cube count for dataset B")
    parser.add_argument("--cube-size", type=float, default=5.0, help="Fixed cube edge length")
    parser.add_argument("--world-size", type=float, default=2000.0, help="Global world span for anchor placement")
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=30.0,
        help="Radius of each dense cluster (smaller => more concentrated)",
    )
    parser.add_argument(
        "--cluster-margin",
        type=float,
        default=220.0,
        help="Distance of cluster centers from world boundaries",
    )
    parser.add_argument(
        "--anchors-per-dataset",
        type=int,
        default=2,
        help="Number of far anchors per dataset to inflate world AABB (max 2 in this script)",
    )
    parser.add_argument(
        "--output-a",
        type=str,
        default="./benchmarks/mesh_overlap/data/raw/cubes_r1_adversarial_50k_a.obj",
        help="Output OBJ for dataset A",
    )
    parser.add_argument(
        "--output-b",
        type=str,
        default="./benchmarks/mesh_overlap/data/raw/cubes_r1_adversarial_50k_b.obj",
        help="Output OBJ for dataset B",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="./benchmarks/mesh_overlap/data/raw/cubes_r1_adversarial_50k_metadata.json",
        help="Metadata JSON output",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_cubes_a <= 0 or args.num_cubes_b <= 0:
        raise ValueError("num-cubes values must be positive")
    if args.cube_size <= 0.0:
        raise ValueError("cube-size must be positive")
    if args.world_size <= 0.0:
        raise ValueError("world-size must be positive")
    if args.cluster_radius <= 0.0:
        raise ValueError("cluster-radius must be positive")
    if args.cluster_margin <= args.cluster_radius:
        raise ValueError("cluster-margin must be greater than cluster-radius")

    anchors_per_dataset = max(0, min(args.anchors_per_dataset, 2))
    if args.num_cubes_a <= anchors_per_dataset or args.num_cubes_b <= anchors_per_dataset:
        raise ValueError("num-cubes must be larger than anchors-per-dataset")

    rng_a = random.Random(args.seed)
    rng_b = random.Random(args.seed + 1)

    c = args.cluster_margin
    w = args.world_size
    center_a = (c, c, c)
    center_b = (w - c, w - c, w - c)

    base_count_a = args.num_cubes_a - anchors_per_dataset
    base_count_b = args.num_cubes_b - anchors_per_dataset

    centers_a = _gen_cluster_centers(base_count_a, center_a, args.cluster_radius, args.world_size, rng_a)
    centers_b = _gen_cluster_centers(base_count_b, center_b, args.cluster_radius, args.world_size, rng_b)

    centers_a.extend(_anchor_points_for_a(args.world_size)[:anchors_per_dataset])
    centers_b.extend(_anchor_points_for_b(args.world_size)[:anchors_per_dataset])

    out_a = Path(args.output_a)
    out_b = Path(args.output_b)
    _write_obj(out_a, centers_a, args.cube_size)
    _write_obj(out_b, centers_b, args.cube_size)

    metadata = {
        "generator": "generate_resolution1_adversarial_cube_pair.py",
        "seed": args.seed,
        "cube_size": args.cube_size,
        "world_size": args.world_size,
        "cluster_radius": args.cluster_radius,
        "cluster_margin": args.cluster_margin,
        "anchors_per_dataset": anchors_per_dataset,
        "cluster_center_a": center_a,
        "cluster_center_b": center_b,
        "intent": "Adversarial for resolution=1 selectivity (disjoint dense clusters + inflated world bounds)",
        "datasets": {
            "a": {"path": str(out_a), "stats": _summary(centers_a)},
            "b": {"path": str(out_b), "stats": _summary(centers_b)},
        },
    }

    metadata_path = Path(args.metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Generated resolution-1 adversarial datasets:")
    print(f"  A: {out_a}")
    print(f"  B: {out_b}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()