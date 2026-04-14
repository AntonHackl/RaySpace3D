#!/usr/bin/env python3
"""Generate two cube datasets on opposing slanted planes with Gaussian z scatter.

Dataset A ramps from low z to high z along a 2D direction on the XY plane.
Dataset B ramps from high z to low z along the same direction.
At the midpoint of the sweep they are aligned around the same z baseline.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from generate_test_cubes import generate_cube_faces, generate_cube_vertices


def _unit2(x: float, y: float) -> Tuple[float, float]:
    norm = math.hypot(x, y)
    if norm <= 0.0:
        raise ValueError("Plane direction vector must be non-zero")
    return x / norm, y / norm


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _direction_projection_progress(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    half_extent: float,
    dir_x: float,
    dir_y: float,
) -> float:
    rel_x = x - center_x
    rel_y = y - center_y
    signed_dist = rel_x * dir_x + rel_y * dir_y
    # Map direction-aligned distance from [-half_extent, +half_extent] to [0, 1].
    return _clamp((signed_dist + half_extent) / (2.0 * half_extent), 0.0, 1.0)


def _generate_centers(
    count: int,
    cube_size: float,
    xy_extent: float,
    z_base: float,
    z_span: float,
    z_noise_sigma: float,
    dir_x: float,
    dir_y: float,
    ascending: bool,
    rng: random.Random,
) -> List[Tuple[float, float, float]]:
    centers: List[Tuple[float, float, float]] = []
    half_xy = xy_extent / 2.0

    for _ in range(count):
        x = rng.uniform(0.0, xy_extent)
        y = rng.uniform(0.0, xy_extent)
        progress = _direction_projection_progress(
            x,
            y,
            center_x=half_xy,
            center_y=half_xy,
            half_extent=half_xy,
            dir_x=dir_x,
            dir_y=dir_y,
        )

        signed = (progress - 0.5) * z_span
        if not ascending:
            signed = -signed

        z = z_base + signed + rng.gauss(0.0, z_noise_sigma)
        centers.append((x, y, z))

    return centers


def _write_obj(path: Path, centers: Sequence[Tuple[float, float, float]], cube_size: float) -> None:
    faces = generate_cube_faces()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Plane-scatter cube dataset\n")
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
    mean_z = sum(zs) / n if n else None
    var_z = sum((z - mean_z) ** 2 for z in zs) / n if n and mean_z is not None else None
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
        description="Generate paired cube datasets distributed on opposite slanted planes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-cubes-a", type=int, default=50_000, help="Cube count for dataset A")
    parser.add_argument("--num-cubes-b", type=int, default=50_000, help="Cube count for dataset B")
    parser.add_argument("--cube-size", type=float, default=5.0, help="Fixed cube edge length")
    parser.add_argument("--xy-extent", type=float, default=120.0, help="Extent for x/y center placement")
    parser.add_argument(
        "--z-base",
        type=float,
        default=60.0,
        help="Shared z baseline where both planes meet near the center",
    )
    parser.add_argument(
        "--z-span",
        type=float,
        default=80.0,
        help="Total z span from low to high ends of each plane",
    )
    parser.add_argument("--z-noise-sigma", type=float, default=2.0, help="Gaussian z scatter sigma")
    parser.add_argument("--direction-x", type=float, default=1.0, help="Plane sweep direction x")
    parser.add_argument("--direction-y", type=float, default=1.0, help="Plane sweep direction y")
    parser.add_argument(
        "--output-a",
        type=str,
        default="./benchmarks/mesh_overlap/data/raw/cubes_plane_50k_a.obj",
        help="Output OBJ for dataset A (ascending plane)",
    )
    parser.add_argument(
        "--output-b",
        type=str,
        default="./benchmarks/mesh_overlap/data/raw/cubes_plane_50k_b.obj",
        help="Output OBJ for dataset B (descending plane)",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="./benchmarks/mesh_overlap/data/raw/cubes_plane_50k_metadata.json",
        help="Optional metadata JSON output",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_cubes_a <= 0 or args.num_cubes_b <= 0:
        raise ValueError("num-cubes values must be positive")
    if args.cube_size <= 0.0:
        raise ValueError("cube-size must be positive")
    if args.xy_extent <= 0.0:
        raise ValueError("xy-extent must be positive")
    if args.z_span < 0.0:
        raise ValueError("z-span must be non-negative")
    if args.z_noise_sigma < 0.0:
        raise ValueError("z-noise-sigma must be non-negative")

    dir_x, dir_y = _unit2(args.direction_x, args.direction_y)

    rng_a = random.Random(args.seed)
    rng_b = random.Random(args.seed + 1)

    centers_a = _generate_centers(
        count=args.num_cubes_a,
        cube_size=args.cube_size,
        xy_extent=args.xy_extent,
        z_base=args.z_base,
        z_span=args.z_span,
        z_noise_sigma=args.z_noise_sigma,
        dir_x=dir_x,
        dir_y=dir_y,
        ascending=True,
        rng=rng_a,
    )
    centers_b = _generate_centers(
        count=args.num_cubes_b,
        cube_size=args.cube_size,
        xy_extent=args.xy_extent,
        z_base=args.z_base,
        z_span=args.z_span,
        z_noise_sigma=args.z_noise_sigma,
        dir_x=dir_x,
        dir_y=dir_y,
        ascending=False,
        rng=rng_b,
    )

    output_a = Path(args.output_a)
    output_b = Path(args.output_b)
    _write_obj(output_a, centers_a, args.cube_size)
    _write_obj(output_b, centers_b, args.cube_size)

    summary = {
        "generator": "generate_plane_scatter_cube_pair.py",
        "seed": args.seed,
        "direction_unit": [dir_x, dir_y],
        "cube_size": args.cube_size,
        "xy_extent": args.xy_extent,
        "z_base": args.z_base,
        "z_span": args.z_span,
        "z_noise_sigma": args.z_noise_sigma,
        "datasets": {
            "a": {
                "mode": "ascending",
                "path": str(output_a),
                "stats": _summary(centers_a),
            },
            "b": {
                "mode": "descending",
                "path": str(output_b),
                "stats": _summary(centers_b),
            },
        },
    }

    metadata_path = Path(args.metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Generated plane-scatter datasets:")
    print(f"  A: {output_a}")
    print(f"  B: {output_b}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
