#!/usr/bin/env python3
"""Visualize paired plane-scatter cube datasets from OBJ files.

Produces:
- XY scatter (colored by z)
- Direction-projection profile (z vs projected XY coordinate)
- 3D scatter preview
"""

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_obj_centers(path: Path) -> List[Tuple[float, float, float]]:
    centers: List[Tuple[float, float, float]] = []
    current: List[Tuple[float, float, float]] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("o "):
                if current:
                    arr = np.array(current, dtype=float)
                    centers.append((float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())))
                    current = []
                continue

            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    current.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if current:
        arr = np.array(current, dtype=float)
        centers.append((float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())))

    return centers


def _project_xy(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    d = direction / np.linalg.norm(direction)
    return points[:, 0] * d[0] + points[:, 1] * d[1]


def _downsample(points: np.ndarray, max_points: int, rng: random.Random) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = rng.sample(range(len(points)), max_points)
    return points[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize plane-scatter cube pair datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-a", type=str, required=True, help="Path to dataset A OBJ")
    parser.add_argument("--dataset-b", type=str, required=True, help="Path to dataset B OBJ")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--direction-x", type=float, default=1.0, help="Projection direction x")
    parser.add_argument("--direction-y", type=float, default=1.0, help="Projection direction y")
    parser.add_argument("--max-points-xy", type=int, default=12000, help="Max points per dataset in XY panel")
    parser.add_argument("--max-points-3d", type=int, default=5000, help="Max points per dataset in 3D panel")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    direction = np.array([args.direction_x, args.direction_y], dtype=float)
    if np.linalg.norm(direction) == 0.0:
        raise ValueError("Projection direction must be non-zero")

    path_a = Path(args.dataset_a)
    path_b = Path(args.dataset_b)

    points_a = np.array(_parse_obj_centers(path_a), dtype=float)
    points_b = np.array(_parse_obj_centers(path_b), dtype=float)
    if points_a.size == 0 or points_b.size == 0:
        raise ValueError("One of the input datasets has no parseable cube centers")

    points_xy_a = _downsample(points_a, args.max_points_xy, rng)
    points_xy_b = _downsample(points_b, args.max_points_xy, rng)
    points_3d_a = _downsample(points_a, args.max_points_3d, rng)
    points_3d_b = _downsample(points_b, args.max_points_3d, rng)

    proj_a = _project_xy(points_xy_a, direction)
    proj_b = _project_xy(points_xy_b, direction)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 5.6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    c1 = ax1.scatter(points_xy_a[:, 0], points_xy_a[:, 1], c=points_xy_a[:, 2], s=4, alpha=0.60, cmap="viridis", label="A")
    ax1.scatter(points_xy_b[:, 0], points_xy_b[:, 1], c=points_xy_b[:, 2], s=4, alpha=0.60, cmap="plasma", label="B")
    ax1.set_title("XY placement (color=z)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="best", fontsize=8)
    cb = fig.colorbar(c1, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("z (dataset A color scale)")

    ax2.scatter(proj_a, points_xy_a[:, 2], s=4, alpha=0.55, label="A: ascending", color="#1f77b4")
    ax2.scatter(proj_b, points_xy_b[:, 2], s=4, alpha=0.55, label="B: descending", color="#d62728")
    ax2.set_title("Plane profile")
    ax2.set_xlabel("projected position along direction")
    ax2.set_ylabel("z")
    ax2.grid(True, alpha=0.35, linestyle="--")
    ax2.legend(loc="best", fontsize=8)

    ax3.scatter(points_3d_a[:, 0], points_3d_a[:, 1], points_3d_a[:, 2], s=2, alpha=0.45, label="A", color="#1f77b4")
    ax3.scatter(points_3d_b[:, 0], points_3d_b[:, 1], points_3d_b[:, 2], s=2, alpha=0.45, label="B", color="#d62728")
    ax3.set_title("3D center preview")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Plane-scatter cube validation | A={len(points_a):,} cubes, B={len(points_b):,} cubes",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(output), dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {output}")


if __name__ == "__main__":
    main()
