#!/usr/bin/env python3
"""
Generates an adversarial dataset for grid-based selectivity estimation.
This version uses multiple clusters (clumps) to ensure that the estimation
remains inaccurate for multiple lower resolutions, only converging as
the grid becomes fine enough to isolate the clusters.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

def _gen_cluster_centers(n_clusters: int, span: float, rng: random.Random) -> List[Tuple[float, float, float]]:
    centers = []
    for _ in range(n_clusters):
        centers.append((
            rng.uniform(0, span),
            rng.uniform(0, span),
            rng.uniform(0, span)
        ))
    return centers

def _write_cubes(path: Path, centers: List[Tuple[float, float, float]], size: float, name_prefix: str) -> None:
    half = size / 2.0
    with open(path, "w", encoding="utf-8") as f:
        for i, (cx, cy, cz) in enumerate(centers):
            f.write(f"o {name_prefix}_{i}\n")
            # Vertices
            v = [
                (cx-half, cy-half, cz-half), (cx+half, cy-half, cz-half),
                (cx+half, cy+half, cz-half), (cx-half, cy+half, cz-half),
                (cx-half, cy-half, cz+half), (cx+half, cy-half, cz+half),
                (cx+half, cy+half, cz+half), (cx-half, cy+half, cz+half)
            ]
            for vx, vy, vz in v:
                f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            # Faces (standard cube)
            f.write(f"f {8*i+1} {8*i+2} {8*i+3} {8*i+4}\n")
            f.write(f"f {8*i+5} {8*i+6} {8*i+7} {8*i+8}\n")
            f.write(f"f {8*i+1} {8*i+2} {8*i+6} {8*i+5}\n")
            f.write(f"f {8*i+2} {8*i+3} {8*i+7} {8*i+6}\n")
            f.write(f"f {8*i+3} {8*i+4} {8*i+8} {8*i+7}\n")
            f.write(f"f {8*i+4} {8*i+1} {8*i+5} {8*i+8}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--cluster-radius", type=float, default=2.0)
    parser.add_argument("--span", type=float, default=500.0)
    parser.add_argument("--cube-size", type=float, default=0.5)
    parser.add_argument("--output-a", type=str, required=True)
    parser.add_argument("--output-b", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    
    # Generate centers for A and B that are spatially separated
    # but still within the same large world bounding box.
    centers_a = []
    clump_centers_a = _gen_cluster_centers(args.clusters, args.span, rng)
    for _ in range(args.count):
        cc = rng.choice(clump_centers_a)
        centers_a.append((
            cc[0] + rng.uniform(-args.cluster_radius, args.cluster_radius),
            cc[1] + rng.uniform(-args.cluster_radius, args.cluster_radius),
            cc[2] + rng.uniform(-args.cluster_radius, args.cluster_radius)
        ))

    # For B, we pick clusters that are offset from A's clusters to ensure 0 ground truth
    # until we want it non-zero. Let's make them disjoint.
    centers_b = []
    # Offset B clusters by 1.0 units (clump radius is 2.0, cube size is 0.5)
    # This ensures a high density of overlap within each cluster pair.
    # A cluster: [c-2, c+2], B cluster: [c-1, c+3]. Overlap region: [c-1, c+2]
    clump_centers_b = [(c[0] + 1.0, c[1] + 1.0, c[2] + 1.0) for c in clump_centers_a]
    for _ in range(args.count):
        cc = rng.choice(clump_centers_b)
        centers_b.append((
            cc[0] + rng.uniform(-args.cluster_radius, args.cluster_radius),
            cc[1] + rng.uniform(-args.cluster_radius, args.cluster_radius),
            cc[2] + rng.uniform(-args.cluster_radius, args.cluster_radius)
        ))

    # Add anchors to force consistent world AABB for all resolutions
    anchors = [(0, 0, 0), (args.span + 30, args.span + 30, args.span + 30)]
    centers_a.extend(anchors)
    centers_b.extend(anchors)

    _write_cubes(Path(args.output_a), centers_a, args.cube_size, "A")
    _write_cubes(Path(args.output_b), centers_b, args.cube_size, "B")
    print(f"Generated {len(centers_a)} and {len(centers_b)} cubes with {args.clusters} clusters.")

if __name__ == "__main__":
    main()
