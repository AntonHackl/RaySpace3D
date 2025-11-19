import os
import sys
import argparse
from math import hypot
import numpy as np
import matplotlib.pyplot as plt

def iter_chunks(total, size):
    remaining = int(total)
    size = int(size)
    while remaining > 0:
        take = size if remaining > size else remaining
        yield take
        remaining -= take

def parse_args():
    p = argparse.ArgumentParser(
        description="Stream-generate diagonal-correlated 3D points and write as WKT (POINT Z)."
    )
    p.add_argument("--n", type=int, required=True, help="Total number of points to generate (e.g., 100000000).")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("minx","miny","maxx","maxy"), required=True,
                   help="2D bounding box for diagonal (x,y).")
    p.add_argument("--zrange", type=float, nargs=2, metavar=("minz","maxz"), default=[0.0, 0.0],
                   help="Z-range [minz, maxz]. If equal, z is constant.")
    p.add_argument("--sigma", type=float, default=0.01,
                   help="Perpendicular noise as fraction of diagonal length (default 0.01).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--out", type=str, required=True, help="Output .txt path for WKT POINT Z.")
    p.add_argument("--chunk", type=int, default=2_000_000, help="Points per chunk (memory/speed trade-off).")
    p.add_argument("--precision", type=int, default=6, help="Decimal precision (default 6).")
    p.add_argument("--clip", action="store_true",
                   help="Clip noisy points back to bbox (x,y) if they fall outside.")
    p.add_argument("--plot", action="store_true", help="Create a 3D scatter plot of a sampled subset after generation.")
    p.add_argument("--sample-size", type=int, default=None, help="Number of points to sample for plotting (default: min(10000, 1% of n)).")
    p.add_argument("--plot-out", type=str, default=None, help="Path to save the sample plot PNG. If omitted, uses output path with _sample.png suffix.")
    return p.parse_args()

def main():
    args = parse_args()

    minx, miny, maxx, maxy = map(float, args.bbox)
    minz, maxz = map(float, args.zrange)

    if not (maxx > minx and maxy > miny):
        raise ValueError("Each max (x,y) must be greater than corresponding min.")
    if maxz < minz:
        raise ValueError("maxz must be >= minz.")

    # Diagonal direction and its perpendicular (unit)
    dx = maxx - minx
    dy = maxy - miny
    diag_len = hypot(dx, dy)
    if diag_len == 0.0:
        raise ValueError("Degenerate bbox: diagonal length is zero.")

    # Unit perpendicular to diagonal (dx,dy) is (-dy, dx)/diag_len
    vx = -dy / diag_len
    vy =  dx / diag_len

    sigma_abs = float(args.sigma) * diag_len

    rng = np.random.default_rng(args.seed)

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Large write buffer helps throughput
    float_fmt = f"%.{args.precision}f"
    wkt_fmt = f"POINT Z ({float_fmt} {float_fmt} {float_fmt})"

    total = int(args.n)
    written = 0

    # Determine sample size for plotting: default min(10000, 1% of total)
    if args.sample_size is None:
        default_k = max(1, int(total * 0.01))
        sample_k = min(10000, default_k)
    else:
        sample_k = max(1, int(args.sample_size))

    # Prepare sample indices (deterministic given seed)
    sample_indices = None
    if args.plot:
        if sample_k >= total:
            # sample everything
            sample_indices = np.arange(total)
        else:
            # choose without replacement
            sample_indices = np.sort(rng.choice(total, size=sample_k, replace=False))
        sample_buffer = np.empty((sample_k, 3), dtype=np.float64)
        sample_filled = 0
        next_sample_idx_ptr = 0  # pointer into sample_indices

    # Note: Using numpy vectorization + np.savetxt for speed (as you suggested).
    # We format full WKT rows directly via a single fmt string.
    with open(out_path, "w", buffering=8 * 1024 * 1024) as f:
        global_idx = 0
        for take in iter_chunks(total, args.chunk):
            # Parameterize along diagonal with t in [0,1)
            t = rng.random(take)  # shape (take,)

            base_x = minx + t * dx
            base_y = miny + t * dy

            # Gaussian noise applied perpendicular to the diagonal
            noise = rng.normal(loc=0.0, scale=sigma_abs, size=take)
            x = base_x + vx * noise
            y = base_y + vy * noise

            if args.clip:
                np.clip(x, minx, maxx, out=x)
                np.clip(y, miny, maxy, out=y)

            if maxz == minz:
                z = np.full(take, minz, dtype=np.float64)
            else:
                z = rng.uniform(minz, maxz, size=take)

            pts = np.column_stack((x, y, z))  # shape (take, 3)

            # Write as WKT POINT Z using numpy's fast writer
            np.savetxt(f, pts, fmt=wkt_fmt)

            # If plotting, collect sampled points that fall into this chunk
            if args.plot:
                if sample_k >= total:
                    # copy whole chunk (may be partial at end)
                    to_take = min(take, sample_k - sample_filled)
                    if to_take > 0:
                        sample_buffer[sample_filled:sample_filled+to_take, :] = pts[:to_take, :]
                        sample_filled += to_take
                else:
                    # check which sample indices lie in [global_idx, global_idx + take)
                    while next_sample_idx_ptr < sample_indices.size and sample_indices[next_sample_idx_ptr] < global_idx + take:
                        si = sample_indices[next_sample_idx_ptr] - global_idx  # local index in this chunk
                        sample_buffer[sample_filled, :] = pts[si, :]
                        sample_filled += 1
                        next_sample_idx_ptr += 1

            written += take
            print(f"Wrote {written:,}/{total:,} points", file=sys.stderr)

            global_idx += take

    print(f"Done. Output: {out_path}")

    # If requested, create a 3D scatter plot of the sampled points
    if args.plot:
        # If sample_buffer not fully filled (should not happen), truncate
        if sample_filled < sample_k:
            sample_data = sample_buffer[:sample_filled]
        else:
            sample_data = sample_buffer

        # Plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sample_data[:, 0], sample_data[:, 1], sample_data[:, 2], s=2, alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sampled {sample_data.shape[0]} points from {total:,} generated')

        plot_out = args.plot_out if args.plot_out else out_path.replace('.wkt', '_sample.png')
        fig.tight_layout()
        fig.savefig(plot_out, dpi=300)
        print(f"Saved sample plot to: {plot_out}")
if __name__ == "__main__":
    main()