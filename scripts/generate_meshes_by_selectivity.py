#!/usr/bin/env python3
"""
Fast Parallel OBJ Generator for RaySpace3D benchmarks.

Key optimisations vs. the old StringIO/np.savetxt-per-object version
----------------------------------------------------------------------
* Workers build **bytes** objects rather than Python strings.
* Each batch is formatted with exactly TWO np.savetxt calls: one for all
  vertices stacked into a (n*num_v, 3) array, one for all faces stacked
  into a (n*num_f, 3) array.  This avoids the per-object Python loop that
  was the bottleneck in the previous version.
* np.savetxt is called with a str fmt (required by NumPy ≥1.24) into a
  StringIO, then the whole block is encoded to bytes in one shot.
* The main process opens the file in **binary mode** with a 32 MiB write
  buffer, so it writes the returned bytes directly without further encoding.
* Batch size is tuned upward (default 500) to keep worker IPC overhead low
  for large templates.
"""

import argparse
import io
import random
from pathlib import Path
from typing import Optional

import multiprocessing as mp
import numpy as np


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def compute_universe_for_selectivity(target_selectivity: float,
                                     min_size: float,
                                     max_size: float) -> float:
    avg_size = (min_size + max_size) / 2.0
    if target_selectivity <= 0:
        raise ValueError("Target selectivity must be positive")
    return (2.0 * avg_size) / (target_selectivity ** (1.0 / 3.0))


def load_template_obj(filepath):
    vertices = []
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                faces.append([int(p.split('/')[0]) - 1 for p in parts[1:]])

    if not vertices:
        raise ValueError(f"No vertices found in {filepath}")

    v_arr = np.array(vertices, dtype=np.float64)
    min_coords = v_arr.min(axis=0)
    max_coords = v_arr.max(axis=0)
    size = max_coords - min_coords
    max_ext = float(size.max()) or 1.0
    center = (min_coords + max_coords) / 2.0
    v_normalized = ((v_arr - center) / max_ext).astype(np.float32)
    f_arr = np.array(faces, dtype=np.int32)
    return v_normalized, f_arr


# ---------------------------------------------------------------------------
# worker function – executed in subprocesses
# ---------------------------------------------------------------------------

def format_obj_batch_bytes(args):
    """
    Format a batch of OBJ objects and return them as a single bytes object.

    Two vectorised np.savetxt calls handle all vertices and all faces for the
    entire batch.  The cheap per-object loop only inserts the 'o objN' header
    line and blank separators.
    """
    start_idx, centers, sizes, template_v, template_f_arr, vertex_offset_start = args

    num_v = len(template_v)
    num_f = len(template_f_arr)
    n = len(centers)

    # ---- transformed vertices: broadcast (n,1,3) * ... ----
    # all_v shape: (n, num_v, 3)
    all_v = template_v[np.newaxis, :, :] * sizes[:, np.newaxis, :] \
            + centers[:, np.newaxis, :]
    all_v_flat = all_v.reshape(-1, 3)           # (n*num_v, 3)

    # ---- face indices with per-object global offset (1-based) ----
    offsets = (vertex_offset_start
               + np.arange(n, dtype=np.int32) * num_v
               + 1)                              # (n,)
    # all_f shape: (n, num_f, 3)
    all_f = (template_f_arr[np.newaxis, :, :]
             + offsets[:, np.newaxis, np.newaxis])
    all_f_flat = all_f.reshape(-1, 3)           # (n*num_f, 3)

    # ---- format vertex block with a single savetxt call ----
    # np.savetxt requires a str fmt (bytes not accepted in NumPy >= 1.24),
    # so write to StringIO then encode the whole block at once.
    v_sbuf = io.StringIO()
    np.savetxt(v_sbuf, all_v_flat, fmt='v %.6f %.6f %.6f')
    v_raw = v_sbuf.getvalue().encode()

    # ---- format face block with a single savetxt call ----
    f_sbuf = io.StringIO()
    np.savetxt(f_sbuf, all_f_flat, fmt='f %d %d %d')
    f_raw = f_sbuf.getvalue().encode()

    # Split into per-line byte lists for slicing; drop trailing empty element
    v_lines = v_raw.split(b'\n')
    if v_lines and not v_lines[-1]:
        v_lines = v_lines[:-1]
    f_lines = f_raw.split(b'\n')
    if f_lines and not f_lines[-1]:
        f_lines = f_lines[:-1]

    # ---- assemble per-object blocks ----
    out = io.BytesIO()
    for i in range(n):
        out.write(b'o obj_')
        out.write(str(start_idx + i).encode())
        out.write(b'\n')
        # vertices
        for line in v_lines[i * num_v:(i + 1) * num_v]:
            out.write(line)
            out.write(b'\n')
        out.write(b'\n')
        # faces
        for line in f_lines[i * num_f:(i + 1) * num_f]:
            out.write(line)
            out.write(b'\n')
        out.write(b'\n')

    return out.getvalue()


# ---------------------------------------------------------------------------
# file writer
# ---------------------------------------------------------------------------

def write_obj_file_fast(
    filepath: Path,
    num_objs: int,
    min_size: float,
    max_size: float,
    extent: float,
    seed: Optional[int],
    template_v: np.ndarray,
    template_f: np.ndarray,
    batch_size: int = 500,
) -> None:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_v = len(template_v)

    centers_all = np.random.uniform(0, extent, size=(num_objs, 3)).astype(np.float32)
    sizes_all   = np.random.uniform(min_size, max_size, size=(num_objs, 1)).astype(np.float32)

    num_procs = mp.cpu_count()
    print(f"  Generating {num_objs:,} objects → {filepath.name} "
          f"({num_procs} cores, batch_size={batch_size})")

    tasks = []
    for start_idx in range(0, num_objs, batch_size):
        end_idx = min(start_idx + batch_size, num_objs)
        v_off = start_idx * num_v
        tasks.append((
            start_idx,
            centers_all[start_idx:end_idx],
            sizes_all[start_idx:end_idx],
            template_v,
            template_f,
            v_off,
        ))

    # Binary file with 32 MiB write buffer – avoids encode/decode overhead
    with open(filepath, 'wb', buffering=1 << 25) as f:
        f.write(b"# Generated mesh for RaySpace3D testing\n")
        f.write(f"# Number of objects: {num_objs}\n\n".encode())

        with mp.Pool(processes=num_procs) as pool:
            for i, batch_bytes in enumerate(
                pool.imap(format_obj_batch_bytes, tasks, chunksize=1)
            ):
                f.write(batch_bytes)
                if (i + 1) % 20 == 0:
                    done = min((i + 1) * batch_size, num_objs)
                    print(f"    ... {done:,} / {num_objs:,} objects written")

    print(f"  Done → {filepath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fast Parallel Generation of templated OBJ meshes'
    )
    parser.add_argument('--template-obj', type=str, required=True,
                        help='Path to template .obj file')
    parser.add_argument('--num-objs-a', '-na', type=int, required=True)
    parser.add_argument('--num-objs-b', '-nb', type=int, required=True)
    parser.add_argument('--min-size',   type=float, required=True)
    parser.add_argument('--max-size',   type=float, required=True)
    parser.add_argument('--selectivity', '-s', type=float, required=True)
    parser.add_argument('--output-a',  '-oa', type=str, required=True)
    parser.add_argument('--output-b',  '-ob', type=str, required=True)
    parser.add_argument('--seed',       type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Objects per worker batch (default 500)')
    args = parser.parse_args()

    t_v, t_f = load_template_obj(args.template_obj)
    universe_extent = compute_universe_for_selectivity(
        args.selectivity, args.min_size, args.max_size
    )

    write_obj_file_fast(
        Path(args.output_a), args.num_objs_a,
        args.min_size, args.max_size, universe_extent,
        args.seed, t_v, t_f, args.batch_size,
    )
    seed_b = args.seed + 1 if args.seed is not None else None
    write_obj_file_fast(
        Path(args.output_b), args.num_objs_b,
        args.min_size, args.max_size, universe_extent,
        seed_b, t_v, t_f, args.batch_size,
    )


if __name__ == '__main__':
    main()
