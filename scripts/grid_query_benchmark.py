#!/usr/bin/env python3
"""Grid-based query benchmark for raytracer.

This script translates preprocessed geometry to different positions in a 3x3x3 grid
and runs the raytracer at each position, measuring query performance.

The grid is constructed over the bounding box of the point dataset, with each
cell representing a different spatial location for the query geometry.

Example:
    python scripts/grid_query_benchmark.py \\
        --geometry generated_files/query.txt \\
        --points generated_files/shape_net_points_1000000.wkt \\
        --bbox generated_files/shape_net_bbox.json \\
        --output results/grid_benchmark.json \\
        --runs 5 \\
        --grid-size 3 3 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import re
# from mpl_toolkits.mplot3d import Axes3D


def load_bbox(bbox_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load bounding box from JSON file."""
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)
    return np.array(bbox['min']), np.array(bbox['max'])


def load_geometry(geometry_path: str) -> Dict[str, Any]:
    """Load geometry data from preprocessed text file.
    
    Format expected:
      vertices: x1 y1 z1 x2 y2 z2 ...
      normals: nx1 ny1 nz1 nx2 ny2 nz2 ...
      indices: i1 i2 i3 i4 i5 i6 ...
      triangleToObject: obj1 obj2 obj3 ...
      total_triangles: N
    """
    if not os.path.exists(geometry_path):
        raise FileNotFoundError(f"Geometry file not found: {geometry_path}")
    
    with open(geometry_path, 'r') as f:
        lines = f.readlines()
    
    geometry = {
        'vertices': [],
        'normals': [],
        'indices': [],
        'triangleToObject': [],
        'total_triangles': 0
    }
    
    for line in lines:
        line = line.strip()
        if line.startswith('vertices:'):
            data = line[9:].strip().split()
            # Parse triplets of floats
            for i in range(0, len(data), 3):
                if i + 2 < len(data):
                    geometry['vertices'].append([
                        float(data[i]),
                        float(data[i+1]),
                        float(data[i+2])
                    ])
        elif line.startswith('normals:'):
            data = line[8:].strip().split()
            for i in range(0, len(data), 3):
                if i + 2 < len(data):
                    geometry['normals'].append([
                        float(data[i]),
                        float(data[i+1]),
                        float(data[i+2])
                    ])
        elif line.startswith('indices:'):
            data = line[8:].strip().split()
            for i in range(0, len(data), 3):
                if i + 2 < len(data):
                    geometry['indices'].append([
                        int(data[i]),
                        int(data[i+1]),
                        int(data[i+2])
                    ])
        elif line.startswith('triangleToObject:'):
            data = line[17:].strip().split()
            geometry['triangleToObject'] = [int(x) for x in data]
        elif line.startswith('total_triangles:'):
            geometry['total_triangles'] = int(line[16:].strip())
    
    return geometry


def translate_geometry(geometry: Dict[str, Any], translation: np.ndarray) -> Dict[str, Any]:
    """Translate geometry by given offset vector."""
    translated = {
        'vertices': [],
        'normals': geometry['normals'].copy(),  # Don't translate normals
        'indices': geometry['indices'].copy(),
        'triangleToObject': geometry['triangleToObject'].copy(),
        'total_triangles': geometry['total_triangles']
    }
    
    # Translate vertices
    for vertex in geometry['vertices']:
        translated['vertices'].append([
            vertex[0] + translation[0],
            vertex[1] + translation[1],
            vertex[2] + translation[2]
        ])
    
    return translated


def save_geometry(geometry: Dict[str, Any], output_path: str):
    """Save geometry to file in the expected format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write vertices
        f.write('vertices:')
        for v in geometry['vertices']:
            f.write(f' {v[0]} {v[1]} {v[2]}')
        f.write('\n')
        
        # Write normals
        f.write('normals:')
        for n in geometry['normals']:
            f.write(f' {n[0]} {n[1]} {n[2]}')
        f.write('\n')
        
        # Write indices
        f.write('indices:')
        for idx in geometry['indices']:
            f.write(f' {idx[0]} {idx[1]} {idx[2]}')
        f.write('\n')
        
        # Write triangleToObject
        f.write('triangleToObject:')
        for obj_id in geometry['triangleToObject']:
            f.write(f' {obj_id}')
        f.write('\n')
        
        # Write total triangles
        f.write(f'total_triangles: {geometry["total_triangles"]}\n')



# Note: batch processing uses a single subprocess call; per-task parsing of
# inside counts/ratios is handled after the batch stdout is available.


def generate_grid_positions(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid_size: Tuple[int, int, int]
) -> List[Tuple[int, int, int, np.ndarray]]:
    """Generate grid positions within the bounding box.
    
    Returns:
        List of tuples: (grid_x, grid_y, grid_z, translation_vector)
    """
    positions = []
    
    # Calculate cell dimensions
    bbox_range = bbox_max - bbox_min
    cell_size = bbox_range / np.array(grid_size)
    
    for ix in range(grid_size[0]):
        for iy in range(grid_size[1]):
            for iz in range(grid_size[2]):
                # Calculate center of this grid cell
                grid_center = bbox_min + (np.array([ix, iy, iz]) + 0.5) * cell_size
                positions.append((ix, iy, iz, grid_center))
    
    return positions


def compute_statistics(query_times: List[float]) -> Dict[str, float]:
    """Compute mean and standard deviation of query times."""
    if not query_times:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    times_array = np.array(query_times)
    return {
        'mean': float(np.mean(times_array)),
        'std': float(np.std(times_array)),
        'min': float(np.min(times_array)),
        'max': float(np.max(times_array))
    }


def plot_3d_results(
    results: List[Dict[str, Any]],
    output_path: str
):
    """Create 3D visualization of query performance across grid positions."""
    if not results:
        print("No results to plot", file=sys.stderr)
        return
    
    # Extract data
    positions = []
    query_times = []

    for r in results:
        # prefer explicit translation field
        translation = r.get('translation') or r.get('grid_position')
        if translation is None:
            # skip entries without a known position
            continue

        # Determine query time: support new per-result timing ('query_ms') and
        # legacy timer JSON ('phases' -> 'Query' entry)
        q_ms = None
        td = r.get('timing_data')
        if isinstance(td, dict):
            # new format: {'query_ms': ..., 'output_ms': ...}
            if 'query_ms' in td and td['query_ms'] is not None:
                q_ms = float(td['query_ms'])
            # legacy nested phases
            elif 'phases' in td and isinstance(td['phases'], dict) and 'Query' in td['phases']:
                entry = td['phases']['Query']
                if 'duration_ms' in entry:
                    q_ms = float(entry['duration_ms'])

        # If still None, try to read results.results.query_ms (some entries store timing there)
        if q_ms is None:
            nested = r.get('results', {})
            if isinstance(nested, dict) and 'query_ms' in nested and nested['query_ms'] is not None:
                q_ms = float(nested['query_ms'])

        if q_ms is None:
            # skip entries without timing
            continue

        positions.append(translation)
        query_times.append(q_ms)
    
    if not positions:
        print("No valid timing data to plot", file=sys.stderr)
        return
    
    positions = np.array(positions)
    query_times = np.array(query_times)
    
    # Normalize query times for color mapping
    min_time = np.min(query_times)
    max_time = np.max(query_times)
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with color gradient
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=query_times,
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Query Time (ms)', rotation=270, labelpad=20, fontsize=10)
    
    # Labels and title
    ax.set_xlabel('X Position', fontsize=10)
    ax.set_ylabel('Y Position', fontsize=10)
    ax.set_zlabel('Z Position', fontsize=10)
    ax.set_title(
        f'Query Performance Across Grid\n'
        f'Mean: {np.mean(query_times):.2f} ms, Std: {np.std(query_times):.2f} ms\n'
        f'Min: {min_time:.2f} ms, Max: {max_time:.2f} ms',
        fontsize=12,
        pad=20
    )
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved 3D visualization to: {output_path}")
    
    # Also save as interactive HTML if plotly is available
    try:
        import plotly.graph_objects as go
        
        fig_plotly = go.Figure(data=[go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=query_times,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Query Time (ms)"),
                line=dict(color='black', width=0.5)
            ),
            text=[f'Query Time: {t:.2f} ms' for t in query_times],
            hovertemplate='<b>Position:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                          '%{text}<extra></extra>'
        )])
        
        fig_plotly.update_layout(
            title=f'Query Performance Across Grid<br>' +
                  f'Mean: {np.mean(query_times):.2f} ms, Std: {np.std(query_times):.2f} ms',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            width=900,
            height=700
        )
        
        html_path = output_path.replace('.png', '_interactive.html')
        fig_plotly.write_html(html_path)
        print(f"Saved interactive 3D visualization to: {html_path}")
    
    except ImportError:
        print("Note: Install plotly for interactive 3D visualization: pip install plotly")


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    
    p = argparse.ArgumentParser(
        description="Run grid-based query benchmark for raytracer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    p.add_argument(
        "--geometry",
        required=True,
        help="Path to preprocessed query geometry file (.txt)"
    )
    p.add_argument(
        "--points",
        required=True,
        help="Path to point dataset file (.wkt)"
    )
    p.add_argument(
        "--bbox",
        help="Path to bounding box JSON file"
    )
    p.add_argument(
        "--bbox-min",
        nargs=3,
        type=float,
        help="Bounding box minimum coordinates: x y z (overrides --bbox if provided)"
    )
    p.add_argument(
        "--bbox-max",
        nargs=3,
        type=float,
        help="Bounding box maximum coordinates: x y z (overrides --bbox if provided)"
    )
    p.add_argument(
        "--raytracer",
        default="build/bin/Debug/raytracer.exe",
        help="Path to raytracer executable (default: build/bin/Debug/raytracer.exe)"
    )
    p.add_argument(
        "--output",
        default="results/grid_benchmark.json",
        help="Output JSON file for benchmark results"
    )
    p.add_argument(
        "--plot",
        default=None,
        help="Output path for 3D plot (default: same as --output but .png)"
    )
    p.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per grid position (default: 5)"
    )
    p.add_argument(
        "--grid-size",
        nargs=3,
        type=int,
        default=[3, 3, 3],
        metavar=("X", "Y", "Z"),
        help="Grid dimensions (default: 3 3 3)"
    )
    p.add_argument(
        "--temp-dir",
        default=None,
        help="Temporary directory for intermediate files (default: system temp)"
    )
    
    args = p.parse_args(argv)
    
    # Validate inputs
    if not os.path.exists(args.geometry):
        print(f"Error: Geometry file not found: {args.geometry}", file=sys.stderr)
        return 1

    if not os.path.exists(args.points):
        print(f"Error: Points file not found: {args.points}", file=sys.stderr)
        return 1

    if args.bbox_min is None or args.bbox_max is None:
        # if bbox min/max not provided, require a bbox file
        if not args.bbox:
            print("Error: Either provide --bbox file or both --bbox-min and --bbox-max", file=sys.stderr)
            return 1
        if not os.path.exists(args.bbox):
            print(f"Error: Bounding box file not found: {args.bbox}", file=sys.stderr)
            return 1

    if not os.path.exists(args.raytracer):
        print(f"Error: Raytracer executable not found: {args.raytracer}", file=sys.stderr)
        return 1
    
    # Create temp directory
    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)
        temp_dir = args.temp_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix="raytracer_benchmark_")
    
    print(f"Using temporary directory: {temp_dir}")
    
    # Load data
    print("\nLoading data...")
    # Prefer explicit bbox values if provided on the command line
    if args.bbox_min is not None and args.bbox_max is not None:
        bbox_min = np.array(args.bbox_min)
        bbox_max = np.array(args.bbox_max)
        print(f"  Bounding box from args: min={bbox_min}, max={bbox_max}")
    else:
        bbox_min, bbox_max = load_bbox(args.bbox)
        print(f"  Bounding box from file {args.bbox}: min={bbox_min}, max={bbox_max}")
    
    geometry = load_geometry(args.geometry)
    print(f"  Loaded geometry:")
    print(f"    Vertices: {len(geometry['vertices'])}")
    print(f"    Triangles: {len(geometry['indices'])}")
    
    # Generate grid positions
    print(f"\nGenerating {args.grid_size[0]}x{args.grid_size[1]}x{args.grid_size[2]} grid...")
    grid_positions = generate_grid_positions(bbox_min, bbox_max, tuple(args.grid_size))
    total_queries = len(grid_positions)
    print(f"  Total queries to run: {total_queries}")
    
    # Pre-generate all translated geometries
    print(f"\nPre-generating {total_queries} translated geometries...")
    geometry_paths = []
    grid_positions_list = []
    
    for idx, (gx, gy, gz, translation) in enumerate(grid_positions, 1):
        print(f"  Generating geometry {idx}/{total_queries}: Grid position ({gx}, {gy}, {gz})")
        
        # Translate geometry
        translated_geometry = translate_geometry(geometry, translation)
        
        # Save to temporary file
        temp_geometry_path = os.path.join(temp_dir, f"geometry_{gx}_{gy}_{gz}.txt")
        save_geometry(translated_geometry, temp_geometry_path)
        geometry_paths.append(temp_geometry_path)
        grid_positions_list.append((gx, gy, gz, translation))
    
    print(f"All {total_queries} geometries pre-generated")
    
    # Run raytracer once with all geometries as a comma-separated list
    print(f"\nRunning batch raytracer with {args.runs} runs per position...\n")
    geometry_list_arg = ','.join(geometry_paths)
    
    timing_json = os.path.join(temp_dir, f"timing_{os.getpid()}.json")
    
    cmd = [
        args.raytracer,
        "--geometry", geometry_list_arg,
        "--points", args.points,
        "--runs", str(args.runs),
        "--output", timing_json,
        "--no-export"
    ]
    
    print(f"Running: {cmd[0]} with {total_queries} geometries (comma-separated)")
    
    try:
        result_proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600  # 1 hour timeout for batch processing
        )
        
        print("Batch processing complete!")
        print(result_proc.stdout)
        
        # Read timing data (timer now writes numbered phase keys: query_1, output_1, ...)
        with open(timing_json, 'r') as f:
            timing_data = json.load(f)

        stdout = result_proc.stdout

        # Parse per-task textual summaries (Total rays / Points INSIDE / Inside ratio)
        total_matches = re.findall(r"Total rays:\s*(\d+)", stdout)
        inside_matches = re.findall(r"Points INSIDE polygons:\s*(\d+)", stdout)
        ratio_matches = re.findall(r"Inside ratio:\s*([0-9\.]+)\s*%", stdout)

        # Build results; also extract per-task timings from timing_data['phases']
        results = []
        query_times = []
        inside_percentages = []

        phases = timing_data.get('phases', {}) if timing_data else {}

        # Helper to get phase duration for a given phase name and task index (1-based)
        def phase_duration_ms(phase_base: str, idx: int):
            key = f"{phase_base.lower()}_{idx}"
            entry = phases.get(key)
            if entry and 'duration_ms' in entry:
                return float(entry['duration_ms'])
            return None

        for idx, (gx, gy, gz, translation) in enumerate(grid_positions_list, start=1):
            # per-task timing extraction
            q_ms = phase_duration_ms('query', idx)
            o_ms = phase_duration_ms('output', idx)

            # textual metrics
            total_rays = int(total_matches[idx-1]) if idx-1 < len(total_matches) else None
            inside_count = int(inside_matches[idx-1]) if idx-1 < len(inside_matches) else None
            inside_pct = float(ratio_matches[idx-1]) if idx-1 < len(ratio_matches) else None

            if inside_pct is None and inside_count is not None and total_rays:
                try:
                    inside_pct = (inside_count / total_rays) * 100.0 if total_rays > 0 else 0.0
                except Exception:
                    inside_pct = None

            result = {
                'grid_position': [gx, gy, gz],
                'translation': translation.tolist(),
                'timing_data': {
                    'query_ms': q_ms,
                    'output_ms': o_ms
                },
                'results': {
                    'total_rays': total_rays,
                    'inside_count': inside_count,
                    'inside_percentage': inside_pct
                }
            }
            results.append(result)

            if q_ms is not None:
                query_times.append(q_ms)
            elif timing_data and 'phases' in timing_data and 'query_1' in timing_data['phases']:
                # fallback: use average per task if only batch timing present
                # handled below
                pass

            if inside_pct is not None:
                inside_percentages.append(inside_pct)

        # If we didn't obtain per-task query times from phases, fall back to averaged batch Query
        if not query_times and timing_data and 'phases' in timing_data:
            # find any 'query_1' or similar; if absent, try to find any 'query_*' keys
            q_keys = [k for k in phases.keys() if k.startswith('query_')]
            if q_keys:
                # sort keys by suffix number
                def key_idx(k):
                    try:
                        return int(k.split('_')[-1])
                    except Exception:
                        return 0
                q_keys.sort(key=key_idx)
                for k in q_keys:
                    entry = phases.get(k)
                    if entry and 'duration_ms' in entry:
                        query_times.append(float(entry['duration_ms']))
            else:
                # last-resort: if there's any 'query_1' missing, try to use overall 'query_1' naming rules not present
                if 'query' in phases and 'duration_ms' in phases['query']:
                    total_query_time_ms = float(phases['query']['duration_ms'])
                    avg_query_time_ms = total_query_time_ms / total_queries if total_queries > 0 else 0
                    query_times = [avg_query_time_ms] * total_queries
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Raytracer timed out after 3600 seconds", file=sys.stderr)
        results = []
        query_times = []
        inside_percentages = []
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Raytracer failed with exit code {e.returncode}", file=sys.stderr)
        print(f"  STDOUT: {e.stdout}", file=sys.stderr)
        print(f"  STDERR: {e.stderr}", file=sys.stderr)
        results = []
        query_times = []
        inside_percentages = []
    
    # Compute statistics
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    stats = compute_statistics(query_times)
    print(f"\nQuery Time Statistics ({len(query_times)} successful queries):")
    print(f"  Mean:     {stats['mean']:.2f} ms")
    print(f"  Std Dev:  {stats['std']:.2f} ms")
    print(f"  Min:      {stats['min']:.2f} ms")
    print(f"  Max:      {stats['max']:.2f} ms")

    # Compute statistics for inside percentages if any
    if inside_percentages:
        pct_stats = compute_statistics(inside_percentages)
        print(f"\nPoints Inside Statistics ({len(inside_percentages)} successful queries):")
        print(f"  Mean:     {pct_stats['mean']:.2f} %")
        print(f"  Std Dev:  {pct_stats['std']:.2f} %")
        print(f"  Min:      {pct_stats['min']:.2f} %")
        print(f"  Max:      {pct_stats['max']:.2f} %")
    else:
        print("\nPoints Inside Statistics: no data collected")
    
    # Save results
    output_data = {
        'configuration': {
            'geometry_file': args.geometry,
            'points_file': args.points,
            'bbox_file': args.bbox,
            'grid_size': args.grid_size,
            'runs_per_position': args.runs,
            'total_queries': total_queries
        },
        'statistics': stats,
        'results': results
    }
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved benchmark results to: {args.output}")
    
    # Generate plot
    plot_path = args.plot if args.plot else args.output.replace('.json', '.png')
    plot_3d_results(results, plot_path)
    
    # Cleanup temp directory if we created it
    if not args.temp_dir:
        import shutil
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
