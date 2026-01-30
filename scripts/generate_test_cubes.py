#!/usr/bin/env python3
"""
Generate test cube meshes for RaySpace3D mesh overlap testing.
Creates triangulated cubes with correct winding as .obj files.
"""

import argparse
import os
import random
import math
from pathlib import Path


def generate_cube_vertices(center_x, center_y, center_z, size):
    """
    Generate 8 vertices for a cube centered at (center_x, center_y, center_z)
    with the given size (edge length).
    
    Returns list of (x, y, z) tuples.
    """
    half_size = size / 2.0
    vertices = [
        # Bottom face (z = center_z - half_size)
        (center_x - half_size, center_y - half_size, center_z - half_size),  # 0
        (center_x + half_size, center_y - half_size, center_z - half_size),  # 1
        (center_x + half_size, center_y + half_size, center_z - half_size),  # 2
        (center_x - half_size, center_y + half_size, center_z - half_size),  # 3
        # Top face (z = center_z + half_size)
        (center_x - half_size, center_y - half_size, center_z + half_size),  # 4
        (center_x + half_size, center_y - half_size, center_z + half_size),  # 5
        (center_x + half_size, center_y + half_size, center_z + half_size),  # 6
        (center_x - half_size, center_y + half_size, center_z + half_size),  # 7
    ]
    return vertices


def generate_cube_faces():
    """
    Generate 12 triangular faces for a cube with counter-clockwise winding
    when viewed from outside.
    
    Returns list of (v1, v2, v3) tuples (0-indexed).
    """
    faces = [
        # Bottom face (looking from below, CCW)
        (0, 2, 1), (0, 3, 2),
        # Top face (looking from above, CCW)
        (4, 5, 6), (4, 6, 7),
        # Front face (y-)
        (0, 1, 5), (0, 5, 4),
        # Right face (x+)
        (1, 2, 6), (1, 6, 5),
        # Back face (y+)
        (2, 3, 7), (2, 7, 6),
        # Left face (x-)
        (3, 0, 4), (3, 4, 7),
    ]
    return faces


def write_obj_file(filepath, cubes_data):
    """
    Write cubes to a single .obj file.
    
    Args:
        filepath: Path to output .obj file
        cubes_data: List of (cube_id, vertices, faces) tuples
    """
    with open(filepath, 'w') as f:
        f.write("# Generated cube mesh for RaySpace3D testing\n")
        f.write(f"# Number of cubes: {len(cubes_data)}\n\n")
        
        vertex_offset = 0
        
        for cube_id, vertices, faces in cubes_data:
            f.write(f"o cube_{cube_id}\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (1-indexed in .obj format)
            for face in faces:
                f.write(f"f {face[0] + vertex_offset + 1} {face[1] + vertex_offset + 1} {face[2] + vertex_offset + 1}\n")
            
            f.write("\n")
            vertex_offset += len(vertices)


def calculate_theoretical_selectivity(u_dims, s_range, count_a, count_b):
    """
    Calculates the theoretical selectivity using the geometric probability formula.
    Selectivity = Probability that any single pair (a, b) intersects.
    """
    ux, uy, uz = u_dims
    s_min, s_max = s_range
    
    # 1. Determine Average Size
    # Assuming uniform distribution of sizes
    avg_s = (s_min + s_max) / 2.0
    
    # 2. Calculate Probability of Intersection per Dimension
    # P_dim = (avg_size_A + avg_size_B) / Universe_Length
    # Since both datasets use the same size distribution, avg_size_A + avg_size_B = 2 * avg_s
    p_x = (2 * avg_s) / ux if ux > 0 else 1.0
    p_y = (2 * avg_s) / uy if uy > 0 else 1.0
    p_z = (2 * avg_s) / uz if uz > 0 else 1.0
    
    # Total Probability (Selectivity) capped at 1.0 per dimension
    p_x = min(1.0, p_x)
    p_y = min(1.0, p_y)
    p_z = min(1.0, p_z)
    
    # Total Probability (Selectivity)
    # The probability that they intersect in X AND Y AND Z
    selectivity = p_x * p_y * p_z
    
    return selectivity

def estimate_selectivity_sampling(u_dims, s_range, count_a, count_b, samples=100000):
    """
    Estimates selectivity by Monte Carlo sampling.
    Generates random pairs and checks for SURFACE intersection (Overlap AND NOT Containment).
    """
    ux, uy, uz = u_dims
    s_min, s_max = s_range
    
    matches = 0
    
    for _ in range(samples):
        # Generate Cube A
        sa = random.uniform(s_min, s_max)
        ca_x = random.uniform(0, ux)
        ca_y = random.uniform(0, uy)
        ca_z = random.uniform(0, uz)
        
        # Generate Cube B
        sb = random.uniform(s_min, s_max)
        cb_x = random.uniform(0, ux)
        cb_y = random.uniform(0, uy)
        cb_z = random.uniform(0, uz)
        
        # Check Overlap (AABB)
        # Overlap in dim D if distance between centers < sum of half sizes
        sum_half_sizes = (sa + sb) / 2.0
        
        if abs(ca_x - cb_x) >= sum_half_sizes: continue
        if abs(ca_y - cb_y) >= sum_half_sizes: continue
        if abs(ca_z - cb_z) >= sum_half_sizes: continue
        
        # If we are here, we have AABB overlap.
        # Check Containment (one inside another)
        # A inside B?
        diff_half_sizes = (sb - sa) / 2.0
        if diff_half_sizes > 0:
            # B is larger, check if A centers are close enough to B centers
            if (abs(ca_x - cb_x) <= diff_half_sizes and 
                abs(ca_y - cb_y) <= diff_half_sizes and 
                abs(ca_z - cb_z) <= diff_half_sizes):
                continue # A is inside B -> No surface intersection
        elif diff_half_sizes < 0:
            # A is larger, check if B centers are close enough to A centers
            # diff_half_sizes is negative, so use (sa - sb)/2 = -diff_half_sizes
            limit = -diff_half_sizes
            if (abs(ca_x - cb_x) <= limit and 
                abs(ca_y - cb_y) <= limit and 
                abs(ca_z - cb_z) <= limit):
                continue # B is inside A -> No surface intersection
        else:
            # Equal size, exact overlap (containment) only if centers match exactly
            # With floats, practically impossible, but logically if centers match they contain each other.
            # But surface DOES intersect if identical cubes?
            # Actually if A==B, surfaces touch everywhere. It's an intersection.
            # But strict containment usually implies no surface crossing if open sets, 
            # but for closed sets surfaces touch.
            # For "Overlap" benchmark, usually we want "Intersection != Empty".
            # If A contains B, Intersection = B != Empty.
            # BUT user said "Does not account for containment".
            # This usually means "Surface Intersection".
            # If A contains B entirely, their surfaces might not touch.
            pass

        matches += 1
        
    selectivity = matches / samples
    expected_matches = selectivity * count_a * count_b
    return selectivity, expected_matches



def generate_random_cube_data(num_cubes, min_size, max_size, extent_x, extent_y, extent_z, seed=None):
    """
    Generate random cube data within the specified space extent.
    
    Args:
        num_cubes: Number of cubes to generate
        min_size: Minimum cube edge length
        max_size: Maximum cube edge length
        extent_x: (min_x, max_x) tuple
        extent_y: (min_y, max_y) tuple
        extent_z: (min_z, max_z) tuple
        seed: Random seed for reproducibility
    
    Returns:
        List of (cube_id, vertices, faces) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    cubes_data = []
    cube_faces = generate_cube_faces()
    
    for i in range(num_cubes):
        # Generate random position within extent
        center_x = random.uniform(extent_x[0], extent_x[1])
        center_y = random.uniform(extent_y[0], extent_y[1])
        center_z = random.uniform(extent_z[0], extent_z[1])
        
        # Generate random size
        size = random.uniform(min_size, max_size)
        
        # Generate cube vertices
        vertices = generate_cube_vertices(center_x, center_y, center_z, size)
        
        cubes_data.append((i, vertices, cube_faces))
    
    return cubes_data


def main():
    parser = argparse.ArgumentParser(
        description='Generate test cube meshes for RaySpace3D overlap testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num-cubes-a', '-na', type=int, default=10,
                        help='Number of cubes to generate in dataset A')
    parser.add_argument('--num-cubes-b', '-nb', type=int, default=10,
                        help='Number of cubes to generate in dataset B')
    parser.add_argument('--min-size', type=float, default=1.0,
                        help='Minimum cube edge length')
    parser.add_argument('--max-size', type=float, default=5.0,
                        help='Maximum cube edge length')
    parser.add_argument('--extent-x', type=float, nargs=2, default=[0.0, 100.0],
                        metavar=('MIN', 'MAX'),
                        help='X extent (min max) for cube centers')
    parser.add_argument('--extent-y', type=float, nargs=2, default=[0.0, 100.0],
                        metavar=('MIN', 'MAX'),
                        help='Y extent (min max) for cube centers')
    parser.add_argument('--extent-z', type=float, nargs=2, default=[0.0, 100.0],
                        metavar=('MIN', 'MAX'),
                        help='Z extent (min max) for cube centers')
    parser.add_argument('--output-a', '-oa', type=str, default='generated_files/Query/test_cubes_a.obj',
                        help='Output path for dataset A .obj file')
    parser.add_argument('--output-b', '-ob', type=str, default='generated_files/Query/test_cubes_b.obj',
                        help='Output path for dataset B .obj file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_cubes_a <= 0 or args.num_cubes_b <= 0:
        parser.error("Number of cubes must be positive")
    if args.min_size <= 0 or args.max_size <= 0:
        parser.error("Cube sizes must be positive")
    if args.min_size > args.max_size:
        parser.error("Minimum size cannot be greater than maximum size")
    
    # Create output directories if they don't exist
    output_path_a = Path(args.output_a)
    output_path_b = Path(args.output_b)
    output_path_a.parent.mkdir(parents=True, exist_ok=True)
    output_path_b.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating two cube datasets for join selectivity estimation...")
    print(f"  Dataset A: {args.num_cubes_a} cubes")
    print(f"  Dataset B: {args.num_cubes_b} cubes")
    print(f"  Size range: [{args.min_size}, {args.max_size}]")
    print(f"  X extent: [{args.extent_x[0]}, {args.extent_x[1]}]")
    print(f"  Y extent: [{args.extent_y[0]}, {args.extent_y[1]}]")
    print(f"  Z extent: [{args.extent_z[0]}, {args.extent_z[1]}]")
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")
    
    # Calculate selectivity estimation for join between dataset A and B
    u_dims = (
        abs(args.extent_x[1] - args.extent_x[0]),
        abs(args.extent_y[1] - args.extent_y[0]),
        abs(args.extent_z[1] - args.extent_z[0])
    )
    s_range = (args.min_size, args.max_size)
    
    # Use Sampling Estimation for better accuracy (accounting for containment)
    theo_sel, theo_matches = estimate_selectivity_sampling(u_dims, s_range, args.num_cubes_a, args.num_cubes_b)
    
    print(f"\n{'=' * 50}")
    print(f"ESTIMATED JOIN SELECTIVITY (A × B) [Sampled]:")
    print(f"{'=' * 50}")
    print(f"  Theoretical Selectivity: {theo_sel:.8f}")
    print(f"  Total possible pairs:    {args.num_cubes_a * args.num_cubes_b:,}")
    print(f"  Expected intersections:  {theo_matches:.2f}")
    print(f"{'=' * 50}\n")
    
    # Generate dataset A
    print(f"Generating dataset A ({args.num_cubes_a} cubes)...")
    cubes_data_a = generate_random_cube_data(
        args.num_cubes_a,
        args.min_size,
        args.max_size,
        tuple(args.extent_x),
        tuple(args.extent_y),
        tuple(args.extent_z),
        args.seed
    )
    
    # Generate dataset B with different seed to ensure different cubes
    seed_b = (args.seed + 1) if args.seed is not None else None
    print(f"Generating dataset B ({args.num_cubes_b} cubes)...")
    cubes_data_b = generate_random_cube_data(
        args.num_cubes_b,
        args.min_size,
        args.max_size,
        tuple(args.extent_x),
        tuple(args.extent_y),
        tuple(args.extent_z),
        seed_b
    )
    
    # Write outputs
    print(f"\nWriting dataset A to: {output_path_a}")
    write_obj_file(output_path_a, cubes_data_a)
    print(f"Successfully wrote {args.num_cubes_a} cubes to {output_path_a}")
    
    print(f"\nWriting dataset B to: {output_path_b}")
    write_obj_file(output_path_b, cubes_data_b)
    print(f"Successfully wrote {args.num_cubes_b} cubes to {output_path_b}")
    
    # Print statistics
    print(f"\nDataset A statistics:")
    print(f"  Total vertices: {args.num_cubes_a * 8}")
    print(f"  Total triangles: {args.num_cubes_a * 12}")
    print(f"\nDataset B statistics:")
    print(f"  Total vertices: {args.num_cubes_b * 8}")
    print(f"  Total triangles: {args.num_cubes_b * 12}")


if __name__ == '__main__':
    main()
