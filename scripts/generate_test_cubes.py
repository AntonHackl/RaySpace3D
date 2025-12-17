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


def write_separate_obj_files(output_dir, cubes_data):
    """
    Write each cube to a separate .obj file.
    
    Args:
        output_dir: Directory to write files to
        cubes_data: List of (cube_id, vertices, faces) tuples
    """
    for cube_id, vertices, faces in cubes_data:
        filepath = os.path.join(output_dir, f"cube_{cube_id:04d}.obj")
        with open(filepath, 'w') as f:
            f.write(f"# Generated cube mesh {cube_id} for RaySpace3D testing\n\n")
            f.write(f"o cube_{cube_id}\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (1-indexed in .obj format)
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


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
    
    parser.add_argument('--num-cubes', '-n', type=int, default=10,
                        help='Number of cubes to generate')
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
    parser.add_argument('--output-dir', '-o', type=str, default='generated_files/Query',
                        help='Output directory for generated mesh files')
    parser.add_argument('--single-file', action='store_true',
                        help='Write all cubes to a single .obj file instead of separate files')
    parser.add_argument('--output-name', type=str, default='test_cubes.obj',
                        help='Output filename when using --single-file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_cubes <= 0:
        parser.error("Number of cubes must be positive")
    if args.min_size <= 0 or args.max_size <= 0:
        parser.error("Cube sizes must be positive")
    if args.min_size > args.max_size:
        parser.error("Minimum size cannot be greater than maximum size")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_cubes} test cubes...")
    print(f"  Size range: [{args.min_size}, {args.max_size}]")
    print(f"  X extent: [{args.extent_x[0]}, {args.extent_x[1]}]")
    print(f"  Y extent: [{args.extent_y[0]}, {args.extent_y[1]}]")
    print(f"  Z extent: [{args.extent_z[0]}, {args.extent_z[1]}]")
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")
    
    # Generate cube data
    cubes_data = generate_random_cube_data(
        args.num_cubes,
        args.min_size,
        args.max_size,
        tuple(args.extent_x),
        tuple(args.extent_y),
        tuple(args.extent_z),
        args.seed
    )
    
    # Write output
    if args.single_file:
        output_path = output_dir / args.output_name
        print(f"Writing to single file: {output_path}")
        write_obj_file(output_path, cubes_data)
        print(f"Successfully wrote {args.num_cubes} cubes to {output_path}")
    else:
        print(f"Writing to directory: {output_dir}")
        write_separate_obj_files(str(output_dir), cubes_data)
        print(f"Successfully wrote {args.num_cubes} cube files to {output_dir}")
    
    # Print statistics
    total_vertices = args.num_cubes * 8
    total_triangles = args.num_cubes * 12
    print(f"\nMesh statistics:")
    print(f"  Total vertices: {total_vertices}")
    print(f"  Total triangles: {total_triangles}")
    print(f"  Average vertices per cube: 8")
    print(f"  Average triangles per cube: 12")


if __name__ == '__main__':
    main()
