import os
import sys
import json
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate points inside AABB or at object centers of a dataset.")
    parser.add_argument("datafile", type=str, help="Path to the preprocessed .txt data file")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of points to generate (for bbox mode)")
    parser.add_argument("--mode", type=str, choices=["bbox", "object_centers"], default="bbox", help="Generation mode: 'bbox' for uniform in bbox, 'object_centers' for object centers")
    return parser.parse_args()
def extract_object_centers(datafile):
    """
    Extracts object centers as centroids of vertices from the datafile.
    File structure:
    - Line 1: vertices: x1 y1 z1 x2 y2 z2 ...
    - Line 2: indices: i1 i2 i3 ... (triangles)
    - Line 3: triangleToObject: objId1 objId2 ... (one per triangle)
    
    For each object, we find all triangles belonging to it, get their vertex indices,
    and compute the centroid of all unique vertices used by that object.
    
    Returns a list of [x, y, z] centroids (one per object).
    """
    with open(datafile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Parse vertices line
    vertices_line = lines[0]
    if not vertices_line.startswith('vertices:'):
        raise ValueError("Expected first line to start with 'vertices:'")
    vertex_data = [float(x) for x in vertices_line.split()[1:]]
    vertices = np.array(vertex_data).reshape(-1, 3)
    
    # Parse indices line
    indices_line = lines[1]
    if not indices_line.startswith('indices:'):
        raise ValueError("Expected second line to start with 'indices:'")
    indices = np.array([int(x) for x in indices_line.split()[1:]])
    
    # Parse triangleToObject line
    tri_to_obj_line = lines[2]
    if not tri_to_obj_line.startswith('triangleToObject:'):
        raise ValueError("Expected third line to start with 'triangleToObject:'")
    triangle_to_object = np.array([int(x) for x in tri_to_obj_line.split()[1:]])
    
    # Group triangles by object
    unique_objects = np.unique(triangle_to_object)
    centers = []
    
    for obj_id in unique_objects:
        # Find all triangles belonging to this object
        triangle_mask = triangle_to_object == obj_id
        triangle_indices_for_obj = np.where(triangle_mask)[0]
        
        # Get vertex indices for these triangles (each triangle has 3 vertices)
        vertex_indices_set = set()
        for tri_idx in triangle_indices_for_obj:
            idx_start = tri_idx * 3
            vertex_indices_set.add(indices[idx_start])
            vertex_indices_set.add(indices[idx_start + 1])
            vertex_indices_set.add(indices[idx_start + 2])
        
        # Get the actual vertices and compute centroid
        object_vertices = vertices[list(vertex_indices_set)]
        centroid = object_vertices.mean(axis=0)
        centers.append(centroid.tolist())
    
    return np.array(centers)

def get_bbox_json_path(datafile):
    base, _ = os.path.splitext(datafile)
    return base + "_bbox.json"

def load_or_calculate_bbox(datafile):
    bbox_json = get_bbox_json_path(datafile)
    if os.path.exists(bbox_json):
        bbox_mtime = os.path.getmtime(bbox_json)
        data_mtime = os.path.getmtime(datafile)
        if bbox_mtime >= data_mtime:
            with open(bbox_json, 'r') as f:
                return json.load(f)
    # Calculate bbox: only use the line that starts with 'vertices:'
    points = []
    with open(datafile, 'r') as f:
        for line in f:
            if line.strip().startswith('vertices:'):
                # Remove 'vertices:' and parse the rest as floats
                parts = line.strip().split()[1:]
                # If the line is long, it may contain all points flattened
                # Try to reshape into Nx3 if possible
                try:
                    floats = [float(x) for x in parts]
                    if len(floats) % 3 == 0:
                        points = [floats[i:i+3] for i in range(0, len(floats), 3)]
                    else:
                        points = [floats]
                except ValueError:
                    continue
                break
    data = np.array(points)
    min_xyz = data.min(axis=0)
    max_xyz = data.max(axis=0)
    bbox = {
        "min": min_xyz.tolist(),
        "max": max_xyz.tolist()
    }
    with open(bbox_json, 'w') as f:
        json.dump(bbox, f, indent=2)
    return bbox

def generate_points_in_bbox(bbox, num_points):
    min_xyz = np.array(bbox["min"])
    max_xyz = np.array(bbox["max"])
    points = np.random.uniform(min_xyz, max_xyz, size=(num_points, len(min_xyz)))
    return points

def main():
    args = parse_args()
    if args.mode == "bbox":
        bbox = load_or_calculate_bbox(args.datafile)
        points = generate_points_in_bbox(bbox, args.num_points)
        wkt_file = os.path.splitext(args.datafile)[0] + f"_points_{args.num_points}.wkt"
        with open(wkt_file, 'w') as f:
            for x, y, z in points:
                f.write(f"POINT Z ({x:.6f} {y:.6f} {z:.6f})\n")
        print(f"Generated individual WKT POINTs in {wkt_file}")
        print(f"Bounding box: {bbox}")
    elif args.mode == "object_centers":
        centers = extract_object_centers(args.datafile)
        wkt_file = os.path.splitext(args.datafile)[0] + f"_object_centers.wkt"
        with open(wkt_file, 'w') as f:
            for x, y, z in centers:
                f.write(f"POINT Z ({x:.6f} {y:.6f} {z:.6f})\n")
        print(f"Generated WKT POINTs at object centers in {wkt_file}")
        print(f"Number of centers: {len(centers)}")

if __name__ == "__main__":
    main()
