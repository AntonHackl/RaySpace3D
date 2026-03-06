#pragma once

#include "common.h"
#include <optix.h>
#include <cuda_runtime.h>

struct MeshIntersectionLaunchParams {
    // Mesh1 data (plain on GPU)
    float3* mesh1_vertices;
    uint3* mesh1_indices;
    int* mesh1_triangle_to_object;
    int mesh1_num_triangles;
    int mesh1_num_objects;

    // Precomputed source edge rays (for overlap phase)
    float3* edge_starts;
    float3* edge_ends;
    int* edge_source_object_counts;
    int* edge_source_objects;
    int* edge_source_object_offsets;
    int num_edges;
    
    // Mesh2 acceleration structure
    OptixTraversableHandle mesh2_handle;
    float3* mesh2_vertices;
    uint3* mesh2_indices;
    int* mesh2_triangle_to_object;
    int mesh2_num_objects;
    
    // Hash Table for on-the-fly deduplication
    unsigned long long* hash_table;
    int hash_table_size;
    int use_hash_table; // Use int instead of bool
    
    // Exactly one launch triangle per source object for containment fallback
    int* first_triangle_index_per_object;
    
    // Two-pass results (legacy)
    int* collision_counts;
    long long* collision_offsets;
    MeshQueryResult* results;
    int pass;
    int swap_result_ids; // 0 = (src_obj, hit_obj), 1 = (hit_obj, src_obj)
};
