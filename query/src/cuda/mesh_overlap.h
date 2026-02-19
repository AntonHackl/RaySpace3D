#pragma once

#include "common.h"
#include <optix.h>
#include <cuda_runtime.h>

struct MeshOverlapLaunchParams {
    // Mesh1 data (plain on GPU)
    float3* mesh1_vertices;
    uint3* mesh1_indices;
    int* mesh1_triangle_to_object;
    int mesh1_num_triangles;
    
    // Mesh2 acceleration structure
    OptixTraversableHandle mesh2_handle;
    float3* mesh2_vertices;
    uint3* mesh2_indices;
    int* mesh2_triangle_to_object;
    
    // Hash Table for on-the-fly deduplication
    unsigned long long* hash_table;
    unsigned long long hash_table_size;
    int use_hash_table;
    int use_bitwise_hash; // Optimization for power-of-two sizes
    
    // Two-pass results
    int* collision_counts;      // Pass 1: per-triangle collision counts
    long long* collision_offsets;     // Exclusive scan of counts (output positions)
    MeshOverlapResult* results; // Pass 2: actual collision pairs
    int pass;                   // 1 = count only, 2 = write results
};
