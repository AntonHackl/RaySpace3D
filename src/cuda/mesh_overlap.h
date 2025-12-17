#pragma once

#include "../common.h"
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
    
    // Results
    MeshOverlapResult* results;
    int* hit_counter;
    int max_results;
};

// Launch functions are provided by MeshOverlapLauncher class
// See src/raytracing/MeshOverlapLauncher.h

