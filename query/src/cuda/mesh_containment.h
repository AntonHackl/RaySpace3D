#pragma once

#include "common.h"
#include <optix.h>
#include <cuda_runtime.h>

struct MeshContainmentLaunchParams {
    // Source mesh data (edges to cast)
    float3* src_vertices;
    uint3*  src_indices;
    int*    src_triangle_to_object;
    int     src_num_triangles;

    // Target mesh acceleration structure (to trace against)
    OptixTraversableHandle target_handle;
    int*    target_triangle_to_object;

    // Intersection hash table (Phase 1: populated by edge checks)
    unsigned long long* intersection_hash_table;
    int                 intersection_hash_table_size;

    // Direction flag for edge kernel:
    //   0 = B edges vs A AS  → insert(hit_obj, src_obj) i.e. (A_obj, B_obj)
    //   1 = A edges vs B AS  → insert(src_obj, hit_obj) i.e. (A_obj, B_obj)
    int swap_ids;

    // Phase 2: Point-in-mesh data
    float3* b_first_vertices;       // One vertex per B object (pre-computed on host)
    int     b_num_objects;

    // Containment result hash table (Phase 2 output)
    unsigned long long* containment_hash_table;
    int                 containment_hash_table_size;
};
