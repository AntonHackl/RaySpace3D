#pragma once
#include <optix.h>
#include <cuda_runtime.h>

// Simple data structures used by both host and device

struct RayGenData {
    float3 origin;
    float3 direction;
};

struct HitGroupData {
    float3 color;
};

struct RayResult {
    int ray_id;        // Original ray index for mapping back to point
    int polygon_index; // Polygon that was hit (-1 for miss)
};

struct LaunchParams {
    RayGenData ray_gen;
    HitGroupData hit_group;
    OptixTraversableHandle handle;
    RayResult* result;

    float3* ray_origins;
    uint3* indices;
    int* triangle_to_object;
    int num_rays;
    
    // For direct GPU-side compaction during ray tracing
    RayResult* compact_result;  // Output buffer for hits only
    int* hit_counter;           // Atomic counter for compact array index
};