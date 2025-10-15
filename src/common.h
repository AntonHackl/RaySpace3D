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
    int hit;           
    float t;            
    float3 hit_point;  
    float2 barycentrics;
    int polygon_index;
};

struct LaunchParams {
    RayGenData ray_gen;
    HitGroupData hit_group;
    OptixTraversableHandle handle;
    RayResult* result;

    float3* ray_origins;
    int* triangle_to_polygon;
    int num_rays;
};