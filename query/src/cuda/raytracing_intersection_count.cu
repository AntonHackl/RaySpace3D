#include <optix_device.h>
#include <optix.h>
#include <cuda_runtime.h>
#include "common.h"
#include "optix_common_shaders.cuh"

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int ray_id = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (ray_id >= params.num_rays) {
        return;
    }
    
    const float3 orig = params.ray_origins[ray_id];
    const float3 dir = make_float3(0.0f, 0.0f, 1.0f);

    int intersection_count = 0;
    float t_min = 0.0f;
    const float epsilon = 1e-6f;
    const float max_distance = 1e16f;
    const int max_iterations = 10000;

    for (int iter = 0; iter < max_iterations; ++iter) {
        unsigned int hitFlag = 0;
        unsigned int distance = __float_as_uint(max_distance);
        unsigned int triangleIndex = 0;

        optixTrace(
            params.handle,
            orig,
            dir,
            t_min + epsilon,
            max_distance,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            /*SBT offset*/ 0,
            /*SBT stride*/ 1,
            /*missSBTIndex*/ 0,
            /*payload*/ hitFlag, distance, triangleIndex);

        float t = __uint_as_float(distance);
        int validHit = hitFlag && (t >= epsilon) && (t < max_distance);
        
        intersection_count += validHit;
        t_min = validHit ? t : t_min;

        if (!validHit) {
            break;
        }
    }

    bool isInside = (intersection_count & 1);
    
    if (isInside) {
        params.result[ray_id].ray_id = ray_id;
        params.result[ray_id].polygon_index = 0;
        
        if (params.compact_result != nullptr && params.hit_counter != nullptr) {
            int compact_idx = atomicAdd(params.hit_counter, 1);
            params.compact_result[compact_idx].ray_id = ray_id;
            params.compact_result[compact_idx].polygon_index = 0;
        }
    } else {
        params.result[ray_id].ray_id = ray_id;
        params.result[ray_id].polygon_index = -1;
    }
}

