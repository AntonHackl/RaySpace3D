#include <optix_device.h>
#include <optix.h>
#include <cuda_runtime.h>
#include "common.h"

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
    unsigned int firstTriangleIndex = 0;

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

        if (!hitFlag) {
            break;
        }

        float t = __uint_as_float(distance);
        if (t < epsilon || t >= max_distance) {
            break;
        }

        if (intersection_count == 0) {
            firstTriangleIndex = triangleIndex;
        }

        intersection_count++;
        t_min = t;

        if (t_min >= max_distance) {
            break;
        }
    }

    bool isInside = (intersection_count % 2 == 1);
    
    params.result[ray_id].ray_id = ray_id;
    params.result[ray_id].polygon_index = isInside ? static_cast<int>(params.triangle_to_object[firstTriangleIndex]) : -1;
    
    if (isInside && params.compact_result != nullptr && params.hit_counter != nullptr) {
        int compact_idx = atomicAdd(params.hit_counter, 1);
        params.compact_result[compact_idx].ray_id = ray_id;
        params.compact_result[compact_idx].polygon_index = static_cast<int>(params.triangle_to_object[firstTriangleIndex]);
    }
}

extern "C" __global__ void __miss__ms()
{
}

extern "C" __global__ void __anyhit__ah()
{
}

extern "C" __global__ void __closesthit__ch()
{
    const float t = optixGetRayTmax();
    const unsigned int triangleIndex = optixGetPrimitiveIndex();
    
    optixSetPayload_0(1);
    optixSetPayload_1(__float_as_uint(t));
    optixSetPayload_2(triangleIndex);
}

