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

    unsigned int hitFlag = 0;
    unsigned int distance = __float_as_uint(1e16f);
    unsigned int triangleIndex = 0;
    unsigned int isInside = 0;

    optixTrace(
        params.handle,
        orig,
        dir,
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        /*SBT offset*/ 0,
        /*SBT stride*/ 1,
        /*missSBTIndex*/ 0,
        /*payload*/ hitFlag, distance, triangleIndex, isInside);

    float t = __uint_as_float(distance);
    int hit = hitFlag && isInside;
    
    // Write only minimal result fields (ray_id, polygon_index)
    params.result[ray_id].ray_id = ray_id;
    params.result[ray_id].polygon_index = hit ? static_cast<int>(params.triangle_to_object[triangleIndex]) : -1;
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
    
    const unsigned int isInside = optixIsBackFaceHit();
    
    optixSetPayload_0(1);
    optixSetPayload_1(__float_as_uint(t));
    optixSetPayload_2(triangleIndex);
    optixSetPayload_3(isInside);
    
    // If compact output is enabled and this is a hit from inside (backface)
    if (isInside && params.compact_result != nullptr && params.hit_counter != nullptr) {
        // Get ray information
        const uint3 idx = optixGetLaunchIndex();
        const uint3 dim = optixGetLaunchDimensions();
        const int ray_id = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
        
        const float3 orig = params.ray_origins[ray_id];
        const float3 dir = make_float3(0.0f, 0.0f, 1.0f);
        
        // Atomically get next available slot in compact array
        int compact_idx = atomicAdd(params.hit_counter, 1);
        
        // Write minimal hit data directly to compact array
        params.compact_result[compact_idx].ray_id = ray_id;
        params.compact_result[compact_idx].polygon_index = static_cast<int>(params.triangle_to_object[triangleIndex]);
    }
} 