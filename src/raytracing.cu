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
    unsigned int distance = __float_as_uint(0.0f);
    unsigned int triangleIndex = 0;

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
        /*payload*/ hitFlag, distance, triangleIndex);

    params.result[ray_id].hit = hitFlag;
    params.result[ray_id].t = __uint_as_float(distance);
    params.result[ray_id].polygon_index = static_cast<int>(params.triangle_to_polygon[triangleIndex]);

    float t = __uint_as_float(distance);
    params.result[ray_id].hit_point = make_float3(
        orig.x + t * dir.x,
        orig.y + t * dir.y,
        orig.z + t * dir.z
    );
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0); // hit = 0 (miss)
    optixSetPayload_1(__float_as_uint(0.0f)); // t = 0.0f
    optixSetPayload_2(0); // triangle index = 0
}

extern "C" __global__ void __closesthit__ch()
{
    const float2 bc = optixGetTriangleBarycentrics();
    const float t = optixGetRayTmax();
    const unsigned int triangleIndex = optixGetPrimitiveIndex();
    
    optixSetPayload_0(1); // hit = 1 (hit)
    optixSetPayload_1(__float_as_uint(t)); // distance to hit point
    optixSetPayload_2(triangleIndex); // triangle index
    
    params.result->barycentrics = bc;
} 