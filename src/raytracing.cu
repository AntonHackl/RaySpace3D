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
    
    params.result[ray_id].hit = hit;
    params.result[ray_id].hit_count = hit;
    params.result[ray_id].t = hit ? t : 0.0f;
    params.result[ray_id].polygon_index = hit ? static_cast<int>(params.triangle_to_object[triangleIndex]) : -1;
    params.result[ray_id].hit_point = hit ? make_float3(orig.x + t * dir.x, orig.y + t * dir.y, orig.z + t * dir.z) : make_float3(0.0f, 0.0f, 0.0f);
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
    const float2 barycentrics = optixGetTriangleBarycentrics();
    
    const uint3 triangle = params.indices[triangleIndex];
    
    const float3 n0 = params.normals[triangle.x];
    const float3 n1 = params.normals[triangle.y];
    const float3 n2 = params.normals[triangle.z];
    
    const float w = 1.0f - barycentrics.x - barycentrics.y;
    const float3 interpolated_normal = make_float3(
        w * n0.x + barycentrics.x * n1.x + barycentrics.y * n2.x,
        w * n0.y + barycentrics.x * n1.y + barycentrics.y * n2.y,
        w * n0.z + barycentrics.x * n1.z + barycentrics.y * n2.z
    );
    
    const float3 ray_dir = optixGetWorldRayDirection();
    const float dot_product = interpolated_normal.x * ray_dir.x + interpolated_normal.y * ray_dir.y + interpolated_normal.z * ray_dir.z;
    
    const unsigned int isInside = dot_product > 0.0f;
    
    optixSetPayload_0(1);
    optixSetPayload_1(__float_as_uint(t));
    optixSetPayload_2(triangleIndex);
    optixSetPayload_3(isInside);
} 