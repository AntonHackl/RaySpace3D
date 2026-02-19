#include "mesh_overlap.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"
#include <math.h>

extern "C" __constant__ MeshOverlapLaunchParams mesh_overlap_params;

__device__ void insert_hash_table(int id1, int id2);



static __forceinline__ __device__ int trace_edge_multi_hits(
    const float3& edgeStart,
    const float3& dirNormalized,
    float edgeLength,
    int objectIdSource,
    bool swapPairOrder,
    int triangleIdx,
    long long& writeCursor,
    float epsilon
) {
    const int kMaxIterations = 100;
    float current_t_min = epsilon;
    int hitsFound = 0;
    
    for (int iter = 0; iter < kMaxIterations; ++iter) {
        if (current_t_min > edgeLength + epsilon) break;
        
        unsigned int hitFlag = 0;
        unsigned int distance = 0;
        unsigned int primitiveIndex = 0;

        optixTrace(
            mesh_overlap_params.mesh2_handle,
            edgeStart,
            dirNormalized,
            current_t_min,
            edgeLength + epsilon,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0,
            hitFlag, distance, primitiveIndex);

        if (!hitFlag) break;
        
        const float t = __uint_as_float(distance);
        if (t > edgeLength + epsilon) break;

        const int objectIdTarget = mesh_overlap_params.mesh2_triangle_to_object[primitiveIndex];
        hitsFound++;
        
        if (mesh_overlap_params.use_hash_table) {
            if (swapPairOrder) insert_hash_table(objectIdTarget, objectIdSource);
            else insert_hash_table(objectIdSource, objectIdTarget);
        } else if (mesh_overlap_params.pass == 2) {
            const long long outIdx = writeCursor++;
            if (swapPairOrder) mesh_overlap_params.results[outIdx] = {objectIdTarget, objectIdSource};
            else mesh_overlap_params.results[outIdx] = {objectIdSource, objectIdTarget};
        }
        
        float next_t_min = t + epsilon;
        if (next_t_min <= t) {
            next_t_min = t + 2e-4f;
        }
        current_t_min = next_t_min;
    }

    return hitsFound;
}

__device__ void insert_hash_table(int id1, int id2) {
    // Pack 2 integers into a 64-bit key
    unsigned long long key = (static_cast<unsigned long long>(id1) << 32) | static_cast<unsigned long long>(id2);
    
    // Simple hash function to distribute keys
    unsigned long long k = key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    
    unsigned long long size = mesh_overlap_params.hash_table_size;
    if (size <= 0) return;

    // Use bitwise AND for power-of-two sizes (avoids expensive integer division on GPU)
    unsigned int h;
    if (mesh_overlap_params.use_bitwise_hash) {
        h = (unsigned int)(k & (size - 1));
    } else {
        h = (unsigned int)(k % size);
    }

    // Linear probing with limit
    for (int i = 0; i < 5000; ++i) {
        // Attempt to insert key
        unsigned long long old = atomicCAS(&mesh_overlap_params.hash_table[h], 0xFFFFFFFFFFFFFFFFULL, key);

        // Success if slot was empty or already contained our key (deduplication!)
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == key) {
            return;
        }

        // Collision: advance to next slot
        if (mesh_overlap_params.use_bitwise_hash) {
            h = (h + 1) & (unsigned int)(size - 1);
        } else {
            h = (h + 1) % (unsigned int)size;
        }
    }
}

__device__ float distance3f(const float3& a, const float3& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ float3 normalize3f(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(v.x / len, v.y / len, v.z / len);
}

extern "C" __global__ void __raygen__mesh1_to_mesh2() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int triangleIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (triangleIdx >= mesh_overlap_params.mesh1_num_triangles) {
        return;
    }
    
    uint3 triIndices = mesh_overlap_params.mesh1_indices[triangleIdx];
    
    float3 v0 = mesh_overlap_params.mesh1_vertices[triIndices.x];
    float3 v1 = mesh_overlap_params.mesh1_vertices[triIndices.y];
    float3 v2 = mesh_overlap_params.mesh1_vertices[triIndices.z];
    
    int objectIdMesh1 = mesh_overlap_params.mesh1_triangle_to_object[triangleIdx];
    
    float3 edgeStarts[3] = {v0, v1, v2};
    float3 edgeEnds[3] = {v1, v2, v0};
    
    const float epsilon = 1e-6f;

    int totalHits = 0;
    long long writeCursor = 0;
    if (!mesh_overlap_params.use_hash_table && mesh_overlap_params.pass == 2) {
        writeCursor = mesh_overlap_params.collision_offsets[triangleIdx];
    }

    for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx) {
        const float3 edgeStart = edgeStarts[edgeIdx];
        const float3 edgeEnd = edgeEnds[edgeIdx];

        const float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x,
                                           edgeEnd.y - edgeStart.y,
                                           edgeEnd.z - edgeStart.z);
        const float edgeLength = distance3f(edgeStart, edgeEnd);
        if (edgeLength < epsilon) continue;

        const float3 normalizedDir = normalize3f(edgeDir);
        totalHits += trace_edge_multi_hits(
            edgeStart,
            normalizedDir,
            edgeLength,
            objectIdMesh1,
            false,
            triangleIdx,
            writeCursor,
            epsilon);
    }

    if (!mesh_overlap_params.use_hash_table && mesh_overlap_params.pass == 1) {
        mesh_overlap_params.collision_counts[triangleIdx] = totalHits;
    }
}

extern "C" __global__ void __raygen__mesh2_to_mesh1() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int triangleIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (triangleIdx >= mesh_overlap_params.mesh1_num_triangles) {
        return;
    }
    
    uint3 triIndices = mesh_overlap_params.mesh1_indices[triangleIdx];
    
    float3 v0 = mesh_overlap_params.mesh1_vertices[triIndices.x];
    float3 v1 = mesh_overlap_params.mesh1_vertices[triIndices.y];
    float3 v2 = mesh_overlap_params.mesh1_vertices[triIndices.z];
    
    // Note: in this raygen we iterate triangles of Mesh2 (uploaded as mesh1_*) against Mesh1 GAS.
    // To keep MeshOverlapResult ordering consistent as (mesh1_id, mesh2_id), we swap pair order on write.
    int objectIdMesh2 = mesh_overlap_params.mesh1_triangle_to_object[triangleIdx];
    
    float3 edgeStarts[3] = {v0, v1, v2};
    float3 edgeEnds[3] = {v1, v2, v0};
    
    const float epsilon = 1e-6f;

    int totalHits = 0;
    long long writeCursor = 0;
    if (!mesh_overlap_params.use_hash_table && mesh_overlap_params.pass == 2) {
        writeCursor = mesh_overlap_params.collision_offsets[triangleIdx];
    }

    for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx) {
        const float3 edgeStart = edgeStarts[edgeIdx];
        const float3 edgeEnd = edgeEnds[edgeIdx];

        const float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x,
                                           edgeEnd.y - edgeStart.y,
                                           edgeEnd.z - edgeStart.z);
        const float edgeLength = distance3f(edgeStart, edgeEnd);
        if (edgeLength < epsilon) continue;

        const float3 normalizedDir = normalize3f(edgeDir);
        totalHits += trace_edge_multi_hits(
            edgeStart,
            normalizedDir,
            edgeLength,
            objectIdMesh2,
            true,
            triangleIdx,
            writeCursor,
            epsilon);
    }

    if (!mesh_overlap_params.use_hash_table && mesh_overlap_params.pass == 1) {
        mesh_overlap_params.collision_counts[triangleIdx] = totalHits;
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


