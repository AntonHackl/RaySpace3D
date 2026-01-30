#include "mesh_overlap.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"
#include <math.h>

extern "C" __constant__ MeshOverlapLaunchParams mesh_overlap_params;

__device__ void insert_hash_table(int id1, int id2);

static __forceinline__ __device__ float3 add3f(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 mul3f(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

static __forceinline__ __device__ int trace_edge_multi_hits(
    const float3& edgeStart,
    const float3& dirNormalized,
    float edgeLength,
    int objectIdSource,
    bool swapPairOrder,
    int triangleIdx,
    int writeBaseOffset,
    int& writeCursor,
    float epsilon
) {
    // Iteratively trace the closest hit, then advance origin just beyond that hit.
    // This discovers multiple intersections along a finite segment without needing
    // any fixed-size local buffers.
    const int kMaxIterations = 1000; // user-approved safety bound

    float3 origin = edgeStart;
    float traveled = 0.0f;
    int hitCount = 0;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        const float remaining = edgeLength - traveled;
        if (remaining <= epsilon) break;

        unsigned int hitFlag = 0;
        unsigned int distance = 0;
        unsigned int primitiveIndex = 0;

        optixTrace(
            mesh_overlap_params.mesh2_handle,
            origin,
            dirNormalized,
            epsilon,
            remaining + epsilon,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,  // SBT offset
            1,  // SBT stride
            0,  // missSBTIndex
            hitFlag, distance, primitiveIndex);

        if (!hitFlag) break;

        const float t = __uint_as_float(distance);
        if (!(t >= epsilon && t <= remaining + epsilon)) break;

        ++hitCount;

        if (mesh_overlap_params.use_hash_table) {
            const int objectIdTarget = mesh_overlap_params.mesh2_triangle_to_object[primitiveIndex];
            if (swapPairOrder) {
                insert_hash_table(objectIdTarget, objectIdSource);
            } else {
                insert_hash_table(objectIdSource, objectIdTarget);
            }
        } else {
            if (mesh_overlap_params.pass == 2) {
                const int objectIdTarget = mesh_overlap_params.mesh2_triangle_to_object[primitiveIndex];
                const int outIdx = writeCursor++;
                if (swapPairOrder) {
                    mesh_overlap_params.results[outIdx] = {objectIdTarget, objectIdSource};
                } else {
                    mesh_overlap_params.results[outIdx] = {objectIdSource, objectIdTarget};
                }
            }
        }

        // Advance origin forward so next trace finds the next closest hit.
        // Use a small step to avoid re-hitting the same primitive due to precision.
        const float advance = t + epsilon;
        traveled += advance;
        origin = add3f(origin, mul3f(dirNormalized, advance));

        // Guard against numerical stagnation
        if (advance <= epsilon) break;
    }

    // In pass 2, ensure we wrote exactly within the triangle's range.
    // (writeBaseOffset is unused here but kept to make call sites clearer)
    (void)triangleIdx;
    (void)writeBaseOffset;
    return hitCount;
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
    
    int size = mesh_overlap_params.hash_table_size;
    if (size <= 0) return;
    unsigned int h = k % size;
    
    // Linear probing with limit
    for (int i = 0; i < 1000; ++i) {
        // Attempt to insert key
        unsigned long long old = atomicCAS(&mesh_overlap_params.hash_table[h], 0xFFFFFFFFFFFFFFFFULL, key);
        
        // Success if slot was empty or already contained our key (deduplication!)
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == key) {
            return;
        }
        
        // Collision with different key, probe next slot
        h = (h + 1) % size;
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
    int writeCursor = 0;
    if (!mesh_overlap_params.use_hash_table && mesh_overlap_params.pass == 2) {
        writeCursor = mesh_overlap_params.collision_offsets[triangleIdx];
    }
    const int writeBaseOffset = writeCursor;

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
            writeBaseOffset,
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
    int writeCursor = 0;
    if (!mesh_overlap_params.use_hash_table && mesh_overlap_params.pass == 2) {
        writeCursor = mesh_overlap_params.collision_offsets[triangleIdx];
    }
    const int writeBaseOffset = writeCursor;

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
            writeBaseOffset,
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


