#include "mesh_intersection.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"
#include <math.h>
#include "optix_common_shaders.cuh"

extern "C" __constant__ MeshIntersectionLaunchParams mesh_intersection_params;

__device__ void insert_hash_table(int id1, int id2);

static __forceinline__ __device__ int trace_edge_multi_hits(
    const float3& edgeStart,
    const float3& dirNormalized,
    float edgeLength,
    int objectIdSource,
    bool swapPairOrder,
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
        unsigned int triangleIndex = 0;

        optixTrace(
            mesh_intersection_params.mesh2_handle,
            edgeStart,
            dirNormalized,
            current_t_min,
            edgeLength + epsilon,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            hitFlag, distance, triangleIndex);

        if (!hitFlag) break;

        const float t = __uint_as_float(distance);
        if (t > edgeLength + epsilon) break;

        const int objectIdTarget = mesh_intersection_params.mesh2_triangle_to_object[triangleIndex];
        hitsFound++;

        if (mesh_intersection_params.use_hash_table) {
            if (swapPairOrder) {
                insert_hash_table(objectIdTarget, objectIdSource);
            } else {
                insert_hash_table(objectIdSource, objectIdTarget);
            }
        } else if (mesh_intersection_params.pass == 2) {
            const long long outIdx = writeCursor++;
            if (swapPairOrder) {
                mesh_intersection_params.results[outIdx] = {objectIdTarget, objectIdSource};
            } else {
                mesh_intersection_params.results[outIdx] = {objectIdSource, objectIdTarget};
            }
        }

        float next_t_min = t + epsilon;
        if (next_t_min <= t) {
            next_t_min = t + 2e-4f;
        }
        current_t_min = next_t_min;
    }

    return hitsFound;
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

__device__ void insert_hash_table(int id1, int id2) {
    unsigned long long key = (static_cast<unsigned long long>(id1) << 32) | static_cast<unsigned long long>(id2);
    
    unsigned long long k = key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    
    int size = mesh_intersection_params.hash_table_size;
    if (size <= 0) return;
    unsigned int h = k % size;
    
    for (int i = 0; i < 1000; ++i) {
        unsigned long long old = atomicCAS(&mesh_intersection_params.hash_table[h], 0xFFFFFFFFFFFFFFFFULL, key);
        
        // Success if slot was empty or already contained our key (deduplication!)
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == key) {
            return;
        }
        
        h = (h + 1) % size;
    }
}

__device__ void mark_object_tested(int objectId) {
    if (objectId >= 0 && objectId < mesh_intersection_params.mesh1_num_objects) {
        mesh_intersection_params.object_tested[objectId] = 1;
    }
}

// Check if a point is inside a mesh using ray casting (odd-even test)
__device__ bool point_inside_mesh(const float3& point, OptixTraversableHandle mesh_handle) {
    float3 rayOrigin = point;
    float3 rayDir = make_float3(0.0f, 0.0f, 1.0f);
    
    int hitCount = 0;
    float tmin = 1e-4f;
    float tmax = 1e10f;
    
    for (int i = 0; i < 1000; ++i) {
        unsigned int hitFlag = 0;
        unsigned int distance = 0;
        unsigned int triangleIndex = 0;
        
        optixTrace(
            mesh_handle,
            rayOrigin,
            rayDir,
            tmin,
            tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,  // SBT offset
            1,  // SBT stride
            0,  // missSBTIndex
            hitFlag, distance, triangleIndex);
        
        float t = __uint_as_float(distance);
        int continueLoop = hitFlag && (t < tmax);
        hitCount += hitFlag;
        tmin = t + 1e-4f;
        
        if (!continueLoop) {
            break;
        }
    }
    
    // Odd count means inside
    return (hitCount & 1);
}

// Mesh1 to Mesh2: Edge ray casting with containment fallback
extern "C" __global__ void __raygen__mesh1_to_mesh2() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int triangleIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (triangleIdx >= mesh_intersection_params.mesh1_num_triangles) {
        return;
    }
    
    uint3 triIndices = mesh_intersection_params.mesh1_indices[triangleIdx];
    
    float3 v0 = mesh_intersection_params.mesh1_vertices[triIndices.x];
    float3 v1 = mesh_intersection_params.mesh1_vertices[triIndices.y];
    float3 v2 = mesh_intersection_params.mesh1_vertices[triIndices.z];
    
    int objectIdMesh1 = mesh_intersection_params.mesh1_triangle_to_object[triangleIdx];
    
    float3 edgeStarts[3] = {v0, v1, v2};
    float3 edgeEnds[3] = {v1, v2, v0};
    
    const float epsilon = 1e-6f;
    bool edgeHitFound = false;
    int totalHits = 0;
    long long writeCursor = 0;
    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 2) {
        writeCursor = mesh_intersection_params.collision_offsets[triangleIdx];
    }
    
    // Test all edges first
    for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx) {
        float3 edgeStart = edgeStarts[edgeIdx];
        float3 edgeEnd = edgeEnds[edgeIdx];
        
        float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x, 
                                     edgeEnd.y - edgeStart.y, 
                                     edgeEnd.z - edgeStart.z);
        float edgeLength = distance3f(edgeStart, edgeEnd);
        
        if (edgeLength < epsilon) {
            continue;
        }

        float3 normalizedDir = normalize3f(edgeDir);
        int hitsFound = trace_edge_multi_hits(
            edgeStart,
            normalizedDir,
            edgeLength,
            objectIdMesh1,
            false,
            writeCursor,
            epsilon);

        if (hitsFound > 0) {
            edgeHitFound = true;
            mark_object_tested(objectIdMesh1);
        }
        totalHits += hitsFound;
    }
    
    // Containment fallback: if no edge hits found, test first vertex
    // Only test once per object (first triangle of that object)
    if (!edgeHitFound && triangleIdx == 0) {
        if (point_inside_mesh(v0, mesh_intersection_params.mesh2_handle)) {
            // Object is fully contained - record intersection with all mesh2 objects
            // This is conservative but correct
            if (mesh_intersection_params.use_hash_table) {
                for (int objId2 = 0; objId2 < mesh_intersection_params.mesh2_num_objects; ++objId2) {
                    insert_hash_table(objectIdMesh1, objId2);
                }
            } else if (mesh_intersection_params.pass == 1) {
                totalHits += mesh_intersection_params.mesh2_num_objects;
            } else if (mesh_intersection_params.pass == 2) {
                for (int objId2 = 0; objId2 < mesh_intersection_params.mesh2_num_objects; ++objId2) {
                    mesh_intersection_params.results[writeCursor++] = {objectIdMesh1, objId2};
                }
            }
        }
    }

    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[triangleIdx] = totalHits;
    }
}

// Mesh2 to Mesh1: Same logic but roles reversed
extern "C" __global__ void __raygen__mesh2_to_mesh1() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int triangleIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (triangleIdx >= mesh_intersection_params.mesh1_num_triangles) {
        return;
    }
    
    uint3 triIndices = mesh_intersection_params.mesh1_indices[triangleIdx];
    
    float3 v0 = mesh_intersection_params.mesh1_vertices[triIndices.x];
    float3 v1 = mesh_intersection_params.mesh1_vertices[triIndices.y];
    float3 v2 = mesh_intersection_params.mesh1_vertices[triIndices.z];
    
    int objectIdMesh2 = mesh_intersection_params.mesh1_triangle_to_object[triangleIdx];
    
    float3 edgeStarts[3] = {v0, v1, v2};
    float3 edgeEnds[3] = {v1, v2, v0};
    
    const float epsilon = 1e-6f;
    bool edgeHitFound = false;
    int totalHits = 0;
    long long writeCursor = 0;
    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 2) {
        writeCursor = mesh_intersection_params.collision_offsets[triangleIdx];
    }
    
    // Test all edges first
    for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx) {
        float3 edgeStart = edgeStarts[edgeIdx];
        float3 edgeEnd = edgeEnds[edgeIdx];
        
        float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x, 
                                     edgeEnd.y - edgeStart.y, 
                                     edgeEnd.z - edgeStart.z);
        float edgeLength = distance3f(edgeStart, edgeEnd);
        
        if (edgeLength < epsilon) {
            continue;
        }

        float3 normalizedDir = normalize3f(edgeDir);
        int hitsFound = trace_edge_multi_hits(
            edgeStart,
            normalizedDir,
            edgeLength,
            objectIdMesh2,
            true,
            writeCursor,
            epsilon);

        if (hitsFound > 0) {
            edgeHitFound = true;
        }
        totalHits += hitsFound;
    }
    
    // Containment fallback for mesh2 objects
    if (!edgeHitFound && triangleIdx == 0) {
        if (point_inside_mesh(v0, mesh_intersection_params.mesh2_handle)) {
            if (mesh_intersection_params.use_hash_table) {
                for (int objId1 = 0; objId1 < mesh_intersection_params.mesh1_num_objects; ++objId1) {
                    insert_hash_table(objId1, objectIdMesh2);
                }
            } else if (mesh_intersection_params.pass == 1) {
                totalHits += mesh_intersection_params.mesh1_num_objects;
            } else if (mesh_intersection_params.pass == 2) {
                for (int objId1 = 0; objId1 < mesh_intersection_params.mesh1_num_objects; ++objId1) {
                    mesh_intersection_params.results[writeCursor++] = {objId1, objectIdMesh2};
                }
            }
        }
    }

    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[triangleIdx] = totalHits;
    }
}

