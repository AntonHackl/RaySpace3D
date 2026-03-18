#include "mesh_intersection.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"
#include <math.h>
#include "optix_common_shaders.cuh"

extern "C" __constant__ MeshIntersectionLaunchParams mesh_intersection_params;

__device__ void insert_hash_table(int id1, int id2);

__device__ __forceinline__ void update_max_u32(unsigned int* addr, unsigned int value) {
    atomicMax(addr, value);
}

static __forceinline__ __device__ int trace_edge_multi_hits(
    const float3& edgeStart,
    const float3& dirNormalized,
    float edgeLength,
    int objectIdSource,
    bool swapPairOrder,
    long long& writeCursor,
    float epsilon
) {
    int kMaxIterations = mesh_intersection_params.overlap_max_iterations;
    if (kMaxIterations <= 0) {
        kMaxIterations = 100;
    }
    float current_t_min = epsilon;
    int hitsFound = 0;
    int iterations = 0;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        iterations++;
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

    if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
        atomicAdd(&mesh_intersection_params.profiling_stats->overlap_trace_calls, 1ULL);
        atomicAdd(&mesh_intersection_params.profiling_stats->overlap_iterations_total, static_cast<unsigned long long>(iterations));
        atomicAdd(&mesh_intersection_params.profiling_stats->overlap_hits_total, static_cast<unsigned long long>(hitsFound));
        update_max_u32(&mesh_intersection_params.profiling_stats->overlap_max_iterations_per_trace, static_cast<unsigned int>(iterations));
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

__device__ int collect_containing_target_objects(const float3& point, int* outObjectIds, int maxOutObjects) {
    constexpr int kMaxCandidates = 32;
    int kMaxIterations = mesh_intersection_params.containment_max_iterations;
    if (kMaxIterations <= 0) {
        kMaxIterations = 2048;
    }
    const float epsilon = 1e-6f;
    const float tmax = 1e10f;

    int candidateObjects[kMaxCandidates];
    unsigned char parity[kMaxCandidates];
    int candidateCount = 0;

    float current_t_min = 1e-4f;
    float lastT = -1.0f;
    unsigned int lastTriangle = 0xFFFFFFFFu;
    int iterations = 0;
    int hitsFound = 0;
    int sameHitSuppressed = 0;
    int candidateAdditions = 0;
    int candidateToggles = 0;
    int candidateOverflow = 0;

    for (int i = 0; i < kMaxIterations; ++i) {
        iterations++;
        if (current_t_min > tmax) break;

        unsigned int hitFlag = 0;
        unsigned int distance = 0;
        unsigned int triangleIndex = 0;

        optixTrace(
            mesh_intersection_params.mesh2_handle,
            point,
            make_float3(0.0f, 0.0f, 1.0f),
            current_t_min,
            tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,
            1,
            0,
            hitFlag, distance, triangleIndex);

        if (!hitFlag) {
            break;
        }

        hitsFound++;

        const float t = __uint_as_float(distance);
        if (t < current_t_min || t > tmax) {
            break;
        }

        const bool sameHit = (triangleIndex == lastTriangle) && (lastT >= 0.0f) && (fabsf(t - lastT) <= 1e-5f);
        if (!sameHit) {
            const int objId = mesh_intersection_params.mesh2_triangle_to_object[triangleIndex];

            int pos = -1;
            for (int j = 0; j < candidateCount; ++j) {
                if (candidateObjects[j] == objId) {
                    pos = j;
                    break;
                }
            }

            if (pos >= 0) {
                parity[pos] ^= 1;
                candidateToggles++;
            } else if (candidateCount < kMaxCandidates) {
                candidateObjects[candidateCount] = objId;
                parity[candidateCount] = 1;
                candidateCount++;
                candidateAdditions++;
            } else {
                candidateOverflow++;
            }
        } else {
            sameHitSuppressed++;
        }

        lastT = t;
        lastTriangle = triangleIndex;

        float next_t_min = t + epsilon;
        if (next_t_min <= t) {
            next_t_min = t + 2e-4f;
        }
        if (next_t_min <= current_t_min) {
            break;
        }
        current_t_min = next_t_min;
    }

    int outCount = 0;
    for (int i = 0; i < candidateCount; ++i) {
        if ((parity[i] & 1) != 0) {
            if (outCount < maxOutObjects) {
                outObjectIds[outCount] = candidateObjects[i];
            }
            outCount++;
        }
    }

    if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_rays_total, 1ULL);
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_iterations_total, static_cast<unsigned long long>(iterations));
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_hits_total, static_cast<unsigned long long>(hitsFound));
        update_max_u32(&mesh_intersection_params.profiling_stats->containment_max_iterations_per_ray, static_cast<unsigned int>(iterations));
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_same_hit_suppressed, static_cast<unsigned long long>(sameHitSuppressed));
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_candidate_additions, static_cast<unsigned long long>(candidateAdditions));
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_candidate_toggles, static_cast<unsigned long long>(candidateToggles));
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_candidate_overflow, static_cast<unsigned long long>(candidateOverflow));
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_targets_total, static_cast<unsigned long long>((outCount > maxOutObjects) ? maxOutObjects : outCount));
    }

    return (outCount > maxOutObjects) ? maxOutObjects : outCount;
}

extern "C" __global__ void __raygen__mesh_overlap() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int edgeIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (edgeIdx >= mesh_intersection_params.num_edges) {
        return;
    }

    const float3 edgeStart = mesh_intersection_params.edge_starts[edgeIdx];
    const float3 edgeEnd = mesh_intersection_params.edge_ends[edgeIdx];
    const int numSourceObjects = mesh_intersection_params.edge_source_object_counts[edgeIdx];
    const int sourceOffset = mesh_intersection_params.edge_source_object_offsets[edgeIdx];
    const int* sourceObjectIds = &mesh_intersection_params.edge_source_objects[sourceOffset];
    
    const float epsilon = 1e-6f;
    int totalHits = 0;
    long long writeCursor = 0;
    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 2) {
        writeCursor = mesh_intersection_params.collision_offsets[edgeIdx];
    }

    float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x,
                                 edgeEnd.y - edgeStart.y,
                                 edgeEnd.z - edgeStart.z);
    float edgeLength = distance3f(edgeStart, edgeEnd);

    if (edgeLength >= epsilon) {
        float3 normalizedDir = normalize3f(edgeDir);
        for (int i = 0; i < numSourceObjects; ++i) {
            const int sourceObjectId = sourceObjectIds[i];
            int hitsFound = trace_edge_multi_hits(
                edgeStart,
                normalizedDir,
                edgeLength,
                sourceObjectId,
                mesh_intersection_params.swap_result_ids != 0,
                writeCursor,
                epsilon);
            totalHits += hitsFound;
        }
    }

    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[edgeIdx] = totalHits;
    }
}

extern "C" __global__ void __raygen__mesh_containment() {
    const uint3 idx = optixGetLaunchIndex();
    const int sourceObjectId = static_cast<int>(idx.x);
    if (sourceObjectId >= mesh_intersection_params.mesh1_num_objects) {
        return;
    }

    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[sourceObjectId] = 0;
    }

    const int firstTri = mesh_intersection_params.first_triangle_index_per_object[sourceObjectId];
    if (firstTri < 0 || firstTri >= mesh_intersection_params.mesh1_num_triangles) {
        return;
    }

    const uint3 triIndices = mesh_intersection_params.mesh1_indices[firstTri];
    float3 queryPoint = mesh_intersection_params.mesh1_vertices[triIndices.x];
    if (mesh_intersection_params.containment_query_point_mode == 1) {
        const float3 v0 = mesh_intersection_params.mesh1_vertices[triIndices.x];
        const float3 v1 = mesh_intersection_params.mesh1_vertices[triIndices.y];
        const float3 v2 = mesh_intersection_params.mesh1_vertices[triIndices.z];
        queryPoint = make_float3(
            (v0.x + v1.x + v2.x) / 3.0f,
            (v0.y + v1.y + v2.y) / 3.0f,
            (v0.z + v1.z + v2.z) / 3.0f
        );
    }
    int targetObjects[32];
    const int numTargets = collect_containing_target_objects(queryPoint, targetObjects, 32);
    if (numTargets <= 0) {
        return;
    }

    if (mesh_intersection_params.use_hash_table) {
        for (int i = 0; i < numTargets; ++i) {
            if (mesh_intersection_params.swap_result_ids == 0) {
                insert_hash_table(sourceObjectId, targetObjects[i]);
            } else {
                insert_hash_table(targetObjects[i], sourceObjectId);
            }
        }
    } else if (mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[sourceObjectId] = numTargets;
    } else if (mesh_intersection_params.pass == 2) {
        const long long outIdx = mesh_intersection_params.collision_offsets[sourceObjectId];
        for (int i = 0; i < numTargets; ++i) {
            if (mesh_intersection_params.swap_result_ids == 0) {
                mesh_intersection_params.results[outIdx + i] = {sourceObjectId, targetObjects[i]};
            } else {
                mesh_intersection_params.results[outIdx + i] = {targetObjects[i], sourceObjectId};
            }
        }
    }
}
