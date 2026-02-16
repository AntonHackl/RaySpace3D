#include "mesh_containment.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"
#include <math.h>

extern "C" __constant__ MeshContainmentLaunchParams containment_params;

// -----------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------

__device__ float cont_distance3f(const float3& a, const float3& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ float3 cont_normalize3f(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8f) return make_float3(0.0f, 0.0f, 0.0f);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

// -----------------------------------------------------------------------
// Hash table helpers
// -----------------------------------------------------------------------

__device__ unsigned long long cont_hash_key(unsigned long long key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

__device__ unsigned long long cont_pack_pair(int id1, int id2) {
    return (static_cast<unsigned long long>(id1) << 32) |
            static_cast<unsigned long long>(id2);
}

// Insert (id1, id2) into a hash table. id1 = A obj, id2 = B obj.
__device__ void cont_insert_table(unsigned long long* table, int size,
                                  int id1, int id2) {
    unsigned long long key = cont_pack_pair(id1, id2);
    unsigned int h = cont_hash_key(key) % size;
    for (int i = 0; i < 1000; ++i) {
        unsigned long long old = atomicCAS(&table[h], 0xFFFFFFFFFFFFFFFFULL, key);
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == key) return;
        h = (h + 1) % size;
    }
}

// Lookup whether (id1, id2) exists in a hash table.
__device__ bool cont_lookup_table(const unsigned long long* table, int size,
                                  int id1, int id2) {
    unsigned long long key = cont_pack_pair(id1, id2);
    unsigned int h = cont_hash_key(key) % size;
    for (int i = 0; i < 1000; ++i) {
        unsigned long long val = table[h];
        if (val == key) return true;
        if (val == 0xFFFFFFFFFFFFFFFFULL) return false;
        h = (h + 1) % size;
    }
    return false;
}

// -----------------------------------------------------------------------
// Phase 1 – Edge intersection check
// Launched once for B→A (swap_ids=0) and once for A→B (swap_ids=1).
// Each thread handles one source triangle (3 edge rays).
// -----------------------------------------------------------------------

extern "C" __global__ void __raygen__check_edges() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int triangleIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;

    if (triangleIdx >= containment_params.src_num_triangles) return;

    uint3 tri = containment_params.src_indices[triangleIdx];
    float3 v0 = containment_params.src_vertices[tri.x];
    float3 v1 = containment_params.src_vertices[tri.y];
    float3 v2 = containment_params.src_vertices[tri.z];

    int src_obj = containment_params.src_triangle_to_object[triangleIdx];

    float3 edgeStarts[3] = {v0, v1, v2};
    float3 edgeEnds[3]   = {v1, v2, v0};

    const float epsilon = 1e-6f;

    for (int e = 0; e < 3; ++e) {
        float3 dir = make_float3(edgeEnds[e].x - edgeStarts[e].x,
                                 edgeEnds[e].y - edgeStarts[e].y,
                                 edgeEnds[e].z - edgeStarts[e].z);
        float edgeLen = cont_distance3f(edgeStarts[e], edgeEnds[e]);
        if (edgeLen < epsilon) continue;

        float3 normDir = cont_normalize3f(dir);

        unsigned int hitFlag      = 0;
        unsigned int distBits     = __float_as_uint(edgeLen + epsilon);
        unsigned int triangleIndex = 0;

        optixTrace(
            containment_params.target_handle,
            edgeStarts[e],
            normDir,
            epsilon,
            edgeLen + epsilon,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            hitFlag, distBits, triangleIndex);

        float t = __uint_as_float(distBits);
        if (hitFlag && t >= epsilon && t <= edgeLen + epsilon) {
            int hit_obj = containment_params.target_triangle_to_object[triangleIndex];

            // Always store as (A_obj, B_obj) regardless of direction
            if (containment_params.swap_ids == 0) {
                // B→A: src=B, hit=A  → (A=hit, B=src)
                cont_insert_table(containment_params.intersection_hash_table,
                                  containment_params.intersection_hash_table_size,
                                  hit_obj, src_obj);
            } else {
                // A→B: src=A, hit=B  → (A=src, B=hit)
                cont_insert_table(containment_params.intersection_hash_table,
                                  containment_params.intersection_hash_table_size,
                                  src_obj, hit_obj);
            }
        }
    }
}

// -----------------------------------------------------------------------
// Phase 2 – Point-in-mesh test
// One thread per B object.  Casts a single ray from the first vertex
// of each B object through A's acceleration structure.  Counts hits
// per A-object (parity/odd-even rule).  If odd AND pair is NOT in the
// intersection table → containment.
// -----------------------------------------------------------------------

#define MAX_UNIQUE_A_OBJECTS 512

extern "C" __global__ void __raygen__point_in_mesh() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int b_obj = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;

    if (b_obj >= containment_params.b_num_objects) return;

    float3 origin = containment_params.b_first_vertices[b_obj];
    float3 dir    = make_float3(0.0f, 0.0f, 1.0f);   // +Z ray

    // Local parity tracking per A-object
    int a_ids[MAX_UNIQUE_A_OBJECTS];
    int a_par[MAX_UNIQUE_A_OBJECTS];   // 0 or 1 (toggled on each hit)
    int num_unique = 0;

    float tmin = 1e-4f;
    float tmax = 1e10f;

    for (int iter = 0; iter < 2000; ++iter) {
        unsigned int hitFlag      = 0;
        unsigned int distBits     = 0;
        unsigned int triangleIndex = 0;

        optixTrace(
            containment_params.target_handle,
            origin,
            dir,
            tmin,
            tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            hitFlag, distBits, triangleIndex);

        if (!hitFlag) break;

        float t = __uint_as_float(distBits);
        if (t >= tmax) break;

        int a_obj = containment_params.target_triangle_to_object[triangleIndex];

        // Update parity for this A object
        bool found = false;
        for (int i = 0; i < num_unique; ++i) {
            if (a_ids[i] == a_obj) {
                a_par[i] ^= 1;
                found = true;
                break;
            }
        }
        if (!found && num_unique < MAX_UNIQUE_A_OBJECTS) {
            a_ids[num_unique] = a_obj;
            a_par[num_unique] = 1;   // first hit → parity 1 (inside)
            num_unique++;
        }

        tmin = t + 1e-4f;
    }

    // Emit containment pairs
    for (int i = 0; i < num_unique; ++i) {
        if (a_par[i] == 1) {
            // B vertex is inside A object – check it's not an intersection pair
            if (!cont_lookup_table(containment_params.intersection_hash_table,
                                   containment_params.intersection_hash_table_size,
                                   a_ids[i], b_obj)) {
                cont_insert_table(containment_params.containment_hash_table,
                                  containment_params.containment_hash_table_size,
                                  a_ids[i], b_obj);
            }
        }
    }
}

// -----------------------------------------------------------------------
// Miss / AnyHit / ClosestHit  (shared by all raygen programs)
// -----------------------------------------------------------------------

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __anyhit__ah() {
}

extern "C" __global__ void __closesthit__ch() {
    const float t = optixGetRayTmax();
    const unsigned int triangleIndex = optixGetPrimitiveIndex();

    optixSetPayload_0(1);
    optixSetPayload_1(__float_as_uint(t));
    optixSetPayload_2(triangleIndex);
}
