#include "mesh_overlap.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"
#include <math.h>

extern "C" __constant__ MeshOverlapLaunchParams mesh_overlap_params;

// Helper function to compute distance between two points
__device__ float distance3f(const float3& a, const float3& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// Helper function to normalize a vector
__device__ float3 normalize3f(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(v.x / len, v.y / len, v.z / len);
}

// OptiX Raygen program: Process Mesh1 edges against Mesh2 acceleration structure
// Each thread processes one triangle, checking all 3 edges
extern "C" __global__ void __raygen__mesh1_to_mesh2() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int triangleIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (triangleIdx >= mesh_overlap_params.mesh1_num_triangles) {
        return;
    }
    
    // Get triangle indices
    uint3 triIndices = mesh_overlap_params.mesh1_indices[triangleIdx];
    
    // Get triangle vertices
    float3 v0 = mesh_overlap_params.mesh1_vertices[triIndices.x];
    float3 v1 = mesh_overlap_params.mesh1_vertices[triIndices.y];
    float3 v2 = mesh_overlap_params.mesh1_vertices[triIndices.z];
    
    // Get object ID for this triangle
    int objectIdMesh1 = mesh_overlap_params.mesh1_triangle_to_object[triangleIdx];
    
    // Define the 3 edges of the triangle
    float3 edgeStarts[3] = {v0, v1, v2};
    float3 edgeEnds[3] = {v1, v2, v0};
    
    const float epsilon = 1e-6f;
    
    // Check each edge
    for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx) {
        float3 edgeStart = edgeStarts[edgeIdx];
        float3 edgeEnd = edgeEnds[edgeIdx];
        
        // Compute edge direction and length
        float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x, 
                                     edgeEnd.y - edgeStart.y, 
                                     edgeEnd.z - edgeStart.z);
        float edgeLength = distance3f(edgeStart, edgeEnd);
        
        if (edgeLength < epsilon) {
            continue; // Skip degenerate edges
        }
        
        float3 normalizedDir = normalize3f(edgeDir);
        
        // Cast ray along the edge against Mesh2 acceleration structure
        unsigned int hitFlag = 0;
        unsigned int distance = __float_as_uint(edgeLength + epsilon);
        unsigned int triangleIndex = 0;
        
        optixTrace(
            mesh_overlap_params.mesh2_handle,
            edgeStart,
            normalizedDir,
            epsilon,
            edgeLength + epsilon,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,  // SBT offset
            1,  // SBT stride
            0,  // missSBTIndex
            hitFlag, distance, triangleIndex);
        
        // If we got a hit, record the overlap
        if (hitFlag) {
            float t = __uint_as_float(distance);
            // Check if hit is within the edge length
            if (t >= epsilon && t <= edgeLength + epsilon) {
                // Get object ID from Mesh2
                int objectIdMesh2 = mesh_overlap_params.mesh2_triangle_to_object[triangleIndex];
                
                // Record the overlap pair (atomically to avoid race conditions)
                int resultIdx = atomicAdd(mesh_overlap_params.hit_counter, 1);
                if (resultIdx < mesh_overlap_params.max_results) {
                    mesh_overlap_params.results[resultIdx].object_id_mesh1 = objectIdMesh1;
                    mesh_overlap_params.results[resultIdx].object_id_mesh2 = objectIdMesh2;
                }
            }
        }
    }
}

// OptiX Raygen program: Process Mesh2 edges against Mesh1 acceleration structure
// Same as raygen_mesh1_to_mesh2 but with reversed roles
extern "C" __global__ void __raygen__mesh2_to_mesh1() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int triangleIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (triangleIdx >= mesh_overlap_params.mesh1_num_triangles) {
        return;
    }
    
    // Get triangle indices (note: mesh1_* here refers to Mesh2 data due to parameter structure)
    uint3 triIndices = mesh_overlap_params.mesh1_indices[triangleIdx];
    
    // Get triangle vertices
    float3 v0 = mesh_overlap_params.mesh1_vertices[triIndices.x];
    float3 v1 = mesh_overlap_params.mesh1_vertices[triIndices.y];
    float3 v2 = mesh_overlap_params.mesh1_vertices[triIndices.z];
    
    // Get object ID for this triangle (Mesh2)
    int objectIdMesh2 = mesh_overlap_params.mesh1_triangle_to_object[triangleIdx];
    
    // Define the 3 edges of the triangle
    float3 edgeStarts[3] = {v0, v1, v2};
    float3 edgeEnds[3] = {v1, v2, v0};
    
    const float epsilon = 1e-6f;
    
    // Check each edge
    for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx) {
        float3 edgeStart = edgeStarts[edgeIdx];
        float3 edgeEnd = edgeEnds[edgeIdx];
        
        // Compute edge direction and length
        float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x, 
                                     edgeEnd.y - edgeStart.y, 
                                     edgeEnd.z - edgeStart.z);
        float edgeLength = distance3f(edgeStart, edgeEnd);
        
        if (edgeLength < epsilon) {
            continue; // Skip degenerate edges
        }
        
        float3 normalizedDir = normalize3f(edgeDir);
        
        // Cast ray along the edge against Mesh1 acceleration structure
        unsigned int hitFlag = 0;
        unsigned int distance = __float_as_uint(edgeLength + epsilon);
        unsigned int triangleIndex = 0;
        
        optixTrace(
            mesh_overlap_params.mesh2_handle,
            edgeStart,
            normalizedDir,
            epsilon,
            edgeLength + epsilon,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,  // SBT offset
            1,  // SBT stride
            0,  // missSBTIndex
            hitFlag, distance, triangleIndex);
        
        // If we got a hit, record the overlap
        if (hitFlag) {
            float t = __uint_as_float(distance);
            // Check if hit is within the edge length
            if (t >= epsilon && t <= edgeLength + epsilon) {
                // Get object ID from Mesh1
                int objectIdMesh1 = mesh_overlap_params.mesh2_triangle_to_object[triangleIndex];
                
                // Record the overlap pair (atomically to avoid race conditions)
                // Note: order is reversed (mesh2, mesh1) compared to kernel_1
                int resultIdx = atomicAdd(mesh_overlap_params.hit_counter, 1);
                if (resultIdx < mesh_overlap_params.max_results) {
                    mesh_overlap_params.results[resultIdx].object_id_mesh1 = objectIdMesh2;
                    mesh_overlap_params.results[resultIdx].object_id_mesh2 = objectIdMesh1;
                }
            }
        }
    }
}

// Miss shader (reused from raytracing.cu)
extern "C" __global__ void __miss__ms()
{
}

// Anyhit shader (reused from raytracing.cu)
extern "C" __global__ void __anyhit__ah()
{
}

// Closesthit shader (reused from raytracing.cu)
extern "C" __global__ void __closesthit__ch()
{
    const float t = optixGetRayTmax();
    const unsigned int triangleIndex = optixGetPrimitiveIndex();
    
    optixSetPayload_0(1);
    optixSetPayload_1(__float_as_uint(t));
    optixSetPayload_2(triangleIndex);
}

