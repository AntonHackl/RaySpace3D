#pragma once

#include <vector>
#include <map>
#include <set>
#include <cuda_runtime.h>
#include "Geometry.h"  // For float3, uint3 types

// Pinned memory allocator (forward declaration)
template<typename T>
struct PinnedAllocator;

// GPU-compatible structure for edge data (flattened version)
struct EdgeMeshData {
    float3* d_edge_starts;           // Start vertex of each edge (GPU)
    float3* d_edge_ends;             // End vertex of each edge (GPU)
    int* d_source_objects;           // Flattened array of source object IDs (GPU)
    int* d_source_object_offsets;    // Offset into d_source_objects for each edge (GPU)
    int* d_source_object_counts;     // Count of source objects per edge (GPU)
    int num_edges;
    
    // Constructor
    EdgeMeshData() : d_edge_starts(nullptr), d_edge_ends(nullptr), 
                   d_source_objects(nullptr), d_source_object_offsets(nullptr),
                   d_source_object_counts(nullptr), num_edges(0) {}
};

class EdgeExtractor {
public:
    // Extract unique edges from a mesh and prepare GPU data (supports PinnedAllocator)
    template<
        typename IndexAllocatorT,
        typename ObjectAllocatorT,
        typename VertexAllocatorT
    >
    static EdgeMeshData extractAndPrepareEdges(
        const std::vector<uint3, IndexAllocatorT>& indices,
        const std::vector<int, ObjectAllocatorT>& triangleToObject,
        const std::vector<float3, VertexAllocatorT>& vertices);
    
    // Free GPU memory allocated in EdgeMeshData
    static void freeEdgeData(EdgeMeshData& edgeData);
};

// Template implementation
#include "../optix/OptixHelpers.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>

template<typename IndexAllocatorT, typename ObjectAllocatorT, typename VertexAllocatorT>
inline EdgeMeshData EdgeExtractor::extractAndPrepareEdges(
    const std::vector<uint3, IndexAllocatorT>& indices,
    const std::vector<int, ObjectAllocatorT>& triangleToObject,
    const std::vector<float3, VertexAllocatorT>& vertices) {
    
    // Structure to track edge info during extraction
    struct EdgeInfoInternal {
        float3 p0, p1;
        std::vector<int> sourceObjects;
    };

    struct QuantizedPoint {
        std::int64_t x;
        std::int64_t y;
        std::int64_t z;

        bool operator<(const QuantizedPoint& other) const {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
    };

    struct EdgeKey {
        QuantizedPoint a;
        QuantizedPoint b;

        bool operator<(const EdgeKey& other) const {
            if (a < other.a) return true;
            if (other.a < a) return false;
            return b < other.b;
        }
    };

    const double quantScale = 1e5;
    auto quantizePoint = [&](const float3& p) -> QuantizedPoint {
        return {
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.x) * quantScale)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.y) * quantScale)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.z) * quantScale))
        };
    };
    
    // Map to store unique edges: normalized geometric edge -> EdgeInfoInternal
    std::map<EdgeKey, EdgeInfoInternal> uniqueEdgesMap;
    
    // Process each triangle to extract edges
    for (size_t triIdx = 0; triIdx < indices.size(); ++triIdx) {
        uint3 tri = indices[triIdx];
        int sourceObject = triangleToObject[triIdx];
        
        // Extract 3 edges from the triangle
        unsigned int verts[3] = {tri.x, tri.y, tri.z};
        
        for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx) {
            const unsigned int v0 = verts[edgeIdx];
            const unsigned int v1 = verts[(edgeIdx + 1) % 3];

            const float3 p0 = vertices[v0];
            const float3 p1 = vertices[v1];
            QuantizedPoint q0 = quantizePoint(p0);
            QuantizedPoint q1 = quantizePoint(p1);

            // Normalize edge representation (always (min, max) for undirected comparison)
            if (q1 < q0) {
                std::swap(q0, q1);
            }

            EdgeKey edgeKey{q0, q1};
            
            // Create or update EdgeInfoInternal for this edge
            if (uniqueEdgesMap.find(edgeKey) == uniqueEdgesMap.end()) {
                EdgeInfoInternal newEdge;
                newEdge.p0 = p0;
                newEdge.p1 = p1;
                newEdge.sourceObjects.push_back(sourceObject);
                uniqueEdgesMap[edgeKey] = newEdge;
            } else {
                // Add source object ID if not already present (handles multiple triangles from same object)
                EdgeInfoInternal& edge = uniqueEdgesMap[edgeKey];
                auto it = std::find(edge.sourceObjects.begin(), 
                                   edge.sourceObjects.end(), 
                                   sourceObject);
                if (it == edge.sourceObjects.end()) {
                    edge.sourceObjects.push_back(sourceObject);
                }
            }
        }
    }
    
    // Convert to GPU format with flattened arrays
    EdgeMeshData result;
    result.num_edges = uniqueEdgesMap.size();
    
    if (result.num_edges == 0) {
        return result;
    }
    
    // Allocate GPU memory
    std::vector<float3> h_edge_starts;
    std::vector<float3> h_edge_ends;
    std::vector<int> h_source_objects;
    std::vector<int> h_offsets;
    std::vector<int> h_counts;
    
    h_edge_starts.reserve(result.num_edges);
    h_edge_ends.reserve(result.num_edges);
    h_offsets.reserve(result.num_edges);
    h_counts.reserve(result.num_edges);
    
    // Build flattened arrays
    for (auto& pair : uniqueEdgesMap) {
        const EdgeInfoInternal& edge = pair.second;
        
        h_edge_starts.push_back(edge.p0);
        h_edge_ends.push_back(edge.p1);
        h_offsets.push_back(static_cast<int>(h_source_objects.size()));
        h_counts.push_back(static_cast<int>(edge.sourceObjects.size()));
        
        // Add source object IDs to flattened array
        for (int objId : edge.sourceObjects) {
            h_source_objects.push_back(objId);
        }
    }
    
    // Copy to GPU
    size_t edge_starts_bytes = result.num_edges * sizeof(float3);
    size_t edge_ends_bytes = result.num_edges * sizeof(float3);
    size_t source_objects_bytes = h_source_objects.size() * sizeof(int);
    size_t offsets_bytes = result.num_edges * sizeof(int);
    size_t counts_bytes = result.num_edges * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&result.d_edge_starts, edge_starts_bytes));
    CUDA_CHECK(cudaMalloc(&result.d_edge_ends, edge_ends_bytes));
    CUDA_CHECK(cudaMalloc(&result.d_source_objects, source_objects_bytes));
    CUDA_CHECK(cudaMalloc(&result.d_source_object_offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc(&result.d_source_object_counts, counts_bytes));
    
    CUDA_CHECK(cudaMemcpy(result.d_edge_starts, h_edge_starts.data(), edge_starts_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_edge_ends, h_edge_ends.data(), edge_ends_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_source_objects, h_source_objects.data(), source_objects_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_source_object_offsets, h_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_source_object_counts, h_counts.data(), counts_bytes, cudaMemcpyHostToDevice));
    
    const double totalTriangleEdges = static_cast<double>(indices.size()) * 3.0;
    const double reductionPct = (totalTriangleEdges > 0.0)
        ? (100.0 * (1.0 - static_cast<double>(result.num_edges) / totalTriangleEdges))
        : 0.0;

    std::cout << "Edge extraction: " << result.num_edges << " unique edges from "
              << indices.size() << " triangles (reduction: "
              << std::fixed << std::setprecision(1) 
              << reductionPct << "%)" << std::endl;
    
    return result;
}
