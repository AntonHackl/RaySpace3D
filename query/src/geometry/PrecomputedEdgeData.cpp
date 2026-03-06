#include "PrecomputedEdgeData.h"

#include "../optix/OptixHelpers.h"

EdgeMeshData PrecomputedEdgeData::uploadFromGeometry(const GeometryData& geometry) {
    EdgeMeshData result;

    if (!geometry.edges.hasEdges()) {
        return result;
    }

    result.num_edges = static_cast<int>(geometry.edges.edgeStarts.size());
    const size_t edge_bytes = geometry.edges.edgeStarts.size() * sizeof(float3);
    const size_t source_objects_bytes = geometry.edges.sourceObjects.size() * sizeof(int);
    const size_t offsets_bytes = geometry.edges.sourceObjectOffsets.size() * sizeof(int);
    const size_t counts_bytes = geometry.edges.sourceObjectCounts.size() * sizeof(int);

    CUDA_CHECK(cudaMalloc(&result.d_edge_starts, edge_bytes));
    CUDA_CHECK(cudaMalloc(&result.d_edge_ends, edge_bytes));
    CUDA_CHECK(cudaMalloc(&result.d_source_object_offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc(&result.d_source_object_counts, counts_bytes));
    if (source_objects_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&result.d_source_objects, source_objects_bytes));
    }

    CUDA_CHECK(cudaMemcpy(result.d_edge_starts, geometry.edges.edgeStarts.data(), edge_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_edge_ends, geometry.edges.edgeEnds.data(), edge_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_source_object_offsets, geometry.edges.sourceObjectOffsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_source_object_counts, geometry.edges.sourceObjectCounts.data(), counts_bytes, cudaMemcpyHostToDevice));
    if (source_objects_bytes > 0) {
        CUDA_CHECK(cudaMemcpy(result.d_source_objects, geometry.edges.sourceObjects.data(), source_objects_bytes, cudaMemcpyHostToDevice));
    }

    return result;
}

void PrecomputedEdgeData::freeEdgeData(EdgeMeshData& edgeData) {
    if (edgeData.d_edge_starts) {
        CUDA_CHECK(cudaFree(edgeData.d_edge_starts));
        edgeData.d_edge_starts = nullptr;
    }
    if (edgeData.d_edge_ends) {
        CUDA_CHECK(cudaFree(edgeData.d_edge_ends));
        edgeData.d_edge_ends = nullptr;
    }
    if (edgeData.d_source_objects) {
        CUDA_CHECK(cudaFree(edgeData.d_source_objects));
        edgeData.d_source_objects = nullptr;
    }
    if (edgeData.d_source_object_offsets) {
        CUDA_CHECK(cudaFree(edgeData.d_source_object_offsets));
        edgeData.d_source_object_offsets = nullptr;
    }
    if (edgeData.d_source_object_counts) {
        CUDA_CHECK(cudaFree(edgeData.d_source_object_counts));
        edgeData.d_source_object_counts = nullptr;
    }
    edgeData.num_edges = 0;
}
