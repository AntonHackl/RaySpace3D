#include "EdgeExtractor.h"
#include "../optix/OptixHelpers.h"

void EdgeExtractor::freeEdgeData(EdgeMeshData& edgeData) {
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
